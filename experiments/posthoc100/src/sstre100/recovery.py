"""Explicit, fail-closed weighted recovery; no automatic resume/submission.

Only a separately approved, SHA-pinned plan can select a previous post-hoc run.
All parent reads are restricted to that run. No parent writes or replacements.
"""
import hashlib, io, json, random, re
from pathlib import Path
import numpy as np
import torch
from .checkpoint import SCHEMA, MODEL_SCHEMA, restore_rng
from .native import state_hash
from .extension import PREFIX, TOKEN, condition, validate_run_id, uses_n, FINGERPRINTS

FIELDS=('epoch','learning_rate','train_weighted_loss','train_augmented_unweighted_ce',
 'deterministic_noisy_train_ce','validation_ce','validation','sample_weight_summary',
 'coefficients','N_refresh','N_bank_epoch','data_order_sha256','augmentation_sha256',
 'observed_counts','gradient_norm','analysis_state_invariant','model_sha256',
 'accuracy_drop_flag','accuracy_drop_action')

def demand(ok,message):
    if not bool(ok): raise ValueError(message)
def info(raw): return {'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
def json_equal(a,b): return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)

def validate_plan(plan,run_key):
    r=condition(run_key)
    demand(set(plan)=={'schema','run_key','parent_run_id','parent_directory','checkpoint_epoch',
        'last_durable_epoch','verified_marker_hashes'},'Recovery plan fields')
    demand(plan['schema']=='posthoc100_recovery_v1' and plan['run_key']==run_key,'Recovery plan identity')
    rid=plan['parent_run_id'];match=re.fullmatch(r'(.+)_seed(42|43|44)_('+TOKEN+')',rid)
    demand(match is not None and rid==validate_run_id(r,match[3]),'Recovery parent identity')
    demand(plan['parent_directory']==f"/{PREFIX}/{r['dataset']}/{rid}",'Recovery parent path')
    e,last=plan['checkpoint_epoch'],plan['last_durable_epoch']
    demand(type(e) is int and type(last) is int and 0<=e<=last<100 and last-e<=9,'Recovery epoch range')
    demand(e in (0,9,10,11,20,30,40,50,60,70,80,90),'Not a retained full checkpoint')
    demand(e==max(x for x in (0,9,10,11,20,30,40,50,60,70,80,90) if x<=last),'Use latest durably verified full checkpoint')
    demand(set(plan['verified_marker_hashes'])=={str(x) for x in range(last+1)},'Missing durable parent markers')
    for item in plan['verified_marker_hashes'].values():
        demand(set(item)=={'sha256','bytes'} and re.fullmatch('[0-9a-f]{64}',item['sha256']) and item['bytes']>0,'Invalid pinned marker')
    return r

class ParentReader:
    def __init__(self,read_file,plan,run_key):
        validate_plan(plan,run_key);self.plan=plan;self.read_file=read_file;self.stages={};self.rows={}
    def raw(self,name,expected):
        demand(re.fullmatch(r'(?:provenance|expected_epoch_\d{3}|verified_epoch_\d{3}|metrics_epoch_\d{3})\.json|(?:checkpoint|best_model)_epoch_\d{3}\.pt|(?:trajectory|validation)_epoch_\d{3}\.npz',name),'Forbidden parent read')
        size=0;blocks=[];cap=140_000_000 if name.endswith('.pt') else 20_000_000
        for block in self.read_file(self.plan['parent_directory']+'/'+name):
            size+=len(block);demand(size<=cap,'Parent file too large');blocks.append(block)
        raw=b''.join(blocks);demand(info(raw)==expected,'Parent SHA/size mismatch: '+name);return raw
    def stage(self,e):
        if e not in self.stages:
            tag=f'epoch_{e:03d}'
            marker=json.loads(self.raw('verified_'+tag+'.json',self.plan['verified_marker_hashes'][str(e)]))
            expected=json.loads(self.raw('expected_'+tag+'.json',marker['expected_manifest']))
            demand(marker['stage']==expected['stage']==tag,'Marker epoch')
            demand(expected['remote_directory']==self.plan['parent_directory'],'Marker parent')
            demand(marker['verified_files']==expected['expected_files'],'Manifest disagreement')
            self.stages[e]=marker['verified_files']
        return self.stages[e]
    def verified(self,e,name):
        expected=self.stage(e).get(name);demand(expected is not None,'Unverified parent artifact: '+name)
        return self.raw(name,expected)
    def row(self,e):
        if e not in self.rows:
            self.rows[e]=json.loads(self.verified(e,f'metrics_epoch_{e:03d}.json'))
            demand(self.rows[e]['epoch']==e,'Metric epoch mismatch')
        return self.rows[e]
    def arrays(self,e,kind):
        demand(kind in ('validation','trajectory'),'Parent array kind')
        with np.load(io.BytesIO(self.verified(e,f'{kind}_epoch_{e:03d}.npz')),allow_pickle=False) as z:
            return {k:z[k].copy() for k in z.files}
    def full_checkpoint(self,path):
        e=self.plan['checkpoint_epoch'];raw=self.verified(e,f'checkpoint_epoch_{e:03d}.pt')
        with Path(path).open('xb') as f:f.write(raw)
        return info(raw)

def restore_full(path,model,optimizer,scheduler,signals,identity,p,previous_row,*,train_samples,cuda_required=True):
    """No optimizer step. CUDA branch is mandatory in the GPU runner."""
    v=torch.load(path,map_location='cpu',weights_only=True)
    demand(v['schema']==SCHEMA and v['identity']==identity,'Full-state checkpoint identity')
    e=v['epoch'];demand(type(e) is int and 0<=e<100,'Checkpoint epoch')
    demand(v['steps']==e*((train_samples+127)//128),'Global optimizer-step counter')
    demand(identity['seed']==p['seed'] and identity['fingerprint']==FINGERPRINTS[identity['dataset']],'Seed/split identity')
    order=v['order_state'];demand(order['next_epoch']==e+1 and order['policy']==p['seed_rules'],'Data-order policy')
    if e:
        for saved,key in [('last_order_sha256','data_order_sha256'),('last_augmentation_sha256','augmentation_sha256')]:
            demand(order[saved]==previous_row[key],'Order/augmentation evidence mismatch')
        g=torch.Generator();g.set_state(order['training_loader_generator'])
        demand(torch.equal(g.get_state(),order['training_loader_generator']),'Loader RNG state invalid')
    rng=v['rng']
    if cuda_required:
        demand(torch.cuda.is_available() and torch.cuda.device_count()==1 and len(rng['cuda'])==1,'Required CUDA RNG missing')
    model.load_state_dict(v['model'],strict=True)
    demand(state_hash(model)==v['model_hash'],'Model hash mismatch')
    if e:demand(v['model_hash']==previous_row['model_sha256'],'Previous model trajectory mismatch')
    optimizer.load_state_dict(v['optimizer']);scheduler.load_state_dict(v['scheduler']);signals.load_state_dict(v['signals'])
    demand(scheduler.last_epoch==e,'Scheduler epoch')
    demand(dict(scheduler.milestones)=={60:1,85:1} and scheduler.gamma==.1 and scheduler.base_lrs==[.1],'Scheduler protocol changed')
    lr=.1*.1**sum(x<=e for x in [60,85])
    demand(all(abs(group['lr']-lr)<1e-12 and group['momentum']==.9 and group['weight_decay']==.0005
        and group['dampening']==0 and not group['nesterov'] and not group['maximize'] for group in optimizer.param_groups),'Optimizer settings changed')
    demand(scheduler.get_last_lr()==[group['lr'] for group in optimizer.param_groups],'Scheduler/optimizer LR disagreement')
    params=[x for group in optimizer.param_groups for x in group['params']]
    demand(not e or len(optimizer.state)==len(params),'Momentum state missing')
    for x in params:
        if not e:break
        m=optimizer.state[x].get('momentum_buffer')
        demand(m is not None and m.shape==x.shape and torch.isfinite(m).all(),'Momentum buffer invalid')
    demand(signals.epochs==list(range(max(1,e-5),e+1)) if e else not signals.epochs,'Signal ring missing')
    if e:
        demand(signals.latest is not None and signals.latest.shape==(train_samples,signals.classes),'Latest predictions missing')
        demand(np.array_equal(signals.argmax[-1],signals.latest.argmax(1)),'Latest prediction/ring disagreement')
    if uses_n(identity['method']) and e>=30:
        bank=max(x for x in p['signals']['neighborhood_refresh_completed_epochs'] if x<=e)
        demand(signals.bank_epoch==bank and signals.N is not None and signals.N_vote is not None,'Required neighborhood state missing/stale')
    else:demand(signals.bank_epoch is None and signals.N is None and signals.N_vote is None,'Inactive/premature neighborhood state')
    restore_rng(rng if cuda_required else {**rng,'cuda':[]})
    demand(torch.equal(torch.get_rng_state(),rng['torch'].cpu()) and random.getstate()==rng['python'],'CPU/Python RNG readback')
    n=rng['numpy'];actual=np.random.get_state()
    demand(actual[0]==n['name'] and np.array_equal(actual[1],np.asarray(n['keys'],np.uint32)) and actual[2:]==(n['pos'],n['has_gauss'],n['cached_gaussian']),'NumPy RNG readback')
    if cuda_required:demand(torch.equal(torch.cuda.get_rng_state(0).cpu(),rng['cuda'][0].cpu()),'CUDA RNG readback failed')
    demand(v['best']['epoch'] is None if not e else 1<=v['best']['epoch']<=e,'Previous best state invalid')
    return v,{'status':'PASS','restored_epoch':e,'next_epoch':e+1,'finish_epoch':100,
        'cuda_rng':'PASS' if cuda_required else 'NOT_TESTED_CPU_ONLY','weights_only':True}

def validate_previous_best(raw,model,identity,best,parent_row):
    v=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    demand(v['schema'] in (SCHEMA,MODEL_SCHEMA) and v['identity']==identity,'Previous-best identity')
    demand(v['epoch']==best['epoch']==parent_row['epoch'] and best['macro_f1']==parent_row['validation']['macro_f1'],'Previous-best selection')
    model.load_state_dict(v['model'],strict=True)
    demand(state_hash(model)==v['model_hash']==parent_row['model_sha256'],'Previous-best model hash')
    return v

def compare_replay(epoch,row,trajectory,validation,parent_row,parent_trajectory,parent_validation):
    differences=[key for key in FIELDS if key not in row or key not in parent_row or not json_equal(row[key],parent_row[key])]
    for kind,new,old in [('trajectory',trajectory,parent_trajectory),('validation',validation,parent_validation)]:
        if set(new)!=set(old):differences.append(kind+'.keys')
        for key in set(new)&set(old):
            a,b=np.asarray(new[key]),np.asarray(old[key])
            if a.dtype!=b.dtype or a.shape!=b.shape or not np.array_equal(a,b):differences.append(kind+'.'+key)
    return {'epoch':epoch,'status':'FAIL' if differences else 'PASS','differences':sorted(differences),'criterion':'EXACT_NO_RELAXATION'}

class ReplayGate:
    def __init__(self,restored,last):self.restored=restored;self.last=last;self.passed=[]
    def before_epoch(self,e):
        demand(self.restored<e<=100,'Resume outside original budget')
        demand(self.passed==list(range(self.restored+1,min(e,self.last+1))),'Replay incomplete: optimizer step forbidden')
    def accept(self,report):
        demand(report['status']=='PASS','Replay divergence: '+json.dumps(report))
        demand(report['epoch']==self.restored+1+len(self.passed) and report['epoch']<=self.last,'Replay order')
        self.passed.append(report['epoch'])
