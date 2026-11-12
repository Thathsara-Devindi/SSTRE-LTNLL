"""One approved post-hoc condition; optional explicit verified-parent recovery."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,time,tempfile,io,copy
import numpy as np
import torch
from .protocol import load,sha256,BUNDLE,ROOT
from .native import seed_all,make_model,optimizer_scheduler,optimizer_step,infer,knn,state_hash,transform
from .data import load_frozen,TrainImages,ValidationImages,loader
from .signals import SignalState
from .checkpoint import save
from .retention import POLICY,policy_identity,retain_epoch
from .store import MountedRunStore
from .extension import load_condition,uses_n,PREFIX,validate_run_id,verify_data_receipt,verify_bundle
from .recovery import ParentReader,restore_full,validate_previous_best,ReplayGate,compare_replay,demand
from .checkpoint import save_selected_model

def summary(x):
    a=np.asarray(x)
    return {'min':float(a.min()),'mean':float(a.astype(np.float64).mean()),'max':float(a.max()),
            'quantiles':np.quantile(a,[0,.25,.5,.75,1]).tolist()}

def native_preflight(p,classes,method,device):
    seed_all(p['seed']);m=make_model(classes).to(device);o,_=optimizer_scheduler(m,p)
    x=torch.randn(2,3,32,32,device=device);y=torch.tensor([0,classes-1],device=device)
    before=state_hash(m)
    result=optimizer_step(m,o,x,y,torch.tensor([.5,1.5],device=device))
    if state_hash(m)==before:raise ValueError('Native optimizer preflight did not update model')
    result.update({'status':'PASS','classes':classes,'method':method,'synthetic_only':True})
    del m,o,x,y
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    return result

def train_one(input_root,output_root,run_key,token,commit,read_file,*,execute=False,resume=None):
    if not execute:raise PermissionError('Separate cloud execution approval required')
    bundle_hash=verify_bundle()
    p,d,run,tools=load_condition(run_key)
    dataset,method=run['dataset'],run['method'];run_id=validate_run_id(run,token)
    writer_policy=json.loads((ROOT/'CHECKPOINT_WRITER_V2.json').read_text(encoding='utf-8'))
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('One approved T4 GPU required')
    if 'T4' not in torch.cuda.get_device_name(0):raise ValueError('Unexpected GPU model')
    device=torch.device('cuda:0');started=time.monotonic()
    pixels,train,val,data_receipt=load_frozen(input_root,d,tools)
    verify_data_receipt(run,data_receipt)
    preflight=native_preflight(p,d['num_classes'],method,device)
    seed_all(p['seed']);model=make_model(d['num_classes']).to(device)
    optimizer,scheduler=optimizer_scheduler(model,p)
    signals=SignalState(train['observed_labels'],d['num_classes'])
    tx=TrainImages(pixels,train,transform(p,True));analysis=TrainImages(pixels,train,transform(p))
    validation=ValidationImages(pixels,val,transform(p))
    store=MountedRunStore(output_root,{'dataset':dataset,'output_parent':PREFIX,'run_id':run_id,'run_key':run_key},commit,read_file)
    identity={'protocol_sha256':sha256(BUNDLE/'protocol.json'),'dataset':dataset,'method':method,
              'fingerprint':data_receipt['fingerprint'],'run_id':run_id,'seed':p['seed'],'run_key':run_key,
              'post_hoc':True,'dataset_split_seed':42,'extension_bundle_sha256':bundle_hash,'checkpoint_retention':policy_identity(),
              'checkpoint_writer':{'id':writer_policy['id'],'sha256':sha256(ROOT/'CHECKPOINT_WRITER_V2.json')}}
    init_hash=state_hash(model);best={'epoch':None,'macro_f1':-1.,'metrics':None};steps=0;rows=[]
    provenance={'identity':identity,'reviewed_run':run,'data':data_receipt,'initialization_sha256':init_hash,
                'torch':str(torch.__version__),'cuda':str(torch.version.cuda),'device':torch.cuda.get_device_name(),
                'precision':p['precision'],'native_preflight':preflight,'test_set_enabled':False,
                'checkpoint_retention':POLICY,'checkpoint_writer':writer_policy,'resume_implemented_for_execution':True,
                'effective_protocol':p,'frozen_split_seed':42,'post_hoc':True,'parent':resume}
    start_epoch=1;parent=None;gate=None;restoration=None
    if resume is not None:
        parent=ParentReader(read_file,resume,run_key);restored=resume['checkpoint_epoch']
        old=json.loads(parent.verified(0,'provenance.json'));old_id=old['identity']
        for key in ('protocol_sha256','dataset','method','fingerprint','seed','run_key','extension_bundle_sha256','checkpoint_retention','checkpoint_writer'):
            demand(old_id[key]==identity[key],'Parent provenance differs: '+key)
        demand(old_id['run_id']==resume['parent_run_id'],'Wrong parent run')
        demand(old['effective_protocol']==p and not old['test_set_enabled'],'Parent protocol differs')
        demand(old['torch']==str(torch.__version__) and old['cuda']==str(torch.version.cuda)
            and old['device']==torch.cuda.get_device_name(),'Recovery runtime differs')
        rows=[copy.deepcopy(parent.row(e)) for e in range(1,restored+1)]
        with tempfile.TemporaryDirectory(prefix='sstre-posthoc-restore-') as td:
            cp=Path(td)/'checkpoint.pt';checkpoint_info=parent.full_checkpoint(cp)
            value,restoration=restore_full(cp,model,optimizer,scheduler,signals,old_id,p,
                rows[-1] if rows else None,train_samples=len(tx),cuda_required=True)
            best=copy.deepcopy(value['best']);steps=value['steps']
            if restored:
                winner=max(rows,key=lambda row:row['validation']['macro_f1'])
                demand(best['epoch']==winner['epoch'] and best['macro_f1']==winner['validation']['macro_f1'],'Parent best rule differs')
                bm=make_model(d['num_classes'])
                raw=parent.verified(best['epoch'],best['checkpoint'])
                validate_previous_best(raw,bm,old_id,best,parent.row(best['epoch']))
                selected=f"best_model_epoch_{best['epoch']:03d}.pt";best['checkpoint']=selected
                save_selected_model(store.path(selected),bm,epoch=best['epoch'],best=best,identity=identity)
                del bm
                # Best-model construction consumes RNG; restore again and verify before training.
                value,restoration=restore_full(cp,model,optimizer,scheduler,signals,old_id,p,rows[-1],train_samples=len(tx),cuda_required=True)
            order_state=value['order_state']
        start_epoch=restored+1;gate=ReplayGate(restored,resume['last_durable_epoch'])
        provenance['initialization_sha256']=old['initialization_sha256']
        provenance['restoration']={**restoration,'parent_checkpoint':checkpoint_info,'parent_best_model_verified':bool(restored)}
    else:order_state={'next_epoch':1,'policy':p['seed_rules']}
    store.json('provenance.json',provenance)
    initial_epoch=start_epoch-1
    save(store.path(f'checkpoint_epoch_{initial_epoch:03d}.pt'),model,optimizer,scheduler,signals,epoch=initial_epoch,steps=steps,best=best,identity=identity,order_state=order_state)
    store.seal(f'epoch_{initial_epoch:03d}')
    try:
        for epoch in range(start_epoch,p['epochs']+1):
            if gate is not None:gate.before_epoch(epoch)
            epoch_started=time.monotonic();seed_all(p['seed']+1009*epoch)
            weights,diagnostics=signals.weights(p,tools,method,epoch)
            if method=='ce' or epoch<=p['warmup_epochs']:
                if not np.array_equal(weights,np.ones(len(tx),np.float32)):raise ValueError('Warmup/CE must have unit weights')
            if not np.isfinite(weights).all() or not (weights>0).all():raise ValueError('Nonpositive sample weight')
            lr=float(optimizer.param_groups[0]['lr'])
            if abs(lr-tools.lr_for_epoch(p,epoch))>1e-12:raise ValueError('LR protocol mismatch')
            training_loader=loader(tx,p,epoch,'training',device)
            order_hash=hashlib.sha256();augmentation_hash=hashlib.sha256();seen=np.zeros(len(tx),np.int8)
            observed_counts=np.zeros(d['num_classes'],np.int64);batch_counts=[];loss_sum=0.;raw_sum=0.;grad_norms=[]
            for x,y,idx in training_loader:
                indices=idx.numpy()
                if seen[indices].any() or len(np.unique(indices))!=len(indices):raise ValueError('Duplicate train sample')
                if not np.array_equal(y.numpy(),train['observed_labels'][indices]):raise ValueError('Training label alignment changed')
                seen[indices]=1;order_hash.update(indices.astype('<i8').tobytes());augmentation_hash.update(x.numpy().tobytes())
                counts=np.bincount(y.numpy(),minlength=d['num_classes']);observed_counts+=counts;batch_counts.append(counts)
                result=optimizer_step(model,optimizer,x.to(device),y.to(device),torch.from_numpy(weights[indices]).to(device))
                loss_sum+=result['loss']*len(y);raw_sum+=result['unweighted_ce']*len(y);grad_norms.append(result['gradient_norm']);steps+=1
            if not seen.all():raise ValueError('Incomplete epoch data order')
            refresh=uses_n(method) and epoch in p['signals']['neighborhood_refresh_completed_epochs']
            probs,features,analysis_hash=infer(model,loader(analysis,p,epoch,'analysis',device),len(tx),d['num_classes'],device,refresh)
            neighbors=similarities=None
            if refresh:neighbors,similarities=knn(features,p['signals']['neighbor_k'],p['signals']['query_batch_size'],device)
            signals.complete(epoch,probs,neighbors)
            vp,_,validation_hash=infer(model,loader(validation,p,epoch,'validation',device),len(validation),d['num_classes'],device)
            if validation_hash!=analysis_hash:raise ValueError('Analysis changed model state')
            m=tools.metrics(val['targets'],vp.argmax(1),d['num_classes'],d['evaluation_groups'])
            cm=np.array(m['confusion_matrix']);m['per_class_recall']=np.divide(np.diag(cm),cm.sum(1),out=np.zeros(d['num_classes']),where=cm.sum(1)>0).tolist()
            val_ce=float(-np.log(np.maximum(vp[np.arange(len(vp)),val['targets']],np.finfo(np.float32).tiny)).mean(dtype=np.float64))
            train_ce=float(-np.log(np.maximum(probs[np.arange(len(probs)),train['observed_labels']],np.finfo(np.float32).tiny)).mean(dtype=np.float64))
            previous_accuracy=rows[-1]['validation']['accuracy'] if rows else None
            drop=previous_accuracy is not None and previous_accuracy-m['accuracy']>=p['monitoring']['drop_flag_accuracy_pp']/100
            row={'epoch':epoch,'learning_rate':lr,'train_weighted_loss':loss_sum/len(tx),'train_augmented_unweighted_ce':raw_sum/len(tx),
                 'deterministic_noisy_train_ce':train_ce,'validation_ce':val_ce,'validation':m,'sample_weight_summary':summary(weights),
                 'coefficients':diagnostics.get('coefficients',np.zeros(3)).tolist(),'N_refresh':refresh,'N_bank_epoch':signals.bank_epoch,
                 'data_order_sha256':order_hash.hexdigest(),'augmentation_sha256':augmentation_hash.hexdigest(),
                 'observed_counts':observed_counts.tolist(),'gradient_norm':summary(grad_norms),'analysis_state_invariant':True,
                 'model_sha256':analysis_hash,'accuracy_drop_flag':drop,'accuracy_drop_action':'LOG_ONLY',
                 'epoch_seconds_before_publication':time.monotonic()-epoch_started}
            if gate is not None and epoch<=resume['last_durable_epoch']:
                trajectory={'original_ids':train['original_ids'],'observed_labels':train['observed_labels'],'probabilities':probs,
                    'batch_observed_counts':np.array(batch_counts),'weights':weights,**{k:v for k,v in diagnostics.items() if k!='weights'}}
                validation_arrays={'original_ids':val['original_ids'],'targets':val['targets'],'probabilities':vp,'predictions':vp.argmax(1)}
                replay=compare_replay(epoch,row,trajectory,validation_arrays,parent.row(epoch),parent.arrays(epoch,'trajectory'),parent.arrays(epoch,'validation'))
                store.json(f'replay_epoch_{epoch:03d}.json',replay)
                gate.accept(replay)  # Any divergence raises before further optimizer steps.
            is_best=m['macro_f1']>best['macro_f1']
            if is_best:best={'epoch':epoch,'macro_f1':m['macro_f1'],'metrics':m}
            scheduler.step()
            tag=f'epoch_{epoch:03d}'
            store.npz('validation_'+tag+'.npz',original_ids=val['original_ids'],targets=val['targets'],probabilities=vp,predictions=vp.argmax(1))
            store.npz('trajectory_'+tag+'.npz',original_ids=train['original_ids'],observed_labels=train['observed_labels'],probabilities=probs,
                      batch_observed_counts=np.array(batch_counts),weights=weights,**{k:v for k,v in diagnostics.items() if k!='weights'})
            if refresh:store.npz('neighbors_'+tag+'.npz',neighbors=neighbors,similarities=similarities,original_ids=train['original_ids'])
            best,checkpoint_record=retain_epoch(store,model,optimizer,scheduler,signals,epoch=epoch,steps=steps,best=best,identity=identity,is_best=is_best,
                 order_state={'next_epoch':epoch+1,'policy':p['seed_rules'],'last_order_sha256':order_hash.hexdigest(),
                              'last_augmentation_sha256':augmentation_hash.hexdigest(),'training_loader_generator':training_loader.generator.get_state()})
            row['checkpoint']=checkpoint_record
            store.json('metrics_'+tag+'.json',row);store.seal(tag)
            row['epoch_seconds_including_publication']=time.monotonic()-epoch_started;rows.append(row)
            print(json.dumps({'status':'epoch_verified','run_id':run_id,'epoch':epoch,'validation':m,'seconds':row['epoch_seconds_including_publication']}),flush=True)
        report={'status':'completed','identity':identity,'completed_epochs':100,'best':best,'final':rows[-1]['validation'],
                'execution_start_epoch':start_epoch,'epochs_executed_this_attempt':101-start_epoch,
                'parent_lineage':resume,'replay_passed':gate.passed if gate is not None else [],
                'epochs':rows,'runtime_seconds':time.monotonic()-started,'remote_directory':store.remote,'verified_checkpoint_count':sum(name.endswith('.pt') for name in store.verified),
                'verified_full_recovery_count':sum(name.startswith('checkpoint_epoch_') and name.endswith('.pt') for name in store.verified),
                'verified_model_only_best_count':sum(name.startswith('best_model_epoch_') and name.endswith('.pt') for name in store.verified),
                'best_checkpoint_verification':store.verified[best['checkpoint']],
                'test_evaluated':False,'retry':False,'next_method_started':False,'actual_cost_usd':None}
        store.json('RESULT.json',report);store.seal('completion');return report
    except Exception as exc:
        # A verification failure blocks all further writes/publication. Preserve its original traceback.
        if not store.blocked:
            try:
                store.json('FAILURE.json',{'type':type(exc).__name__,'error':str(exc),'completed_epochs':len(rows),'retry':False})
                store.seal('failure')
            except Exception as evidence_error:
                exc.add_note('Secondary evidence preservation failed: '+str(evidence_error))
        raise
