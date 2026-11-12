"""Protocol mathematics/data contracts only. No training, network, Modal, downloads."""
from pathlib import Path
from fractions import Fraction
import hashlib,json,re
import numpy as np

ROOT=Path(__file__).resolve().parent
MATRIX={('cifar10n','ce'),('cifar10n','fixed'),('cifar10n','adaptive'),('cifar10n','c_only'),('cifar100n','ce'),('cifar100n','fixed'),('cifar100n','adaptive')}

def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def fractions(rows):
    return tuple(Fraction(n,d) for n,d in rows)

def lr_for_epoch(p,e):
    if type(e) is not int or not 1<=e<=p['epochs']:raise ValueError('Invalid epoch')
    return p['optimizer']['lr']*p['scheduler']['gamma']**sum(m<e for m in p['scheduler']['milestones'])

def coefficients(p,method,e):
    lr_for_epoch(p,e)
    if method not in {'ce','fixed','adaptive','c_only'}:raise ValueError('Unapproved method')
    if method=='ce' or e<=p['warmup_epochs']:return None
    if method=='fixed':return fractions(p['fusion']['fixed_fractions'])
    if method=='c_only':return fractions(p['fusion']['c_only_fractions'])
    stages=[s for s in p['fusion']['stages'] if s['first_epoch']<=e<=s['last_epoch']]
    if len(stages)!=1:raise ValueError('Stage partition')
    return fractions(stages[0]['fractions'])

def evidence_epochs(p,e):
    lr_for_epoch(p,e)
    if e<=p['warmup_epochs']:return None
    bank=max(t for t in p['signals']['neighborhood_refresh_completed_epochs'] if t<e)
    return {'C':e-1,'T':list(range(e-6,e-1)),'N':bank}

def temporal_from_ring(history_epochs,predicted_labels,observed,loss_epoch):
    epochs=np.asarray(history_epochs)
    pred=np.asarray(predicted_labels);y=np.asarray(observed)
    if not np.array_equal(epochs,np.arange(loss_epoch-6,loss_epoch)):
        raise ValueError('History must be last6 completed epochs; exclude confidence anchor')
    if pred.shape!=(6,len(y)):raise ValueError('History shape')
    return (pred[:5]==y[None,:]).mean(axis=0).astype(np.float32)

def fuse(C,T,N,a):
    arrays=[np.asarray(x,dtype=np.float64) for x in [C,T,N]]
    if any(x.ndim!=1 or x.shape!=arrays[0].shape or not len(x) or not np.isfinite(x).all() or not ((x>=0)&(x<=1)).all() for x in arrays):
        raise ValueError('Invalid signals')
    a=np.asarray(a,dtype=np.float64)
    if a.shape!=(3,) or not np.isfinite(a).all() or (a<0).any() or abs(a.sum()-1)>1e-12:raise ValueError('Coefficients')
    return np.clip(sum(v*x for v,x in zip(a,arrays)),0,1).astype(np.float32)

def positive_weights(R):
    r=np.asarray(R,dtype=np.float64)
    if r.ndim!=1 or not len(r) or not np.isfinite(r).all() or not ((r>=0)&(r<=1)).all():raise ValueError('Invalid score')
    q=.25+.75*r
    return (q/q.mean()).astype(np.float32)

def validate_train_view(view,classes):
    if set(view)!={'original_ids','observed_labels'}:raise ValueError('Training view must exclude audit/clean/validation arrays')
    ids=np.asarray(view['original_ids']);y=np.asarray(view['observed_labels'])
    if ids.ndim!=1 or y.shape!=ids.shape or ids.dtype.kind not in 'iu' or y.dtype.kind not in 'iu':raise ValueError('IDs/labels')
    if len(set(ids.tolist()))!=len(ids) or not ((ids>=0)&(ids<50000)).all() or not ((y>=0)&(y<classes)).all():raise ValueError('Bounds or duplicate IDs')
    return True

def build_cifar100_contract_fixture(source_clean,source_observed,contract):
    """Pure frozen-split constructor, tested on synthetic arrays only in this task.
    REAL source provenance/annotation key must be verified before future export.
    Clean source labels are used ONLY here for controlled construction/validation.
    """
    clean=np.asarray(source_clean);observed=np.asarray(source_observed)
    if clean.shape!=(50000,) or observed.shape!=(50000,) or clean.dtype.kind not in 'iu' or observed.dtype.kind not in 'iu':raise ValueError('Original train source arrays')
    if not ((clean>=0)&(clean<100)).all() or not ((observed>=0)&(observed<100)).all():raise ValueError('Not100 fine classes')
    if not np.array_equal(np.bincount(clean,minlength=100),np.full(100,500)):raise ValueError('Source class count')
    if not np.array_equal(np.unique(observed),np.arange(100)):raise ValueError('Official fine-label source must cover100 classes; reject coarse20 labels')
    rng=np.random.Generator(np.random.PCG64(contract['split_seed']));lt=[];val=[]
    for c,n in enumerate(contract['expected_source_class_counts']):
        shuffled=rng.permutation(np.flatnonzero(clean==c))
        val.append(shuffled[:50]);lt.append(shuffled[50:50+n])
    lt=rng.permutation(np.concatenate(lt)).astype('<i8');val=rng.permutation(np.concatenate(val)).astype('<i8')
    y=observed[lt].astype('<i8')
    h=hashlib.sha256(b'cifar100n_fine_IR10_seed42_v1\0')
    for x in [lt,val,y]:h.update(x.tobytes())
    train={'original_ids':lt,'observed_labels':y}
    validate_train_view(train,100)
    if np.intersect1d(lt,val).size or len(val)!=5000 or len(lt)!=17614:raise ValueError('Split contamination/count')
    return {'train':train,'validation':{'original_ids':val,'targets':clean[val].copy()},
            'audit_only':{'clean_labels':clean[lt].copy(),'is_noisy':(clean[lt]!=y)},
            'fingerprint':h.hexdigest(),'numpy_version':np.__version__}

def metrics(y,pred,classes,groups):
    y=np.asarray(y);pred=np.asarray(pred)
    if y.shape!=pred.shape or y.ndim!=1 or not len(y) or y.dtype.kind not in 'iu' or pred.dtype.kind not in 'iu':raise ValueError('Metric inputs')
    if not ((y>=0)&(y<classes)&(pred>=0)&(pred<classes)).all():raise ValueError('Class range')
    cm=np.bincount(y.astype(np.int64)*classes+pred,minlength=classes*classes).reshape(classes,classes)
    tp=np.diag(cm).astype(float)
    recall=np.divide(tp,cm.sum(1),out=np.zeros(classes),where=cm.sum(1)>0)
    precision=np.divide(tp,cm.sum(0),out=np.zeros(classes),where=cm.sum(0)>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(classes),where=(precision+recall)>0)
    result={'accuracy':float(tp.sum()/len(y)),'balanced_accuracy':float(recall[cm.sum(1)>0].mean()),'macro_precision':float(precision.mean()),'macro_recall':float(recall.mean()),'macro_f1':float(f1.mean()),'confusion_matrix':cm.tolist()}
    for name,indices in groups.items():
        total=cm[indices].sum();result[name+'_accuracy']=float(tp[indices].sum()/total) if total else None
    return result

def output_path(mount,dataset,method,seed,token):
    if (dataset,method) not in MATRIX or seed not in [42,43,44] or not re.fullmatch(r'[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}',token):raise ValueError('Unapproved/unsafe output identity')
    base=Path(mount).resolve();p=base/'Phase6_Main100_V2'/dataset/(method+'_seed'+str(seed)+'_'+token)
    if not p.resolve().is_relative_to(base):raise ValueError('Output escaped mount')
    return p

def exclusive_json(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def validate_bundle(p,datasets,runs):
    assert p['epochs']==100 and p['warmup_epochs']==30 and p['seed']==42
    assert p['training_implementation']=='NOT_IMPLEMENTED_IN_THIS_PREPARATION'
    assert p['optimizer']=={'name':'SGD','lr':.1,'momentum':.9,'weight_decay':.0005}
    assert p['scheduler']=={'name':'MultiStepLR','milestones':[60,85],'gamma':.1,'step_timing':'after_epoch'}
    assert p['loader']['batch_size']==128 and p['signals']['temporal_window']==5
    assert p['signals']['neighborhood_refresh_completed_epochs']==list(range(30,100,5))
    assert p['persistence']['rolling_prediction_argmax_epochs']==6
    assert p['persistence']['overwrite'] is False and p['persistence']['automatic_recovery'] is False
    assert p['persistence']['create_output_volume_if_missing'] is False
    assert p['resources_proposed']['retries']==0 and p['resources_proposed']['automatic_next_method'] is False
    assert p['resources_proposed']['cloud_approval_granted'] is False
    assert p['evaluation']['evaluate_official_test'] is False and p['evaluation']['build_test_loader'] is False
    assert p['evaluation']['selection']=='validation_macro_f1' and p['evaluation']['tie_rule']=='strict_improvement_first_tie'
    weighted=[e for s in p['fusion']['stages'] for e in range(s['first_epoch'],s['last_epoch']+1)]
    assert weighted==list(range(31,101))
    mean=tuple(sum((s['last_epoch']-s['first_epoch']+1)*fractions(s['fractions'])[i] for s in p['fusion']['stages'])/70 for i in range(3))
    assert mean==fractions(p['fusion']['fixed_fractions'])==(Fraction(23,70),Fraction(24,70),Fraction(23,70))
    assert set((r['dataset'],r['method']) for r in runs)==MATRIX and len(runs)==7
    allowed={'run_key','execution_order','dataset','method','seed','epochs','protocol_sha256','dataset_contract_sha256','scientific_overrides','output_volume','output_prefix','estimated_minutes','estimated_compute_usd','approval_required','cloud_training_ready'}
    for r in runs:
        assert set(r)==allowed and r['scientific_overrides']=={} and r['seed']==42 and r['epochs']==100
        assert r['cloud_training_ready'] is False and r['approval_required'] is True
        assert r['output_prefix']=='/Phase6_Main100_V2/'+r['dataset']
    for d in datasets.values():
        groups=d['evaluation_groups'];flat=[c for ids in groups.values() for c in ids]
        assert sorted(flat)==list(range(d['num_classes'])) and len(flat)==len(set(flat))
        assert sum(d['expected_source_class_counts'])==d['train_samples'] and d['validation_samples']==5000
    assert datasets['cifar10n']['regenerate_split'] is False
    assert datasets['cifar10n']['split_fingerprint']!=''
    assert datasets['cifar100n']['split_fingerprint'] is None and datasets['cifar100n']['source_annotation_key'] is None
    assert datasets['cifar100n']['train_samples']==17614 and datasets['cifar100n']['num_classes']==100
    for e in range(31,101):
        evidence=evidence_epochs(p,e)
        assert evidence['C']<e and max(evidence['T'])<evidence['C'] and evidence['T']==list(range(e-6,e-1))
        assert 1<=e-evidence['N']<=5
    return True
