"""Pure native inference and metrics. No dataset/cloud access on import."""
import hashlib,io,json
from pathlib import Path,PurePosixPath
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image

MODEL_LOCK_SHA='e5f62e9e024986df640a23bfdef880000d97ead4609afa35d7be1ae515d7d12b'
FINGERPRINTS={'cifar10n':'9a8f859c871027638440913cdf4678e3febc924951c2ee0b3c01a5a61dc17839',
              'cifar100n':'0a9f072cee264e5bfbe715eda22252afaebc009f6104fb8f4e7847ad8e597fc2'}
CONDITIONS={'cifar10n_ce','cifar10n_fixed','cifar10n_adaptive','cifar10n_c_only','cifar100n_ce','cifar100n_fixed','cifar100n_adaptive'}
SCHEMAS={'main100_retention_v1_model_only','main100_v2_rolling6_adapter01'}

def require(value,message):
    if not value:raise ValueError(message)
def file_info(path):
    h=hashlib.sha256();size=0
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block);size+=len(block)
    return {'sha256':h.hexdigest(),'bytes':size}
def source_path(root,remote):
    rel=PurePosixPath(remote)
    require(rel.is_absolute() and '..' not in rel.parts and '\\' not in remote,'Unsafe source path')
    root=Path(root).resolve();target=root.joinpath(*rel.parts[1:])
    require(target.resolve().is_relative_to(root) and not target.is_symlink(),'Source escapes mount')
    return target
def validate_lock(raw):
    require(hashlib.sha256(raw).hexdigest()==MODEL_LOCK_SHA,'Frozen model-lock SHA mismatch')
    lock=json.loads(raw);models=lock['selected_models']
    require(lock['status']=='FROZEN_BEFORE_OFFICIAL_TEST' and lock['official_test_evaluated'] is False,'Unexpected model lock')
    require(len(models)==7 and {m['condition'] for m in models}==CONDITIONS,'Not the seven predeclared models')
    for m in models:
        require(m['selection_metric']=='validation_macro_f1' and 1<=m['selected_epoch']<=100,'Non-validation selection')
        require(m['classes']==(10 if m['dataset']=='cifar10n' else 100),'Class count mapping')
        require(m['condition'].startswith(m['dataset']+'_') and m['checkpoint_path'].startswith('/Phase6_Main100_V2/'+m['dataset']+'/'+m['run_id']+'/'),'Checkpoint identity/path mapping')
    return lock
def load_bundle(root,expected_bundle_sha):
    root=Path(root);raw=(root/'BUNDLE_MANIFEST.json').read_bytes()
    require(len(expected_bundle_sha)==64 and hashlib.sha256(raw).hexdigest()==expected_bundle_sha,'Evaluator bundle lock mismatch')
    entries=json.loads(raw)
    for name,expected in entries.items():
        path=root/name
        require(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink(),'Bundle path')
        require(file_info(path)['sha256']==expected,'Evaluator source/config changed: '+name)
    lock=validate_lock((root/'LOCKED_MODELS.json').read_bytes())
    rules=json.loads((root/'EVALUATION_RULES.json').read_text());p=json.loads((root/'TRAINING_PROTOCOL.json').read_text())
    require(rules['model_lock_sha256']==MODEL_LOCK_SHA and file_info(root/'TRAINING_PROTOCOL.json')['sha256']==rules['training_protocol_sha256'],'Training protocol reference')
    require(rules['preprocessing']['mean']==p['preprocessing']['mean'] and rules['preprocessing']['std']==p['preprocessing']['std'],'Changed normalization')
    require(rules['preprocessing']['operations']==p['preprocessing']['evaluation']==['ToTensor','Normalize'],'Changed evaluation transform')
    return lock,rules
def build_model(classes):
    require(classes in (10,100),'Unsupported classifier')
    model=resnet18(weights=None)
    model.conv1=nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
    model.maxpool=nn.Identity();model.fc=nn.Linear(512,classes)
    return model
def state_hash(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        a=tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()
def load_verified_checkpoint(path,selected,protocol_sha,fingerprint):
    # Hash and deserialize the SAME bytes, preventing a hash/load file race.
    with Path(path).open('rb') as f:raw=f.read(selected['checkpoint_bytes']+1)
    require(len(raw)==selected['checkpoint_bytes'] and hashlib.sha256(raw).hexdigest()==selected['checkpoint_sha256'],'Checkpoint SHA/size mismatch')
    value=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    require(isinstance(value,dict) and value.get('schema') in SCHEMAS,'Unapproved checkpoint schema')
    identity=value['identity'];method=selected['condition'][len(selected['dataset'])+1:]
    expected={'dataset':selected['dataset'],'method':method,'run_id':selected['run_id'],'seed':42,
              'protocol_sha256':protocol_sha,'fingerprint':fingerprint}
    require(all(identity.get(k)==v for k,v in expected.items()),'Checkpoint training provenance mismatch')
    require(value['epoch']==value['best']['epoch']==selected['selected_epoch'],'Checkpoint was not the frozen selected epoch')
    require(all(not t.is_floating_point() or (t.dtype==torch.float32 and torch.isfinite(t).all()) for t in value['model'].values()),'Unexpected parameter precision/values')
    model=build_model(selected['classes']);model.load_state_dict(value['model'],strict=True)
    require(state_hash(model)==value['model_hash'],'Checkpoint model-state hash mismatch')
    model.eval();model.requires_grad_(False)
    return model,{'epoch':value['epoch'],'model_sha256':value['model_hash'],'checkpoint_sha256':selected['checkpoint_sha256'],'weights_only':True,'strict':True}
def make_transform(rules):
    p=rules['preprocessing'];require(p['operations']==['ToTensor','Normalize'],'Random/changed evaluation transform')
    require(len(p['mean'])==len(p['std'])==3 and all(np.isfinite(p['mean'])) and all(np.isfinite(p['std'])) and min(p['std'])>0,'Normalization values')
    return transforms.Compose([transforms.ToTensor(),transforms.Normalize(p['mean'],p['std'])])

class Images(Dataset):
    def __init__(self,images,transform):self.images=images;self.transform=transform
    def __len__(self):return len(self.images)
    def __getitem__(self,i):return self.transform(Image.fromarray(self.images[i])),i

@torch.inference_mode()
def predict(model,images,classes,rules,device='cpu',batch_size=256,num_workers=0):
    require(images.dtype==np.uint8 and images.ndim==4 and images.shape[1:]==(32,32,3) and len(images)>0,'Inference images')
    require(classes in (10,100) and model.fc.out_features==classes,'Prediction class count')
    model.eval();model.requires_grad_(False);before=state_hash(model)
    require(all(not m.training for m in model.modules()) and all(p.grad is None for p in model.parameters()),'Evaluation state/gradient contamination')
    model.to(device);n=len(images);probabilities=np.empty((n,classes),np.float32);seen=np.zeros(n,bool)
    loader=DataLoader(Images(images,make_transform(rules)),batch_size=batch_size,shuffle=False,num_workers=num_workers,drop_last=False)
    for x,ids in loader:
        idx=ids.numpy();require(not seen[idx].any() and len(np.unique(idx))==len(idx),'Duplicate test ID')
        logits=model(x.to(device));require(logits.shape==(len(idx),classes) and torch.isfinite(logits).all(),'Invalid prediction dimensions/values')
        probabilities[idx]=torch.softmax(logits.float(),dim=1).cpu().numpy();seen[idx]=True
    require(seen.all() and state_hash(model)==before,'Incomplete predictions or model/buffer mutation')
    require(np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1),1,rtol=0,atol=1e-5),'Invalid probabilities')
    return probabilities,{'model_state_before':before,'model_state_after':before,'evaluation_mode':True,'grad_enabled':False,'inference_mode':True,'sample_count':n}
def classification_metrics(targets,predictions,classes,groups):
    y=np.asarray(targets);pred=np.asarray(predictions)
    require(y.ndim==1 and y.shape==pred.shape and len(y)>0 and y.dtype.kind in 'iu' and pred.dtype.kind in 'iu','Metric labels')
    require(((y>=0)&(y<classes)&(pred>=0)&(pred<classes)).all(),'Metric label range')
    require(sorted(c for cs in groups.values() for c in cs)==list(range(classes)),'Evaluation group partition')
    cm=np.bincount(y*classes+pred,minlength=classes*classes).reshape(classes,classes)
    tp=np.diag(cm);support=cm.sum(1);predicted=cm.sum(0)
    recall=np.divide(tp,support,out=np.zeros(classes,float),where=support>0)
    precision=np.divide(tp,predicted,out=np.zeros(classes,float),where=predicted>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(classes,float),where=precision+recall>0)
    result={'accuracy':float(tp.sum()/len(y)),'balanced_accuracy':float(recall[support>0].mean()),
            'macro_precision':float(precision.mean()),'macro_recall':float(recall.mean()),'macro_f1':float(f1.mean()),
            'confusion_matrix':cm.tolist(),'per_class_recall':recall.tolist(),'per_class_precision':precision.tolist(),'per_class_f1':f1.tolist(),'per_class_support':support.tolist()}
    for name,cs in groups.items():result[name+'_accuracy']=float(tp[cs].sum()/support[cs].sum()) if support[cs].sum() else None
    return result
