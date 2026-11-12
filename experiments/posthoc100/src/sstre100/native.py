"""Native PyTorch paths. No datasets or cloud activity on import."""
import hashlib, os, random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from torchvision import transforms
from .reliability import normalized_features, validate_neighbors

def seed_all(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
def make_model(classes):
    if classes not in (10,100):raise ValueError('Only approved class counts')
    m=resnet18(weights=None)
    m.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
    m.maxpool=nn.Identity();m.fc=nn.Linear(512,classes)
    return m

def optimizer_scheduler(model,p):
    o=p['optimizer'];s=p['scheduler']
    optimizer=torch.optim.SGD(model.parameters(),lr=o['lr'],momentum=o['momentum'],weight_decay=o['weight_decay'])
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=s['milestones'],gamma=s['gamma'])
    return optimizer,scheduler

def transform(p,training=False):
    steps=[transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip()] if training else []
    return transforms.Compose(steps+[transforms.ToTensor(),transforms.Normalize(p['preprocessing']['mean'],p['preprocessing']['std'])])

def state_hash(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():
        a=t.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def optimizer_step(model,optimizer,x,observed,weights):
    if observed.dtype!=torch.int64 or weights.requires_grad or weights.shape!=observed.shape:raise ValueError('Detached per-sample weights required')
    if not torch.isfinite(weights).all() or not ((weights>=.25-1e-6)&(weights<=4+1e-6)).all():raise ValueError('Invalid bounded weights')
    model.train();optimizer.zero_grad(set_to_none=True)
    logits=model(x);individual=F.cross_entropy(logits,observed,reduction='none')
    loss=F.cross_entropy(logits,observed) if torch.equal(weights,torch.ones_like(weights)) else (individual*weights).mean()
    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
    loss.backward()
    sq=0.
    for parameter in model.parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():raise ValueError('Nonfinite gradient before optimizer step')
            sq+=float(parameter.grad.detach().float().square().sum())
    if sq<=0 or not np.isfinite(sq):raise ValueError('Missing/nonfinite gradient norm')
    optimizer.step()
    return {'loss':float(loss.detach()),'unweighted_ce':float(individual.detach().mean()),'gradient_norm':sq**.5}

@torch.inference_mode()
def infer(model,loader,n,classes,device,features=False):
    model.eval();before=state_hash(model)
    probs=np.empty((n,classes),np.float32);seen=np.zeros(n,np.int8)
    embeddings=np.empty((n,512),np.float32) if features else None
    for x,_,ids in loader:
        idx=ids.numpy()
        if len(np.unique(idx))!=len(idx) or (idx<0).any() or (idx>=n).any() or seen[idx].any():raise ValueError('Inference local-ID mapping')
        x=x.to(device)
        if features:
            z=model.maxpool(model.relu(model.bn1(model.conv1(x))))
            z=model.layer4(model.layer3(model.layer2(model.layer1(z))))
            z=torch.flatten(model.avgpool(z),1);logits=model.fc(z)
            embeddings[idx]=z.cpu().numpy()
        else:logits=model(x)
        probs[idx]=torch.softmax(logits,dim=1).cpu().numpy();seen[idx]=1
    if not seen.all() or state_hash(model)!=before:raise ValueError('Incomplete inference or model/buffer mutation')
    return probs,embeddings,before

@torch.inference_mode()
def knn(features,k=10,query_batch=128,device='cpu'):
    normalized_features(features)
    f=F.normalize(torch.from_numpy(np.asarray(features,np.float32)),p=2,dim=1)
    bank=f.to(device);n=len(bank)
    if not 0<k<n or query_batch<1:raise ValueError('K/query batch')
    ids=np.empty((n,k),np.int64);sim=np.empty((n,k),np.float32)
    for start in range(0,n,query_batch):
        stop=min(n,start+query_batch);scores=bank[start:stop]@bank.T
        scores[torch.arange(stop-start,device=device),torch.arange(start,stop,device=device)]=-torch.inf
        order=torch.argsort(scores,dim=1,descending=True,stable=True)[:,:k]
        ids[start:stop]=order.cpu().numpy();sim[start:stop]=scores.gather(1,order).cpu().numpy()
    validate_neighbors(ids,n)
    if not np.isfinite(sim).all():raise ValueError('Nonfinite neighbor similarities')
    return ids,sim
