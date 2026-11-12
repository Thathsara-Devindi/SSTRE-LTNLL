"""One tiny synthetic checkpoint on ONE new directory. No training/datasets."""
from pathlib import Path
import random
import numpy as np
import torch
from . import checkpoint
from .native import state_hash
from .signals import SignalState
from .store import MountedRunStore,file_info
SYNTHETIC_REMOTE='/SSTRE_Checkpoint_Compatibility_V2/writer_v2_20261026_01'
class SyntheticStore(MountedRunStore):
    """Reuse the unchanged publisher, restricting this test to its exact prefix."""
    def __init__(self,mount,commit,read_file):
        mount=Path(mount).resolve();root=mount/SYNTHETIC_REMOTE.lstrip('/')
        if not root.resolve().is_relative_to(mount) or root.parent.is_symlink():raise ValueError('Unsafe synthetic directory')
        root.parent.mkdir(parents=True,exist_ok=True);root.mkdir(exist_ok=False)
        self.root=root;self.remote=SYNTHETIC_REMOTE;self.commit=commit;self.read_file=read_file
        self.blocked=False;self.verified={};self.stats={};self.markers={}
def fixture():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    model=torch.nn.Linear(3,10)
    optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,weight_decay=.0005)
    # Synthetic momentum fields exercise serialization; NO optimizer step occurs.
    for parameter in model.parameters():optimizer.state[parameter]['momentum_buffer']=torch.zeros_like(parameter)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[60,85],gamma=.1)
    signals=SignalState(np.arange(12,dtype=np.int64)%10,10)
    return model,optimizer,scheduler,signals

def run_compatibility(mount,commit,read_file,*,approved=False):
    if not approved:raise PermissionError('Separate CPU cloud-test approval required')
    if torch.cuda.is_available():raise RuntimeError('This compatibility test must be CPU-only')
    store=SyntheticStore(mount,commit,read_file)
    model,optimizer,scheduler,signals=fixture()
    identity={'synthetic_checkpoint_writer_v2':True,'directory':SYNTHETIC_REMOTE}
    best={'epoch':None,'macro_f1':-1.};name='checkpoint_epoch_000.pt'
    try:
        checkpoint.save(store.path(name),model,optimizer,scheduler,signals,epoch=0,steps=0,best=best,identity=identity,order_state={'next_epoch':1,'synthetic':True})
    except Exception:
        store.blocked=True
        raise
    store.seal('epoch_000')
    if name not in store.verified:raise RuntimeError('Checkpoint not independently verified; do not load')
    before=file_info(store.path(name))
    try:
        checkpoint.save(store.path(name),model,optimizer,scheduler,signals,epoch=0,steps=0,best=best,identity=identity,order_state={})
    except FileExistsError:no_overwrite=True
    else:raise RuntimeError('Exclusive checkpoint creation failed')
    if file_info(store.path(name))!=before:raise RuntimeError('No-overwrite probe changed the existing synthetic file')
    store._remote_verify(name,before)
    clone,o2,s2,h2=fixture()
    restored=checkpoint.restore(store.path(name),clone,o2,s2,h2,identity)
    if state_hash(model)!=state_hash(clone) or restored['epoch']!=0 or s2.last_epoch!=0:raise RuntimeError('Safe checkpoint roundtrip failed')
    if len(o2.state)!=len(optimizer.state):raise RuntimeError('Synthetic optimizer state missing')
    if len(store.verified)!=3:raise RuntimeError('Unexpected synthetic file count')
    return {'status':'PASS','synthetic_only':True,'remote_directory':SYNTHETIC_REMOTE,
            'files':store.verified,'safe_weights_only_load':'PASS','no_overwrite':'PASS',
            'cpu_only':True,'optimizer_steps':0,'training_launched':False,'datasets_accessed':False,
            'retries':0,'cleanup':False}
