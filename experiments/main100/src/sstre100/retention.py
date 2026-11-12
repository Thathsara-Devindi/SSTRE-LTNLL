"""The user-approved retention-only amendment. Never changes optimization."""
from pathlib import Path
import json
from .protocol import ROOT,sha256
from .checkpoint import save,save_selected_model
from .store import file_info
POLICY_PATH=ROOT/'CHECKPOINT_RETENTION_V1.json'
POLICY=json.loads(POLICY_PATH.read_text(encoding='utf-8'))
RECOVERY_EPOCHS=tuple(POLICY['overrides']['full_recovery_epochs'])
if POLICY['id']!='main100_retention_v1_20261026' or RECOVERY_EPOCHS!=(9,10,11,20,30,40,50,60,70,80,90,100):raise ValueError('Unreviewed retention schedule')
def policy_identity():return {'id':POLICY['id'],'sha256':sha256(POLICY_PATH)}
def retain_epoch(store,model,optimizer,scheduler,signals,*,epoch,steps,best,identity,order_state,is_best):
    if type(epoch) is not int or not 1<=epoch<=100:raise ValueError('Retention epoch')
    full=epoch in RECOVERY_EPOCHS
    name=f'checkpoint_epoch_{epoch:03d}.pt' if full else (f'best_model_epoch_{epoch:03d}.pt' if is_best else None)
    best=dict(best)
    if is_best:best['checkpoint']=name
    record={'file':name,'kind':'full_recovery' if full else ('model_only_best' if is_best else None),
            'best_checkpoint':best.get('checkpoint'),
            'verification_record':f'verified_epoch_{epoch:03d}.json' if name else None,
            'valid_only_with_verified_marker':True}
    if name:
        try:
            path=store.path(name)
            if full:save(path,model,optimizer,scheduler,signals,epoch=epoch,steps=steps,best=best,identity=identity,order_state=order_state)
            else:save_selected_model(path,model,epoch=epoch,best=best,identity=identity)
            record.update(file_info(path))
        except Exception:
            # Incomplete local writes must not enter the failure-publication path.
            store.blocked=True
            raise
    return best,record
