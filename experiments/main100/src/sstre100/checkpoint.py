"""Exclusive safe checkpoints. No unpickling fallback or automatic recovery."""
from pathlib import Path
import random
import numpy as np
import torch
from .native import state_hash
from .atomic import atomic_torch_save
SCHEMA='main100_v2_rolling6_adapter01'
def rng_state():
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':{'name':n[0],'keys':n[1].astype(np.int64).tolist(),
            'pos':int(n[2]),'has_gauss':int(n[3]),'cached_gaussian':float(n[4])},
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
def restore_rng(s):
    random.setstate(s['python']);n=s['numpy']
    np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['pos'],n['has_gauss'],n['cached_gaussian']))
    torch.set_rng_state(s['torch'].cpu())
    if s['cuda']:
        if not torch.cuda.is_available() or len(s['cuda'])!=torch.cuda.device_count():raise ValueError('CUDA RNG cannot be restored')
        torch.cuda.set_rng_state_all(s['cuda'])
def save(path,model,optimizer,scheduler,signals,*,epoch,steps,best,identity,order_state):
    value={'schema':SCHEMA,'model':model.state_dict(),'model_hash':state_hash(model),
        'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'signals':signals.state_dict(),
        'epoch':int(epoch),'steps':int(steps),'best':best,'identity':identity,'rng':rng_state(),'order_state':order_state}
    atomic_torch_save(value,path)
    return value['model_hash']
def restore(path,model,optimizer,scheduler,signals,expected_identity):
    value=torch.load(path,map_location='cpu',weights_only=True)
    if value['schema']!=SCHEMA or value['identity']!=expected_identity:raise ValueError('Checkpoint provenance mismatch')
    if not 0<=value['epoch']<=100 or value['steps']<0:raise ValueError('Checkpoint counters')
    model.load_state_dict(value['model'],strict=True)
    if state_hash(model)!=value['model_hash']:raise ValueError('Restored model checksum')
    optimizer.load_state_dict(value['optimizer']);scheduler.load_state_dict(value['scheduler'])
    signals.load_state_dict(value['signals'])
    if signals.epochs and signals.epochs[-1]!=value['epoch']:raise ValueError('Signal epoch mismatch')
    if scheduler.last_epoch!=value['epoch']:raise ValueError('Scheduler epoch mismatch')
    restore_rng(value['rng'])
    return value

MODEL_SCHEMA='main100_retention_v1_model_only'
def save_selected_model(path,model,*,epoch,best,identity):
    payload={'schema':MODEL_SCHEMA,'model':model.state_dict(),'model_hash':state_hash(model),
             'epoch':int(epoch),'best':best,'identity':identity,'recovery_capable':False}
    atomic_torch_save(payload,path)
    return payload['model_hash']
def load_selected_model(path,model,expected_identity):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['schema'] not in (SCHEMA,MODEL_SCHEMA) or payload['identity']!=expected_identity:raise ValueError('Selected model provenance mismatch')
    model.load_state_dict(payload['model'],strict=True)
    if state_hash(model)!=payload['model_hash']:raise ValueError('Selected model hash mismatch')
    return {'epoch':payload['epoch'],'best':payload['best'],'recovery_capable':payload['schema']==SCHEMA}
