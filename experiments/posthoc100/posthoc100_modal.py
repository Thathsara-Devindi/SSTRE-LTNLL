"""One explicitly approved post-hoc condition. Local startup uses stdlib only.

Defining lazy Modal objects is not a connection. No work on direct invocation.
Cloud submission, claim creation, and result writes each require future approval.
"""
from pathlib import Path
import hashlib,json,sys,traceback
from datetime import datetime,timezone
import uuid
import os
import modal

def bundle_root(local,script):
    return Path(script).resolve().parent if local else Path('/app')

ROOT=bundle_root(modal.is_local(),__file__)
MANIFEST=json.loads((ROOT/'POSTHOC_SOURCE_MANIFEST.json').read_text())
for name,expected in MANIFEST.items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Post-hoc source/config hash mismatch: '+name)
REGISTER=json.loads((ROOT/'POSTHOC_REGISTER.json').read_text())
RUNTIME=json.loads((ROOT/'reviewed_protocol/protocol.json').read_text())['image']

app=modal.App('sstre-phase6-posthoc100-v1')
resource_settings = {name: os.environ[name] for name in ('SSTRE_INPUT_VOLUME', 'SSTRE_OUTPUT_VOLUME', 'SSTRE_MODAL_ENVIRONMENT', 'SSTRE_CLAIMS_DICT')}
environment = resource_settings['SSTRE_MODAL_ENVIRONMENT']
inputs=modal.Volume.from_name(resource_settings['SSTRE_INPUT_VOLUME'],environment_name=environment,create_if_missing=False)
outputs=modal.Volume.from_name(resource_settings['SSTRE_OUTPUT_VOLUME'],environment_name=environment,create_if_missing=False)
# Server-atomic conditional claims, not a mounted-filesystem lock. Never reset/delete.
claims=modal.Dict.from_name(resource_settings['SSTRE_CLAIMS_DICT'],environment_name=environment,create_if_missing=True)
image=(modal.Image.debian_slim(python_version=RUNTIME['python']).pip_install(*RUNTIME['packages'])
    .pip_install('torch=='+RUNTIME['torch'],'torchvision=='+RUNTIME['torchvision'],index_url=RUNTIME['torch_index_url'])
    .env({**resource_settings,'PYTHONPATH':'/app/src','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONDONTWRITEBYTECODE':'1'}))
for name in ['src','reviewed_protocol']:
    image=image.add_local_dir(ROOT/name,remote_path='/app/'+name,copy=True)
for name in [*{Path(k).parts[0] for k in MANIFEST if '/' not in k},'POSTHOC_SOURCE_MANIFEST.json']:
    image=image.add_local_file(ROOT/name,remote_path='/app/'+name,copy=True)

@app.function(image=image,volumes={'/research':inputs.with_mount_options(read_only=True),'/results':outputs},
    gpu='T4',cpu=(2.,2.),memory=(8192,8192),timeout=10800,startup_timeout=180,retries=0,
    max_containers=1,single_use_containers=True,scaledown_window=2)
def train(run_key:str,token:str,resume:dict|None=None):
    from sstre100.extension import acquire_claim,persist_claim
    from sstre100.training import train_one
    from sstre100.recovery import validate_plan
    try:
        if resume is not None:validate_plan(resume,run_key)
        claim=acquire_claim(claims.put,run_key,token,resume=resume)
        persist_claim('/results',claim,token,outputs.commit,outputs.read_file)
        return train_one('/research/LTNLL_Research','/results',run_key,token,outputs.commit,outputs.read_file,
            execute=True,resume=resume)
    except Exception as exc:
        traceback.print_exc()
        return {'status':'FAIL','run_key':run_key,'error_type':type(exc).__name__,'error':str(exc),
                'retry':False,'next_method_started':False}

@app.local_entrypoint()
def main(run_key:str='',confirm_cloud_execution:bool=False,confirm_training:bool=False,
         confirm_result_writes:bool=False,confirm_claim_write:bool=False,resume_plan_file:str=''):
    if not all([confirm_cloud_execution,confirm_training,confirm_result_writes,confirm_claim_write]):
        raise PermissionError('Separate approval for ONE job, permanent duplicate claim, and new result writes required')
    if run_key not in {r['run_key'] for r in REGISTER['runs']}:raise ValueError('Not one of thirteen reviewed conditions')
    resume=None
    if resume_plan_file:
        path=Path(resume_plan_file)
        if not path.is_file() or path.stat().st_size>200000:raise ValueError('Invalid recovery plan file')
        resume=json.loads(path.read_text(encoding='utf-8'))
        if resume.get('run_key')!=run_key:raise ValueError('Recovery plan condition mismatch')
    token=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    # Token is generated ONCE locally, not on container startup/replacement.
    print(json.dumps({'run_key':run_key,'launch_token':token,'post_hoc':True,'official_test':False}),flush=True)
    report=train.remote(run_key,token,resume)
    print(json.dumps(report,indent=2,allow_nan=False))
    if report.get('status')!='completed':raise RuntimeError('Run failed. No automatic retry or next experiment.')
    return report

if __name__=='__main__':print('Preparation only. Use a separately approved modal run command.')
