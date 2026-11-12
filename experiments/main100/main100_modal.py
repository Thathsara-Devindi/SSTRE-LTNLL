"""Exactly one explicitly approved main100 condition; never downloads data."""
from pathlib import Path
import hashlib,json,traceback
import os
import modal
ROOT=Path(__file__).resolve().parent if modal.is_local() else Path('/app')
for name,expected in json.loads((ROOT/'SOURCE_MANIFEST.json').read_text()).items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Adapter source/config hash changed: '+name)
PROTOCOL=json.loads((ROOT/'reviewed_protocol/protocol.json').read_text())
RUNTIME=PROTOCOL['image']
app=modal.App('sstre-phase6-main100-v2-retention-v1-writer-v2')
resource_settings = {name: os.environ[name] for name in ('SSTRE_INPUT_VOLUME', 'SSTRE_OUTPUT_VOLUME', 'SSTRE_MODAL_ENVIRONMENT')}
environment = resource_settings['SSTRE_MODAL_ENVIRONMENT']
inputs=modal.Volume.from_name(resource_settings['SSTRE_INPUT_VOLUME'],environment_name=environment,create_if_missing=False)
outputs=modal.Volume.from_name(resource_settings['SSTRE_OUTPUT_VOLUME'],environment_name=environment,create_if_missing=False)
image=(modal.Image.debian_slim(python_version=RUNTIME['python']).pip_install(*RUNTIME['packages'])
    .pip_install('torch=='+RUNTIME['torch'],'torchvision=='+RUNTIME['torchvision'],index_url=RUNTIME['torch_index_url'])
    .env({**resource_settings,'PYTHONPATH':'/app/src','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONDONTWRITEBYTECODE':'1'}))
for name in ['src','reviewed_protocol']:image=image.add_local_dir(ROOT/name,remote_path='/app/'+name,copy=True)
for name in ['main100_modal.py','SOURCE_MANIFEST.json','CHECKPOINT_RETENTION_V1.json','CHECKPOINT_WRITER_V2.json']:image=image.add_local_file(ROOT/name,remote_path='/app/'+name,copy=True)
@app.function(image=image,volumes={'/research':inputs.with_mount_options(read_only=True),'/results':outputs},
    gpu='T4',cpu=(2.,2.),memory=(8192,8192),timeout=10800,startup_timeout=180,retries=0,
    max_containers=1,single_use_containers=True,scaledown_window=2)
def train(dataset:str,method:str):
    from sstre100.training import train_one
    try:return train_one('/research/LTNLL_Research','/results',dataset,method,outputs.commit,outputs.read_file,execute=True)
    except Exception as exc:
        traceback.print_exc()
        return {'status':'FAIL','dataset':dataset,'method':method,'error_type':type(exc).__name__,'error':str(exc),'retry':False,'next_method_started':False}
@app.local_entrypoint()
def main(dataset:str='',method:str='',confirm_cloud_execution:bool=False,confirm_training:bool=False,confirm_result_writes:bool=False):
    if not all([confirm_cloud_execution,confirm_training,confirm_result_writes]):raise PermissionError('Separate approval for exactly one GPU run and new output files required')
    allowed={('cifar10n','ce'),('cifar10n','fixed'),('cifar10n','adaptive'),('cifar10n','c_only'),('cifar100n','ce'),('cifar100n','fixed'),('cifar100n','adaptive')}
    if (dataset,method) not in allowed:raise ValueError('Not one of the seven reviewed conditions')
    report=train.remote(dataset,method)
    print(json.dumps(report,indent=2,allow_nan=False))
    if report.get('status')!='completed':raise RuntimeError('Run failed; no automatic retry or next experiment')
    return report
if __name__=='__main__':print('Prepared only. No execution on direct invocation.')
