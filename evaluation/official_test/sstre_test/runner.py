"""Exactly seven locked checkpoints; official data access requires explicit approval."""
import time
from pathlib import Path
import numpy as np
import torch
from .core import require,load_bundle,source_path,load_verified_checkpoint,predict,classification_metrics,file_info,FINGERPRINTS,MODEL_LOCK_SHA
from .data import load_official_test
from .publication import Publisher

def run_official(bundle,input_mount,result_mount,token,bundle_sha,commit,read_file,*,authorized=False):
    if authorized is not True:raise PermissionError('Official held-out test execution is not approved')
    lock,rules=load_bundle(bundle,bundle_sha)
    require(torch.cuda.is_available(),'Approved T4 GPU unavailable; no CPU fallback')
    require('T4' in torch.cuda.get_device_name(),'Unexpected GPU allocation')
    started=time.monotonic();torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    # The token is generated once locally, so platform restarts cannot reopen a
    # partial test output directory and silently repeat test evaluation.
    output=Publisher(result_mount,token,commit,read_file)
    models={};preflight={};paths={}
    for selected in lock['selected_models']:
        path=source_path(result_mount,selected['checkpoint_path'])
        model,proof=load_verified_checkpoint(path,selected,rules['training_protocol_sha256'],FINGERPRINTS[selected['dataset']])
        models[selected['condition']]=model;preflight[selected['condition']]=proof;paths[selected['condition']]=path
    require(len(models)==7,'Missing locked model preflight')
    # Every checkpoint must pass before ANY official test image/label is decoded.
    output.json('RUN_CONFIG.json',{'token':token,'model_lock_sha256':MODEL_LOCK_SHA,'bundle_sha256':bundle_sha,
        'selected_models':lock['selected_models'],'rules':rules,'all_seven_checkpoint_preflights':preflight,
        'model_selection_uses_test':False,'training':False,'torch':str(torch.__version__),'device':torch.cuda.get_device_name()})
    data={};rows=[]
    for selected in lock['selected_models']:
        condition=selected['condition'];dataset=selected['dataset'];classes=selected['classes']
        if dataset not in data:
            data[dataset]=load_official_test(input_mount,lock,rules,dataset,authorized=True)
        images,targets,receipt=data[dataset]
        model=models[condition]
        probabilities,inference=predict(model,images,classes,rules,'cuda',rules['batch_size'],rules['num_workers'])
        predictions=probabilities.argmax(1);metrics=classification_metrics(targets,predictions,classes,rules['groups'][dataset])
        require(file_info(paths[condition])=={'bytes':selected['checkpoint_bytes'],'sha256':selected['checkpoint_sha256']},'Checkpoint file changed during evaluation')
        row={'condition':condition,'dataset':dataset,'selected_epoch':selected['selected_epoch'],'run_id':selected['run_id'],
            'checkpoint_sha256':selected['checkpoint_sha256'],'test_data':receipt,'inference':inference,'test_metrics':metrics}
        output.npz('predictions_'+condition+'.npz',test_ids=np.arange(len(images),dtype=np.int64),targets=targets,
                   probabilities=probabilities,predictions=predictions.astype(np.int64))
        output.json('metrics_'+condition+'.json',row);rows.append(row);model.to('cpu')
    require({r['condition'] for r in rows}=={m['condition'] for m in lock['selected_models']} and len(rows)==7,'Incomplete seven-model evaluation')
    result={'status':'completed','token':token,'output_directory':output.remote,'model_lock_sha256':MODEL_LOCK_SHA,
        'bundle_sha256':bundle_sha,'models_evaluated':7,'results':rows,'runtime_seconds':time.monotonic()-started,
        'test_based_selection':False,'test_based_tuning':False,'training_started':False,'retry':False,'next_job_started':False,
        'actual_cost_usd':None}
    output.json('RESULT.json',result);publication=output.seal()
    return {'status':'completed','output_directory':output.remote,'models_evaluated':7,'model_lock_sha256':MODEL_LOCK_SHA,
        'runtime_seconds':result['runtime_seconds'],'publication':publication,'results':rows,'training_started':False,'test_based_selection':False}
