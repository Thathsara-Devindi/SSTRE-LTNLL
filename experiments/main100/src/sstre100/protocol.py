"""Verify the unchanged reviewed bundle before exposing any run."""
from pathlib import Path
import hashlib, importlib.util, json
ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / 'reviewed_protocol'
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def load(dataset,method):
    lock=read(BUNDLE/'CONFIG_LOCK.json')
    for name,expected in lock.items():
        if sha256(BUNDLE/name)!=expected:raise ValueError('Reviewed configuration changed: '+name)
    spec=importlib.util.spec_from_file_location('reviewed_protocol_tools',BUNDLE/'protocol_tools.py')
    tools=importlib.util.module_from_spec(spec);spec.loader.exec_module(tools)
    p=read(BUNDLE/'protocol.json')
    datasets={k:read(BUNDLE/'datasets'/(k+'.json')) for k in ['cifar10n','cifar100n']}
    runs=[read(f) for f in sorted((BUNDLE/'runs').glob('*.json'))]
    tools.validate_bundle(p,datasets,runs)
    matches=[r for r in runs if (r['dataset'],r['method'])==(dataset,method)]
    if len(matches)!=1:raise ValueError('Not one of the seven approved configurations')
    run=matches[0]
    if run['protocol_sha256']!=sha256(BUNDLE/'protocol.json') or run['dataset_contract_sha256']!=sha256(BUNDLE/'datasets'/(dataset+'.json')):raise ValueError('Run hash mismatch')
    return p,datasets[dataset],run,tools
