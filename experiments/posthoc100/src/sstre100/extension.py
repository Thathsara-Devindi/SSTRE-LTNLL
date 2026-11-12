"""Explicit 13-condition extension. Frozen construction seed is never changed."""
import copy, hashlib, json, re
from pathlib import Path
from fractions import Fraction
from .protocol import ROOT, load as load_original, sha256

PREFIX = 'Phase6_PostHoc100_V1'
TOKEN = r'[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}'
SUBSETS = {'cn': (Fraction(1,2),0,Fraction(1,2)),
           'ct': (Fraction(23,47),Fraction(24,47),0),
           'tn': (0,Fraction(24,47),Fraction(23,47)),
           't_only': (0,1,0), 'n_only': (0,0,1)}
EXPECTED = {('cifar10n',m,42) for m in SUBSETS} | {
    (d,m,s) for d in ('cifar10n','cifar100n') for m in ('fixed','adaptive') for s in (43,44)}
FINGERPRINTS = {
 'cifar10n':'9a8f859c871027638440913cdf4678e3febc924951c2ee0b3c01a5a61dc17839',
 'cifar100n':'0a9f072cee264e5bfbe715eda22252afaebc009f6104fb8f4e7847ad8e597fc2'}
C100_FILES = {'train.npz':'dacdf2958aeef632b5321a1d021459dfc9759e509e1fe8a1969da2bf9f251254',
 'validation.npz':'b86582b4e1c3a503162f85e76d65058f22d4cb241722f712e81cea16ff6797fe'}

def register():
    result=json.loads((ROOT/'POSTHOC_REGISTER.json').read_text())
    runs=result['runs']
    if len(runs)!=13 or {(r['dataset'],r['method'],r['seed']) for r in runs}!=EXPECTED:
        raise ValueError('Not the reviewed thirteen conditions')
    if len({r['run_key'] for r in runs})!=13: raise ValueError('Duplicate run keys')
    for r in runs:
        if r['dataset_split_seed']!=42 or r['split_fingerprint']!=FINGERPRINTS[r['dataset']]:
            raise ValueError('Frozen split changed')
        if r['epochs']!=100 or r['warmup_epochs']!=30 or r['official_test_enabled'] or r['audit_labels_in_training']:
            raise ValueError('Scientific safety contract changed')
    return result

def condition(key):
    found=[r for r in register()['runs'] if r['run_key']==key]
    if len(found)!=1: raise ValueError('Unknown post-hoc condition')
    return copy.deepcopy(found[0])

def coefficients(p, original_tools, method, epoch):
    original_tools.lr_for_epoch(p,epoch)
    if method not in ('fixed','adaptive',*SUBSETS): raise ValueError('Unapproved extension method')
    if epoch<=p['warmup_epochs']: return None
    return SUBSETS[method] if method in SUBSETS else original_tools.coefficients(p,method,epoch)

def uses_n(method):
    if method not in ('fixed','adaptive',*SUBSETS): raise ValueError('Unapproved method')
    return method not in ('ct','t_only')

def load_condition(key):
    r=condition(key)
    # Validate the untouched seven-run seed42 bundle FIRST, with its original validator.
    p,d,_,tools=load_original(r['dataset'],r['method'] if r['method'] in ('fixed','adaptive') else 'fixed')
    p=copy.deepcopy(p);p['seed']=r['seed']
    return p,d,r,tools

def validate_run_id(r, token):
    if re.fullmatch(TOKEN,token) is None: raise ValueError('Invalid launch token')
    return f"{r['method']}_seed{r['seed']}_{token}"

def verify_data_receipt(r, receipt):
    if receipt['fingerprint']!=FINGERPRINTS[r['dataset']]: raise ValueError('Frozen dataset fingerprint mismatch')
    if r['dataset']=='cifar100n':
        if receipt['annotation_key']!='noisy_label': raise ValueError('Wrong noisy labels')
        if any(receipt['files_sha256'][k]!=v for k,v in C100_FILES.items()):
            raise ValueError('Frozen CIFAR100 training/validation bytes changed')

def verify_bundle(root=ROOT):
    entries=json.loads((root/'POSTHOC_SOURCE_MANIFEST.json').read_text())
    for name,expected in entries.items():
        path=root/name
        if path.is_symlink() or not path.is_file() or sha256(path)!=expected:
            raise ValueError('Post-hoc bundle changed: '+name)
    return sha256(root/'POSTHOC_SOURCE_MANIFEST.json')

def acquire_claim(put, key, token, *, resume=None):
    """Server-side conditional insert. Never reset/delete a claim on failure.

    put is Modal Dict.put(...,skip_if_exists=True), tested with a local fake.
    The caller must hold separate approval for this one submission and claim write.
    """
    r=condition(key);validate_run_id(r,token)
    claim='condition:'+key if resume is None else 'recovery:'+key+':'+resume['parent_run_id']
    if not put(claim,{'run_key':key,'token':token,'post_hoc':True},skip_if_exists=True):
        raise RuntimeError('Duplicate/platform restart refused: claim already exists; no automatic retry')
    return claim

def persist_claim(mount,claim,token,commit,read_file):
    """Durable exclusive marker complements the server-atomic launch guard.

    Dict entries expire after inactivity; this committed marker does not. A failed
    or incomplete claim remains blocking. No cleanup/reset/retry is implemented.
    """
    base=Path(mount).resolve();parent=base/PREFIX/'_claims'
    for folder in (base/PREFIX,parent):
        if folder.is_symlink() or not folder.resolve().is_relative_to(base):raise ValueError('Claim path escapes mount')
    parent.mkdir(parents=True,exist_ok=True)
    name=hashlib.sha256(claim.encode()).hexdigest()+'.json';path=parent/name
    if path.is_symlink():raise ValueError('Claim symlink forbidden')
    payload=json.dumps({'claim':claim,'launch_token':token,'post_hoc':True},sort_keys=True).encode()
    with path.open('xb') as f:
        f.write(payload);f.flush()
        import os
        os.fsync(f.fileno())
    commit()
    h=hashlib.sha256();count=0
    for block in read_file('/'+PREFIX+'/_claims/'+name):
        count+=len(block)
        if count>len(payload):raise ValueError('Durable claim size mismatch')
        h.update(block)
    if count!=len(payload) or h.hexdigest()!=hashlib.sha256(payload).hexdigest():raise ValueError('Durable claim SHA mismatch')
    return {'path':'/'+PREFIX+'/_claims/'+name,'sha256':h.hexdigest(),'bytes':count}
