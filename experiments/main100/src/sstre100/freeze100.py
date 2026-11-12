"""Preparation for separately approved local CIFAR100 freeze. No network."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from .data import cached_images
from .protocol import sha256

def prepare_from_verified_sources(archive,annotation,destination,contract,tools,*,expected_archive_sha256,expected_annotation_sha256,provenance,approved=False):
    if not approved:raise PermissionError('Separate data-freezing approval required')
    if not provenance or len(expected_archive_sha256)!=64 or len(expected_annotation_sha256)!=64:raise ValueError('Actual source hashes/provenance required')
    archive=Path(archive);annotation=Path(annotation);dest=Path(destination)
    if dest.exists():raise FileExistsError('Never overwrite a freeze')
    if sha256(archive)!=expected_archive_sha256 or sha256(annotation)!=expected_annotation_sha256:raise ValueError('Source SHA256 mismatch')
    md5=hashlib.md5(usedforsecurity=False)
    with archive.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):md5.update(b)
    if md5.hexdigest()!=contract['image_archive']['md5']:raise ValueError('Official CIFAR100 archive MD5')
    _,source_clean=cached_images(archive,expected_archive_sha256,100)
    # No unsafe numpy/pickle fallback. Inspect a different release separately if this fails.
    value=torch.load(annotation,map_location='cpu',weights_only=True)
    if not isinstance(value,dict) or not {'noisy_label','clean_label'}<=value.keys():raise ValueError('Official Python release keys not found')
    def integers(x):
        if torch.is_tensor(x):x=x.cpu().numpy()
        a=np.asarray(x)
        if a.shape!=(50000,) or a.dtype.kind not in 'iu':raise ValueError('Annotation shape/dtype')
        return a.astype(np.int64)
    clean=integers(value['clean_label']);observed=integers(value['noisy_label'])
    if not np.array_equal(clean,source_clean):raise ValueError('Original image-ID/annotation clean-label alignment failed')
    frozen=tools.build_cifar100_contract_fixture(source_clean,observed,contract)
    dest.mkdir(parents=True,exist_ok=False)
    for name,view in [('train.npz',frozen['train']),('validation.npz',frozen['validation']),('AUDIT_ONLY.npz',frozen['audit_only'])]:
        with (dest/name).open('xb') as f:np.savez_compressed(f,**view)
    receipt={'dataset':'cifar100n','classes':100,'annotation_key':'noisy_label','annotation_sha256':expected_annotation_sha256,
        'archive_sha256':expected_archive_sha256,'archive_md5':md5.hexdigest(),'provenance':provenance,
        'original_train_order_alignment':True,'numpy_version':np.__version__,'fingerprint':frozen['fingerprint'],
        'files_sha256':{n:sha256(dest/n) for n in ['train.npz','validation.npz','AUDIT_ONLY.npz']}}
    with (dest/'FREEZE_RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2,allow_nan=False)
    return receipt
