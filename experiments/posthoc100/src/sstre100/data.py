"""Cached, checksum-gated TRAIN images only. No network or official test loader."""
from pathlib import Path
import hashlib, io, json, pickle, tarfile
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from .protocol import sha256
from .native import transform
BATCH_MD5={
 'cifar-10-batches-py/data_batch_1':'c99cafc152244af753f735de768cd75f',
 'cifar-10-batches-py/data_batch_2':'d4bba439e000b95fd0a9bffe97cbabec',
 'cifar-10-batches-py/data_batch_3':'54ebc095f3ab1f0389bbae665268c751',
 'cifar-10-batches-py/data_batch_4':'634d18415352ddfa80567beed471001a',
 'cifar-10-batches-py/data_batch_5':'482c414d41f54cd18b22e5b47cb7c3cb'}
class ImageUnpickler(pickle.Unpickler):
    # Official CIFAR Python files contain NumPy arrays; reject arbitrary globals.
    def find_class(self,module,name):
        allowed={('numpy.core.multiarray','_reconstruct'):np.core.multiarray._reconstruct,
                 ('numpy','ndarray'):np.ndarray,('numpy','dtype'):np.dtype,
                 ('_codecs','encode'):__import__('codecs').encode}
        if (module,name) not in allowed:raise pickle.UnpicklingError('Forbidden CIFAR global: '+module+'.'+name)
        return allowed[module,name]
def digest_bytes(data,algorithm):return hashlib.new(algorithm,data).hexdigest()
def cached_images(archive,sha,classes):
    archive=Path(archive)
    if not sha or not archive.is_file() or archive.is_symlink() or sha256(archive)!=sha:raise ValueError('Missing/unverified image archive; no download fallback')
    names=BATCH_MD5 if classes==10 else {'cifar-100-python/train':'16019d7e3df5f24257cddd939b257f8d'}
    seen=set();arrays=[];labels=[]
    with tarfile.open(archive,mode='r|gz') as tar:
        for member in tar:
            if member.name not in names:continue  # Never load test batches/labels.
            if member.name in seen or not member.isfile() or member.issparse() or not 0<member.size<170000000:raise ValueError('Unsafe/repeated archive member')
            seen.add(member.name)
            with tar.extractfile(member) as f:raw=f.read(member.size+1)
            if len(raw)!=member.size or digest_bytes(raw,'md5')!=names[member.name]:raise ValueError('Official train-file MD5 mismatch')
            record=ImageUnpickler(io.BytesIO(raw),encoding='latin1').load()
            x=np.asarray(record['data']);y=np.asarray(record['labels' if classes==10 else 'fine_labels'],np.int64)
            if x.dtype!=np.uint8 or x.shape!=(len(y),3072):raise ValueError('Official train-image format')
            arrays.append((member.name,x));labels.append((member.name,y))
    if seen!=set(names):raise ValueError('Missing official TRAIN batch')
    x=np.concatenate([v for _,v in sorted(arrays)]).reshape(-1,3,32,32).transpose(0,2,3,1)
    clean=np.concatenate([v for _,v in sorted(labels)])
    if x.shape!=(50000,32,32,3) or not np.array_equal(np.bincount(clean,minlength=classes),np.full(classes,50000//classes)):raise ValueError('Image/source-label alignment')
    return x,clean

def logical_fingerprint(ids,val,y,prefix=b''):
    h=hashlib.sha256(prefix)
    for a in (ids,val,y):h.update(np.asarray(a,dtype='<i8').tobytes())
    return h.hexdigest()

def load_frozen(input_root,contract,tools):
    root=Path(input_root);classes=contract['num_classes']
    if classes==10:
        archive=root/'datasets/cifar10_official/cifar-10-python.tar.gz'
        pixels,source_clean=cached_images(archive,contract['image_archive']['sha256'],classes)
        split=root/'Phase1_V2/dataset_split_v2.npz'
        if sha256(split)!=contract['local_split_sha256']:raise ValueError('Frozen CIFAR10 split bytes changed')
        with np.load(split,allow_pickle=False) as z:
            ids=z['lt_indices'].copy();val=z['val_indices'].copy();y=z['observed_labels'].copy()
        fp=logical_fingerprint(ids,val,y)
        if fp!=contract['split_fingerprint']:raise ValueError('Frozen CIFAR10 fingerprint')
        train={'original_ids':ids,'observed_labels':y}
        validation={'original_ids':val,'targets':source_clean[val].copy()}
        receipt={'fingerprint':fp,'archive_sha256':contract['image_archive']['sha256']}
    else:
        base=root/'Phase1_CIFAR100N_V1';r=json.loads((base/'FREEZE_RECEIPT.json').read_text())
        if r['annotation_key']!='noisy_label' or r['dataset']!='cifar100n' or r['classes']!=100:raise ValueError('Unverified CIFAR100 annotation identity')
        for name in ['train.npz','validation.npz']:
            if sha256(base/name)!=r['files_sha256'][name]:raise ValueError('Frozen CIFAR100 file hash')
        pixels,source_clean=cached_images(root/'datasets/cifar100_official/cifar-100-python.tar.gz',r['archive_sha256'],classes)
        with np.load(base/'train.npz',allow_pickle=False) as z:train={k:z[k].copy() for k in z.files}
        with np.load(base/'validation.npz',allow_pickle=False) as z:validation={k:z[k].copy() for k in z.files}
        if set(validation)!={'original_ids','targets'} or not np.array_equal(source_clean[validation['original_ids']],validation['targets']):raise ValueError('Validation image-label alignment')
        fp=logical_fingerprint(train['original_ids'],validation['original_ids'],train['observed_labels'],b'cifar100n_fine_IR10_seed42_v1\0')
        if fp!=r['fingerprint']:raise ValueError('Frozen CIFAR100 fingerprint')
        receipt=r
    tools.validate_train_view(train,classes)
    if len(train['original_ids'])!=contract['train_samples'] or len(validation['original_ids'])!=contract['validation_samples']:raise ValueError('Frozen sample counts')
    vi=validation['original_ids']
    if vi.dtype.kind not in 'iu' or len(np.unique(vi))!=len(vi) or not ((vi>=0)&(vi<50000)).all() or np.intersect1d(train['original_ids'],vi).size:raise ValueError('Validation contamination/IDs')
    if not np.array_equal(np.bincount(validation['targets'],minlength=classes),np.full(classes,contract['validation_per_class'])):raise ValueError('Validation balance')
    return pixels,train,validation,receipt

class TrainImages(Dataset):
    def __init__(self,pixels,view,tx):
        if set(view)!={'original_ids','observed_labels'}:raise ValueError('Training accepts only IDs and noisy labels')
        self.pixels=pixels;self.ids=np.asarray(view['original_ids']);self.labels=np.asarray(view['observed_labels']);self.tx=tx
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):return self.tx(Image.fromarray(self.pixels[self.ids[i]])),int(self.labels[i]),i
class ValidationImages(Dataset):
    def __init__(self,pixels,view,tx):self.data=TrainImages(pixels,{'original_ids':view['original_ids'],'observed_labels':view['targets']},tx)
    def __len__(self):return len(self.data)
    def __getitem__(self,i):return self.data[i]
def worker_seed(_):
    import random
    seed=torch.initial_seed()%2**32;random.seed(seed);np.random.seed(seed)
def loader(dataset,p,epoch,kind,device):
    if kind not in ('training','analysis','validation'):raise ValueError('Loader kind')
    multiplier={'training':101,'analysis':103,'validation':107}[kind]
    generator=torch.Generator().manual_seed(p['seed']+multiplier*epoch)
    return DataLoader(dataset,batch_size=p['loader']['batch_size'] if kind=='training' else p['loader']['analysis_batch_size'],
        shuffle=kind=='training',num_workers=p['loader']['num_workers'],drop_last=False,persistent_workers=False,
        worker_init_fn=worker_seed,generator=generator,pin_memory=str(device).startswith('cuda'))
