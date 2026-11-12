"""Checksum-gated official test reader. No activity/downloads on import.

Local tests call pure decoding helpers with generated synthetic archives only.
"""
import hashlib,io,pickle,tarfile
import numpy as np
from .core import require,file_info,source_path

class RestrictedImageUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('numpy.core.multiarray','_reconstruct'):np.core.multiarray._reconstruct,
            ('numpy','ndarray'):np.ndarray,('numpy','dtype'):np.dtype,('_codecs','encode'):__import__('codecs').encode}
        if (module,name) not in allowed:raise pickle.UnpicklingError('Forbidden image-record global')
        return allowed[module,name]

def decode_record(raw,spec):
    require(hashlib.md5(raw).hexdigest()==spec['member_md5'],'Official test member checksum mismatch')
    record=RestrictedImageUnpickler(io.BytesIO(raw),encoding='latin1').load()
    images=np.asarray(record['data']);labels=np.asarray(record[spec['label_key']])
    n=spec['samples'];classes=spec['classes']
    require(images.dtype==np.uint8 and images.shape==(n,3072),'Image format/count')
    require(labels.dtype.kind in 'iu' and labels.shape==(n,) and ((labels>=0)&(labels<classes)).all(),'Official test label range/order')
    require(np.array_equal(np.bincount(labels.astype(np.int64),minlength=classes),np.full(classes,spec['per_class'])),'Official balanced test class counts')
    return images.reshape(n,3,32,32).transpose(0,2,3,1).copy(),labels.astype(np.int64)

def read_archive_bytes(path,archive_sha,spec):
    """Internal helper, independently tested only on synthetic tar fixtures."""
    require(file_info(path)['sha256']==archive_sha,'Official archive SHA mismatch')
    result=None;member_sha=None
    with tarfile.open(path,mode='r|gz') as tar:
        for member in tar:
            if member.name!=spec['member']:continue
            require(result is None and member.isfile() and not member.issparse() and 0<member.size<50_000_000,'Unsafe/repeated test archive member')
            with tar.extractfile(member) as f:raw=f.read(member.size+1)
            require(len(raw)==member.size,'Test member byte count')
            result=decode_record(raw,spec);member_sha=hashlib.sha256(raw).hexdigest()
    require(result is not None,'Missing official test member')
    return *result,{'archive_sha256':archive_sha,'member':spec['member'],'member_md5':spec['member_md5'],'member_sha256':member_sha,'sample_count':spec['samples']}

def load_official_test(input_mount,lock,rules,dataset,*,authorized=False):
    # This guard precedes any file lookup/open. Local synthetic tests never pass it.
    if authorized is not True:raise PermissionError('Separate explicit official-test execution approval required')
    require(dataset in ('cifar10n','cifar100n'),'Unknown official dataset')
    archive=next(a for a in lock['verified_source_archives'] if a['dataset']==dataset)
    return read_archive_bytes(source_path(input_mount,archive['remote_path']),archive['sha256'],rules['datasets'][dataset])
