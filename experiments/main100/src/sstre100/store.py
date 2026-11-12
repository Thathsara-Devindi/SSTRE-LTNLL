"""Minimal direct mounted-Volume saving. Exclusive files, commit, durable readback.

No batch_upload, overwrite, deletion, implicit retry or old result-path access.
An expected manifest is committed before verification; only verified markers may
be used for selection/resume. Incomplete files remain unverified on failure.
"""
import hashlib,json,re
from pathlib import Path,PurePosixPath
import numpy as np
from .reliability import require,InvariantError

class PersistenceError(InvariantError):pass
def file_info(path):
    h=hashlib.sha256();count=0
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block);count+=len(block)
    return {'sha256':h.hexdigest(),'bytes':count}

class MountedRunStore:
    def __init__(self,mount,spec,commit,read_file):
        mount=Path(mount).resolve()
        require(spec['output_parent']=='Phase6_Main100_V2' and spec['dataset'] in ('cifar10n','cifar100n'),'Wrong output parent')
        require(re.fullmatch(r'(?:ce|fixed|adaptive|c_only)_seed42_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}',spec['run_id']) is not None,'Unsafe run ID')
        parent=mount/spec['output_parent']/spec['dataset']
        require(parent.resolve().is_relative_to(mount) and not parent.is_symlink(),'Output parent escapes mount')
        parent.mkdir(parents=True,exist_ok=True)
        self.root=parent/spec['run_id'];self.root.mkdir(exist_ok=False)
        require(mount in self.root.resolve().parents,'Result directory escapes mount')
        self.remote='/'+spec['output_parent']+'/'+spec['dataset']+'/'+spec['run_id']
        self.commit=commit;self.read_file=read_file;self.blocked=False
        self.verified={};self.stats={};self.markers={}
    def path(self,relative):
        rel=PurePosixPath(relative)
        require(not rel.is_absolute() and '..' not in rel.parts and '\\' not in relative,'Unsafe result path')
        path=self.root/relative
        require(self.root in path.resolve().parents and not path.is_symlink(),'Output escapes run')
        require(path.suffix in {'.json','.npz','.pt'},'Unexpected artifact type')
        require(not self.blocked,'Store blocked; no retry or republication')
        return path
    def json(self,name,value):
        path=self.path(name)
        with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    def npz(self,name,**arrays):
        with self.path(name).open('xb') as stream:np.savez_compressed(stream,**arrays)
    def _remote_verify(self,name,expected):
        h=hashlib.sha256();count=0
        try:
            for block in self.read_file(self.remote+'/'+name):
                h.update(block);count+=len(block)
                if count>expected['bytes']:break
        except Exception as exc:
            raise PersistenceError('Durable readback failed: '+json.dumps({'file':name,'expected':expected,
                'observed_sha256':h.hexdigest(),'observed_bytes':count,'error_type':type(exc).__name__,'error':str(exc)})) from exc
        actual={'sha256':h.hexdigest(),'bytes':count}
        if actual!=expected:raise PersistenceError('Durable hash/byte mismatch: '+json.dumps({'file':name,'expected':expected,'observed':actual}))
    def seal(self,tag):
        require(not self.blocked,'Store blocked; no retry or republication')
        require(re.fullmatch(r'(?:evaluation_gate|epoch_[0-9]{3}|completion|failure)',tag) is not None,'Unexpected persistence stage')
        try:
            pending={}
            for path in sorted(self.root.iterdir()):
                require(path.is_file() and not path.is_symlink(),'Unexpected output entry')
                name=path.name;stat=(path.stat().st_size,path.stat().st_mtime_ns)
                if name in self.stats:
                    require(self.stats[name]==stat,'Previously verified artifact changed')
                else:pending[name]=file_info(path)
            require(pending,'Nothing to commit')
            receipt=f'expected_{tag}.json'
            self.json(receipt,{'schema':1,'stage':tag,'remote_directory':self.remote,'expected_files':pending})
            receipt_info=file_info(self.root/receipt)
            self.commit()
            for name,info in {**pending,receipt:receipt_info}.items():
                self._remote_verify(name,info)
                require(file_info(self.root/name)==info,'Local file changed during commit/readback')
            marker=f'verified_{tag}.json'
            self.json(marker,{'schema':1,'stage':tag,'verified_files':pending,'expected_manifest':receipt_info})
            marker_info=file_info(self.root/marker)
            self.commit();self._remote_verify(marker,marker_info)
            for name,info in {**pending,receipt:receipt_info,marker:marker_info}.items():
                self.verified[name]=info
                path=self.root/name;self.stats[name]=(path.stat().st_size,path.stat().st_mtime_ns)
            self.markers[tag]=marker
            return pending
        except Exception:
            self.blocked=True
            raise
