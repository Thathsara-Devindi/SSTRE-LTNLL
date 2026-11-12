"""Small adaptation of the validated exclusive-create/commit/SHA writer pattern."""
import json,os,re
from pathlib import Path
import numpy as np
from .core import require,file_info

class Publisher:
    def __init__(self,mount,token,commit,read_file):
        require(re.fullmatch(r'test_seed42_[0-9]{8}T[0-9]{6}Z_[a-f0-9]{8}',token) is not None,'Unsafe test token')
        mount=Path(mount).resolve();parent=mount/'Phase6_OfficialTest_V1'
        require(parent.resolve().is_relative_to(mount) and not parent.is_symlink(),'Output prefix escape')
        parent.mkdir(exist_ok=True);self.root=parent/token;self.root.mkdir(exist_ok=False)
        self.remote='/Phase6_OfficialTest_V1/'+token;self.commit=commit;self.read_file=read_file;self.blocked=False
    def path(self,name):
        require(not self.blocked and '/' not in name and '\\' not in name and name not in ('.','..'),'Unsafe/blocked result path')
        p=self.root/name;require(p.suffix in ('.json','.npz') and not p.is_symlink(),'Invalid result type')
        return p
    def json(self,name,value):
        with self.path(name).open('x',encoding='utf-8',newline='\n') as f:
            json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    def npz(self,name,**values):
        with self.path(name).open('xb') as f:np.savez_compressed(f,**values);f.flush();os.fsync(f.fileno())
    def verify(self,name,expected):
        import hashlib
        h=hashlib.sha256();size=0
        for block in self.read_file(self.remote+'/'+name):
            h.update(block);size+=len(block);require(size<=expected['bytes'],'Oversized durable readback')
        require({'bytes':size,'sha256':h.hexdigest()}==expected,'Durable SHA/byte mismatch: '+name)
    def seal(self):
        require(not self.blocked,'No automatic republication')
        try:
            files={p.name:file_info(p) for p in sorted(self.root.iterdir()) if p.is_file()}
            require(len(files)>0 and all(not p.is_symlink() for p in self.root.iterdir()),'Invalid result entries')
            self.json('EXPECTED_COMPLETION.json',{'remote_directory':self.remote,'expected_files':files})
            expected_info=file_info(self.root/'EXPECTED_COMPLETION.json');self.commit()
            for name,info in {**files,'EXPECTED_COMPLETION.json':expected_info}.items():self.verify(name,info)
            self.json('VERIFIED_COMPLETION.json',{'status':'PASS','remote_directory':self.remote,'verified_files':files,'expected_manifest':expected_info})
            self.commit();self.verify('VERIFIED_COMPLETION.json',file_info(self.root/'VERIFIED_COMPLETION.json'))
            return {'status':'PASS','files':files,'marker':'VERIFIED_COMPLETION.json'}
        except Exception:
            self.blocked=True
            raise
