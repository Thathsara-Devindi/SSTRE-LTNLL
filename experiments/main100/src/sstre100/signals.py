"""Run-local causal evidence, with six completed argmax epochs only."""
import numpy as np
import torch
from .reliability import confidence, check_probabilities, neighborhood_support
class SignalState:
    def __init__(self,observed,classes):
        self.observed=np.asarray(observed,np.int64).copy();self.classes=classes
        if self.observed.ndim!=1 or not ((self.observed>=0)&(self.observed<classes)).all():raise ValueError('Observed labels')
        self.epochs=[];self.argmax=[];self.latest=None;self.bank_epoch=None;self.N=None;self.N_vote=None
    def complete(self,epoch,probabilities,neighbors=None):
        expected=(self.epochs[-1]+1) if self.epochs else 1
        if epoch!=expected:raise ValueError('Nonconsecutive trajectory history')
        p=check_probabilities(probabilities)
        if p.shape!=(len(self.observed),self.classes):raise ValueError('Probability mapping')
        self.latest=p.astype(np.float32).copy()
        self.epochs=(self.epochs+[epoch])[-6:]
        self.argmax=(self.argmax+[p.argmax(1).astype(np.int64)])[-6:]
        if neighbors is not None:
            n=neighborhood_support(p,self.observed,neighbors)
            self.N=n['N_soft'].copy();self.N_vote=n['N_vote'].copy();self.bank_epoch=epoch
    def weights(self,p,tools,method,epoch):
        a=tools.coefficients(p,method,epoch)
        if a is None:return np.ones(len(self.observed),np.float32),{}
        if self.latest is None or not self.epochs or self.epochs[-1]!=epoch-1:raise ValueError('Missing own completed trajectory')
        C=confidence(self.latest,self.observed)
        # C-only does not need temporal or neighborhood evidence for training.
        if method=='c_only':
            R=C;diagnostic={'C':C,'reliability':R,'coefficients':np.array([1.,0.,0.])}
        else:
            evidence=tools.evidence_epochs(p,epoch)
            if self.bank_epoch!=evidence['N'] or self.N is None:raise ValueError('Missing/stale/future neighborhood bank')
            T=tools.temporal_from_ring(self.epochs,np.stack(self.argmax),self.observed,epoch)
            R=tools.fuse(C,T,self.N,[float(x) for x in a])
            diagnostic={'C':C,'T':T,'N_soft':self.N.copy(),'N_vote':self.N_vote.copy(),
                        'reliability':R,'coefficients':np.array([float(x) for x in a]),
                        'C_epoch':np.array(epoch-1),'T_epochs':np.array(evidence['T']),
                        'N_epoch':np.array(self.bank_epoch)}
        w=tools.positive_weights(R);diagnostic['weights']=w.copy()
        return w,diagnostic
    def state_dict(self):
        def tensor(x):return None if x is None else torch.from_numpy(np.asarray(x).copy())
        return {'observed':tensor(self.observed),'classes':self.classes,'epochs':self.epochs.copy(),
                'argmax':[tensor(x) for x in self.argmax],'latest':tensor(self.latest),
                'bank_epoch':self.bank_epoch,'N':tensor(self.N),'N_vote':tensor(self.N_vote)}
    def load_state_dict(self,s):
        if s['classes']!=self.classes or not np.array_equal(s['observed'].cpu().numpy(),self.observed):raise ValueError('Signal identity changed')
        self.epochs=list(s['epochs']);self.argmax=[x.cpu().numpy().copy() for x in s['argmax']]
        self.latest=None if s['latest'] is None else s['latest'].cpu().numpy().copy()
        self.bank_epoch=s['bank_epoch']
        self.N=None if s['N'] is None else s['N'].cpu().numpy().copy()
        self.N_vote=None if s['N_vote'] is None else s['N_vote'].cpu().numpy().copy()
        expected_epochs=list(range(max(1,self.epochs[-1]-5),self.epochs[-1]+1)) if self.epochs else []
        if len(self.epochs)!=len(self.argmax) or len(self.epochs)>6 or self.epochs!=expected_epochs:
            raise ValueError('Malformed signal history')
        if self.latest is not None:
            check_probabilities(self.latest)
            if self.latest.shape!=(len(self.observed),self.classes):raise ValueError('Latest history shape')
        for x in self.argmax:
            if x.shape!=self.observed.shape or not ((x>=0)&(x<self.classes)).all():raise ValueError('Argmax history bounds')
        if self.bank_epoch is not None:
            if not self.epochs or self.bank_epoch>self.epochs[-1]:raise ValueError('Future bank')
            for x in [self.N,self.N_vote]:
                if x is None or x.shape!=self.observed.shape or not np.isfinite(x).all() or not ((x>=0)&(x<=1)).all():raise ValueError('Neighborhood state')
