"""Reproducible experiment metadata registry."""
from dataclasses import dataclass,asdict
from datetime import datetime,timezone
import hashlib,json,os
@dataclass
class Experiment:
    name:str;dataset_fingerprint:str;signal_version:str;model_version:str;train_policy:str;evaluation_policy:str;metrics:dict;seed:int=42;created_at:str="";fingerprint:str=""
    def finalize(self):
        if not self.created_at:self.created_at=datetime.now(timezone.utc).isoformat()
        d=asdict(self);d["fingerprint"]=""
        self.fingerprint=hashlib.sha256(json.dumps(d,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()[:16];return self
class ExperimentRegistry:
    def __init__(self,path):self.path=path
    def save(self,e):
        e.finalize()
        try:
            with open(self.path,encoding="utf-8") as f:data=json.load(f)
        except (FileNotFoundError,json.JSONDecodeError):data=[]
        data.append(asdict(e));tmp=self.path+".tmp"
        with open(tmp,"w",encoding="utf-8") as f:json.dump(data,f,ensure_ascii=False,indent=2)
        os.replace(tmp,self.path);return e
