import os
import importlib

CRITERION_REGISTRY={}

def register_criterion(criterion_name):
    def register_criterion_(cls):
        if cls in CRITERION_REGISTRY.values():
            raise ValueError("Cannot register duplicate criterion ({})".format(criterion_name))
        CRITERION_REGISTRY[criterion_name] = cls
        return cls

    return register_criterion_


# automatically import any Python files in the criterions/ directory
for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("ppseq.criterions." + file_name)



def build_criterion(criterion_args):
    criterion_name = criterion_args.name
    criterion_args.pop("name")
    names = ",".join(CRITERION_REGISTRY.keys())
    assert criterion_name in CRITERION_REGISTRY.keys(), f"Only the following criteria are supported: {names}"

    criterion = CRITERION_REGISTRY[criterion_name](**criterion_args)
    return criterion

'''
usage:
xx.yaml:
criterion:
    name: ce
    label_smooth_eps: 0.1
    pad_idx: 1
    
################################   
from yacs.config import CfgNode
from ppseq.criterions import build_criterion
cfg_path="zhen.yaml"
args = CfgNode.load_cfg(open(cfg_path, encoding="utf-8"))
criterion_args = args.criterion

criterion = build_criterion(criterion_args)
print(criterion)


'''