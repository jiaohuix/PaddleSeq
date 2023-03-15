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
    cargs = {k:v for k,v in criterion_args.items() if k!="name"}
    # criterion_args.pop("name") # 不能pop，否则恢复训练无name了
    names = ", ".join(CRITERION_REGISTRY.keys())
    assert criterion_name in CRITERION_REGISTRY.keys(), f"Criterion [{criterion_name}] not exists, only support:  {names}"

    criterion = CRITERION_REGISTRY[criterion_name](**cargs)
    return criterion

