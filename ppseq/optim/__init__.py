import os
import importlib
import paddle.nn as nn
from paddle.optimizer import SGD,Momentum,Adam,AdamW
from paddle.optimizer.lr import CosineAnnealingDecay, NoamDecay
from paddlenlp.transformers import LinearDecayWithWarmup

OPTIMIZER_REGISTRY = {
    "sgd": SGD,
    "mom": Momentum,
    "adam": Adam,
    "adamw": AdamW,
}

LR_SCHEDULER_REGISTRY = {
    "cosine": CosineAnnealingDecay,
    "noam": NoamDecay,
    "linear": LinearDecayWithWarmup,
}

def register_lr_scheduler(lr_scheduler_name):
    def register_lr_scheduler_(cls):
        if cls in LR_SCHEDULER_REGISTRY.values():
            raise ValueError("Cannot register duplicate criterion ({})".format(lr_scheduler_name))
        LR_SCHEDULER_REGISTRY[lr_scheduler_name] = cls
        return cls

    return register_lr_scheduler_


def import_pkgs(pkgs_dir, namespace):
    for file in os.listdir(pkgs_dir):
        path = os.path.join(pkgs_dir, file)
        if (
                not file.startswith("_")
                and not file.startswith(".")
                and (file.endswith(".py") or os.path.isdir(path))
        ):
            pkg_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + pkg_name)

pkgs_dir = os.path.dirname(__file__)
import_pkgs(pkgs_dir, namespace="ppseq.optim")


def check_float(val):
    if type(val) == str and val.find("e-")!=-1:
        val = float(val)
    return val

def build_lr_scheduler(conf, dataloader=None,global_step=-1):
    scheduler_args = conf.lr_scheduler
    scheduler_name = scheduler_args.name
    # step
    epoch_steps = len(dataloader)
    total_steps = conf.train.max_epoch * epoch_steps

    # common params
    sargs = {k: check_float(v) for k,v in scheduler_args.items() if k!= "name"}
    del sargs["reset_lr"]

    # extra params
    if "last_epoch" in sargs.keys(): sargs["last_epoch"] = global_step
    if "total_steps" in sargs.keys(): sargs["total_steps"] = total_steps
    if scheduler_name == "knee":
        sargs["explore_steps"] = scheduler_args.explore_epochs * epoch_steps
        sargs["peak_lr"] = sargs["learning_rate"]
        del sargs["explore_epochs"]
        del sargs["learning_rate"]

    names = ", ".join(LR_SCHEDULER_REGISTRY.keys())
    assert scheduler_name in LR_SCHEDULER_REGISTRY.keys(), f"LR Scheduler [{scheduler_name}] not exists, only support:  {names}"
    scheduler = LR_SCHEDULER_REGISTRY[scheduler_name](**sargs)

    return scheduler


def get_grad_clip(clip_args):
    assert clip_args.type in ["lnorm","gnorm","value"], "clip_type should in [lnor|gnorm|value]."
    clip_map = {"lnorm": "ClipGradByNorm", "gnorm": "ClipGradByGlobalNorm", "value": "ClipGradByValue"}
    clip_name = clip_map[clip_args.type]
    if clip_args.type == "value":
        args = {"min":clip_args.min,"max": clip_args.max}
    else:
        clip_value = clip_args.value
        if clip_value <= 0: return None
        args = {"clip_norm": clip_value}
    grad_clip = getattr(nn, clip_name)(**args)
    return grad_clip


def build_optimizer(optim_args, paramaters, scheduler):
    optim_name = optim_args.name
    # common params
    oargs = {
                "learning_rate": scheduler,
                "weight_decay": float(optim_args.weight_decay),
                "grad_clip": get_grad_clip(optim_args.clip),
                "parameters": paramaters
    }

    # extra params
    if optim_args.name == "mom":
        oargs["use_nesterov"] = optim_args.use_nesterov
        oargs["momentum"] = optim_args.momentum
    elif optim_args.name.find("adam") != -1:
        oargs["beta1"] = optim_args.beta1
        oargs["beta2"] = optim_args.beta2

    names = ", ".join(OPTIMIZER_REGISTRY.keys())
    assert optim_name in OPTIMIZER_REGISTRY.keys(), f"Optimizer [{optim_name}] not exists, only support:  {names}"
    optimizer = OPTIMIZER_REGISTRY[optim_name](**oargs)

    return optimizer

