'''
TODO: add translation task
'''

import os
import paddle
import importlib

TASK_REGISTRY={}

def register_task(arch_name):
    def register_task_(fn):
        if fn in TASK_REGISTRY.values():
            raise ValueError("Cannot register duplicate model architecture ({})".format(arch_name))
        TASK_REGISTRY[arch_name] = fn
        return fn

    return register_task_


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
import_pkgs(pkgs_dir, namespace="ppseq.tasks")