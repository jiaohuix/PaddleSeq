#!/usr/bin/env python
import setuptools
from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ppseq',
    description='PaddlePaddle Seq2seq',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='1.0',
    packages=setuptools.find_packages(
        where=".",
        # exclude=("examples*", "scripts*"),
        exclude=("examples*", "datasets*", "scripts*", "visual*"),
    ),
    project_urls={
        "Gitter": "https://github.com/MiuGod0126/PaddleSeq",
    },
    python_requires=">=3.6",
    install_requires=[
        "paddlepaddle-gpu>=2.3.0",
        "attrdict",
        "yacs",
        "sacremoses==0.0.53",
        "sacrebleu==1.5",
        "fastcore==1.5.21",
        "pandas==1.1.5",
        "paddlenlp==2.1.1",
        "tqdm>=4.27.0",
        "hydra-core",
    ],
    entry_points={
        "console_scripts": [
            "ppseq_preprocess=ppseq_cli.preprocess:cli_main",
            "ppseq_train=ppseq_cli.train:main",
            "hydra_train=ppseq_cli.hydra_train:main",
            "ppseq_generate=ppseq_cli.generate:main",
        ],
    }
)
