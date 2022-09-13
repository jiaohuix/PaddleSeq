#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
__dir__=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__,"../")))
import argparse
import logging
from paddlenlp.data import Vocab
from paddleseq.reader import data_utils
import paddleseq.reader.indexed_dataset as indexed_dataset
logger = logging.getLogger(__name__)


def get_parser():
    print("eg:  python scripts/read_binarized.py --dataset-impl mmap --dict data-bin/src_tgt/vocab.src --input data-bin/src_tgt/train.src-tgt.src ")
    parser = argparse.ArgumentParser(
        description="writes text from binarized file to stdout"
    )
    # fmt: off
    parser.add_argument('--dataset-impl', help='dataset implementation',
                        choices=indexed_dataset.get_available_dataset_impl())
    parser.add_argument('--dict', metavar='FP', help='dictionary containing known words', default=None)
    parser.add_argument('--input', metavar='FP', required=True, help='binarized file to read')
    # fmt: on

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # dictionary = Dictionary.load(args.dict) if args.dict is not None else None
    dictionary = Vocab.load_vocabulary(
        args.dict,
        bos_token="<s>",
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>"
    ) if args.dict is not None else None

    dataset = data_utils.load_indexed_dataset(
        args.input,
        dictionary,
        dataset_impl=args.dataset_impl,
        default="lazy",
    )

    for tensor_line in dataset:
        if dictionary is None:
            line = " ".join([str(int(x)) for x in tensor_line])
        else:
            # line = dictionary.string(tensor_line)
            line = " ".join(dictionary.to_tokens(tensor_line))

        print(line)


if __name__ == "__main__":
    main()
