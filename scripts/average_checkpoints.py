#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import collections
import os
import paddle

def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights.

    Args:
      inputs: An iterable of string paths of checkpoints to load from.

    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict() # for accumulate params
    params_keys = None
    new_state = None
    num_models = len(inputs)

    for fpath in inputs:
        state = paddle.load(fpath)
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(fpath, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        averaged_params[k] /= num_models
    new_state = averaged_params
    return new_state


def last_n_best_checkpoints(paths, n):
    assert len(paths) == 1
    path = paths[0]
    all_names = os.listdir(path)
    ckpt_names = [name for name in all_names if os.path.isdir(os.path.join(path, name)) and name.startswith('model_best_')]
    print(ckpt_names)
    best_names = [name for name in ckpt_names if name.find('best') != -1]
    best_names = list(sorted(best_names, key=lambda name: float(name.replace('model_best_', ''))))
    if len(best_names) < n:
        raise Exception(
            "Found {} checkpoint files but need at least {}", len(best_names), n
        )

    return [os.path.join(path, name,"model.pdparams") for name in best_names[-n:]]

def main():
    parser = argparse.ArgumentParser(
        description="Tool to average the params of input checkpoints to "
        "produce a new checkpoint",
    )
    # fmt: off
    parser.add_argument('--inputs', required=True, nargs='+',
                        help='Input checkpoint file paths.')
    parser.add_argument('--output', default=None, metavar='FILE',
                        help='Write the new checkpoint containing the averaged weights to this path.') # outputdir
    num_group = parser.add_mutually_exclusive_group()
    num_group.add_argument('--num-ckpts', type=int,
                           help='if set, will try to find checkpoints with names checkpoint_xx.pt in the path specified by input, '
                           'and average last this many of them.')
    # fmt: on
    args = parser.parse_args()
    print(args)

    num = None
    if args.num_ckpts is not None:
        num = args.num_ckpts

    args.output=args.output if args.output is not None else os.path.join(args.inputs[0],f"average{args.num_ckpts}")
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    if num is not None:
        args.inputs = last_n_best_checkpoints(
            args.inputs,
            num,
        )
        print("averaging checkpoints: ", args.inputs)

    new_state = average_checkpoints(args.inputs)

    paddle.save(new_state,os.path.join(args.output,"model.pdparams"))
    print("Finished writing averaged checkpoint to {}".format(args.output))


if __name__ == "__main__":
    main()
