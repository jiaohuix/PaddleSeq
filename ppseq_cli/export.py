'''
参考generate文件实现参数获取，模型、解码器加载。然后参考seq2seq导出模型
'''
# -*- coding: utf-8 -*-
import os
import sys
__dir__=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__,"../")))

import argparse
import logging
from yacs.config import CfgNode
import paddle
from paddle import inference
from ppseq.deploy import NMTInferModel
from ppseq.models import build_model,SequenceGenerator

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ppseq_cli.export")


def parse_export_args():
    """return argumeents, this will overwrite the config after loading yaml file"""
    parser = argparse.ArgumentParser(description="Export model", add_help=True)
    parser.add_argument("-c", "--cfg", default=None, type=str,required=True, metavar="FILE", help="yaml file path")
    parser.add_argument("-b","--beam-size", default=5, type=int, help="beam search size")
    parser.add_argument(
        "-e",
        "--export-path",
        type=str,
        default=None,
        help="The output file prefix used to save the exported inference model.",
    )
    parser.add_argument(
        "-d",
        "--device", default="gpu", choices=["gpu", "cpu"], help="Device selected for inference."
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_export_args()
    conf = CfgNode.load_cfg(open(args.cfg, encoding="utf-8"))
    # Set device
    paddle.set_device(args.device)

    # Load the trained model
    model = build_model(conf, is_test=True)

    # Switch to eval model
    model.eval()

    # Build generator
    generator = SequenceGenerator(model, vocab_size=model.tgt_vocab_size, beam_size=args.beam_size)

    # Create infer model
    infer_model = NMTInferModel(model=model,generator=generator)

    # Convert to static graph with specific input description
    static_model = paddle.jit.to_static(
        infer_model,
        input_spec=[
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # src tokens
        ],
    )

    # Save converted static graph model
    paddle.jit.save(static_model, args.export_path)
    logging.info(f"Saved to static graph model success:")
    os.system(f"ls {args.export_path}*")

if __name__ == '__main__':
    main()
