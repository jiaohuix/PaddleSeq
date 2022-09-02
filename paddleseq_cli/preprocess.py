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
import shutil
from collections import Counter
from multiprocessing import Pool
from paddlenlp.data import Vocab

from paddleseq.reader.binarizer import Binarizer
from paddleseq.reader import indexed_dataset
from paddleseq.reader.file_chunker_utils import find_offsets

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("paddleseq_cli.preprocess")


# 创建字典
def load_dictionary(filename):
    vocab = Vocab.load_vocabulary(
        filename,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>"
    )
    return vocab

def get_vocab_path(args,return_src=True):
    target = not args.only_source

    if args.joined_dictionary:
        assert (
                not args.srcdict or not args.tgtdict
        ), "cannot use both --srcdict and --tgtdict with --joined-dictionary"

        if args.srcdict:
            src_dict = args.srcdict
        elif args.tgtdict:
            src_dict = args.tgtdict
        else:
            assert (
                args.trainpref
            ), "--trainpref must be set if --srcdict is not specified"
        tgt_dict = src_dict
    else:
        if args.srcdict:
            src_dict = args.srcdict
        else:
            assert (
                args.trainpref
            ), "--trainpref must be set if --srcdict is not specified"

        if target:
            if args.tgtdict:
                tgt_dict = args.tgtdict
            else:
                assert (
                    args.trainpref
                ), "--trainpref must be set if --tgtdict is not specified"
        else:
            tgt_dict = None
    return src_dict if return_src else tgt_dict

def main(args):
    os.makedirs(args.destdir, exist_ok=True)

    logger.addHandler(
        logging.FileHandler(
            filename=os.path.join(args.destdir, "preprocess.log"),
        )
    )
    logger.info(args)


    def train_path(lang):
        return "{}{}".format(args.trainpref, ("." + lang) if lang else "")

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args.destdir, file_name(prefix, lang))

    def dict_path(lang):
        return dest_path("dict", lang) + ".txt"


    target = not args.only_source

    if not args.srcdict and os.path.exists(dict_path(args.source_lang)):
        raise FileExistsError(dict_path(args.source_lang))
    if target and not args.tgtdict and os.path.exists(dict_path(args.target_lang)):
        raise FileExistsError(dict_path(args.target_lang))

    def make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers): # output_prefix就没有1
        logger.info("[{}] Dictionary: {} types".format(lang, len(load_dictionary(vocab))))
        n_seq_tok = [0, 0]
        replaced = Counter()

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

        input_file = "{}{}".format(
            input_prefix, ("." + lang) if lang is not None else ""
        )
        offsets = find_offsets(input_file, num_workers)
        (first_chunk, *more_chunks) = zip(offsets, offsets[1:])
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id, (start_offset, end_offset) in enumerate(
                more_chunks, start=1
            ):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    func=binarize,
                    args=(
                        args,
                        input_file,
                        vocab,
                        prefix,
                        lang,
                        start_offset,
                        end_offset,
                    ),
                    callback=merge_result,
                )


            pool.close()

        ds = indexed_dataset.make_builder(
            dataset_dest_file(args, output_prefix, lang, "bin"),
            impl=args.dataset_impl,
            vocab_size=len(load_dictionary(vocab)),
        )
        merge_result(
            Binarizer.binarize(
                input_file,
                load_dictionary(vocab),
                lambda t: ds.add_item(t),
                offset=first_chunk[0],
                end=first_chunk[1],
            )
        )

        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, lang)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))
        # write to final file
        ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))

        logger.info(
            "[{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                lang,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
                100 * sum(replaced.values()) / n_seq_tok[1],
                load_dictionary(vocab).unk_token,
            )
        )

    def make_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1):
        if args.dataset_impl == "raw":
            # Copy original text file to destination folder
            output_text_file = dest_path(
                output_prefix + ".{}-{}".format(args.source_lang, args.target_lang),
                lang,
            )
            shutil.copyfile(file_name(input_prefix, lang), output_text_file)
        else:
            make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers)

    def make_all(lang, vocab):
        if args.trainpref:
            make_dataset(vocab, args.trainpref, "train", lang, num_workers=args.workers)
        if args.validpref:
            for k, validpref in enumerate(args.validpref.split(",")):
                outprefix = "valid{}".format(k) if k > 0 else "valid"
                make_dataset(
                    vocab, validpref, outprefix, lang, num_workers=args.workers
                )
        if args.testpref:
            for k, testpref in enumerate(args.testpref.split(",")):
                outprefix = "test{}".format(k) if k > 0 else "test"
                make_dataset(vocab, testpref, outprefix, lang, num_workers=args.workers)

    src_dict_path=get_vocab_path(args,return_src=True)
    make_all(args.source_lang, src_dict_path)
    if target:
        tgt_dict_path = get_vocab_path(args, return_src=False)
        make_all(args.target_lang, tgt_dict_path)

    # copy vocab to destdir
    shutil.copyfile(args.srcdict,os.path.join(args.destdir,os.path.basename(args.srcdict)))
    shutil.copyfile(args.tgtdict,os.path.join(args.destdir,os.path.basename(args.tgtdict)))
    logger.info("Wrote preprocessed data to {}".format(args.destdir))


def binarize(args, filename, vocab, output_prefix, lang, offset, end,append_eos = False):

    vocab=load_dictionary(vocab)
    ds = indexed_dataset.make_builder(
        dataset_dest_file(args, output_prefix, lang, "bin"),
        impl=args.dataset_impl,
        vocab_size=len(vocab),
    )

    def consumer(tensor):
        ds.add_item(tensor)
    res = Binarizer.binarize(
        filename, vocab, consumer, append_eos=append_eos, offset=offset, end=end
    )
    # write to each process's file
    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
    return res

def binarize_alignments(args, filename, parse_alignment, output_prefix, offset, end):
    ds = indexed_dataset.make_builder(
        dataset_dest_file(args, output_prefix, None, "bin"),
        impl=args.dataset_impl,
        vocab_size=None,
    )

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize_alignments(
        filename, parse_alignment, consumer, offset=offset, end=end
    )
    ds.finalize(dataset_dest_file(args, output_prefix, None, "idx"))
    return res

def dataset_dest_prefix(args, output_prefix, lang):
    base = "{}/{}".format(args.destdir, output_prefix)
    if lang is not None:
        lang_part = ".{}-{}.{}".format(args.source_lang, args.target_lang, lang)
    elif args.only_source:
        lang_part = ""
    else:
        lang_part = ".{}-{}".format(args.source_lang, args.target_lang)

    return "{}{}".format(base, lang_part)

def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return "{}.{}".format(base, extension)

def add_preprocess_args(parser):
    group = parser.add_argument_group("Preprocessing")
    # fmt: off
    group.add_argument("-s", "--source-lang", default='en', metavar="SRC",
                       help="source language")
    group.add_argument("-t", "--target-lang", default='ro', metavar="TARGET",
                       help="target language")
    group.add_argument("--trainpref", metavar="FP", default=None,
                       help="train file prefix (also used to build dictionaries)")
    group.add_argument("--validpref", metavar="FP", default=None,
                       help="comma separated, valid file prefixes "
                            "(words missing from train set are replaced with <unk>)")
    group.add_argument("--testpref", metavar="FP", default=None,
                       help="comma separated, test file prefixes "
                            "(words missing from train set are replaced with <unk>)")
    group.add_argument("--align-suffix", metavar="FP", default=None,
                       help="alignment file suffix")
    group.add_argument("--destdir", metavar="DIR", default=None,
                       help="destination dir")
    group.add_argument("--thresholdtgt", metavar="N", default=0, type=int,
                       help="map words appearing less than threshold times to unknown")
    group.add_argument("--thresholdsrc", metavar="N", default=0, type=int,
                       help="map words appearing less than threshold times to unknown")
    group.add_argument("--tgtdict", metavar="FP",
                       help="reuse given target dictionary")
    group.add_argument("--srcdict", metavar="FP",
                       help="reuse given source dictionary")
    group.add_argument("--nwordstgt", metavar="N", default=-1, type=int,
                       help="number of target words to retain")
    group.add_argument("--nwordssrc", metavar="N", default=-1, type=int,
                       help="number of source words to retain")
    group.add_argument("--alignfile", metavar="ALIGN", default=None,
                       help="an alignment file (optional)")
    parser.add_argument('--dataset-impl', metavar='FORMAT', default='mmap',
                        choices=["raw", "lazy", "cached", "mmap", "fasta"],
                        help='output dataset implementation')
    group.add_argument("--joined-dictionary", action="store_true",
                       help="Generate joined dictionary")
    group.add_argument("--only-source", action="store_true",
                       help="Only process the source language")
    group.add_argument("--padding-factor", metavar="N", default=8, type=int,
                       help="Pad dictionary size to be multiple of N")
    group.add_argument("--workers", metavar="N", default=1, type=int, # 只支持1worker
                       help="number of parallel workers")
    group.add_argument("--dict-only", action='store_true',
                       help="if true, only builds a dictionary and then exits")
    # fmt: on
    return parser


def get_parser(desc, default_task="translation"):
    # Before creating the true parser, we need to import optional user module
    # in order to eagerly import custom tasks, optimizers, architectures, etc.
    usr_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    usr_parser.add_argument("--user-dir", default=None) # nnone
    usr_args, _ = usr_parser.parse_known_args()
    # utils.import_user_module(usr_args)

    parser = argparse.ArgumentParser(allow_abbrev=False)
    # Task definitions can be found under fairseq/tasks/

    parser.add_argument(
        "--task",
        metavar="TASK",
        default=default_task,
        help="task",
    )
    # fmt: on
    return parser


def get_preprocessing_parser(default_task="translation"):
    parser = get_parser("Preprocessing", default_task)
    add_preprocess_args(parser)
    return parser


def cli_main():
    parser = get_preprocessing_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()

