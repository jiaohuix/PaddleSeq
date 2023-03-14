# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import paddle
from collections import Counter
from typing import Dict
from .file_chunker_utils import Chunker
from .file_io import PathManager


# from .tokenizer import tokenize_line


class Binarizer:
    @staticmethod
    def binarize(
            filename,
            dict,
            consumer,
            # tokenize=tokenize_line,
            append_eos=False,
            reverse_order=False,
            offset=0,
            end=-1,
            already_numberized=False,
    ) -> Dict[str, int]:
        nseq, ntok = 0, 0
        replaced = Counter()
        def replaced_consumer(word, idx):
            if idx == dict.to_indices(dict.unk_token) and word != dict.unk_token:
                replaced.update([word])
        with Chunker(
                PathManager.get_local_path(filename), offset, end
        ) as line_iterator:
            for line in line_iterator:
                if already_numberized:  # 已经数值化
                    id_strings = line.strip().split()
                    id_list = [int(id_string) for id_string in id_strings]
                    if reverse_order:
                        id_list.reverse()
                    if append_eos:
                        id_list.append(dict.eos())
                    ids = paddle.to_tensor(id_list, dtype='int32')
                else:  # 需要用dict 变成索引
                    ids = dict.to_indices(tokens=line.strip().split())
                    ids = paddle.to_tensor(ids, dtype='int32')
                nseq += 1
                ntok += len(ids)
                consumer(ids)
        return {
            "nseq": nseq,
            "nunk": sum(replaced.values()),
            "ntok": ntok,
            "replaced": replaced,
        }

    @staticmethod
    def binarize_alignments(
            filename, alignment_parser, consumer, offset=0, end=-1
    ) -> Dict[str, int]:
        nseq = 0

        with Chunker(
                PathManager.get_local_path(filename), offset, end
        ) as line_iterator:
            for line in line_iterator:
                ids = alignment_parser(line)
                nseq += 1
                consumer(ids)
        return {"nseq": nseq}
