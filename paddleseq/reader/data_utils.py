import numpy as np
import itertools
import logging
logger = logging.getLogger(__name__)

def load_indexed_dataset(
    path, dictionary=None, dataset_impl=None, combine=False, default="mmap"
):
    """A helper function for loading indexed datasets.

    Args:
        path (str): path to indexed dataset (e.g., 'data-bin/train')
        dictionary (~fairseq.data.Dictionary): data dictionary
        dataset_impl (str, optional): which dataset implementation to use. If
            not provided, it will be inferred automatically. For legacy indexed
            data we use the 'cached' implementation by default.
        combine (bool, optional): automatically load and combine multiple
            datasets. For example, if *path* is 'data-bin/train', then we will
            combine 'data-bin/train', 'data-bin/train1', ... and return a
            single ConcatDataset instance.
    """
    import paddleseq.reader.indexed_dataset as indexed_dataset

    datasets = []
    for k in itertools.count():
        path_k = path + (str(k) if k > 0 else "")
        try:
            path_k = indexed_dataset.get_indexed_dataset_to_local(path_k)
        except Exception as e:
            if "StorageException: [404] Path not found" in str(e):
                logger.warning(f"path_k: {e} not found")
            else:
                raise e

        dataset_impl_k = dataset_impl
        if dataset_impl_k is None:
            dataset_impl_k = indexed_dataset.infer_dataset_impl(path_k)
        dataset = indexed_dataset.make_dataset(
            path_k,
            impl=dataset_impl_k or default,
            fix_lua_indexing=True,
            dictionary=dictionary,
        )
        if dataset is None:
            break
        logger.info("loaded {:,} examples from: {}".format(len(dataset), path_k))
        datasets.append(dataset)
        if not combine:
            break
    if len(datasets) == 0:
        return None
    elif len(datasets) == 1:
        return datasets[0]
    else:
        print("ConcatDataset")
        # return ConcatDataset(datasets)
    return datasets

def num_tokens_vec_fn(indices,src_sizes,tgt_sizes):
    """Return the number of tokens for a set of positions defined by indices.
    This value is used to enforce ``--max-tokens`` during batching.
    返回索引对应的最大句长向量（src和tgt取最大）
    """
    sizes = src_sizes[indices]
    if tgt_sizes is not None:
        sizes = np.maximum(sizes, tgt_sizes[indices])
    return sizes

def ordered_indices(src_sizes,tgt_sizes,common_seed,shuffle=True,buckets=None):
    """Return an ordered list of indices. Batches will be constructed based
    on this order."""
    if shuffle:
        indices = np.random.RandomState(common_seed).permutation(len(src_sizes)).astype(np.int64)
    else:
        indices = np.arange(len(src_sizes), dtype=np.int64)
    if buckets is None:
        # sort by target length, then source length  # 排序
        if tgt_sizes is not None:  # 先按照tgt的tokens数排序
            indices = indices[
                np.argsort(tgt_sizes[indices], kind="mergesort")]  # 把indices把tgtsize打乱后，再用稳定的mergesort排序，得到排序后的索引
        return indices[np.argsort(src_sizes[indices], kind="mergesort")]  # 再按照src tokens排序
    else:
        # 按照最大的进行排序
        # sort by bucketed_num_tokens, which is:
        #   max(padded_src_len, padded_tgt_len)
        bucketed_num_tokens=np.array([max(src_size,tgt_size) for src_size,tgt_size in zip(src_sizes,tgt_sizes)])
        return indices[
            np.argsort(bucketed_num_tokens[indices], kind="mergesort")
        ]

def batch_by_size_vec(indices, num_tokens_vec, max_tokens, max_sentences, bsz_factor):
    if indices.shape[0] == 0:
        return []

    assert max_tokens <= 0 or np.max(num_tokens_vec) <= max_tokens, (
        f"Sentences lengths should not exceed max_tokens={max_tokens}"
    )

    indices_len = indices.shape[0]


    batches_ends = np.zeros(indices_len, dtype=np.int32)
    batches_ends_view = batches_ends
    num_tokens_view = num_tokens_vec

    pos = 0
    new_batch_end = 0

    new_batch_max_tokens = 0
    new_batch_sentences = 0
    new_batch_num_tokens = 0

    overflow = False
    size_matches_with_bsz_factor = False

    batches_count = 0
    batch_start = 0
    tail_max_tokens = 0
    batch_max_tokens = 0

    for pos in range(indices_len):
        # At every pos we keep stats about the last complete batch [batch_start:batch_end),
        #      and tail [batch_end:pos].
        # 1) Every time when (batch + tail) forms a valid batch
        #      (according to max_tokens, max_sentences and bsz_factor) we append tail to batch.
        # 2) When (batch+tail) violates max_tokens or max_sentences constraints
        #      we finalize running batch, and tail becomes a new batch.
        # 3) There is a corner case when tail also violates constraints.
        #      In that situation [batch_end:pos-1] (tail without the current pos)
        #      gets added to the finalized batches, while [pos:pos] becomes a new tail.
        #
        # Important: For the sake of performance try to avoid using function calls within this loop.
        # 如果当前位置tokens长度超过，尾部最大tokens长度，更新尾最长
        tail_max_tokens = tail_max_tokens \
            if tail_max_tokens > num_tokens_view[pos] \
            else num_tokens_view[pos]
        new_batch_end = pos + 1  # 尾巴索引更新，用于split indices
        # batch最长token
        new_batch_max_tokens = batch_max_tokens \
            if batch_max_tokens > tail_max_tokens \
            else tail_max_tokens
        # 句数
        new_batch_sentences = new_batch_end - batch_start
        # tokens为最长句tokens*句子数
        new_batch_num_tokens = new_batch_sentences * new_batch_max_tokens
        # 是否溢出
        overflow = (new_batch_sentences > max_sentences > 0 or
                    new_batch_num_tokens > max_tokens > 0)
        # 是否符合mult
        size_matches_with_bsz_factor = (new_batch_sentences < bsz_factor or
                                      new_batch_sentences % bsz_factor == 0)

        if overflow:
            tail_num_tokens = tail_max_tokens * \
                              (new_batch_end - batches_ends_view[batches_count])
            tail_overflow = tail_num_tokens > max_tokens > 0
            # In case of a tail overflow finalize two batches
            if tail_overflow:
                batches_count += 1
                batches_ends_view[batches_count] = pos
                tail_max_tokens = num_tokens_view[pos]
            batch_start = batches_ends_view[batches_count]
            batches_count += 1
            new_batch_max_tokens = tail_max_tokens

        if overflow or size_matches_with_bsz_factor:
            batches_ends_view[batches_count] = new_batch_end
            batch_max_tokens = new_batch_max_tokens
            tail_max_tokens = 0
    if batches_ends_view[batches_count] != indices_len:
        batches_count += 1
    # Memory and time-efficient split
    batches_indices= np.split(indices, batches_ends[:batches_count])
    batches_indices = list(map(lambda batch_indices: batch_indices.tolist(), batches_indices))
    return batches_indices


def get_batches_indices(
        indices,
        num_tokens_vec=None,
        max_tokens=None,
        max_sentences=None,
        bsz_factor=1,
        ):
    """
        Yield mini-batches of indices bucketed by size. Batches may contain
        sequences of different lengths. # 用桶做的，句子可能含有不同长度！关键是桶子大小多少？

        Args:
            indices (List[int]): ordered list of dataset indices
            num_tokens_vec (List[int], optional): precomputed vector of the number
                of tokens for each index in indices (to enable faster batch generation) # 预先计算所以索引的token数的向量
            max_tokens (int, optional): max number of tokens in each batch # 一个bucket中最大token数
                (default: None).
            max_sentences (int, optional): max number of sentences in each # 最大句子长度
                batch (default: None).
            bsz_factor (int, optional): require batch size to # 没指定最大句子长度时，bsz要符合是该参数的倍数
                be less than N or a multiple of N (default: 1).
        """
    # added int() to avoid TypeError: an integer is required
    max_tokens = (
        int(max_tokens) if max_tokens is not None else -1
    )
    max_sentences = max_sentences if max_sentences is not None else -1
    if not isinstance(indices, np.ndarray):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)
    if num_tokens_vec is not None and not isinstance(num_tokens_vec, np.ndarray):
        num_tokens_vec = np.fromiter(num_tokens_vec, dtype=np.int64, count=-1)

    return batch_by_size_vec(
        indices,
        num_tokens_vec,
        max_tokens,
        max_sentences,
        bsz_factor)