import math
import numpy as np
from paddle.io import BatchSampler
from .data_utils import ordered_indices, num_tokens_vec_fn, get_batches_indices

class DistributedDynamicBatchSampler(BatchSampler):
    ''' 支持多卡训练的动态bsz采样器,与fairseq对齐。 10/2
    '''
    def __init__(self,
                 dataset,
                 mode='train',
                 has_target=False,
                 max_tokens=4000,
                 max_sentences=None,
                 bsz_factor=1,
                 seed=1,
                 num_replicas=None,
                 rank=None,
                 drop_last=False):
        self.dataset = dataset
        assert mode in ['train', 'dev', 'test']
        self.shuffle = mode == 'train'
        self.src_sizes = np.array([len(data[0])+1 for data in dataset])
        self.tgt_sizes = np.array([len(data[1])+1 for data in dataset]) if mode != 'test' or has_target else None
        # self.num_tokens_fn = lambda idx:self.dataset[idx]+1 # 长度dset,一定要加eos或sos！！
        assert max_tokens is not None or max_sentences is not None, \
            "max_tokens and max_sentences should not be null at the same time, please specify one parameter at least"
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        assert isinstance(bsz_factor, int) and bsz_factor > 0, \
            "bsz_factor should be a positive integer"
        self.bsz_factor = bsz_factor
        self.common_seed=seed
        from paddle.fluid.dygraph.parallel import ParallelEnv
        if num_replicas is not None:
            assert isinstance(num_replicas, int) and num_replicas > 0, \
                "num_replicas should be a positive integer"
            self.nranks = num_replicas
        else:
            self.nranks = ParallelEnv().nranks

        if rank is not None:
            assert isinstance(rank, int) and rank >= 0, \
                "rank should be a non-negative integer"
            self.local_rank = rank
        else:
            self.local_rank = ParallelEnv().local_rank
        assert isinstance(drop_last, bool), \
            "drop_last should be a boolean number"
        self.drop_last = drop_last  # 如果多余了就不变，bool不变，然后删除最后一个；如果没多余
        self.epoch = 1  # we use 1-based indexing for epochs

        # get indices and shuffle samples (only calc once)
        indices = ordered_indices(src_sizes=self.src_sizes, tgt_sizes=self.tgt_sizes,
                                  common_seed=self.common_seed,shuffle=self.shuffle)
        # get batches indices and subsample for rank
        self._frozen_batches = self._get_batches_by_max_tokens(indices)

    def __iter__(self):
        # get batches_indices shuffled by epoch+seed
        prev_epoch = self.epoch
        self.epoch+=1
        seed=self.common_seed + prev_epoch
        batches_indices=self._get_batches_for_epoch(seed=seed,shuffle=self.shuffle)
        _batch_iter = iter(batches_indices)

        for batch_indices in _batch_iter:
            yield batch_indices

    def __len__(self):
        return len(self._frozen_batches)

    def _get_batches_for_epoch(self, seed, shuffle):
        batches = self._frozen_batches.copy()
        if shuffle:
            np.random.RandomState(seed=seed).shuffle(batches) # 无返回值
        return batches

    def set_epoch(self, epoch):
        '''
         Sets the epoch number. When :attr:`shuffle=True`, this number is used
        as seeds of random numbers. By default, users may not set this, all
        replicas (workers) use a different random ordering for each epoch.
        If set same number at each epoch, this sampler will yield the same
        ordering at all epoches.
        '''
        self.epoch = epoch

    def _get_batches_by_max_tokens(self, indices):
        ''' get shard data by rank,no shuffle '''
        num_tokens_vec = num_tokens_vec_fn(indices, self.src_sizes, self.tgt_sizes)
        batches_indices = get_batches_indices(indices,
                                              num_tokens_vec=num_tokens_vec,
                                              max_tokens=self.max_tokens,
                                              max_sentences=self.max_sentences,
                                              bsz_factor=self.bsz_factor)

        # process last batch
        if self.drop_last and len(batches_indices[-1]) % self.bsz_factor != 0:
            batches_indices.pop()

        # subsample batches_indices for ranks
        if self.nranks > 1:
            local_batches_indices = []
            last_batches = len(batches_indices) % self.nranks  # 多余的batch
            # 补全batches
            if last_batches > 0:
                batches_indices.extend(batches_indices[:(self.nranks - last_batches)])
            assert len(batches_indices) % self.nranks == 0  # 确保batch数是nrank的倍数

            # sabsample for each process
            for i in range(0, len(batches_indices), self.nranks):
                local_batches_indices.append(batches_indices[i])
            return local_batches_indices
        # single process
        return batches_indices