from paddle.io import DataLoader
import paddle.distributed as dist
from paddlenlp.datasets import MapDataset
from paddlenlp.data.sampler import SamplerHelper
from paddleseq.reader.indexed_dataset import MMapIndexedDataset

class LanguagePairDataset(object):
    """ 使用内存映射文件加载语料对 """
    def __init__(self,src_path,tgt_path):
        super(LanguagePairDataset,self).__init__()
        self.src_data = MMapIndexedDataset(src_path)
        self.tgt_data = MMapIndexedDataset(tgt_path)

    def __iter__(self):
        for idx,(src_tokens,tgt_tokens) in enumerate(zip(self.src_data,self.tgt_data)):
            yield idx,src_tokens,tgt_tokens

    def __len__(self):
        return len(self.src_data)

class BufferedDataloader(object):
    def __init__(self,
                 src_data,
                 tgt_data,
                 buffer_size=200000,
                 sort_type=None,
                 max_tokens=4096,
                 seed=1,
                 shuffle=False,
                 batchify_fn=None):

        self.src_data=src_data
        self.tgt_data=tgt_data
        self.dataset=self.src_data # 打印长度用
        self.buffer_size=buffer_size
        self.sort_type=sort_type
        self.max_tokens=max_tokens
        self.common_seed=seed
        self.shuffle=shuffle
        self.batchify_fn=batchify_fn
        self.epoch = 1

    def __len__(self):
        return int(len(self.dataset)*30/self.max_tokens) # 估计每个文本长30

    def __iter__(self):
        self.epoch+=1 # common_seed+epoch as random seed of shuffle batch
        return self.reader()

    def __call__(self):
        return self.__iter__()

    def buffer_dataloader(self,buffer_data):
        ''' 用于将收集好的buffer大小的data动态组batch,返回dataloader'''
        buffer_dataset = MapDataset(buffer_data)
        batch_sampler = self.dynamic_sampler(buffer_dataset)
        dataloader = DataLoader(
            dataset=buffer_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.batchify_fn,
        )
        return dataloader

    def reader(self):
        ''' 迭代器获取一个buffer的sample，然后用用生成器返回 '''
        buffer_data=[]
        src_iter_=iter(self.src_data)
        tgt_iter_=iter(self.tgt_data)
        for idx,(src_tokens,tgt_tokens) in enumerate(zip(src_iter_,tgt_iter_)):

            buffer_data.append((src_tokens,tgt_tokens,idx))

            if len(buffer_data)==self.buffer_size:
                for batch_data in self.buffer_dataloader(buffer_data):
                    yield batch_data
                buffer_data=[]

        if len(buffer_data)>0:
            for batch_data in self.buffer_dataloader(buffer_data):
                yield batch_data

    def dynamic_sampler(self,dataset):
        '''根据收集到的buffer_data动态组batch，并根据卡数分片'''
        sampler = SamplerHelper(dataset)
        if self.sort_type == SortType.GLOBAL:
            src_key = (lambda idx, data_source: len(data_source[idx][0]))
            tgt_key = (lambda idx, data_source: len(data_source[idx][1]))
            # Sort twice
            sampler = sampler.sort(key=tgt_key).sort(key=src_key)
        else:  # pool
            if self.shuffle:
                sampler = sampler.shuffle(seed=self.common_seed) # shuffle dataset
            max_key = (lambda idx, data_source: max(len(data_source[idx][0]), len(data_source[idx][1])))
            if self.sort_type == SortType.POOL:
                sampler = sampler.sort(key=max_key, buffer_size=self.buffer_size)
        # 输入 idx,length（高）,size（宽）, data_source ,返回新的size，这个size默认是mini batch的句子数，也可以自定义为宽度（最大词数）
        batch_size_fn = lambda idx, count, sofar, data_source: max(sofar, len(data_source[idx][0]),
                                                                   len(data_source[idx][1]))
        batch_sampler = sampler.batch(
            batch_size=self.max_tokens,
            drop_last=False,
            batch_size_fn=batch_size_fn,  # 返回当前的size（宽度）
            key=lambda size_so_far, minibatch_len: size_so_far * minibatch_len)  # 输入宽高，计算token数，和bsz比较

        batch_seed=self.common_seed+self.epoch-1
        batch_sampler = batch_sampler.shuffle(seed=batch_seed)
        if dist.get_world_size()>1:
            batch_sampler = batch_sampler.shard()
        return batch_sampler

    def set_epoch(self, epoch):
        '''
         Sets the epoch number. When :attr:`shuffle=True`, this number is used
        as seeds of random numbers. By default, users may not set this, all
        replicas (workers) use a different random ordering for each epoch.
        If set same number at each epoch, this sampler will yield the same
        ordering at all epoches.
        '''
        self.epoch = epoch


class SortType(object):
    GLOBAL = 'global'
    POOL = 'pool'
    NONE = "none"
