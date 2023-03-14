import struct
import shutil
import numpy as np
from paddle.io import Dataset
from functools import lru_cache
from .file_io import PathManager
from enum import Enum, EnumMeta
from typing import List,Union


_code_to_dtype = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.double,
    8: np.uint16,
    9: np.uint32,
    10: np.uint64,
}

def avoid_int32_overflow(np_val):
    # avoid_int32_overflow
    min_val = -2147483648
    max_val = 2147483647
    np_val = np.array(np_val,dtype="int64")
    if np_val<0:
        np_val = (np_val- min_val) + max_val + 1
    return np_val

def best_fitting_int_dtype(
    max_int_to_represent,
) -> Union[np.uint16, np.uint32, np.int64]:

    if max_int_to_represent is None:
        return np.uint32  # Safe guess
    elif max_int_to_represent < 65500:
        return np.uint16
    elif max_int_to_represent < 4294967295:
        return np.uint32
    else:
        return np.int64
        # we avoid np.uint64 because it doesn't save space and its type promotion behaves unexpectedly
        # https://github.com/numpy/numpy/issues/5745


def infer_dataset_impl(path):
    if MMapIndexedDataset.exists(path):
        with open(index_file_path(path), "rb") as f:
            magic = f.read(8)
            if magic == MMapIndexedDataset.Index._HDR_MAGIC[:8]:
                return "mmap"
    else:
        return None


def make_dataset(path, impl, fix_lua_indexing=False, dictionary=None):
    if impl == "mmap" and MMapIndexedDataset.exists(path):
        print("mmap-------------")
        return MMapIndexedDataset(path)
    return None

class StrEnumMeta(EnumMeta):
    # this is workaround for submitit pickling leading to instance checks failing in hydra for StrEnum, see
    # https://github.com/facebookresearch/hydra/issues/1156
    @classmethod
    def __instancecheck__(cls, other):
        return "enum" in str(type(other))

# 这里没有train1
def make_builder(out_file, impl, vocab_size=None):
    if impl == "mmap":
        return MMapIndexedDatasetBuilder(
            out_file, dtype=best_fitting_int_dtype(vocab_size)
        )
    elif impl == "fasta":
        raise NotImplementedError
    # else:
    #     return IndexedDatasetBuilder(out_file)

class StrEnum(Enum, metaclass=StrEnumMeta):
    def __str__(self):
        return self.value

    def __eq__(self, other: str):
        return self.value == other

    def __repr__(self):
        return self.value

    def __hash__(self):
        return hash(str(self))

def ChoiceEnum(choices: List[str]):
    """return the Enum class used to enforce list of choices"""
    return StrEnum("Choices", {k: k for k in choices})

DATASET_IMPL_CHOICES = ChoiceEnum(["raw", "lazy", "cached", "mmap", "fasta"])

def get_available_dataset_impl():
    return list(map(str, DATASET_IMPL_CHOICES))

def _dtype_header_code(dtype) -> int:
    for k in _code_to_dtype.keys():
        if _code_to_dtype[k] == dtype:
            return k
    raise ValueError(dtype)

def _warmup_mmap_file(path):
    with open(path, "rb") as stream:
        while stream.read(100 * 1024 * 1024):
            pass

def index_file_path(prefix_path):
    return prefix_path + ".idx"

def data_file_path(prefix_path):
    return prefix_path + ".bin"

class MMapIndexedDataset(Dataset):
    class Index:  # 处理idx文件，返回某样本的ptr（首地址）和size
        _HDR_MAGIC = b"MMIDIDX\x00\x00"

        @classmethod
        def writer(cls, path, dtype):
            class _Writer:
                def __enter__(self):
                    self._file = open(path, "wb")

                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack("<Q", 1))
                    self._file.write(struct.pack("<B", _dtype_header_code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes):
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []

                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size

                    return pointers

                def write(self, sizes):
                    pointers = self._get_pointers(sizes)

                    self._file.write(struct.pack("<Q", len(sizes)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order="C"))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order="C"))
                    del pointers

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path):
            with open(path, "rb") as stream:
                magic_test = stream.read(9) #读9个字节
                assert self._HDR_MAGIC == magic_test, (
                    "Index file doesn'align_norm match expected format. "
                    "Make sure that --dataset-impl is configured properly."
                )
                version = struct.unpack("<Q", stream.read(8)) #返回一个由解包数据(string)得到的一个元组(tuple)
                assert (1,) == version
                # read是顺序前往后
                (dtype_code,) = struct.unpack("<B", stream.read(1))
                self._dtype = _code_to_dtype[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()

            _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode="r", order="c") #mode r打开现有文件以供阅读。  order column major。读到一个数组
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            self._sizes = np.frombuffer(
                self._bin_buffer, dtype=np.int32, count=self._len, offset=offset # count 数组长度3003，offset读取的起始位置 offset=26= （9+8+1+8）
            ) # 各行文本长度
            self._pointers = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._len,
                offset=offset + self._sizes.nbytes,
            )

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i] # 返回首地址和长度

        def __len__(self):
            return self._len

    def __init__(self, path):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path):
        self._path = path
        self._index = self.Index(index_file_path(self._path)) # index_file添加idx后缀

        _warmup_mmap_file(data_file_path(self._path)) # bin
        self._bin_buffer_mmap = np.memmap(
            data_file_path(self._path), mode="r", order="C"
        )
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        ptr, size = self._index[i]
        ptr = avoid_int32_overflow(ptr)
        np_array = np.frombuffer(
            self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr
        )
        if self._index.dtype != np.int64:
            np_array = np_array.astype(np.int64)

        return np_array.tolist()

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return PathManager.exists(index_file_path(path)) and PathManager.exists(
            data_file_path(path)
        )


def get_indexed_dataset_to_local(path) -> str:
    local_index_path = PathManager.get_local_path(index_file_path(path))
    local_data_path = PathManager.get_local_path(data_file_path(path))

    assert local_index_path.endswith(".idx") and local_data_path.endswith(".bin"), (
        "PathManager.get_local_path does not return files with expected patterns: "
        f"{local_index_path} and {local_data_path}"
    )

    local_path = local_data_path[:-4]  # stripping surfix ".bin"
    assert local_path == local_index_path[:-4]  # stripping surfix ".idx"
    return local_path


# write binary data
class MMapIndexedDatasetBuilder:
    def __init__(self, out_file, dtype=np.int64):
        self._data_file = open(out_file, "wb")
        self._dtype = dtype
        self._sizes = []

    def add_item(self, tensor):
        np_array = np.array(tensor.numpy(), dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)

    def merge_file_(self, another_file):
        # Concatenate index
        index = MMapIndexedDataset.Index(index_file_path(another_file))
        assert index.dtype == self._dtype
        for size in index.sizes:
            self._sizes.append(size)

        # Concatenate data
        with open(data_file_path(another_file), "rb") as f:
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, index_file):
        self._data_file.close()
        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes)
