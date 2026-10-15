import ctypes
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from lightllm.utils.kv_cache_utils import attach_shm_kv_cache_ptr, create_shm_kv_cache_ptr, register_shm_ptr_to_pin


@dataclass(frozen=True)
class CpuCacheTensorSpec:
    shm_key: int
    shape: Tuple[int, ...]
    dtype: torch.dtype
    size_bytes: int

    # 沐曦上，shm_bytes 是整段共享内存。size_bytes / byte_offset 是本进程要 pin 的那一截。
    shm_bytes: Optional[int] = None
    byte_offset: int = 0


class CpuCacheCreator:
    def __init__(self, tensor_spec: CpuCacheTensorSpec):
        self.tensor_spec = tensor_spec

    def create_or_attach(self, init_shm_data: bool, pin: bool) -> Union[torch.Tensor, List[torch.Tensor]]:
        # shm_bytes 未设置时，共享内存和张量一样大。沐曦按 rank 切时，shm_bytes 是整段缓存。
        if self.tensor_spec.shm_bytes is None:
            shm_bytes = self.tensor_spec.size_bytes
        else:
            shm_bytes = self.tensor_spec.shm_bytes

        offset = self.tensor_spec.byte_offset
        if offset < 0 or self.tensor_spec.size_bytes < 0 or offset + self.tensor_spec.size_bytes > shm_bytes:
            raise RuntimeError(
                f"cpu cache pin span [{offset}, {offset + self.tensor_spec.size_bytes}) is outside shm size {shm_bytes}"
            )

        if init_shm_data:
            shm_ptr = create_shm_kv_cache_ptr(key=self.tensor_spec.shm_key, size=shm_bytes)
        else:
            shm_ptr = attach_shm_kv_cache_ptr(key=self.tensor_spec.shm_key, size=shm_bytes)

        if pin:
            page_num = self.tensor_spec.shape[0]
            if page_num <= 0 or self.tensor_spec.size_bytes % page_num != 0:
                raise RuntimeError("cpu cache size is not divisible by page num")

            page_bytes = self.tensor_spec.size_bytes // page_num
            registered = register_shm_ptr_to_pin(
                shm_ptr=shm_ptr + offset,
                size=self.tensor_spec.size_bytes,
                align_bytes=page_bytes,
            )
            if isinstance(registered, int):
                cpu_cache_tensor = self._build_tensor_view(
                    shm_ptr=registered,
                    nbytes=self.tensor_spec.size_bytes,
                    shape=self.tensor_spec.shape,
                )
                assert registered == cpu_cache_tensor.data_ptr()
                return cpu_cache_tensor
            chunks = []
            for device_ptr, nbytes in registered:
                n_pages = nbytes // page_bytes
                shape = (n_pages,) + tuple(self.tensor_spec.shape[1:])
                chunk = self._build_tensor_view(shm_ptr=device_ptr, nbytes=nbytes, shape=shape)
                assert device_ptr == chunk.data_ptr()
                chunks.append(chunk)
            return chunks

        cpu_cache_tensor = self._build_tensor_view(
            shm_ptr=shm_ptr + offset,
            nbytes=self.tensor_spec.size_bytes,
            shape=self.tensor_spec.shape,
        )
        assert shm_ptr + offset == cpu_cache_tensor.data_ptr()

        return cpu_cache_tensor

    def _build_tensor_view(self, shm_ptr: int, nbytes: int, shape: Tuple[int, ...]) -> torch.Tensor:
        numpy_array = np.frombuffer(
            memoryview((ctypes.c_uint8 * nbytes).from_address(shm_ptr)),
            dtype=np.uint8,
        )
        return torch.from_numpy(numpy_array).view(dtype=self.tensor_spec.dtype).view(shape)
