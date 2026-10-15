from dataclasses import dataclass
from typing import Optional, Tuple

from lightllm.utils.kv_cache_utils import CpuKVCacheMeta


@dataclass(frozen=True)
class RankLocalPin:
    byte_offset: int
    size_bytes: int
    shape: Tuple[int, ...]
    layout: str


def gqa_rank_local_heads(num_heads: int, world_size: int) -> Optional[int]:
    # 单头注意力不需要切分。
    if num_heads <= 1:
        return None

    kv_heads = num_heads // 2
    if kv_heads % world_size != 0:
        return None

    return num_heads // world_size


def _check_huge_page(full_bytes: int, world_size: int) -> bool:
    # 大页映射的注册地址和长度都要落在大页上。这一段对不齐就退回整段 pin。
    from lightllm.utils.envs_utils import enable_huge_page
    
    if enable_huge_page():
        from lightllm.utils.kv_cache_utils import _get_default_hugepage_size
        huge = _get_default_hugepage_size()
        if (full_bytes // world_size) % huge != 0:
            return False

    return True


def build_rank_local_pin(
    meta: CpuKVCacheMeta,
    rank: int,
    world_size: int,
    enable_disk_cache: bool,
) -> Tuple[Optional[RankLocalPin], str]:
    if enable_disk_cache:
        return None, "disk cache needs one contiguous page"

    if world_size <= 1:
        return None, "single rank"

    full_bytes = int(meta.calcu_size())

    if not _check_huge_page(full_bytes, world_size):
        return None, "rank slice is not aligned to the hugepage"

    # 线性注意力的 meta 把整页打成一个 uint8 head。这在 calcu_cpu_cache_meta 里已经判过模型类型。
    if meta.num_heads == 1 and meta.layer_num == 1 and meta.token_page_size == 1:
        from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig

        config = LinearAttCacheConfig.load_from_args()
        if config.full_att_all_num_kv_heads % world_size != 0:
            return None, "linear-att page is not divisible by tp"

        local_page = config.get_cpu_cache_big_page_bytes() // world_size
        local_bytes = meta.page_num * local_page
        return (
            RankLocalPin(
                byte_offset=rank * local_bytes,
                size_bytes=local_bytes,
                shape=(meta.page_num, 1, 1, 1, local_page),
                layout="linear",
            ),
            "ok",
        )

    local_heads = gqa_rank_local_heads(meta.num_heads, world_size)
    if local_heads is None:
        return None, "kv heads are not divisible by tp"

    shape = (
        meta.page_num,
        meta.layer_num,
        meta.token_page_size,
        local_heads,
        meta.get_merged_head_dim(),
    )
    local_bytes = full_bytes // world_size
    return (
        RankLocalPin(
            byte_offset=rank * local_bytes,
            size_bytes=local_bytes,
            shape=shape,
            layout="gqa",
        ),
        "ok",
    )
