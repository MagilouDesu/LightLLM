import torch
from typing import Any, ContextManager, Optional

from lightllm.platform.base.graph import BackendGraph


class MluGraphBackend(BackendGraph):

    def __init__(self) -> None:
        import torch_mlu  # noqa: F401

        self._mlu = torch.mlu

    def create_graph(self) -> Any:
        return self._mlu.MLUGraph()

    def graph(self, graph_obj: Any, pool: Optional[Any] = None, stream: Optional[Any] = None) -> ContextManager:
        return self._mlu.graph(graph_obj, pool=pool, stream=stream)

    def graph_pool_handle(self) -> Any:
        return self._mlu.graph_pool_handle()

    def is_capturing(self) -> bool:
        return self._mlu.is_current_stream_capturing()
