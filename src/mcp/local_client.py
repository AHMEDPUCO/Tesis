from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


SearchLogsHandler = Callable[..., Dict[str, Any]]


@dataclass(frozen=True)
class ToolDescriptor:
    name: str
    backend: str
    version: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    aliases: Dict[str, str]


@dataclass(frozen=True)
class _SearchLogsBinding:
    descriptor: ToolDescriptor
    handler: SearchLogsHandler


class MCPToolError(RuntimeError):
    pass


class LocalMCPClient:
    """
    Local MCP-style registry/dispatcher for tools.
    This gives us an explicit tool contract and backend swap point.
    """

    def __init__(self) -> None:
        self._search_logs_tools: Dict[str, _SearchLogsBinding] = {}

    def register_search_logs(
        self,
        *,
        backend: str,
        handler: SearchLogsHandler,
        aliases: Optional[Dict[str, str]] = None,
        version: str = "1.0",
    ) -> None:
        descriptor = ToolDescriptor(
            name="search_logs",
            backend=backend,
            version=version,
            input_schema={
                "episode_id": "int|None",
                "query": "str|None",
                "start": "iso8601|None",
                "end": "iso8601|None",
                "filters": "dict|None",
                "limit": "int",
                "agg": "dict|None",
            },
            output_schema={
                "matched": "int",
                "returned": "int",
                "events": "list",
                "aggregation": "dict|None",
            },
            aliases=aliases or {},
        )
        self._search_logs_tools[backend] = _SearchLogsBinding(descriptor=descriptor, handler=handler)

    def list_tools(self, *, name: Optional[str] = None) -> List[ToolDescriptor]:
        descriptors = [binding.descriptor for binding in self._search_logs_tools.values()]
        if name:
            return [d for d in descriptors if d.name == name]
        return descriptors

    def available_backends(self, *, tool_name: str = "search_logs") -> List[str]:
        if tool_name != "search_logs":
            return []
        return sorted(self._search_logs_tools.keys())

    def call_tool(
        self,
        *,
        tool_name: str,
        backend: str,
        logs_dir: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if tool_name != "search_logs":
            raise MCPToolError(f"Unsupported tool_name={tool_name}")

        binding = self._search_logs_tools.get(backend)
        if binding is None:
            raise MCPToolError(
                f"Backend '{backend}' not registered for tool '{tool_name}'. "
                f"Available: {self.available_backends(tool_name=tool_name)}"
            )

        result = binding.handler(logs_dir, **kwargs)
        if not isinstance(result, dict):
            raise MCPToolError("Tool handler returned non-dict result")

        out = dict(result)
        out["_tool_meta"] = {
            "mode": "mcp",
            "tool_name": binding.descriptor.name,
            "backend": binding.descriptor.backend,
            "version": binding.descriptor.version,
            "available_backends": self.available_backends(tool_name=tool_name),
            "aliases": binding.descriptor.aliases,
        }
        return out
