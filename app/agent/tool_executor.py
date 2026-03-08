from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolPlan:
    requested: bool
    tool_names: list[str]
    note: str


class ToolExecutor:
    def plan(self, query: str, needs_tools: bool) -> ToolPlan:
        if not needs_tools:
            return ToolPlan(
                requested=False,
                tool_names=[],
                note="No external tool call is required for this request.",
            )
        return ToolPlan(
            requested=True,
            tool_names=["search"],
            note=f"Tool execution is a placeholder in v0.1. Query kept for later wiring: {query[:80]}",
        )
