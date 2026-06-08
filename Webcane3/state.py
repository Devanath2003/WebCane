from typing import TypedDict, List, Dict, Optional, Annotated
import operator


class WebCaneState(TypedDict):
    goal: str
    starting_url: str
    current_url: str
    screenshot: Optional[bytes]
    observation: Optional[str]
    blockers: List[str]
    elements: List[Dict]
    last_action: Optional[Dict]
    last_action_success: Optional[bool]
    execution_history: Annotated[List[Dict], operator.add]
    is_complete: bool
    error: Optional[str]
    loop_count: int
    start_time: float


class SupervisorAction(TypedDict):
    action: str
    target: str
    query: Optional[str]
    reason: Optional[str]


class ExecutionHistoryEntry(TypedDict):
    action: str
    target: str
    success: bool
    timestamp: float


def create_initial_state(goal: str, starting_url: str = "") -> WebCaneState:
    import time
    return {
        "goal": goal,
        "starting_url": starting_url,
        "current_url": starting_url or "about:blank",
        "screenshot": None,
        "observation": None,
        "blockers": [],
        "elements": [],
        "last_action": None,
        "last_action_success": None,
        "execution_history": [],
        "is_complete": False,
        "error": None,
        "loop_count": 0,
        "start_time": time.time(),
    }
