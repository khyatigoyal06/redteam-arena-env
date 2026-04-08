import logging
import time
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from environment.env import RedTeamArenaEnv
from environment.models import Action
from environment.tasks import TASKS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("redteam-arena.server")


class ResetRequest(BaseModel):
    task_id: int = Field(default=1, ge=1, le=5)


@dataclass
class SessionState:
    env: RedTeamArenaEnv
    started_at: float = field(default_factory=time.time)
    reset_count: int = 0
    step_count: int = 0
    done: bool = False
    last_reward: float | None = None


app = FastAPI(title="RedTeam Arena API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: dict[str, SessionState] = {}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        "%s %s %s %.2fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


def get_session_state(session_id: str = "default") -> SessionState:
    """Return the session state, creating a default environment if needed."""

    if session_id not in SESSIONS:
        SESSIONS[session_id] = SessionState(env=RedTeamArenaEnv(task_id=1))
    return SESSIONS[session_id]


def get_session_id(session_id_header: str | None, x_session_id: str | None) -> str:
    """Normalize the incoming session id header."""

    if session_id_header and session_id_header.strip():
        return session_id_header.strip()
    if x_session_id and x_session_id.strip():
        return x_session_id.strip()
    return "default"


def build_metrics(session_state: SessionState) -> dict[str, Any]:
    """Return current episode metrics for the active session."""

    env = session_state.env
    return {
        "task_id": env.task.task_id,
        "task_name": env.task.name,
        "max_turns": env.task.max_turns,
        "current_turn": env.current_turn,
        "episode_active": env.current_persona is not None,
        "done": session_state.done,
        "reset_count": session_state.reset_count,
        "step_count": session_state.step_count,
        "conversation_length": len(env.conversation_history),
        "attacker_persona": env.current_persona.value if env.current_persona else None,
        "harm_category": env.episode_harm_category,
        "guard_score_so_far": env.state().guard_score_so_far if env.current_persona else 0.0,
        "last_reward": session_state.last_reward,
        "session_age_seconds": round(time.time() - session_state.started_at, 3),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "1.0.0"}


@app.get("/openenv/health")
def openenv_health() -> dict[str, str]:
    return health()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "RedTeam Arena API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": ["/health", "/reset", "/step", "/state", "/tasks", "/metrics"],
    }


@app.post("/reset")
async def reset(
    request: Request,
    session_id_header: str | None = Header(default=None, alias="session_id"),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
) -> dict[str, Any]:
    session_id = get_session_id(session_id_header, x_session_id)
    session_state = get_session_state(session_id)

    payload: Any = None
    try:
        payload = await request.json()
    except Exception:
        payload = None

    task_id_raw: Any = None
    if isinstance(payload, dict):
        task_id_raw = payload.get("task_id")
    if task_id_raw is None:
        task_id_raw = request.query_params.get("task_id")

    if task_id_raw is None:
        task_id = 1
    else:
        try:
            task_id = int(task_id_raw)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="task_id must be an integer between 1 and 5") from exc

    if task_id not in TASKS:
        raise HTTPException(status_code=422, detail="task_id must be between 1 and 5")

    session_state.env = RedTeamArenaEnv(task_id=task_id)
    session_state.reset_count += 1
    session_state.step_count = 0
    session_state.done = False
    session_state.last_reward = None
    observation = session_state.env.reset()
    return observation.model_dump()


@app.post("/openenv/reset")
async def openenv_reset(
    request: Request,
    session_id_header: str | None = Header(default=None, alias="session_id"),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
) -> dict[str, Any]:
    return await reset(request, session_id_header, x_session_id)


@app.post("/step")
def step(
    action: Action,
    session_id_header: str | None = Header(default=None, alias="session_id"),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
) -> dict[str, Any]:
    session_id = get_session_id(session_id_header, x_session_id)
    session_state = get_session_state(session_id)
    try:
        observation, reward, done, info = session_state.env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    session_state.step_count += 1
    session_state.done = done
    session_state.last_reward = reward.value
    return {
        "observation": observation.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.post("/openenv/step")
def openenv_step(
    action: Action,
    session_id_header: str | None = Header(default=None, alias="session_id"),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
) -> dict[str, Any]:
    return step(action, session_id_header, x_session_id)


@app.get("/state")
def state(
    session_id_header: str | None = Header(default=None, alias="session_id"),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
) -> dict[str, Any]:
    session_id = get_session_id(session_id_header, x_session_id)
    session_state = get_session_state(session_id)
    return session_state.env.state().model_dump()


@app.get("/openenv/state")
def openenv_state(
    session_id_header: str | None = Header(default=None, alias="session_id"),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
) -> dict[str, Any]:
    return state(session_id_header, x_session_id)


@app.get("/tasks")
def tasks() -> list[dict[str, Any]]:
    return [
        {
            "id": task.task_id,
            "name": task.name,
            "description": task.description,
            "harm_category": task.harm_category,
            "max_turns": task.max_turns,
            "success_threshold": task.success_threshold,
            "attacker_persona_options": [persona.value for persona in task.attacker_persona_options],
        }
        for task in TASKS.values()
    ]


@app.get("/metrics")
def metrics(
    session_id_header: str | None = Header(default=None, alias="session_id"),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
) -> dict[str, Any]:
    session_id = get_session_id(session_id_header, x_session_id)
    session_state = get_session_state(session_id)
    return build_metrics(session_state)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860)
