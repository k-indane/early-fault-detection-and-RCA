from __future__ import annotations
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List
import asyncio
# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Allows Node FE to call API
from pydantic import BaseModel
from tefault.config import MonitorConfig
from apps.backend.session_manager import SessionManager

class CreateSessionRequest(BaseModel):
    run_uid: str
    tick_interval_s: float = 0.0

class CreateSessionResponse(BaseModel):
    session_id: str
    run_uid: str

# Build FastAPI app
def create_app() -> FastAPI:
    app = FastAPI(title="TEFault Live Monitor API", version="1.0.0")

    # CORS for local dev frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], # Set specific origins for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Load config
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = (repo_root / "monitor.yaml")
    cfg = MonitorConfig.from_yaml(str(cfg_path))

    # Cache manager on app state
    app.state.manager = SessionManager(cfg=cfg, project_root=str(repo_root))



    # Endpoints
    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/runs")
    def list_runs() -> List[Dict[str, Any]]:
        mgr: SessionManager = app.state.manager
        df = mgr.list_runs()
        return df.to_dict(orient="records")

    @app.post("/session", response_model=CreateSessionResponse)
    def create_session(req: CreateSessionRequest) -> CreateSessionResponse:
        mgr: SessionManager = app.state.manager
        try:
            sess = mgr.create_session(req.run_uid, tick_interval_s=req.tick_interval_s)
            return CreateSessionResponse(session_id=sess.session_id, run_uid=sess.run_uid)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.websocket("/ws/{session_id}")
    async def ws_stream(websocket: WebSocket, session_id: str):
        await websocket.accept()

        mgr: SessionManager = app.state.manager
        sess = mgr.get_session(session_id)
        if sess is None:
            await websocket.send_json({"type": "error", "message": f"Unknown session_id={session_id}"})
            await websocket.close(code=1008)
            return

        # Build engine and get run df
        try:
            engine = mgr.build_engine_for_session(session_id)
            df_run = mgr.get_run_df(sess.run_uid)
        except Exception as e:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close(code=1011)
            return
        
        # Stream events with pacing
        try:
            for event in engine.stream(df_run):
                await websocket.send_json(event)

                if (
                    event.get("type") == "tick"
                    and getattr(sess, "tick_interval_s", 0.0) > 0
                ):
                    await asyncio.sleep(float(sess.tick_interval_s))

        except WebSocketDisconnect:
            return
        except Exception as e:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close(code=1011)

    return app

app = create_app()
