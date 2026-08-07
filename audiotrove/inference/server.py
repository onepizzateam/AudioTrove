"""Minimal HTTP server wrapping inference sessions.

Uses aiohttp (imported lazily inside :meth:`run`) so importing this module does
not require aiohttp to be installed. Sessions are lazy-loaded on first request
and kept warm, mirroring the audio.cpp server behaviour.
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any, Dict

from audiotrove.inference.base import InferenceSession

logger = logging.getLogger(__name__)


class AudioTroveServer:
    """Serve loaded TTS/ASR inference sessions over HTTP."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = json.load(f)
        self.host = self.config.get("host", "127.0.0.1")
        self.port = self.config.get("port", 8080)
        self._sessions: Dict[str, InferenceSession] = {}

    def _model_spec(self, model_id: str) -> dict:
        for m in self.config.get("models", []):
            if m["id"] == model_id:
                return m
        raise KeyError(f"Unknown model id: {model_id}")

    def _get_session(self, model_id: str) -> InferenceSession:
        """Lazily load and cache the session for ``model_id``."""
        if model_id not in self._sessions:
            spec = self._model_spec(model_id)
            task = spec.get("task", "tts")
            options = dict(spec.get("options", {}))
            if "model_path" in spec:
                options.setdefault("model_path", spec["model_path"])
            if task == "tts":
                from audiotrove.inference.tts import get_tts_session

                session = get_tts_session(spec["family"], **options)
            elif task == "asr":
                from audiotrove.inference.asr import get_asr_session

                session = get_asr_session(spec["family"], **options)
            elif task == "vc":
                from audiotrove.inference.vc import get_vc_session

                session = get_vc_session(spec["family"], **options)
            else:
                raise ValueError(f"Unknown task: {task}")
            session.load()
            self._sessions[model_id] = session
        return self._sessions[model_id]

    # --- Route handlers (aiohttp handlers; request objects duck-typed) ---

    async def handle_health(self, request: Any) -> Any:
        from aiohttp import web

        return web.json_response({"status": "ok"})

    async def handle_models(self, request: Any) -> Any:
        from aiohttp import web

        models = [
            {"id": m["id"], "task": m.get("task", "tts"), "family": m["family"]}
            for m in self.config.get("models", [])
        ]
        return web.json_response({"models": models})

    async def handle_speech(self, request: Any) -> Any:
        from aiohttp import web

        body = await request.json()
        model_id = body.get("model") or body.get("family")
        text = body["text"]
        session = self._get_session(model_id)
        result = session.run(text=text, voice_ref=body.get("voice_ref"))
        wav_bytes = _encode_wav(result.audio, result.sample_rate)
        return web.Response(body=wav_bytes, content_type="audio/wav")

    async def handle_transcription(self, request: Any) -> Any:
        from aiohttp import web

        reader = await request.multipart()
        field = await reader.next()
        data = await field.read()
        model_id = request.query.get("model") or _first_asr_model(self.config)
        # Persist upload to a temp file for the session to read.
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        session = self._get_session(model_id)
        result = session.run(audio_path=tmp_path)
        return web.json_response({"text": result.text})

    async def handle_task_run(self, request: Any) -> Any:
        from aiohttp import web

        body = await request.json()
        model_id = body.get("model")
        params = body.get("params", {})
        session = self._get_session(model_id)
        result = session.run(**params)
        return web.json_response(
            {"text": result.text, "sample_rate": result.sample_rate, "metadata": result.metadata}
        )

    def build_app(self) -> Any:
        """Construct the aiohttp application with all routes registered."""
        from aiohttp import web

        app = web.Application()
        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/v1/models", self.handle_models)
        app.router.add_post("/v1/audio/speech", self.handle_speech)
        app.router.add_post("/v1/audio/transcriptions", self.handle_transcription)
        app.router.add_post("/v1/tasks/run", self.handle_task_run)
        return app

    def run(self) -> None:
        """Start the aiohttp server (blocking)."""
        try:
            from aiohttp import web
        except ImportError as e:
            raise ImportError(
                "Server requires aiohttp: pip install audiotrove[infer]"
            ) from e

        app = self.build_app()
        logger.info("Starting AudioTrove server on %s:%s", self.host, self.port)
        web.run_app(app, host=self.host, port=self.port)


def _first_asr_model(config: dict) -> str:
    for m in config.get("models", []):
        if m.get("task") == "asr":
            return m["id"]
    raise ValueError("No ASR model configured")


def _encode_wav(audio, sample_rate: int) -> bytes:
    """Encode a float32 numpy waveform to WAV bytes."""
    import wave

    import numpy as np

    buf = io.BytesIO()
    pcm = np.clip(np.asarray(audio), -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()
