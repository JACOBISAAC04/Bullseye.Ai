"""Gemini-assisted English coaching scripts with deterministic evidence boundaries.

Gemini may ONLY choose presentation style and order of approved measured facts.
Final speech text is assembled locally from validated 2D observations: the
LLM never supplies arbitrary factual sentences to the audio engine.

Usage is opt-in. No API key or network required for local fallback.
"""
from __future__ import annotations

import copy
import json
import os
import re
from urllib.request import Request, urlopen

from app.coaching.concise_english_voice import (
    ConciseEnglishVoice,
    _pick_observations,
)
from app.coaching.coaching_script_validator import (
    STYLES,
    compose_script,
    validate_plan,
)

DEFAULT_MODEL = "gemini-3.5-flash-lite"
_MODEL_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"


class GeminiRestClient:
    """Small stdlib-only Gemini REST client; secrets remain in HTTP headers."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL,
                 timeout: float = 20.0):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Missing Gemini API key")
        if not isinstance(model, str) or not _MODEL_PATTERN.fullmatch(model):
            raise ValueError("Invalid Gemini model ID")
        if type(timeout) not in (float, int) or not 3 <= timeout <= 60:
            raise ValueError("Timeout must be between 3 and 60 seconds")
        self._api_key = api_key.strip()
        self.model = model
        self.timeout = float(timeout)

    @staticmethod
    def _payload(shot_number: int, facts: list[tuple[str, str]]) -> dict:
        # Do not send raw video, frame data, identities, filenames or source URLs.
        allowed_ids = [identifier for identifier, _ in facts]
        prompt = {
            "task": "Select a conversational English presentation style and order "
                    "for the already verified video-measurement sentences.",
            "shot_number": shot_number,
            "approved_evidence": [
                {"id": identifier, "sentence": sentence}
                for identifier, sentence in facts
            ],
            "allowed_styles": list(STYLES),
            "rules": [
                "Return ONLY a JSON object with style and evidence_ids.",
                "Include every approved evidence ID exactly once, no others.",
                "Do not produce script text, corrections, scores, or shot outcomes.",
                "Never claim ideal mechanics, make/miss outcome or proven errors.",
            ],
        }
        return {
            "contents": [{"role": "user", "parts": [
                {"text": json.dumps(prompt, ensure_ascii=True)}
            ]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 180,
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "style": {
                            "type": "STRING",
                            "enum": list(STYLES),
                        },
                        "evidence_ids": {
                            "type": "ARRAY",
                            "items": {
                                "type": "STRING",
                                "enum": allowed_ids,
                            },
                        },
                    },
                    "required": ["style", "evidence_ids"],
                },
            },
        }

    def select_plan(self, shot_number: int,
                    facts: list[tuple[str, str]]) -> dict:
        payload = self._payload(shot_number, facts)
        request = Request(
            _ENDPOINT.format(self.model),
            data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key,
            },
            method="POST",
        )
        # No retries: avoids unintended extra charge/quota usage.
        with urlopen(request, timeout=self.timeout) as response:
            raw = response.read(65537)
        if len(raw) > 65536:
            raise ValueError("Gemini response too large")
        result = json.loads(raw.decode("utf-8"))
        candidates = result.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("Gemini returned no candidates")
        candidate = candidates[0]
        if candidate.get("finishReason") not in (None, "STOP"):
            raise ValueError("Gemini response was incomplete or blocked")
        parts = candidate.get("content", {}).get("parts", [])
        texts = [p["text"] for p in parts if isinstance(p, dict)
                 and isinstance(p.get("text"), str)]
        if len(texts) != 1:
            raise ValueError("Gemini did not return one JSON text response")
        parsed = json.loads(texts[0])
        validate_plan(parsed, [identifier for identifier, _ in facts])
        return parsed


class GeminiCoachingWriter:
    """Make a manifest compatible with the existing Windows offline TTS."""

    def __init__(self, client: object = None):
        self.client = client

    def build(self, report: dict, *, use_gemini: bool = False,
              model: str = DEFAULT_MODEL, timeout: float = 20) -> dict:
        # Existing Phase 6 validation must run first: timing, source links,
        # no validated errors, no grades, replay consistency, shot alignment.
        base = ConciseEnglishVoice().build(report)
        output = copy.deepcopy(base)
        output["schema_version"] = "1.2"
        output["narration_style"] = "gemini_assisted_grounded_english"
        output["gemini_requested"] = bool(use_gemini)
        output["gemini_model_requested"] = model if use_gemini else None
        output["privacy"] = "Only selected measured sentences sent; no video uploaded"
        client = self.client
        reason_if_no_client = None
        if use_gemini and client is None:
            key = os.environ.get("GEMINI_API_KEY", "")
            if not key.strip():
                reason_if_no_client = "api_key_missing"
            else:
                try:
                    client = GeminiRestClient(key, model=model, timeout=timeout)
                except (ValueError, TypeError):
                    reason_if_no_client = "api_configuration_invalid"

        for shot, original in zip(output["shots"], report["shots"]):
            number = shot["shot_number"]
            facts = _pick_observations(original["feedback_cards"])
            # No LLM call for shots without supported observations.
            if not facts:
                shot["script_source"] = "local_no_qualified_evidence"
                shot["gemini_status"] = "not_needed"
            else:
                fallback = {
                    "style": "focused",
                    "evidence_ids": [identifier for identifier, _ in facts],
                }
                plan = fallback
                status = "not_requested" if not use_gemini else (
                    reason_if_no_client or "fallback"
                )
                if use_gemini and client is not None:
                    try:
                        proposed = client.select_plan(number, facts)
                        validate_plan(
                            proposed, [identifier for identifier, _ in facts]
                        )
                        plan = proposed
                        status = "accepted"
                    except Exception:
                        # Do not disclose private HTTP error bodies or keys.
                        status = "invalid_or_unavailable"
                try:
                    script = compose_script(number, facts, plan)
                except ValueError:
                    script = compose_script(number, facts, fallback)
                    status = "invalid_or_unavailable"
                shot["narration_text"] = script
                shot["spoken_topics"] = plan["evidence_ids"] if status == "accepted" else fallback["evidence_ids"]
                shot["spoken_word_count"] = len(script.split())
                shot["script_source"] = (
                    "gemini_selected_approved_sentences"
                    if status == "accepted"
                    else "local_approved_sentences"
                )
                shot["gemini_status"] = status
            shot["audio_filename"] = f"shot_{number:03d}_english_ai_voice.wav"
            shot["audio_status"] = "script_only"
        return output
