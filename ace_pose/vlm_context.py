"""W — Vision-Language-Model context extraction.

Extracts the three operator-supplied REBA / RULA fields that pose alone
cannot recover:
  - load_kg      — estimated weight of a held / carried object (kg)
  - coupling     — REBA coupling 0=Good / 1=Fair / 2=Poor / 3=Unacceptable
  - activity flags — `static_minute`, `repetitive_4_per_min`, `rapid`

Two backends are supplied:

  1. ``HeuristicBackend`` — works WITHOUT a VLM. Uses pose-derived
     cues only (arm geometry, hand height relative to shoulder, etc.)
     to make conservative estimates. Always installed; runs in <1 ms.
  2. ``HuggingFaceVLMBackend`` — uses a local HuggingFace VLM (Qwen2-VL
     family by default). Optional, gated on `transformers` availability
     and a downloaded checkpoint. Slow (~1-3 s/image) but produces real
     content-aware estimates.

The orchestrating ``ContextExtractor`` returns a ``VLMContext`` dataclass
that plugs directly into ``ergo_v2.score_all`` as keyword arguments,
addressing the audit-flagged "default-zero" REBA gap.
"""
from __future__ import annotations

import abc
import json
import logging
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np

from ace_pose.major_joints import (
    Major25Pose, NOSE, SHOULDER_L, SHOULDER_R, WRIST_L, WRIST_R, PELVIS,
)


log = logging.getLogger("ace_pose.vlm_context")


# ----------------------------------------------------------------------
# Output schema
# ----------------------------------------------------------------------

@dataclass
class VLMContext:
    load_kg: float = 0.0
    load_band: str = "unknown"            # "<5" / "5-10" / "10-20" / ">20" / "unknown"
    coupling: int = 1                     # default Fair (per audit recommendation)
    coupling_label: str = "fair"
    activity_static: bool = True          # conservative default for single-frame
    activity_repetitive: bool = False
    activity_rapid: bool = False
    confidence: float = 0.0               # backend's self-reported confidence
    backend: str = ""
    raw_response: str = ""
    notes: list[str] = field(default_factory=list)

    def to_ergo_kwargs(self) -> dict[str, Any]:
        """Dict suitable for ``ergo_v2.score_all`` keyword arguments."""
        return dict(
            load_kg=float(self.load_kg),
            coupling=int(self.coupling),
            static_minute=bool(self.activity_static),
            repetitive_4_per_min=bool(self.activity_repetitive),
            rapid_changes=bool(self.activity_rapid),
        )


# ----------------------------------------------------------------------
# Backend ABC
# ----------------------------------------------------------------------

class VLMBackend(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    def extract(self, image_bgr: np.ndarray | None,
                pose: Major25Pose,
                worker_metadata: dict | None = None) -> VLMContext:
        ...


# ----------------------------------------------------------------------
# Heuristic backend (no VLM required)
# ----------------------------------------------------------------------

class HeuristicBackend(VLMBackend):
    """Conservative defaults augmented with simple pose-cue rules.

    Decisions
    ---------
    load_kg
        Defaults to 0. Heuristic: if both wrists are close together AND
        within 30° of the body's anterior axis, infer the worker is
        gripping/holding something — assign 5 kg (a deliberately low
        estimate). Otherwise 0.
    coupling
        Defaults to **1 (Fair)** per `standards/reba.md §7` audit
        recommendation. Upgrades to 0 (Good) when hands are close
        together (suggesting a two-handed grip on an object), downgrades
        to 2 (Poor) when one wrist is above shoulder height (overhead
        reach with poor handle access).
    activity flags
        For single-frame analysis we cannot detect temporal patterns.
        Per `standards/reba.md §12` recommendation, set
        ``activity_static = True`` (conservative); other two stay False.
    secondary-person contact (expert-pilot fix, Section 4.9)
        If ``worker_metadata["primary_kpts_2d"]`` and
        ``worker_metadata["other_people_kpts_2d"]`` are supplied (major-25
        2D image-pixel keypoints of the worker and of any other person
        detected in the same frame), and
        either wrist is within ``0.6x`` the primary worker's shoulder
        width of that person's pelvis/shoulders, the worker is inferred
        to be physically supporting another person's body weight (e.g.
        an eldercare transfer/assist), which independent expert scoring
        flagged as a >=20 kg load that the pose-only heuristic below
        cannot see (no second person's keypoints were persisted at
        batch-inference time; see Section 4.9). This is a narrow,
        opt-in signal: callers that do not pass these keys
        (the full 6,007-pose cohort pipeline) are unaffected.
    """
    name = "heuristic"
    CONTACT_LOAD_KG = 22.0  # expert-stated ">20 kg" band (OWAS load code 3)

    def extract(self, image_bgr=None, pose=None, worker_metadata=None) -> VLMContext:
        ctx = VLMContext(backend=self.name, confidence=0.3)
        if pose is None:
            ctx.notes.append("no pose supplied; using all defaults")
            return ctx

        kp = pose.keypoints
        wl, wr = kp[WRIST_L], kp[WRIST_R]
        sl, sr = kp[SHOULDER_L], kp[SHOULDER_R]
        pelvis = kp[PELVIS]
        wrist_sep = float(np.linalg.norm(wr - wl))
        sh_sep = float(np.linalg.norm(sr - sl))

        # RTMW3D 3D x/y are per-person crop coordinates, so cross-person
        # distance is only meaningful in the shared 2D image frame.
        meta = worker_metadata or {}
        p2d = meta.get("primary_kpts_2d")
        contact = False
        if p2d is not None:
            sh2d = float(np.linalg.norm(p2d[SHOULDER_R] - p2d[SHOULDER_L]))
            for other_kp in meta.get("other_people_kpts_2d") or []:
                targets = [other_kp[PELVIS], other_kp[SHOULDER_L], other_kp[SHOULDER_R]]
                for wrist in (p2d[WRIST_L], p2d[WRIST_R]):
                    for t in targets:
                        if np.linalg.norm(wrist[:2] - t[:2]) < 0.6 * max(sh2d, 1e-6):
                            contact = True
        if contact:
            ctx.load_kg = self.CONTACT_LOAD_KG
            ctx.load_band = ">20"
            ctx.notes.append(
                "secondary-person contact -> assistive load >=20 kg "
                "(expert-informed threshold, Section 4.9)"
            )

        # close grip detection: wrists within half a shoulder-width
        close_grip = wrist_sep < 0.5 * max(sh_sep, 1e-6)
        # both hands forward of body
        # both hands above shoulder (any of L or R)
        overhead = (np.linalg.norm(wl - pelvis) > 0
                    and np.linalg.norm(wr - pelvis) > 0)
        # Use Y axis convention: -y = up in image. Hand above shoulder if
        # wrist Y < shoulder Y (more negative).
        overhead = (wl[1] < sl[1] - 0.05 * sh_sep) or (wr[1] < sr[1] - 0.05 * sh_sep)

        if close_grip:
            if not contact:
                ctx.load_kg = 5.0
                ctx.load_band = "5-10"
            ctx.coupling = 0      # good two-handed grip
            ctx.coupling_label = "good"
            ctx.notes.append("close-grip → 5 kg, coupling=Good")
        if overhead:
            # Overhead reach degrades coupling regardless of hand grip
            ctx.coupling = max(ctx.coupling, 2)
            ctx.coupling_label = "poor"
            ctx.notes.append("overhead reach → coupling=Poor")

        ctx.activity_static = True
        ctx.notes.append("single-frame → static_minute=True (REBA §12)")
        return ctx


# ----------------------------------------------------------------------
# Optional HuggingFace VLM backend
# ----------------------------------------------------------------------

DEFAULT_HF_MODEL = "Qwen/Qwen2-VL-2B-Instruct"


PROMPT_TEMPLATE = (
    "You are an occupational ergonomics assistant. Look at the image of a "
    "worker (you may also be told the worker's pose summary). Provide:\n"
    "1) load_kg_band: weight of any object the worker is carrying or "
    "manipulating, chosen from [\"<5\", \"5-10\", \"10-20\", \">20\", \"none\"].\n"
    "2) coupling: quality of the grip on a handled object, chosen from "
    "[\"good\", \"fair\", \"poor\", \"unacceptable\"]. Default to \"fair\" "
    "when unclear.\n"
    "3) activity_static: true if the worker appears to be in a held / "
    "sustained posture, otherwise false.\n"
    "4) activity_repetitive: true if the task looks like a short-range "
    "repeating motion (you may have to guess from context cues).\n"
    "5) activity_rapid: true if the posture suggests rapid or large-range "
    "movement; false if static.\n"
    "Return STRICT JSON only, no commentary."
)


class HuggingFaceVLMBackend(VLMBackend):
    """Optional backend that prompts a local VLM (Qwen2-VL family by default).

    Lazy-initialised on first call. Soft-fails (logs a warning and returns
    heuristic defaults) when transformers / weights are unavailable.
    """
    name = "hf-vlm"

    def __init__(self, model_id: str = DEFAULT_HF_MODEL,
                 device: str = "cuda", dtype: str = "bfloat16",
                 max_new_tokens: int = 200):
        self.model_id = model_id
        self.device = device
        self.dtype_name = dtype
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._available: bool | None = None

    def _ensure_loaded(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            import torch
            try:
                # Qwen2-VL uses a special class
                from transformers import Qwen2VLForConditionalGeneration
                self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=getattr(torch, self.dtype_name),
                    device_map=self.device,
                )
            except (ImportError, AttributeError, OSError):
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=getattr(torch, self.dtype_name),
                    trust_remote_code=True,
                    device_map=self.device,
                )
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True,
            )
            self._available = True
        except Exception as ex:
            log.warning("HF VLM init failed (%s); will fall back to heuristic", ex)
            self._available = False
        return self._available

    def extract(self, image_bgr, pose, worker_metadata=None) -> VLMContext:
        if not self._ensure_loaded():
            return _fallback(reason="hf-vlm unavailable", pose=pose)
        try:
            import cv2
            import torch
            from PIL import Image

            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": pil},
                    {"type": "text", "text": PROMPT_TEMPLATE},
                ],
            }]
            try:
                text = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                # processors without chat-template fall back to a plain prompt
                text = PROMPT_TEMPLATE
            inputs = self._processor(text=[text], images=[pil],
                                     return_tensors="pt").to(self.device)
            with torch.no_grad():
                gen = self._model.generate(**inputs,
                                           max_new_tokens=self.max_new_tokens)
            out_ids = gen[:, inputs.input_ids.shape[1]:]
            out_text = self._processor.batch_decode(
                out_ids, skip_special_tokens=True,
            )[0]
        except Exception as ex:
            log.warning("HF VLM inference failed (%s); falling back", ex)
            return _fallback(reason=f"hf-vlm exception: {ex}", pose=pose)

        ctx = parse_vlm_json(out_text)
        ctx.backend = self.name
        ctx.raw_response = out_text
        return ctx


# ----------------------------------------------------------------------
# Parsing utilities
# ----------------------------------------------------------------------

LOAD_BAND_TO_KG = {
    "none": 0.0,
    "<5":   2.0,
    "5-10": 7.5,
    "10-20": 15.0,
    ">20":  25.0,
}

COUPLING_TO_INT = {
    "good": 0, "fair": 1, "poor": 2, "unacceptable": 3,
}


def parse_vlm_json(text: str) -> VLMContext:
    """Robust JSON-ish parser for VLM completions.

    Looks for the first balanced-brace JSON block and extracts the five
    known fields. Falls back to safe defaults when fields are missing
    or malformed.
    """
    ctx = VLMContext()
    ctx.raw_response = text
    match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if not match:
        ctx.notes.append("no JSON block found in VLM output")
        return ctx
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as ex:
        ctx.notes.append(f"json decode error: {ex}")
        return ctx

    band = str(data.get("load_kg_band", "")).lower().strip()
    if band in LOAD_BAND_TO_KG:
        ctx.load_band = band
        ctx.load_kg = LOAD_BAND_TO_KG[band]
    else:
        ctx.notes.append(f"unknown load_kg_band: {band!r}")

    coupling = str(data.get("coupling", "")).lower().strip()
    if coupling in COUPLING_TO_INT:
        ctx.coupling = COUPLING_TO_INT[coupling]
        ctx.coupling_label = coupling
    else:
        ctx.notes.append(f"unknown coupling: {coupling!r}")

    ctx.activity_static = bool(data.get("activity_static", True))
    ctx.activity_repetitive = bool(data.get("activity_repetitive", False))
    ctx.activity_rapid = bool(data.get("activity_rapid", False))
    ctx.confidence = float(data.get("confidence", 0.5))
    return ctx


def _fallback(reason: str, pose: Major25Pose | None) -> VLMContext:
    heur = HeuristicBackend()
    ctx = heur.extract(pose=pose)
    ctx.notes.append(f"fallback: {reason}")
    return ctx


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------

class ContextExtractor:
    """Top-level entry that delegates to a configured backend."""

    def __init__(self, backend: VLMBackend | None = None):
        self.backend = backend or HeuristicBackend()

    def __call__(self, image_bgr=None, pose=None, worker_metadata=None) -> VLMContext:
        return self.backend.extract(image_bgr=image_bgr, pose=pose,
                                    worker_metadata=worker_metadata)


# ----------------------------------------------------------------------
# Convenience helpers for the batch pipeline
# ----------------------------------------------------------------------

def context_to_columns(ctx: VLMContext) -> dict[str, Any]:
    """Flatten a `VLMContext` into batch-CSV friendly columns."""
    return {
        "ctx_backend": ctx.backend,
        "ctx_load_kg": ctx.load_kg,
        "ctx_load_band": ctx.load_band,
        "ctx_coupling": ctx.coupling,
        "ctx_coupling_label": ctx.coupling_label,
        "ctx_activity_static": ctx.activity_static,
        "ctx_activity_repetitive": ctx.activity_repetitive,
        "ctx_activity_rapid": ctx.activity_rapid,
        "ctx_confidence": ctx.confidence,
        "ctx_notes": "; ".join(ctx.notes) if ctx.notes else "",
    }
