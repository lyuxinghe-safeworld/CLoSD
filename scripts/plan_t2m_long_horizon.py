#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan a long-horizon text prompt into scheduled T2M segments."
    )
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--prompt-file", type=str, default="")
    parser.add_argument("--planning-horizon", type=int, default=40, help="Planning horizon in 20fps units.")
    parser.add_argument("--output-dir", type=str, default="output/long_horizon_plans")
    parser.add_argument("--max-segments", type=int, default=8)
    parser.add_argument("--default-segment-duration", type=float, default=2.0)
    parser.add_argument("--min-duration", type=float, default=0.4)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--openai-timeout-sec", type=float, default=45.0)
    return parser.parse_args()


def _read_prompt(args: argparse.Namespace) -> str:
    if bool(args.prompt) == bool(args.prompt_file):
        raise ValueError("Exactly one of --prompt or --prompt-file must be provided.")

    if args.prompt_file:
        if not os.path.isfile(args.prompt_file):
            raise ValueError(f"--prompt-file does not exist: {args.prompt_file}")
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
    else:
        prompt = args.prompt.strip()

    if not prompt:
        raise ValueError("Prompt is empty after trimming whitespace.")
    return prompt


def _extract_text_from_responses_api(payload: Dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks: List[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text)
    return "\n".join(chunks).strip()


def _json_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "segments": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "action": {"type": "string"},
                        "duration_sec": {"type": "number"},
                        "prompt": {"type": "string"},
                    },
                    "required": ["action", "duration_sec", "prompt"],
                },
            }
        },
        "required": ["segments"],
    }


def _call_openai_segments(prompt: str, max_segments: int, timeout_sec: float) -> List[Dict[str, Any]]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
    if not model:
        model = "gpt-4o-mini"

    url = f"{base_url}/responses"
    system_prompt = (
        "You are a motion-planning assistant. Break the user's long motion intent into "
        f"1 to {max_segments} temporally ordered free-form segments. "
        "Return JSON only following the provided schema."
    )
    user_prompt = (
        "Create segments for text-to-motion generation.\n"
        "Requirements:\n"
        "1) duration_sec must be positive and realistic.\n"
        "2) prompt must start with 'a person is ...' (or 'a person is ... while ...').\n"
        "3) action is a short action phrase.\n\n"
        f"User request:\n{prompt}"
    )

    body = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "t2m_segments",
                "schema": _json_schema(),
                "strict": True,
            }
        },
    }

    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI HTTP error {exc.code}: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"OpenAI request failed: {exc}") from exc

    payload = json.loads(raw)
    text = _extract_text_from_responses_api(payload)
    if not text:
        raise RuntimeError("OpenAI response did not contain output text.")

    parsed = json.loads(text)
    segments = parsed.get("segments")
    if not isinstance(segments, list) or not segments:
        raise RuntimeError("OpenAI output JSON has no valid segments list.")
    return segments


def _heuristic_segments(prompt: str, max_segments: int, default_segment_duration: float) -> List[Dict[str, Any]]:
    pieces = re.split(
        r"\s*(?:,|;|\band then\b|\bthen\b|\bafter that\b|\bnext\b)\s*",
        prompt,
        flags=re.IGNORECASE,
    )
    cleaned = [p.strip(" .") for p in pieces if p and p.strip(" .")]
    if not cleaned:
        cleaned = [prompt.strip()]

    if len(cleaned) > max_segments:
        cleaned = cleaned[:max_segments]

    segments = []
    for p in cleaned:
        sentence = p.strip()
        if not sentence.endswith("."):
            sentence += "."
        segments.append(
            {
                "action": p,
                "duration_sec": default_segment_duration,
                "prompt": sentence,
            }
        )
    return segments


def _safe_float(value: Any, default_value: float) -> float:
    try:
        f = float(value)
    except Exception:
        return default_value
    if not (f == f) or f in [float("inf"), float("-inf")]:
        return default_value
    return f


def _canonicalize_segment_prompt(prompt_text: str, action_text: str, fallback_text: str) -> str:
    raw = (prompt_text or "").strip()
    if not raw:
        raw = (action_text or "").strip()
    if not raw:
        raw = (fallback_text or "").strip()
    raw = re.sub(r"\s+", " ", raw).strip()
    raw = re.sub(r"[ \t\r\n\.\!\?;:,]+$", "", raw)

    lowered = raw.lower()
    known_prefixes = [
        "a person is ",
        "person is ",
        "the person is ",
        "someone is ",
        "a human is ",
    ]
    for prefix in known_prefixes:
        if lowered.startswith(prefix):
            raw = raw[len(prefix):].strip()
            break

    if not raw:
        raw = "moving"
    return f"a person is {raw}."


def _normalize_segments(
    segments: List[Dict[str, Any]],
    source_prompt: str,
    planning_horizon_20fps: int,
    min_duration: float,
    max_duration: float,
    default_segment_duration: float,
) -> Dict[str, Any]:
    planning_horizon_30fps = planning_horizon_20fps * 30 // 20
    normalized = []

    for idx, segment in enumerate(segments):
        action = str(segment.get("action", "")).strip()
        prompt_raw = str(segment.get("prompt", "")).strip()
        duration_sec = _safe_float(segment.get("duration_sec"), default_segment_duration)

        if not action and prompt_raw:
            action = prompt_raw
        if not action:
            action = source_prompt

        prompt = _canonicalize_segment_prompt(
            prompt_text=prompt_raw,
            action_text=action,
            fallback_text=source_prompt,
        )

        duration_sec = max(min_duration, min(max_duration, duration_sec))
        num_horizons = max(1, int(round(duration_sec * 60.0 / float(planning_horizon_30fps))))
        normalized.append(
            {
                "index": idx,
                "action": action,
                "prompt": prompt,
                "duration_sec": round(duration_sec, 3),
                "num_horizons": int(num_horizons),
            }
        )

    if not normalized:
        fallback_prompt = _canonicalize_segment_prompt("", "", source_prompt)
        normalized = [
            {
                "index": 0,
                "action": source_prompt,
                "prompt": fallback_prompt,
                "duration_sec": round(max(min_duration, default_segment_duration), 3),
                "num_horizons": 1,
            }
        ]

    prompt_per_horizon: List[str] = []
    for s in normalized:
        prompt_per_horizon.extend([s["prompt"]] * int(s["num_horizons"]))

    total_horizons = len(prompt_per_horizon)
    episode_length = total_horizons * planning_horizon_30fps
    return {
        "version": 1,
        "source_prompt": source_prompt,
        "planning_horizon_20fps": planning_horizon_20fps,
        "planning_horizon_30fps": planning_horizon_30fps,
        "segments": normalized,
        "prompt_per_horizon": prompt_per_horizon,
        "total_horizons": total_horizons,
        "episode_length": episode_length,
    }


def _build_schedule_filename(prompt: str) -> str:
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", prompt).strip("_").lower()
    if not slug:
        slug = "long_prompt"
    slug = slug[:48]
    return f"{now}_{slug}.json"


def main() -> int:
    args = parse_args()
    try:
        prompt = _read_prompt(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.planning_horizon <= 0:
        print("--planning-horizon must be positive.", file=sys.stderr)
        return 2
    if args.max_segments <= 0:
        print("--max-segments must be positive.", file=sys.stderr)
        return 2
    if args.default_segment_duration <= 0:
        print("--default-segment-duration must be positive.", file=sys.stderr)
        return 2

    used_fallback = False
    planner_error = ""
    segments: List[Dict[str, Any]]
    try:
        segments = _call_openai_segments(prompt, args.max_segments, args.openai_timeout_sec)
    except Exception as exc:
        used_fallback = True
        planner_error = str(exc)
        segments = _heuristic_segments(prompt, args.max_segments, args.default_segment_duration)

    schedule = _normalize_segments(
        segments=segments,
        source_prompt=prompt,
        planning_horizon_20fps=args.planning_horizon,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        default_segment_duration=args.default_segment_duration,
    )
    schedule["planner"] = {
        "source": "heuristic" if used_fallback else "openai",
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "fallback_reason": planner_error if used_fallback else "",
    }

    os.makedirs(args.output_dir, exist_ok=True)
    schedule_path = os.path.abspath(os.path.join(args.output_dir, _build_schedule_filename(prompt)))
    with open(schedule_path, "w", encoding="utf-8") as f:
        json.dump(schedule, f, ensure_ascii=False, indent=2)

    if used_fallback:
        print(f"[planner] OpenAI planner failed, fallback heuristic used: {planner_error}", file=sys.stderr)

    summary = {
        "schedule_path": schedule_path,
        "episode_length": int(schedule["episode_length"]),
        "total_horizons": int(schedule["total_horizons"]),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
