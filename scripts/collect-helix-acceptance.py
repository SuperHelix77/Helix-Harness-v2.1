#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Collect one production chat SSE run without retaining hidden reasoning.

The request JSON is supplied by the caller. Authentication is read from a file so
the bearer never appears in command output or the committed artifact. The output
retains tool events, visible assistant text, final usage/timings/speculative data,
and client wall/first-event/first-visible timings. Reasoning deltas are counted and
dropped rather than persisted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import urllib.request


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888")
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--background-audit",
        action="store_true",
        help="Mark the request as Helix's artifact-only post-task self-audit.",
    )
    return parser.parse_args()


def _sanitize_event(value):
    if isinstance(value, list):
        return [_sanitize_event(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        normalized = str(key).lower()
        if normalized in {"reasoning", "reasoning_content", "thinking", "chain_of_thought"}:
            continue
        result[key] = _sanitize_event(item)
    return result


def main() -> int:
    args = _args()
    token = Path(args.token_file).read_text(encoding="utf-8").strip()
    payload = json.loads(Path(args.request).read_text(encoding="utf-8"))
    payload["stream"] = True
    payload.setdefault("stream_options", {"include_usage": True})
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "X-Unsloth-Events": "1",
    }
    if args.background_audit:
        headers["X-Helix-Background-Audit"] = "1"
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    started = time.perf_counter()
    first_event_ms = None
    first_visible_ms = None
    assistant_parts: list[str] = []
    tool_events: list[dict] = []
    usage = None
    timings = None
    speculative = None
    finish_reason = None
    reasoning_delta_count = 0
    raw_event_count = 0

    with urllib.request.urlopen(request, timeout=1800) as response:
        status = response.status
        while True:
            raw = response.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            raw_event_count += 1
            now_ms = (time.perf_counter() - started) * 1_000.0
            if first_event_ms is None:
                first_event_ms = now_ms
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            event_type = event.get("type")
            if event_type in {"tool_start", "tool_end", "tool_args", "tool_output", "tool_status"}:
                tool_events.append(_sanitize_event(event))
                continue

            choices = event.get("choices") or []
            for choice in choices:
                if choice.get("finish_reason"):
                    finish_reason = choice.get("finish_reason")
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content"):
                    reasoning_delta_count += 1
                content = delta.get("content")
                if isinstance(content, str) and content:
                    if first_visible_ms is None:
                        first_visible_ms = now_ms
                    assistant_parts.append(content)
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            if isinstance(event.get("timings"), dict):
                timings = event["timings"]
            if isinstance(event.get("speculative"), dict):
                speculative = event["speculative"]

    wall_ms = (time.perf_counter() - started) * 1_000.0
    result = {
        "schema_version": "helix.real-agent-stream.v1",
        "http_status": status,
        "wall_ms": wall_ms,
        "first_event_ms": first_event_ms,
        "first_visible_ms": first_visible_ms,
        "assistant_text": "".join(assistant_parts),
        "tool_events": tool_events,
        "usage": usage,
        "timings": timings,
        "speculative": speculative,
        "finish_reason": finish_reason,
        "raw_event_count": raw_event_count,
        "reasoning_delta_count_discarded": reasoning_delta_count,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
