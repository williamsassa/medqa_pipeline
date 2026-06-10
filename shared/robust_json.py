"""
Robust JSON extraction for LLM outputs.

Free-tier models (gpt-oss-20b:free et al.) frequently emit JSON that's *almost* valid:
trailing commas, single quotes, ```json fences, an unterminated final string, or a
truncated object missing its closing brace. The strict `json.loads` throws on all of
these and the sample gets rejected — this was ~22% of the BrainMedCoT rejections
(`json_parse_failed`) plus a chunk of `empty_think`.

`loads_robust` applies a layered recovery:
  1. strict json.loads
  2. json5.loads (trailing commas, single quotes, unquoted keys, comments)
  3. manual repairs: strip fences → slice outermost {...} → drop trailing commas →
     balance braces/brackets → close a dangling string
  4. give up → None

`extract_think` additionally tolerates alternate key names the models use instead of
"think" (reasoning / thinking / thought / analysis), recovering more `empty_think` cases.

No network, no side effects. Pure functions, safe to unit-test.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

try:
    import json5  # type: ignore
    _HAS_JSON5 = True
except ImportError:
    _HAS_JSON5 = False


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
# Alternate keys models use when they don't follow the "think" instruction.
_THINK_KEYS = ("think", "reasoning", "thinking", "thought", "analysis", "rationale")


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # remove leading ```json / ``` and trailing ```
        s = _FENCE_RE.sub("", s)
        s = s.strip("`").strip()
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    return s


def _outermost_object(s: str) -> Optional[str]:
    first = s.find("{")
    last = s.rfind("}")
    if first == -1:
        return None
    if last > first:
        return s[first:last + 1]
    # No closing brace at all — take from the first { to the end (truncated output).
    return s[first:]


def _balance_braces(s: str) -> str:
    """Append the closing braces/brackets needed to balance, ignoring those inside strings."""
    in_str = False
    esc = False
    stack: list[str] = []
    for ch in s:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()
    # If we ended inside a string, close it first.
    repaired = s
    if in_str:
        repaired += '"'
    # Close any still-open containers, innermost first.
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    return repaired


def _manual_repair(s: str) -> Optional[dict]:
    candidate = _outermost_object(s)
    if candidate is None:
        return None
    attempts = [candidate]
    # drop trailing commas
    attempts.append(_TRAILING_COMMA_RE.sub(r"\1", candidate))
    # balance braces (handles truncation / unterminated string)
    balanced = _balance_braces(candidate)
    attempts.append(balanced)
    attempts.append(_TRAILING_COMMA_RE.sub(r"\1", balanced))
    for a in attempts:
        for parser in (_try_json, _try_json5):
            obj = parser(a)
            if isinstance(obj, dict):
                return obj
    return None


def _try_json(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _try_json5(s: str) -> Optional[Any]:
    if not _HAS_JSON5:
        return None
    try:
        return json5.loads(s)
    except Exception:
        return None


def loads_robust(text: str) -> Optional[dict]:
    """Best-effort parse of a JSON object from a (possibly messy) LLM string. None on total failure."""
    if not text or not text.strip():
        return None

    # Layer 1: strict on the raw text
    obj = _try_json(text)
    if isinstance(obj, dict):
        return obj

    s = _strip_fences(text)

    # Layer 2: strict + json5 on the de-fenced outermost object
    candidate = _outermost_object(s)
    if candidate is not None:
        obj = _try_json(candidate) or _try_json5(candidate)
        if isinstance(obj, dict):
            return obj

    # Layer 3: manual repairs
    return _manual_repair(s)


def _regex_extract_think(text: str) -> Optional[str]:
    """Last resort: pull a think-like field's string value straight out of broken JSON."""
    for key in _THINK_KEYS:
        # "key": "....."  — non-greedy, allow escaped quotes, stop at the closing quote
        # that is followed by a comma or closing brace.
        m = re.search(
            r'"' + key + r'"\s*:\s*"((?:[^"\\]|\\.)*)"',
            text, flags=re.DOTALL,
        )
        if m:
            val = m.group(1)
            # unescape common sequences
            val = val.encode().decode("unicode_escape", errors="ignore") if "\\" in val else val
            if val.strip():
                return val.strip()
    return None


def extract_think(text: str) -> tuple[Optional[str], str]:
    """
    Extract the chain-of-thought string from an LLM response.

    Returns (think, reason). reason is "" on success, else a short failure tag matching
    the existing rejection vocabulary so downstream stats stay comparable.
    """
    parsed = loads_robust(text)
    if parsed is not None:
        for key in _THINK_KEYS:
            v = parsed.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip(), ""
        # Parsed but no think-like field with content → try regex before giving up.
        salvaged = _regex_extract_think(text)
        if salvaged:
            return salvaged, ""
        return None, "empty_think"

    # Couldn't parse at all → one more regex shot at the think field.
    salvaged = _regex_extract_think(text)
    if salvaged:
        return salvaged, ""
    return None, "json_parse_failed"
