#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ppt_preprocess.py

PPT preprocessing for LLM continual pretraining.

Step 1: Judge page continuity and merge into coherent segments (supports:
        - iterative + hybrid (cheap signals + LLM) + jump-aware windowing
        - pairwise if desired)
Step 2: (optional) Rewrite merged segments via local LLM into clean training text.

Input format (JSON list):
[
  { "file_name": "...", "page": 1, "content": "..." },
  { "file_name": "...", "page": 2, "content": "..." },
  ...
]

Outputs:
- segments.json : {"segments":[{"file_name","pages":[...],"text":...}, ...]}
- pretrain.jsonl: lines of {"text": "...", "meta": {...}, "original": {...}}  (only if not --dry-run)

Supported backends:
- --api-type openai     : OpenAI-compatible /chat/completions endpoint (e.g., vLLM, LM Studio, etc.)
- --api-type ollama     : Ollama HTTP API (/api/chat)
- --api-type ollama_py  : Using the 'ollama' Python package (import ollama)

Dependencies:
- requests  (always)
- ollama    (only if you use --api-type ollama_py)

Recommended model styles:
- Instruct-tuned models that follow JSON well (e.g., llama3.1:8b-instruct, qwen2.5:7b-instruct, etc.)
"""

import os
import re
import json
import uuid
import time
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

try:
    import requests
except ImportError as e:
    raise SystemExit("This script requires 'requests'. Install via: pip install requests") from e


# =========================
# LLM Client Abstractions
# =========================

class BaseLLMClient:
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 512) -> str:
        raise NotImplementedError


class OpenAICompatClient(BaseLLMClient):
    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 512) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        # Expect OpenAI-style response
        return data["choices"][0]["message"]["content"]


class OllamaHTTPClient(BaseLLMClient):
    """
    Uses Ollama HTTP /api/chat. We pass options.format='json' and num_predict=-1 to
    reduce truncated/invalid JSON risks.
    """
    def __init__(self, base_url: str, model: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 512) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": -1,     # let model finish the JSON
                "format": "json",      # enforce JSON output
            },
            "stream": False
        }
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        # chat returns content under data["message"]["content"]
        return data.get("message", {}).get("content", "")


class OllamaPyClient(BaseLLMClient):
    """
    Uses the official 'ollama' Python package. Honors OLLAMA_HOST if set,
    or you can pass --api-base and we will treat it as host for a per-client instance.
    """
    def __init__(self, model: str, host: Optional[str] = None, timeout: int = 120):
        try:
            import ollama  # noqa: F401
            from ollama import Client as OllamaLibClient  # noqa: F401
        except ImportError as e:
            raise SystemExit("Missing 'ollama' package. Install via: pip install ollama") from e

        self.model = model
        self.timeout = timeout
        self._client = None
        if host:
            from ollama import Client as OllamaLibClient
            self._client = OllamaLibClient(host=host)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 512) -> str:
        import ollama
        options = {
            "temperature": temperature,
            "num_predict": -1,   # ensure completion
            "format": "json",    # enforce JSON
        }
        if self._client is not None:
            resp = self._client.chat(model=self.model, messages=messages, options=options)
        else:
            resp = ollama.chat(model=self.model, messages=messages, options=options)
        return resp.get("message", {}).get("content", "")


def make_client(api_type: str, base_url: str, model: str, api_key: Optional[str]) -> BaseLLMClient:
    api_type = api_type.lower().strip()
    if api_type == "openai":
        return OpenAICompatClient(base_url=base_url, model=model, api_key=api_key)
    elif api_type == "ollama":
        # HTTP API at /api/chat
        # Auto-default base for Ollama if user forgot to set
        if not base_url or base_url == "http://localhost:8000/v1":
            base_url = "http://localhost:11434"
        return OllamaHTTPClient(base_url=base_url, model=model)
    elif api_type == "ollama_py":
        host = None
        if base_url and base_url.startswith("http"):
            host = base_url
        else:
            host = os.getenv("OLLAMA_HOST")
        return OllamaPyClient(model=model, host=host)
    else:
        raise ValueError("Unsupported --api-type. Use 'openai', 'ollama', or 'ollama_py'.")


# =========================
# IO Helpers
# =========================

def load_items(input_path: str) -> List[Dict[str, Any]]:
    """
    Load a JSON list: [{"file_name","page","content"}, ...]
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects.")
    norm = []
    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            raise ValueError(f"Item {i} is not an object.")
        file_name = obj.get("file_name")
        page = obj.get("page")
        content = obj.get("content", "")
        if file_name is None or page is None:
            raise ValueError(f"Item {i} missing 'file_name' or 'page'.")
        norm.append({"file_name": str(file_name), "page": int(page), "content": str(content or "")})
    return norm


def group_by_file(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets = defaultdict(list)
    for it in items:
        buckets[it["file_name"]].append({"page": it["page"], "content": it["content"]})
    for fname in buckets:
        buckets[fname].sort(key=lambda x: x["page"])
    return buckets


# =========================
# Prompts
# =========================

CONTINUITY_SYSTEM = (
    "你是一個嚴謹的投影片段落分段器。給你「當前已合併段落的尾端」與「下一頁」的文字，"
    "判斷下一頁是否延續同一個概念，而不是新章節或新主題。請只輸出 JSON，勿加入多餘文字或 Markdown。"
)

# Keep minimal JSON to reduce truncation risk
CONTINUITY_USER_TMPL = """[當前段落尾端]
{seg_text}

[下一頁]
{next_text}

請嚴格回傳 JSON（不要附加其他文字）：
{{
  "continuous": true/false,
  "confidence": 0.0~1.0
}}
"""

REWRITE_SYSTEM = (
    "You are a professional technical editor. "
    "Rewrite the given merged PPT content into a clean, coherent training paragraph for LLM continual pretraining. "
    "Preserve all factual details across pages, merge bullet points into sentences, expand abbreviations once, "
    "remove slide artifacts (page numbers, headers/footers), and avoid hallucinations. "
    "Respond strictly as JSON."
)

REWRITE_USER_TMPL = """Rewrite the following content into cohesive paragraph(s).
Keep information complete; do not shorten aggressively.

[CONTENT]
{merged_text}

Return STRICTLY JSON:
{{
  "text": "clean rewritten passage preserving all details"
}}
"""


# =========================
# JSON extraction (robust)
# =========================

def extract_json(s: str) -> dict:
    """
    Robustly extract the first complete JSON object from a string.
    Handles extra text, code fences, and trailing commas.
    """
    s = (s or "").strip()

    # Strip common code fences quickly
    if s.startswith("```"):
        # Remove the outermost ```...``` if present
        # (best effort; even if imperfect, bracket matching below will handle)
        s = s.strip("`").strip()

    # If the whole string is a JSON object, try directly
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except Exception:
            pass

    # Bracket matching to find first valid {...}
    start = s.find("{")
    if start == -1:
        raise ValueError(f"No JSON object start found in: {s[:200]}...")

    depth = 0
    in_str = False
    esc = False
    end = None
    for i, ch in enumerate(s[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
    if end is None:
        raise ValueError(f"Unbalanced braces in: {s[:200]}...")

    blob = s[start:end+1]
    try:
        return json.loads(blob)
    except Exception:
        # Remove common trailing commas and retry
        blob2 = re.sub(r",\s*}", "}", blob)
        blob2 = re.sub(r",\s*]", "]", blob2)
        return json.loads(blob2)


# =========================
# Continuity judging utils
# =========================

def judge_continuity_llm(
    client: BaseLLMClient,
    seg_text: str,
    next_text: str,
    temperature: float = 0.0,
    max_tokens: int = 200,
) -> Tuple[bool, float, str]:
    """
    Ask LLM for continuity with minimal JSON schema: {continuous, confidence}.
    Provides a safe fallback to avoid breaking the pipeline.
    """
    try:
        user = CONTINUITY_USER_TMPL.format(seg_text=seg_text, next_text=next_text)
        messages = [
            {"role": "system", "content": CONTINUITY_SYSTEM},
            {"role": "user", "content": user},
        ]
        resp = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
        data = extract_json(resp)
        cont = bool(data.get("continuous", False))
        conf = float(data.get("confidence", 0.0))
        return cont, conf, ""
    except Exception as e:
        # Fallback: treat as non-continuous
        print(f"[WARN] LLM continuity parse failed: {e}")
        return False, 0.0, "parse_failed"


# Cheap signals (language-agnostic + zh)
BULLETS = tuple(["- ", "•", "* ", "– ", "— ", "1.", "2.", "(a)", "(1)", "• "])
RESET_WORDS = {
    # EN
    "outline","agenda","summary","conclusion","references","appendix",
    "overview","contents","table of contents","toc","thank you",
    # ZH
    "大綱","目錄","目次","摘要","結論","參考文獻","附錄","謝謝","感謝","前言"
}

def normalize_lines(s: str) -> List[str]:
    return [ln.strip() for ln in s.splitlines() if ln.strip()]

def is_title_like(line: str) -> bool:
    # short line, title-like capitalization or ALL CAPS
    if len(line) <= 2: return False
    if len(line.split()) <= 8 and (line.isupper() or re.match(r"^([A-Z][a-z0-9]+)(\s+[A-Z][a-z0-9]+)*$", line)):
        return True
    # Chinese titles often short and without punctuation
    if len(line) <= 12 and not re.search(r"[。．.!?？,，:：；;]$", line):
        return True
    return False

def token_set(s: str) -> set:
    toks = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", s.lower())
    return {t for t in toks if len(t) >= 2}

def jaccard(a: set, b: set) -> float:
    if not a or not b: return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def last_sentence_incomplete(lines: List[str]) -> bool:
    if not lines: return False
    tail = lines[-1]
    # no terminal punctuation, or ends with colon/dash/comma
    return not re.search(r"[。．.!?？]$", tail) or tail.endswith((":", "：", "—", "–", ","))

def first_line_bullet_or_continuation(lines: List[str]) -> bool:
    if not lines: return False
    h = lines[0]
    if h.startswith(BULLETS): return True
    # continuation if starts lowercase latin or chinese without title-like pattern
    return h[:1].islower()

def has_reset_signal(lines: List[str]) -> bool:
    if not lines: return False
    first = lines[0].strip()
    if is_title_like(first): return True
    low = first.lower()
    if any(w in low for w in RESET_WORDS):
        return True
    return False

def cheap_continuity_score(prev_text: str, next_text: str) -> float:
    prev_lines = normalize_lines(prev_text)
    next_lines = normalize_lines(next_text)
    j = jaccard(token_set(prev_text), token_set(next_text))

    score = 0.0
    # Positive signals
    if last_sentence_incomplete(prev_lines): score += 0.35
    if first_line_bullet_or_continuation(next_lines): score += 0.35
    score += 0.5 * j  # lexical overlap up to +0.5

    # Negative signals
    if has_reset_signal(next_lines): score -= 0.55

    return max(0.0, min(1.0, score))


def judge_continuity_hybrid(
    client: BaseLLMClient,
    cur_tail_text: str,
    next_text: str,
    cheap_hi: float = 0.80,
    cheap_lo: float = 0.20,
    conf_threshold: float = 0.55,
    lookahead_text: Optional[str] = None,
) -> Tuple[bool, float, str]:
    """
    Hybrid decision: cheap signals first, LLM only in the middle band, with light lookahead smoothing.
    """
    cs = cheap_continuity_score(cur_tail_text, next_text)

    # 1) Fast path
    if cs >= cheap_hi:
        cont, conf, reasons = True, min(0.90, cs), ["cheap:high"]
    elif cs <= cheap_lo:
        cont, conf, reasons = False, min(1.0, 1.0 - cs), ["cheap:low"]
    else:
        # 2) LLM decision
        cont, conf, reason = judge_continuity_llm(client, cur_tail_text, next_text)
        reasons = [reason]

    # 3) Lookahead smoothing
    if cont and lookahead_text is not None:
        la_cs = cheap_continuity_score(next_text, lookahead_text)
        if la_cs <= 0.15:   # next pair likely break
            conf *= 0.85
            reasons.append("lookahead:downweight")

    return cont, conf, "; ".join(reasons)


# =========================
# Jump-aware windowing
# =========================

def continuity_score_combo(client: BaseLLMClient, cur_tail_text: str, cand_text: str, use_llm: bool = True) -> float:
    """
    Combine cheap score and (optional) LLM confidence.
    """
    cs = cheap_continuity_score(cur_tail_text, cand_text)
    llm_conf = 0.0
    if use_llm and 0.2 < cs < 0.8:
        cont, conf, _ = judge_continuity_llm(client, cur_tail_text, cand_text)
        llm_conf = conf if cont else 0.0
    # blend
    return 0.6 * cs + 0.4 * llm_conf

def find_best_jump(
    client: BaseLLMClient,
    cur_tail_text: str,
    page_items: List[Dict[str, Any]],  # [{"page","content"}] sorted
    next_idx: int,                     # index of the immediate next page (0-based)
    W: int = 3,
    lambda_gap: float = 0.15,
) -> Tuple[Optional[int], float]:
    """
    Search within a small window for a better continuation target than next_idx.

    Returns (best_index, best_score), where best_index is 0-based index in page_items,
    or (None, 0.0) if no good candidate.
    """
    n = len(page_items)
    if next_idx + 1 >= n:
        return None, 0.0

    best_j = None
    best_score = 0.0
    # candidates: next_idx+1 ... next_idx+W (0-based index)
    for j in range(next_idx + 1, min(next_idx + W, n - 1) + 1):
        cand_text = page_items[j]["content"]
        base = continuity_score_combo(client, cur_tail_text, cand_text, use_llm=True)

        # distance penalty: number of skipped pages between (next_idx) and j
        gap = j - next_idx - 1
        score = base - lambda_gap * max(0, gap)

        # penalize if any mid page has strong reset signal
        mid_has_reset = False
        for k in range(next_idx, j):
            if has_reset_signal(normalize_lines(page_items[k]["content"])):
                mid_has_reset = True
                break
        if mid_has_reset:
            score -= 0.10

        if score > best_score:
            best_j, best_score = j, score

    return best_j, best_score


# =========================
# Segmentation Logic
# =========================

@dataclass
class Segment:
    file_name: str
    pages: List[int]
    text: str

def truncate_tail(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[-max_chars:]


def make_segments_for_file(
    client: BaseLLMClient,
    file_name: str,
    page_items: List[Dict[str, Any]],   # [{"page", "content"}] sorted
    strategy: str,
    conf_threshold: float,
    context_tail_chars: int,
    cheap_hi: float,
    cheap_lo: float,
    jump_window: int,
    jump_threshold: float,
    jump_gap_penalty: float,
    sleep_sec: float = 0.0,
) -> List[Segment]:
    """
    Iterative + hybrid + jump-aware windowing segmentation.
    """
    if not page_items:
        return []

    segments: List[Segment] = []

    # start first segment with page 0
    cur_pages = [page_items[0]["page"]]
    cur_text = page_items[0]["content"] or ""
    i = 1  # index of the next page candidate (0-based)

    n = len(page_items)
    while i < n:
        nxt = page_items[i]

        # Choose context tail from either current merged segment (iterative) or last single page (pairwise)
        if strategy == "iterative":
            seg_tail = truncate_tail(cur_text, context_tail_chars)
        else:  # pairwise
            prev_single = page_items[i - 1]["content"]
            seg_tail = truncate_tail(prev_single, min(1600, context_tail_chars))

        lookahead_text = page_items[i + 1]["content"] if (i + 1 < n) else None

        cont, conf, _ = judge_continuity_hybrid(
            client=client,
            cur_tail_text=seg_tail,
            next_text=nxt["content"],
            cheap_hi=cheap_hi,
            cheap_lo=cheap_lo,
            conf_threshold=conf_threshold,
            lookahead_text=lookahead_text,
        )

        if sleep_sec > 0:
            time.sleep(sleep_sec)

        if cont and conf >= conf_threshold:
            # merge the immediate next page
            cur_pages.append(nxt["page"])
            cur_text = (cur_text.rstrip() + "\n" + (nxt["content"] or "")).strip()
            i += 1
            continue

        # If not continuous with next page, try jump-aware within window
        seg_tail_for_jump = truncate_tail(cur_text, context_tail_chars)
        best_j, best_score = find_best_jump(
            client=client,
            cur_tail_text=seg_tail_for_jump,
            page_items=page_items,
            next_idx=i,  # the immediate next index we just failed to merge
            W=jump_window,
            lambda_gap=jump_gap_penalty,
        )

        if best_j is not None and best_score >= jump_threshold:
            # Merge the best_j page and skip interlude pages (do not add them into this segment)
            target = page_items[best_j]
            cur_pages.append(target["page"])
            cur_text = (cur_text.rstrip() + "\n" + (target["content"] or "")).strip()
            i = best_j + 1  # jump forward
            continue

        # Otherwise, finalize current segment and start a new one at i
        segments.append(Segment(file_name=file_name, pages=cur_pages[:], text=cur_text.strip()))
        cur_pages = [nxt["page"]]
        cur_text = nxt["content"] or ""
        i += 1

    # flush last
    segments.append(Segment(file_name=file_name, pages=cur_pages[:], text=cur_text.strip()))
    return segments


# =========================
# Rewrite (optional step)
# =========================

def rewrite_segment(
    client: BaseLLMClient,
    merged_text: str,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    user = REWRITE_USER_TMPL.format(merged_text=merged_text)
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": user},
    ]
    resp = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
    data = extract_json(resp)
    text = data.get("text", "").strip()
    if not text:
        raise ValueError("Rewrite returned empty 'text'.")
    return text


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="PPT preprocessing (JSON list input): continuity merge (+ optional rewrite).")
    parser.add_argument("--input", required=True, help="Path to a JSON list with [{'file_name','page','content'}, ...].")
    parser.add_argument("--output-segments", required=True, help="Path to write merged segments JSON.")
    parser.add_argument("--output-jsonl", required=True, help="Path to write rewritten pretrain JSONL.")
    parser.add_argument("--api-type", default="openai", choices=["openai", "ollama", "ollama_py"], help="Local model API type.")
    parser.add_argument("--api-base", default="http://localhost:8000/v1", help="API base URL (OpenAI) or Ollama host (e.g., http://localhost:11434).")
    parser.add_argument("--model", required=True, help="Local model name (e.g., llama3.1:8b-instruct).")
    parser.add_argument("--api-key", default=None, help="API key if using OpenAI-compatible endpoint.")
    parser.add_argument("--strategy", default="iterative", choices=["pairwise", "iterative"], help="Continuity strategy (iterative recommended).")
    parser.add_argument("--conf-threshold", type=float, default=0.55, help="Confidence threshold to merge adjacent page.")
    parser.add_argument("--context-tail-chars", type=int, default=2400, help="Tail chars of current segment used for judgement.")
    parser.add_argument("--cheap-hi", type=float, default=0.80, help="Cheap score high threshold (auto-merge).")
    parser.add_argument("--cheap-lo", type=float, default=0.20, help="Cheap score low threshold (auto-split).")
    parser.add_argument("--jump-window", type=int, default=3, help="Max lookahead window for jump-aware merge.")
    parser.add_argument("--jump-threshold", type=float, default=0.70, help="Score threshold to accept a jump merge.")
    parser.add_argument("--jump-gap-penalty", type=float, default=0.15, help="Per-page penalty for skipped pages during jump merge.")
    parser.add_argument("--sleep-sec", type=float, default=0.0, help="Sleep between LLM calls (rate-limit safety).")
    parser.add_argument("--max-rewrite-tokens", type=int, default=1024, help="Max tokens for rewrite call.")
    parser.add_argument("--dry-run", action="store_true", help="Only produce segments.json, skip rewrite & jsonl.")
    args = parser.parse_args()

    # Load and group
    items = load_items(args.input)
    if not items:
        raise SystemExit("No input items found.")
    grouped = group_by_file(items)

    # Client
    client = make_client(args.api_type, args.api_base, args.model, args.api_key)

    # Process each file_name independently
    all_segments: List[Segment] = []
    for fname in sorted(grouped.keys()):
        page_items = grouped[fname]
        segs = make_segments_for_file(
            client=client,
            file_name=fname,
            page_items=page_items,
            strategy=args.strategy,
            conf_threshold=args.conf_threshold,
            context_tail_chars=args.context_tail_chars,
            cheap_hi=args.cheap_hi,
            cheap_lo=args.cheap_lo,
            jump_window=args.jump_window,
            jump_threshold=args.jump_threshold,
            jump_gap_penalty=args.jump_gap_penalty,
            sleep_sec=args.sleep_sec,
        )
        all_segments.extend(segs)

    # Save segments JSON
    out_dir = os.path.dirname(args.output_segments) or "."
    os.makedirs(out_dir, exist_ok=True)
    seg_out = []
    for s in all_segments:
        seg_out.append({
            "file_name": s.file_name,
            "pages": s.pages,
            "text": s.text,
        })

    # Atomic write
    tmp = args.output_segments + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"segments": seg_out}, f, ensure_ascii=False, indent=4)
    os.replace(tmp, args.output_segments)

    if args.dry_run:
        print(f"[Dry-run] Wrote segments to: {args.output_segments}. Skipped rewrite.")
        return

    # # Rewrite merged segments -> JSONL
    # out_dir2 = os.path.dirname(args.output_jsonl) or "."
    # os.makedirs(out_dir2, exist_ok=True)
    # written = 0
    # with open(args.output_jsonl, "w", encoding="utf-8") as fout:
    #     for s in all_segments:
    #         try:
    #             rewritten = rewrite_segment(
    #                 client=client,
    #                 merged_text=s.text,
    #                 temperature=0.2,
    #                 max_tokens=args.max_rewrite_tokens,
    #             )
    #         except Exception as e:
    #             print(f"[WARN] Rewrite failed for {s.file_name} pages {s.pages}: {e}")
    #             rewritten = s.text  # fallback: keep original

    #         item = {
    #             "text": rewritten,
    #             "meta": {
    #                 "id": str(uuid.uuid4()),
    #                 "source": s.file_name,
    #                 "pages": {"start": s.pages[0], "end": s.pages[-1], "list": s.pages},
    #                 "num_pages": len(s.pages),
    #             },
    #             "original": {
    #                 "text": s.text
    #             }
    #         }
    #         fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    #         written += 1

    # print(f"Done.\n- Segments: {args.output_segments}\n- Pretrain JSONL: {args.output_jsonl}\n- Rewritten segments: {written}")


if __name__ == "__main__":
    main()
