#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ppt_preprocess.py  —  Theme-Driven, Jump-Aware PPT Preprocessor
Version: r1 (2025-08-09)

目的：
1) 以「結構化 slide card」＋「段落主題 theme」來判斷連續性（比 tail 更適合碎片化的投影片）
2) 支援「跳頁容忍」（jump-aware window）：中間插播頁（Outline/References/圖頁）可被跳過，銜接後續延續內容
3) （可選）把合併段落丟給本地 LLM 改寫成連貫文章

輸入（JSON list）：
[
  { "file_name": "...", "page": 1, "content": "..." },
  { "file_name": "...", "page": 2, "content": "..." },
  ...
]

輸出：
- segments.json : {"segments":[{"file_name","pages":[...],"text":...}, ...]}
- pretrain.jsonl: 每行一筆 {"text": "...", "meta": {...}, "original": {...}}  （不帶 --dry-run 時才產出）

支援後端：
- --api-type openai     : OpenAI 相容 /chat/completions（vLLM、LM Studio 等）
- --api-type ollama     : Ollama HTTP /api/chat
- --api-type ollama_py  : 官方 Python 套件 `import ollama`

相依套件：
- requests（必要）
- ollama（若使用 --api-type ollama_py）
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
# LLM Clients
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
        return data["choices"][0]["message"]["content"]


class OllamaHTTPClient(BaseLLMClient):
    """
    Ollama /api/chat with options.format='json' + num_predict=-1 to avoid truncation/invalid JSON.
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
                "num_predict": -1,  # let model finish JSON
                "format": "json",
            },
            "stream": False
        }
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "")


class OllamaPyClient(BaseLLMClient):
    """
    'import ollama' client. Honors OLLAMA_HOST or a host provided via --api-base.
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
            "num_predict": -1,
            "format": "json",
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
        if not base_url or base_url == "http://localhost:8000/v1":
            base_url = "http://localhost:11434"
        return OllamaHTTPClient(base_url=base_url, model=model)
    elif api_type == "ollama_py":
        host = base_url if (base_url and base_url.startswith("http")) else os.getenv("OLLAMA_HOST")
        return OllamaPyClient(model=model, host=host)
    else:
        raise ValueError("Unsupported --api-type. Use 'openai', 'ollama', or 'ollama_py'.")


# =========================
# IO Helpers
# =========================

def load_items(input_path: str) -> List[Dict[str, Any]]:
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
    "你是一個嚴謹的投影片段落分段器。請只輸出 JSON，不要額外解釋或 Markdown。"
)

# 最小 JSON，降低截斷風險
CONTINUITY_THEME_TMPL = """Judge if the NEXT slide continues the SAME topic as the CURRENT SEGMENT THEME.

[CURRENT_SEGMENT_THEME]
{theme_json}

[NEXT_SLIDE_CARD]
{card_json}

Rules:
- "continuous": true if the next slide logically extends the same topic (bullet list continuation, same figure/section, same method steps).
- Titles like Outline/References/Conclusion are NOT continuous.
- Consider title, bullets, flow steps, figure captions, and body text holistically.

Return JSON only:
{{
  "continuous": true/false,
  "confidence": 0.0~1.0
}}
"""

THEME_SYSTEM = "You are a careful technical editor. Work only in JSON."

THEME_INIT_TMPL = """Summarize the topic of this slide for starting a segment.

[SLIDE_CARD_JSON]
{card_json}

Return JSON only:
{{
  "theme_summary": "<=80 words summary of the core topic",
  "key_terms": ["t1","t2","t3","t4","t5"]
}}
"""

THEME_UPDATE_TMPL = """Update the current segment theme with the new slide.

[CURRENT_THEME]
{theme_json}

[NEW_SLIDE_CARD]
{card_json}

Return JSON only:
{{
  "theme_summary": "<=100 words updated summary (cover both)",
  "key_terms": ["t1","t2","t3","t4","t5","t6","t7"]
}}
"""

REWRITE_SYSTEM = (
    "You are a professional technical editor. "
    "Rewrite the given merged PPT content into a clean, coherent training paragraph for LLM continual pretraining. "
    "Preserve all factual details across pages, merge bullets into sentences, expand abbreviations once, "
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
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except Exception:
            pass

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
        blob2 = re.sub(r",\s*}", "}", blob)
        blob2 = re.sub(r",\s*]", "]", blob2)
        return json.loads(blob2)


# =========================
# Slide card parsing
# =========================

def is_title_like(line: str) -> bool:
    if len(line) <= 2: return False
    if len(line) <= 12 and not re.search(r"[。．.!?？,，:：；;]$", line):
        return True
    if len(line.split()) <= 8 and (line.isupper() or re.match(r"^([A-Z][a-z0-9]+)(\s+[A-Z][a-z0-9]+)*$", line)):
        return True
    return False

def make_slide_card(content: str) -> dict:
    lines = [ln.strip() for ln in (content or "").splitlines() if ln.strip()]
    if not lines:
        return {"title":"", "bullets":[], "flow":[], "captions":[], "body":""}

    title = lines[0] if is_title_like(lines[0]) else ""
    start_idx = 1 if title else 0

    bullets, flow, captions, body_lines = [], [], [], []
    for ln in lines[start_idx:]:
        if re.match(r"^(\-|\•|\*|–|—|\d+\.)\s", ln):
            bullets.append(ln)
        elif re.match(r"^(Step\s*\d+|流程\s*\d+|步驟\s*\d+|[（(]?\d+[)）]\s*[:：]?)", ln):
            flow.append(ln)
        elif re.match(r"^(Figure|Fig\.|圖|表|Table)\s*\d*", ln, flags=re.I):
            captions.append(ln)
        else:
            body_lines.append(ln)

    body = "\n".join(body_lines).strip()
    return {"title": title, "bullets": bullets, "flow": flow, "captions": captions, "body": body}


# =========================
# Theme init / update
# =========================

def llm_json(client: BaseLLMClient, system: str, user: str, temperature=0.0, max_tokens=256) -> dict:
    messages = [{"role":"system","content":system},{"role":"user","content":user}]
    resp = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
    return extract_json(resp)

def init_theme_from_card(client: BaseLLMClient, card: dict) -> dict:
    user = THEME_INIT_TMPL.format(card_json=json.dumps(card, ensure_ascii=False))
    data = llm_json(client, THEME_SYSTEM, user, temperature=0.0, max_tokens=256)
    return {
        "theme_summary": data.get("theme_summary",""),
        "key_terms": data.get("key_terms",[])[:7]
    }

def update_theme_with_card(client: BaseLLMClient, theme: dict, card: dict) -> dict:
    user = THEME_UPDATE_TMPL.format(
        theme_json=json.dumps(theme, ensure_ascii=False),
        card_json=json.dumps(card, ensure_ascii=False)
    )
    data = llm_json(client, THEME_SYSTEM, user, temperature=0.0, max_tokens=320)
    return {
        "theme_summary": data.get("theme_summary",""),
        "key_terms": data.get("key_terms",[])[:10]
    }


# =========================
# Continuity judging (theme + card)
# =========================

RESET_WORDS = {
    # English
    "outline","agenda","summary","conclusion","references","appendix","overview","contents","table of contents","toc","thank you",
    # Chinese
    "大綱","目錄","目次","摘要","結論","參考文獻","附錄","謝謝","感謝","前言"
}

def has_reset_signal_from_card(card: dict) -> bool:
    first_line = (card.get("title") or "").strip()
    low = first_line.lower()
    return any(w in low for w in RESET_WORDS)

def tokenize_text(s: str) -> set:
    toks = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", (s or "").lower())
    return {t for t in toks if len(t) >= 2}

def judge_continuity_theme(client: BaseLLMClient, theme: dict, next_card: dict) -> Tuple[bool, float]:
    user = CONTINUITY_THEME_TMPL.format(
        theme_json=json.dumps(theme, ensure_ascii=False),
        card_json=json.dumps(next_card, ensure_ascii=False),
    )
    data = llm_json(client, CONTINUITY_SYSTEM, user, temperature=0.0, max_tokens=200)
    return bool(data.get("continuous", False)), float(data.get("confidence", 0.0))

def continuity_score_theme_combo(client: BaseLLMClient, theme: dict, cand_card: dict, use_llm: bool = True) -> float:
    keyset = set([k.lower() for k in theme.get("key_terms", []) if isinstance(k, str)])
    cand_text = " ".join([cand_card.get("title","")] + cand_card.get("bullets",[]) +
                         cand_card.get("flow",[]) + cand_card.get("captions",[]) + [cand_card.get("body","")])
    cand_tokens = tokenize_text(cand_text)
    union = keyset | cand_tokens
    j = (len(keyset & cand_tokens) / len(union)) if union else 0.0

    llm_conf = 0.0
    if use_llm:
        cont, conf = judge_continuity_theme(client, theme, cand_card)
        llm_conf = conf if cont else 0.0

    return 0.5 * j + 0.5 * llm_conf


# =========================
# Jump-aware window search (theme-based)
# =========================

def find_best_jump_theme(
    client: BaseLLMClient,
    theme: dict,
    cards: List[dict],
    page_items: List[Dict[str, Any]],
    next_idx: int,              # index of immediate next page (0-based)
    W: int = 3,
    lambda_gap: float = 0.15,
) -> Tuple[Optional[int], float]:
    """
    在窗口內（next_idx+1..next_idx+W）尋找最佳的延續頁（允許跨過插播頁）。
    回傳 (best_index, best_score) 或 (None, 0.0)
    """
    n = len(cards)
    if next_idx + 1 >= n:
        return None, 0.0

    best_j, best_score = None, 0.0
    for j in range(next_idx + 1, min(next_idx + W, n - 1) + 1):
        cand_card = cards[j]
        base = continuity_score_theme_combo(client, theme, cand_card, use_llm=True)

        # distance penalty
        gap = j - next_idx - 1
        score = base - lambda_gap * max(0, gap)

        # penalize if mid pages look like resets
        mid_has_reset = False
        for k in range(next_idx, j):
            if has_reset_signal_from_card(cards[k]):
                mid_has_reset = True
                break
        if mid_has_reset:
            score -= 0.10

        if score > best_score:
            best_j, best_score = j, score

    return best_j, best_score


# =========================
# Segmentation
# =========================

@dataclass
class Segment:
    file_name: str
    pages: List[int]
    text: str

def make_segments_for_file(
    client: BaseLLMClient,
    file_name: str,
    page_items: List[Dict[str, Any]],   # [{"page","content"}] sorted
    conf_threshold: float,
    jump_window: int,
    jump_threshold: float,
    jump_gap_penalty: float,
    sleep_sec: float = 0.0,
) -> List[Segment]:
    if not page_items:
        return []

    cards = [make_slide_card(x["content"]) for x in page_items]
    n = len(page_items)

    segments: List[Segment] = []

    # start first segment
    cur_pages = [page_items[0]["page"]]
    cur_text  = page_items[0]["content"] or ""
    cur_theme = init_theme_from_card(client, cards[0])
    i = 1  # index of next candidate

    while i < n:
        next_card = cards[i]

        # theme-based continuity
        cont, conf = judge_continuity_theme(client, cur_theme, next_card)

        # simple lookahead downweight if next-next looks like a break
        if cont and i + 1 < n:
            la_score = continuity_score_theme_combo(client, cur_theme, cards[i+1], use_llm=False)
            if la_score <= 0.15:
                conf *= 0.85

        if sleep_sec > 0:
            time.sleep(sleep_sec)

        if cont and conf >= conf_threshold:
            # merge immediate next
            cur_pages.append(page_items[i]["page"])
            cur_text = (cur_text.rstrip() + "\n" + (page_items[i]["content"] or "")).strip()
            cur_theme = update_theme_with_card(client, cur_theme, next_card)
            i += 1
            continue

        # try jump-aware within window
        best_j, best_score = find_best_jump_theme(
            client=client,
            theme=cur_theme,
            cards=cards,
            page_items=page_items,
            next_idx=i,
            W=jump_window,
            lambda_gap=jump_gap_penalty,
        )

        if best_j is not None and best_score >= jump_threshold:
            # merge best_j and jump forward; skip interlude pages
            cur_pages.append(page_items[best_j]["page"])
            cur_text = (cur_text.rstrip() + "\n" + (page_items[best_j]["content"] or "")).strip()
            cur_theme = update_theme_with_card(client, cur_theme, cards[best_j])
            i = best_j + 1
            continue

        # finalize segment and start new one at i
        segments.append(Segment(file_name=file_name, pages=cur_pages[:], text=cur_text.strip()))
        cur_pages = [page_items[i]["page"]]
        cur_text  = page_items[i]["content"] or ""
        cur_theme = init_theme_from_card(client, cards[i])
        i += 1

    # flush last
    segments.append(Segment(file_name=file_name, pages=cur_pages[:], text=cur_text.strip()))
    return segments


# =========================
# Rewrite (optional)
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
    parser = argparse.ArgumentParser(description="PPT preprocessing (JSON list): theme-driven continuity merge (+ optional rewrite).")
    parser.add_argument("--input", required=True, help="Path to a JSON list with [{'file_name','page','content'}, ...].")
    parser.add_argument("--output-segments", required=True, help="Path to write merged segments JSON.")
    parser.add_argument("--output-jsonl", required=True, help="Path to write rewritten pretrain JSONL.")
    parser.add_argument("--api-type", default="openai", choices=["openai", "ollama", "ollama_py"], help="Local model API type.")
    parser.add_argument("--api-base", default="http://localhost:8000/v1", help="API base URL (OpenAI) or Ollama host (e.g., http://localhost:11434).")
    parser.add_argument("--model", required=True, help="Local model name (e.g., llama3.1:8b-instruct).")
    parser.add_argument("--api-key", default=None, help="API key if using OpenAI-compatible endpoint.")
    parser.add_argument("--conf-threshold", type=float, default=0.60, help="Confidence threshold to merge adjacent or theme-continued page.")
    parser.add_argument("--jump-window", type=int, default=3, help="Max lookahead window for jump-aware merge.")
    parser.add_argument("--jump-threshold", type=float, default=0.72, help="Score threshold to accept a jump merge.")
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
            conf_threshold=args.conf_threshold,
            jump_window=args.jump_window,
            jump_threshold=args.jump_threshold,
            jump_gap_penalty=args.jump_gap_penalty,
            sleep_sec=args.sleep_sec,
        )
        all_segments.extend(segs)

    # Save segments JSON (atomic)
    out_dir = os.path.dirname(args.output_segments) or "."
    os.makedirs(out_dir, exist_ok=True)
    seg_out = [{"file_name": s.file_name, "pages": s.pages, "text": s.text} for s in all_segments]

    tmp = args.output_segments + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"segments": seg_out}, f, ensure_ascii=False, indent=4)
    os.replace(tmp, args.output_segments)

    if args.dry_run:
        print(f"[Dry-run] Wrote segments to: {args.output_segments}. Skipped rewrite.")
        return

    # Rewrite merged segments -> JSONL
    out_dir2 = os.path.dirname(args.output_jsonl) or "."
    os.makedirs(out_dir2, exist_ok=True)
    written = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for s in all_segments:
            try:
                rewritten = rewrite_segment(
                    client=client,
                    merged_text=s.text,
                    temperature=0.2,
                    max_tokens=args.max_rewrite_tokens,
                )
            except Exception as e:
                print(f"[WARN] Rewrite failed for {s.file_name} pages {s.pages}: {e}")
                rewritten = s.text  # fallback

            item = {
                "text": rewritten,
                "meta": {
                    "id": str(uuid.uuid4()),
                    "source": s.file_name,
                    "pages": {"start": s.pages[0], "end": s.pages[-1], "list": s.pages},
                    "num_pages": len(s.pages),
                },
                "original": {"text": s.text}
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1

    print(f"Done.\n- Segments: {args.output_segments}\n- Pretrain JSONL: {args.output_jsonl}\n- Rewritten segments: {written}")


if __name__ == "__main__":
    main()
