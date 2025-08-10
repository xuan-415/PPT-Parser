# -*- coding: utf-8 -*-
"""
Hybrid Sliding-Window × Pairwise Continuity
-------------------------------------------
步驟：
1) 在 [i+1..i+W] 取候選
2) LLM 單輪多候選打分 -> 取 Top-M
3) 只對 Top-M 做 pairwise 對決重排
4) 距離懲罰 + 門檻 + Top-K 保留，若全未達標可回退相鄰頁
"""

from typing import List, Dict, Any, Optional, Tuple
import json, re, math, random

# ========= LLM 介面（你可換成 OpenAI/Ollama 的 complete(prompt, temperature)） =========
class LLMClient:
    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        raise NotImplementedError

# ---- Mock LLM：用關鍵字重疊來近似（可直接跑示範） ----
class MockHybridLLM(LLMClient):
    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        if "候選 A（頁" in prompt and "候選 B（頁" in prompt:
            # Pairwise 問法：回 "A" 或 "B"
            cur = self._between(prompt, "當前頁：", "候選 A").lower()
            ta = self._between(prompt, "候選 A（頁", "候選 B").lower()
            tb = prompt.lower().split("候選 B（頁", 1)[1]
            tb = tb.split("）", 1)[1].strip()
            score_a = self._overlap(cur, ta)
            score_b = self._overlap(cur, tb)
            if abs(score_a - score_b) < 1:  # 小差距視為平手，偏向 A/B 隨機
                return random.choice(["A", "B"])
            return "A" if score_a > score_b else "B"
        else:
            # Seed 多候選：回 JSON scores
            cur = self._between(prompt, "當前頁：", "候選頁").lower()
            cands = self._split_cands(prompt.lower())
            cur_tokens = set(self._tok(cur))
            scores = []
            for i, (_lab, txt) in enumerate(cands, start=1):
                overlap = len(cur_tokens.intersection(set(self._tok(txt))))
                s = min(1.0, overlap/10.0 + 0.35)  # 只是示意
                scores.append({"idx": i, "score": round(s, 2)})
            return json.dumps({"scores": scores}, ensure_ascii=False)

    def _tok(self, s: str): return re.findall(r"[a-z0-9]+", s)
    def _overlap(self, a: str, b: str):
        A, B = set(self._tok(a)), set(self._tok(b))
        return len(A & B)
    def _between(self, text: str, a: str, b: str) -> str:
        try:
            s = text.index(a) + len(a)
            e = text.index(b, s)
            return text[s:e]
        except: return ""
    def _split_cands(self, text: str):
        blocks = []
        for m in re.finditer(r"\[候選\s+(\d+)\s*-\s*頁\s*\d+\]\s*(.+?)(?=(?:\n\[候選|\Z))", text, flags=re.S):
            blocks.append((m.group(1), m.group(2)))
        return blocks

# ========= Prompt =========
PROMPT_RANK = """你是PPT連續性評審。請對每個候選頁作為「當前頁」的自然續接程度給 0~1 分數，僅輸出一行 JSON：
{"scores":[{"idx":1,"score":0.82},{"idx":2,"score":0.15},...]}
評分原則：主題/術語延續 > 段落承接 > 同圖表/流程延展；插頁/題外給低分。

當前頁：
---
{cur}
---

候選頁（1..N）：
{cands}
"""

PROMPT_PAIR = """當前頁：
{cur}

候選 A（頁 {pa}）：
{ta}

候選 B（頁 {pb}）：
{tb}

哪個更自然承接當前頁？只回 "A" 或 "B"。
"""

def _cands_block(cand_pages: List[int], cand_texts: List[str]) -> str:
    parts = []
    for i, (p, t) in enumerate(zip(cand_pages, cand_texts), start=1):
        parts.append(f"[候選 {i} - 頁 {p}]\n{t}\n")
    return "\n".join(parts)

# ========= Seed：單輪多候選打分 =========
def seed_scores(llm: LLMClient, cur_text: str, cand_pages: List[int], cand_texts: List[str],
                temperature: float = 0.0) -> List[Tuple[int, float]]:
    prompt = PROMPT_RANK.format(cur=cur_text, cands=_cands_block(cand_pages, cand_texts))
    raw = llm.complete(prompt, temperature=temperature).strip()
    try:
        obj = json.loads(raw)
        out = []
        for item in obj.get("scores", []):
            k = int(item["idx"]); s = float(item["score"])
            if 1 <= k <= len(cand_texts):
                out.append((k-1, max(0.0, min(1.0, s))))
        return out
    except:
        # 解析失敗 -> 全 0
        return [(i, 0.0) for i in range(len(cand_texts))]

# ========= Pairwise：只對 Top-M 做對決 =========
def pairwise_refine(llm: LLMClient, cur_text: str, cand_pages: List[int], cand_texts: List[str],
                    top_m_indices: List[int], schedule: str = "star",
                    temperature: float = 0.0) -> List[Tuple[int, float]]:
    """
    回傳 [(i_in_window, pair_score_0_1)...]；僅對 top_m_indices 做對決，其餘候選留給 seed 分數。
    schedule:
      - "star": 以 seed 第一名為 pivot，依序與其餘對決
      - "roundrobin": 小規模全對決（M<=5建議）
    """
    if len(top_m_indices) <= 1:
        return [(idx, 1.0) for idx in top_m_indices]

    wins = {idx: 0.0 for idx in top_m_indices}
    games = {idx: 0.0 for idx in top_m_indices}

    def duel(i_idx: int, j_idx: int):
        pa, pb = cand_pages[i_idx], cand_pages[j_idx]
        ta, tb = cand_texts[i_idx], cand_texts[j_idx]
        raw = llm.complete(PROMPT_PAIR.format(cur=cur_text, pa=pa, ta=ta, pb=pb, tb=tb), temperature=temperature).strip().upper()
        if "A" in raw and "B" not in raw:
            wins[i_idx] += 1; games[i_idx] += 1; games[j_idx] += 1
        elif "B" in raw and "A" not in raw:
            wins[j_idx] += 1; games[i_idx] += 1; games[j_idx] += 1
        else:
            wins[i_idx] += 0.5; wins[j_idx] += 0.5; games[i_idx] += 1; games[j_idx] += 1

    if schedule == "star":
        pivot = top_m_indices[0]
        for j in top_m_indices[1:]:
            duel(pivot, j)
    else:  # roundrobin
        for a in range(len(top_m_indices)):
            for b in range(a+1, len(top_m_indices)):
                duel(top_m_indices[a], top_m_indices[b])

    # 正規化 0~1
    out = []
    for idx in top_m_indices:
        g = max(1.0, games[idx])
        out.append((idx, wins[idx]/g))
    return out

# ========= 融合 seed 與 pairwise，再套距離懲罰，挑 Top-K =========
def pick_multi_edges(cur_index: int,
                     cand_global_indices: List[int],
                     seed_ranked: List[Tuple[int, float]],
                     pair_subset_scores: List[Tuple[int, float]],
                     min_conf: float = 0.6, max_keep: int = 2,
                     lambda_gap: float = 0.15) -> List[int]:
    """
    seed_ranked: [(i_in_win, seed_score)]
    pair_subset_scores: 只含 Top-M 的 (i_in_win, pair_score)
    做法：final_score = 0.5*seed + 0.5*pair(若有，否則就是seed)；再乘距離懲罰。
    """
    seed_map = {i: s for i, s in seed_ranked}
    pair_map = {i: s for i, s in pair_subset_scores}

    scored = []
    for i_in_win, g in zip(range(len(cand_global_indices)), cand_global_indices):
        base = seed_map.get(i_in_win, 0.0)
        pair = pair_map.get(i_in_win, None)
        mix = 0.5*base + 0.5*(pair if pair is not None else base)
        gap = g - cur_index
        adj = mix * math.exp(-lambda_gap * max(0, gap-1))
        scored.append((g, mix, adj, gap))

    scored.sort(key=lambda x: x[2], reverse=True)
    kept = [g for (g, mix, adj, gap) in scored if adj >= min_conf][:max_keep]
    return kept

# ========= 主流程：每頁做一次混合排名 =========
def analyze_hybrid_pairwise(llm: LLMClient,
                            page_items: List[Dict[str, Any]],
                            W: int = 4,
                            M: int = 3,                  # 進入 pairwise 的 Top-M
                            schedule: str = "star",      # "star" 或 "roundrobin"
                            min_conf: float = 0.6,
                            max_keep: int = 2,
                            lambda_gap: float = 0.15,
                            prefer_adjacent_if_none: bool = True,
                            temperature_seed: float = 0.0,
                            temperature_pair: float = 0.0):
    results = []
    n = len(page_items)
    for i in range(n):
        if i >= n-1:
            results.append({
                "cur_index": i,
                "cur_page": page_items[i]["page"],
                "next_indices": [],
                "next_pages": [],
                "dbg": {}
            })
            continue

        start, end = i+1, min(i+W, n-1)
        cand_idx = list(range(start, end+1))
        cand_pages = [page_items[j]["page"] for j in cand_idx]
        cand_texts = [page_items[j]["content"] for j in cand_idx]

        # 1) seed 粗排
        seed_ranked = seed_scores(llm, page_items[i]["content"], cand_pages, cand_texts,
                                  temperature=temperature_seed)  # [(i_in_win, score)]
        # Top-M（依 seed 分數）
        seed_sorted = sorted(seed_ranked, key=lambda x: x[1], reverse=True)
        top_m = [idx for idx, _ in seed_sorted[:min(M, len(seed_sorted))]]

        # 2) pairwise 只比較 Top-M
        pair_scores = pairwise_refine(llm, page_items[i]["content"], cand_pages, cand_texts,
                                      top_m_indices=top_m, schedule=schedule,
                                      temperature=temperature_pair)

        # 3) 融合 + 懲罰 + Top-K
        chosen = pick_multi_edges(i, cand_idx, seed_ranked, pair_scores,
                                  min_conf=min_conf, max_keep=max_keep, lambda_gap=lambda_gap)

        if not chosen and prefer_adjacent_if_none:
            chosen = [i+1]

        results.append({
            "cur_index": i,
            "cur_page": page_items[i]["page"],
            "next_indices": chosen,
            "next_pages": [page_items[j]["page"] for j in chosen],
            "dbg": {
                "window_index_range": [start, end],
                "seed_ranked": seed_ranked,
                "top_m": top_m,
                "pair_scores": pair_scores
            }
        })
    return results

# ========= Demo =========
if __name__ == "__main__":
    page_items = [
        {"page": 1, "content": "Intro: define Method A and study objectives; outline Sections A, B."},
        {"page": 2, "content": "Method overview: pipeline of Method A; components and training."},
        {"page": 3, "content": "Method A details: loss functions, optimization, and ablations."},
        {"page": 4, "content": "Method A qualitative results: figures and case studies."},
        {"page": 5, "content": "Method A quantitative results: metrics and tables; ablation study."},
        {"page": 6, "content": "Conclusion and future work."},
    ]
    llm = MockHybridLLM()
    graph = analyze_hybrid_pairwise(
        llm, page_items,
        W=4,           # 視窗看後 4 頁
        M=3,           # 只對 seed 前 3 名做 pairwise
        schedule="star",   # "star" 便宜；"roundrobin" 穩但多次
        min_conf=0.6,
        max_keep=2,
        lambda_gap=0.18,
        temperature_seed=0.0,
        temperature_pair=0.0
    )

    print(json.dumps(graph, ensure_ascii=False, indent=4))
    for row in graph:
        nxt = ", ".join([f"p{p}" for p in row["next_pages"]]) or "None"
        print(f"p{row['cur_page']} → {nxt}")
