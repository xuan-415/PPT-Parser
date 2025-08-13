score_prompt="""你是「PPT 連續性審核員」。目標：為 continual pretraining 的改寫做分段決策，評估 A 與 B 是否應合併（偏向合併以避免資訊遺漏）。
僅依據 A/B 文字，禁止外部常識或猜測。請先分析再計分，最後只輸出 JSON（4 空格縮排）。

【需要的分析與打分】
1) Closures（0-1）：B 是否關閉 A 的缺口（指代/符號、單位/數值、流程承接、圖表說明）。按嚴重度加權：low=1, mid=2, high=3。
   -> closures = (加權已關閉缺口) / (加權缺口總量)。若 A 無缺口，視為 0.7（可被 threads/bridge 拉高）。
2) Threads（0-1）：A 的關鍵「主題實體/變量/符號/目標」在 B 的延續比例；單位一致性與指代清晰度會提升分數。
3) CoverageLift（0-1）：先以 A+B 萃取 5–12 條原子事實（具名詞/變量/單位/關係），估 coverage_A、coverage_B、coverage_AB。
   -> coverage_lift = max(0, coverage_AB − max(coverage_A, coverage_B)).
4) Bridge（0/0.5/1）：能否在不新增新主題/數值的前提下，寫出 1 句自然承接句？
   -> 成功=1；僅有輕微違規（措辭補充非新概念）=0.5；不可行=0。
5) FB-Overlap（0-1）：從 A 前視的「合理下一頁元素」與從 B 後視的「所需前提」的交集比例；是否推進 A 的任務 (+0.1 上限)。
6) Penalties（0-1 各自）：硬性阻斷信號
   - topic_shift：顯著換題（主題/任務皆不同且無共享線索）
   - contradiction：定義/符號/單位/結論矛盾（無法靠改寫統一）
   - missing_prereq：B 需要的前提在 A/B 皆未出現，且 bridge 失敗

【最終輸出 JSON（4 空格縮排）】
{
    "scores": {
        "closures": 0-1,
        "threads": 0-1,
        "coverage_lift": 0-1,
        "bridge": 0|0.5|1,
        "fb_overlap": 0-1,
        "penalty_topic_shift": 0-1,
        "penalty_contradiction": 0-1,
        "penalty_missing_prereq": 0-1
    },
    "final_score": 0-1,
    "decision": "MERGE" | "KEEP_SEPARATE",
    "facts": [{"id":"F1","text":"...","from_A":true/false,"from_B":true/false,"requires_AB":true/false}],
    "closures": [{"type":"符號定義","evidence_A":"...","evidence_B":"..."}],
    "threads_continued": ["實體/變量/符號..."],
    "bridge_sentence": "若 bridge>0 則給 1 句，否則空字串",
    "inconsistencies": ["若有，具體描述"],
    "rewrite_hints": ["定義前置…","單位統一為 mTorr","以橋接句開頭…"],
    "rationale": "限制兩句：為何（不）合併"
}

A:
{{page_A_text}}

B:
{{page_B_text}}
"""


binary_prompt="""你是「PPT 連續性審核員」。目標：為了後續改寫與 continual pretraining，決定 A 與 B 是否應合併，盡量避免因分割造成的資訊遺漏。
禁止使用外部常識；僅依據 A/B 文字。

決策原則（MERGE-biased）：
(0) 預設：MERGE
(1) 僅當命中任一【硬性阻斷】時改判 KEEP_SEPARATE：
    - 顯著換題：A 與 B 主題/任務明顯不同，且【沒有】共享關鍵實體/變量/符號
    - 關鍵矛盾：定義、單位、符號、結論相互衝突（且無法在合併後以「單位統一/定義前置」消解）
    - 缺前提且不可橋接：B 需要的前置在 A/B 皆未出現，且無法用一句不引入新概念的承接句弭平
(2) 若無硬性阻斷，但仍不確定，執行【橋接可行性】：
    - 嘗試用 1 句不新增新主題/數值的承接句把 A 引到 B；若可行則保持 MERGE
(3) 請輸出下列 JSON（4 空格縮排）：
{
    "decision": "MERGE" | "KEEP_SEPARATE",
    "hard_blockers": ["顯著換題","關鍵矛盾","缺前提且不可橋接"] 或 [],
    "shared_threads": ["延續的實體/變量/符號..."],
    "closures": [{"type":"符號定義/單位補齊/步驟承接/圖表說明","evidence_A":"...","evidence_B":"..."}],
    "bridge_sentence": "若可行請給 1 句；否則空字串",
    "risk_flags": ["單位不一致(可統一)","指代稍弱(可補承接句)"] 或 [],
    "rewrite_hints": [
        "將 n 的定義與單位於段首前置",
        "統一壓力單位為 mTorr",
        "以橋接句開頭：『既然上頁定義了X，接著我們…』"
    ],
    "rationale": "限制兩句：為何（不）合併"
}
A:
{{page_A_text}}

B:
{{page_B_text}}
"""