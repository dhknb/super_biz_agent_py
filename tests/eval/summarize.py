"""把多份评测报告并成一张对照表。

用法:
    python -m tests.eval.summarize                      # 汇总 reports/ 下所有报告
    python -m tests.eval.summarize --baseline mq-A      # 指定 Δ 的基准组
    python -m tests.eval.summarize mq-A mq-C mq-E       # 只看这几组(按 tag 前缀)

为什么要有这个脚本:

手工从 JSON 里抄数字算 Δ 出过一次真实的错 —— 拿 82 条样本的报告去和一份
18 条样本的旧报告比,两个数都是真的,差值却毫无意义,而且看不出问题。
这里把「同一评测集、同一 top_k、同一样本数」变成脚本里的断言:
不满足就打警告,而不是安静地算出一个漂亮的提升。

同名 tag 有多份时只取**最新**那份(按文件名里的时间戳)。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPORTS_DIR = Path(__file__).parent / "reports"

# 只看这几个 K:全库 16 篇文档时 Hit@10 等于返回了 62% 的库,没有区分度。
SHOWN_KS = (1, 3, 5)


def load_reports(prefixes: list[str] | None) -> list[dict[str, Any]]:
    """读 reports/*.json,同 tag 取最新一份。

    文件名形如 <tag>-<ISO时间戳>.json,而 tag 本身含 '-',所以不能按 '-'
    切分取 tag —— 直接读文件里的 tag 字段,那是唯一可靠的来源。
    """
    by_tag: dict[str, tuple[str, dict[str, Any]]] = {}

    for path in sorted(REPORTS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[SKIP] {path.name}: 读不了 ({exc})")
            continue

        tag = data.get("tag") or path.stem
        if prefixes and not any(tag.startswith(p) for p in prefixes):
            continue

        stamp = data.get("timestamp") or path.stem
        # 同 tag 多次运行:留时间戳更大的那份
        if tag not in by_tag or stamp > by_tag[tag][0]:
            by_tag[tag] = (stamp, data)

    rows = [data for _stamp, data in by_tag.values()]
    rows.sort(key=lambda d: d.get("tag", ""))
    return rows


def _metrics(report: dict[str, Any]) -> dict[str, float]:
    """抽出要上表的指标。

    hit_at_k 的键经过 JSON 往返后是字符串,这里统一按 str(k) 取 ——
    按 int 取会全部落空,然后表里安静地显示一片 0.0000。
    """
    summary = report.get("summary") or {}
    hit = summary.get("hit_at_k") or {}
    meta = report.get("run_meta") or {}

    out: dict[str, float] = {
        "mrr": float(summary.get("mrr") or 0.0),
        "sec": float(meta.get("avg_seconds_per_case") or 0.0),
    }
    for k in SHOWN_KS:
        out[f"hit{k}"] = float(hit.get(str(k), hit.get(k, 0.0)) or 0.0)
    return out


def _credibility_notes(report: dict[str, Any]) -> list[str]:
    """这份报告有没有已知的可信度问题。

    这些字段哪怕全是 0 也值得检查 —— 一份被网络抖动打脏的报告和一份
    干净的报告,指标格式上完全一样,只有这里能区分。
    """
    meta = report.get("run_meta") or {}
    notes: list[str] = []

    failed = int(meta.get("failed_case_count") or 0)
    if failed:
        notes.append(f"{failed} 条检索失败(记空召回,指标被拉低)")

    retried = meta.get("retried_cases") or []
    if retried:
        notes.append(f"{len(retried)} 条重试过")

    degraded = int(meta.get("rewrite_degraded_cases") or 0)
    if degraded:
        notes.append(f"{degraded} 条改写降级为单查询")

    return notes


def print_table(reports: list[dict[str, Any]], baseline_tag: str | None) -> None:
    if not reports:
        print("没有匹配的报告。")
        return

    # -- 可比性检查:样本数不同的报告放进同一张表,Δ 就是假的 -----------
    totals = {int((r.get("summary") or {}).get("total") or 0) for r in reports}
    if len(totals) > 1:
        print(f"[WARN] 报告样本数不一致 {sorted(totals)} —— 跨组 Δ 不成立,")
        print("       请确认是不是混进了别的评测集的报告。\n")

    rows = [(r.get("tag", "?"), _metrics(r), r) for r in reports]

    baseline: dict[str, float] | None = None
    if baseline_tag:
        for tag, m, _r in rows:
            if tag.startswith(baseline_tag):
                baseline = m
                baseline_tag = tag
                break
        if baseline is None:
            print(f"[WARN] 找不到基准组 {baseline_tag},本次不算 Δ\n")

    width = max(len(tag) for tag, _m, _r in rows)
    header = (
        f"{'tag':<{width}}  {'n':>3}  {'MRR':>6}  "
        + "  ".join(f"{'Hit@' + str(k):>6}" for k in SHOWN_KS)
        + f"  {'s/case':>7}"
    )
    print(header)
    print("-" * len(header))

    for tag, m, report in rows:
        total = int((report.get("summary") or {}).get("total") or 0)
        line = (
            f"{tag:<{width}}  {total:>3}  {m['mrr']:>6.4f}  "
            + "  ".join(f"{m['hit' + str(k)]:>6.4f}" for k in SHOWN_KS)
            + f"  {m['sec']:>7.3f}"
        )
        print(line)

    if baseline:
        print()
        print(f"-- Δ vs {baseline_tag} --")
        for tag, m, _r in rows:
            if m is baseline:
                continue
            deltas = [f"MRR {m['mrr'] - baseline['mrr']:+.4f}"]
            deltas += [
                f"Hit@{k} {m['hit' + str(k)] - baseline['hit' + str(k)]:+.4f}"
                for k in SHOWN_KS
            ]
            # 成本倍数:只报收益不报成本等于替读者把权衡做掉了
            cost = m["sec"] / baseline["sec"] if baseline["sec"] else 0.0
            deltas.append(f"cost ×{cost:.1f}")
            print(f"  {tag:<{width}}  " + "  ".join(deltas))

    # -- 可信度提示 -----------------------------------------------------
    flagged = [(tag, notes) for tag, _m, r in rows if (notes := _credibility_notes(r))]
    if flagged:
        print()
        print("-- 可信度提示 --")
        for tag, notes in flagged:
            print(f"  [{tag}] " + "; ".join(notes))


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总评测报告成对照表")
    parser.add_argument("prefixes", nargs="*", help="只看 tag 以这些前缀开头的报告")
    parser.add_argument("--baseline", default=None, help="算 Δ 的基准组 tag 前缀")
    args = parser.parse_args()

    reports = load_reports(args.prefixes or None)
    print_table(reports, args.baseline)


if __name__ == "__main__":
    main()
