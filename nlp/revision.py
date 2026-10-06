"""校对修改的应用、撤销与偏移重映射。

核心诉求：「改哪一处、怎么改，后续任务（分词/情感/摘要……）给出的结果
都得对得上」。因此本模块不只是简单的字符串替换，还提供：

1. :func:`apply_edits` —— 批量应用一组编辑，要求：
   - 针对**指定版本**的文本（乐观并发控制，版本不符返回冲突而不是静默覆盖）；
   - 编辑彼此**不重叠**（重叠整组拒绝，避免一半成功一半失败）；
   - 自动修正编辑偏移（基于原文，应用时按原文位置定位）；
   - 返回新版本号、新文本、每处编辑的**字符级 span 映射**
     （旧偏移 -> 新偏移），下游结果可据此对齐。
2. :class:`EditSet` —— 编辑集合的规范化、冲突检测、按序应用。
3. :func:`remap_offset` / :func:`remap_finding` —— 把旧文本上的偏移/
   未处理 finding 重映射到新文本，已应用/已忽略的 finding 不重复报，
   其余 finding 随修改移动，真正实现「多处修改彼此不冲突」。

编辑（edit）的统一表示::

    {
      "id":         finding 稳定 id,
      "start/end":  原文字符偏移（end == start 表示插入）,
      "replacement": 替换为的文本（"" 表示删除，插入时用 insert_at/insert_text）,
      "insert_at":  插入位置（可选，仅插入类 finding）,
      "insert_text": 插入内容（可选）,
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


class EditConflict(ValueError):
    """编辑之间重叠、顺序无法确定，或针对的文本版本已过期。"""


@dataclass
class AppliedEdit:
    finding_id: str
    old_start: int
    old_end: int
    new_start: int
    new_end: int
    original: str
    replacement: str
    rule: str = ""
    category: str = ""


def _normalize_edit(edit: dict) -> dict:
    """把 finding / 前端 edit 统一成内部编辑表示。

    - 普通替换：start:end -> replacement；
    - 纯插入（漏字、引号括号补闭合）：insert_at + insert_text，
      start:end 仅用于定位/高亮，不删除原字符。判断依据是 finding
      自带 ``insert_at``，且该位置不是区间替换点。
    """
    insert_at = edit.get("insert_at")
    insert_text = edit.get("insert_text")
    start = int(edit.get("start", insert_at if insert_at is not None else 0))
    end = int(edit.get("end", start))
    # 纯插入：只给了 insert_at + insert_text（漏字、补闭合），
    # 或插入点不在 (start,end] 替换范围（如引号开符号在行尾补闭）。
    if insert_text and insert_at is not None and \
            ("start" not in edit or start == end
             or int(insert_at) != start):
        return {
            "id": edit.get("id", f"e_{insert_at}_ins"),
            "start": int(insert_at), "end": int(insert_at),
            "replacement": insert_text,
            "rule": edit.get("rule", ""),
            "category": edit.get("category", ""),
            "original": "",
        }
    replacement = edit.get("replacement", "")
    return {
        "id": edit.get("id", f"e_{start}_{end}"),
        "start": start,
        "end": end,
        "replacement": replacement,
        "rule": edit.get("rule", ""),
        "category": edit.get("category", ""),
        "original": edit.get("original", ""),
    }


def edits_overlap(a: dict, b: dict) -> bool:
    """两个编辑在原文上是否重叠（端点相接不算重叠）。"""
    return a["start"] < b["end"] and b["start"] < a["end"]


class EditSet:
    """一组针对同一文本、彼此不冲突的编辑。"""

    def __init__(self, edits: Iterable[dict]):
        normalized = [_normalize_edit(e) for e in edits]
        # 同一位置只保留一个（后写覆盖），并检查整体不重叠
        normalized.sort(key=lambda e: (e["start"], e["end"]))
        deduped: list[dict] = []
        for e in normalized:
            if deduped and e["start"] < deduped[-1]["end"] and \
                    deduped[-1]["start"] < e["end"]:
                raise EditConflict(
                    f"编辑位置重叠：{deduped[-1]['id']} 与 {e['id']}")
            if deduped and deduped[-1]["start"] == e["start"] and \
                    deduped[-1]["end"] == e["end"]:
                deduped[-1] = e
            else:
                deduped.append(e)
        self.edits = deduped

    def __len__(self) -> int:
        return len(self.edits)

    def __iter__(self):
        return iter(self.edits)

    def apply(self, text: str) -> tuple[str, list[AppliedEdit]]:
        """按从后往前的顺序安全应用，返回 (新文本, 应用记录)。"""
        new_text = text
        applied: list[AppliedEdit] = []
        # 逆序应用，避免偏移失效；同时计算新偏移
        for e in reversed(self.edits):
            s, t = e["start"], e["end"]
            if s < 0 or t > len(text) or s > t:
                raise EditConflict(f"编辑越界：{e['id']} ({s}:{t})")
            original = text[s:t]
            # 若编辑携带了原片段，做一次一致性校验（防止版本错位下的误改）
            if e["original"] and original != e["original"]:
                raise EditConflict(
                    f"编辑 {e['id']} 的原文不匹配：期望 {e['original']!r}，"
                    f"实际 {original!r}（文本可能已被修改）")
            new_text = new_text[:s] + e["replacement"] + new_text[t:]
            applied.append(AppliedEdit(
                finding_id=e["id"], old_start=s, old_end=t,
                new_start=s, new_end=s + len(e["replacement"]),
                original=original, replacement=e["replacement"],
                rule=e["rule"], category=e["category"]))
        applied.reverse()
        # 前面的编辑应用后，后续编辑的新偏移需要加上累积位移
        delta = 0
        for rec in applied:
            rec.new_start += delta
            rec.new_end += delta
            delta += len(rec.replacement) - (rec.old_end - rec.old_start)
        return new_text, applied


def apply_edits(text: str, edits: Iterable[dict],
                base_version: Optional[int] = None,
                expected_version: Optional[int] = None
                ) -> tuple[str, list[AppliedEdit], int]:
    """批量应用编辑。

    :param text: 原始文本（版本 ``base_version``）。
    :param edits: 编辑列表（finding 或 edit dict）。
    :param base_version: 调用方认为的文本版本。
    :param expected_version: 服务端当前版本；与 ``base_version`` 不一致则
        抛 :class:`EditConflict`，由调用方重新拉取最新文本后再改。
    :returns: (新文本, 应用记录, 新版本号)
    """
    if expected_version is not None and base_version is not None and \
            base_version != expected_version:
        raise EditConflict(
            f"文本版本已过期：当前 v{expected_version}，编辑基于 v{base_version}")
    edit_set = EditSet(edits)
    new_text, applied = edit_set.apply(text)
    new_version = (base_version or 0) + 1
    return new_text, applied, new_version


# ---------------------------------------------------------------------------
# 偏移重映射
# ---------------------------------------------------------------------------

def remap_offset(offset: int, applied: Iterable[AppliedEdit],
                 side: str = "right") -> int:
    """把旧文本偏移映射到新文本。

    :param side: ``right``（默认，插入点/结束点向右推）或 ``left``
        （开始点，落在编辑内时贴到编辑起点）。
    """
    result = offset
    for rec in applied:
        delta = len(rec.replacement) - (rec.old_end - rec.old_start)
        if side == "left":
            if offset > rec.old_start:
                result += delta if offset >= rec.old_end else \
                    (rec.new_start - offset if offset < rec.old_end else 0)
        else:
            if offset >= rec.old_end or (rec.old_start == rec.old_end
                                         and offset == rec.old_start):
                result += delta
    return result


def remap_finding(finding: dict, applied: Iterable[AppliedEdit]) -> Optional[dict]:
    """把一条旧文本上的 finding 重映射到新文本。

    - 与已应用编辑重叠：返回 ``None``（该处已经处理，不再报）；
    - 否则偏移整体平移，``id`` 保持不变（同类别同规则同相对位置），
      但位置会更新。
    """
    start, end = finding["start"], finding["end"]
    delta = 0
    for rec in applied:
        if rec.finding_id == finding.get("id"):
            return None
        # 区间重叠
        if start < rec.old_end and rec.old_start < end:
            return None
        if start >= rec.old_end:
            delta += len(rec.replacement) - (rec.old_end - rec.old_start)
        elif start == rec.old_start == rec.old_end and end == start:
            delta += len(rec.replacement)
    new_f = dict(finding)
    new_f["start"] = start + delta
    new_f["end"] = end + delta
    if "insert_at" in new_f:
        new_f["insert_at"] = new_f["insert_at"] + delta
    return new_f


def remap_findings(findings: Iterable[dict],
                   applied: Iterable[AppliedEdit]) -> list[dict]:
    """批量重映射；过滤掉已处理/重叠的 finding。"""
    applied = list(applied)
    out = []
    for f in findings:
        nf = remap_finding(f, applied)
        if nf is not None:
            out.append(nf)
    return out


def describe_change(old_text: str, new_text: str) -> dict:
    """生成一次修改的简要统计（供历史记录/下游对齐）。"""
    return {
        "old_length": len(old_text),
        "new_length": len(new_text),
        "delta": len(new_text) - len(old_text),
    }
