"""中文文本自动校对（错别字 / 标点检测）。

设计目标（对应用户诉求）：

1. **逐处标出疑似错误并给候选**：每条结果含字符偏移、原片段、候选改法、
   类别、严重度、置信度与判定依据；前端可逐处高亮、一键应用或忽略。
2. **查得准（宁漏勿滥）**：
   - 显式易混字规则（在/再、己/已、的/地/得……）必须有上下文证据才报；
   - 整词错词表只收「错形本身不是合法词」的条目；
   - 同音字/形近字编辑邻域检测，只有「改后分词质量显著变好」才报；
   - NER 实体（人名/地名/机构/时间/数字）、URL、邮箱、英文单词、
     用户白名单（专有名词/方言/刻意新写法）一律豁免。
3. **查得全**：重复叠字、疑似漏字（低置信度，仅供参考）、
   成对引号括号不闭合（跨行栈配对）、标点误用（半角混用、重复标点）。
4. **批量不卡死**：``deadline`` 预算 + 每篇独立调用，超时抛
   :class:`ProofTimeout`，由上层捕获后跳过该篇继续整批。
5. **结果稳定**：每条 finding 有由 (类别, 偏移) 决定的稳定 id，
   供应用/忽略与偏移重映射使用（见 :mod:`nlp.revision`）。

检测器之间通过共享的「豁免区间」协调，并在输出时做重叠去重：
显式规则 > 错词表 > 编辑邻域（同音/形近）> 重复字 > 漏字 > 标点。
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Iterable, Optional

from . import proof_data as pd
from .lexicon import PERSONS
from .ner import NERExtractor
from .proof_data import (
    CATEGORY_NAMES, CONFUSION_RULES, HOMOPHONE_MAP, MISSING_PATTERNS,
    LEGAL_DUPLICATION_PHRASES, LEGAL_REDUPLICATION, SEVERITY_NAMES, SHAPE_MAP,
    WRONG_WORDS,
)
from .segmenter import Segmenter

CJK_RE = re.compile(r"[一-鿿]")
URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s，。！？；：、）)】」』]+", re.I)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
LATIN_RUN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'’.\-]*")
# 成对标点：开 -> 闭
PAIR_PUNCT = {
    "（": "）", "(": ")",
    "【": "】", "[": "]",
    "《": "》", "<": ">",
    "「": "」", "『": "』",
    "“": "”", "\"": "\"",
    "‘": "’", "'": "'",
}
CLOSERS = set(PAIR_PUNCT.values())
OPENERS = set(PAIR_PUNCT.keys())
# 引号允许跨行；硬括号不允许（未闭合大概率是漏字）
QUOTE_OPENERS = {"“", "\"", "‘", "'"}
QUOTE_CLOSERS = {"”", "\"", "’", "'"}
BRACKET_PAIRS = {"（": "）", "(": ")", "【": "】", "[": "]",
                 "《": "》", "<": ">", "「": "」", "『": "』"}

# 半角标点（出现在中文语境中）
HALF_PUNCT_RE = re.compile(r"[,.!?;:]")
FULL_EQUIV = {",": "，", ".": "。", "!": "！", "?": "？", ";": "；", ":": "："}
# 连续重复 / 混用的句读标点
REPEAT_PUNCT_RE = re.compile(r"([，。！？；：,.!?;:])\1+")
MIXED_PUNCT_RE = re.compile(r"(?<=[，。！？])(?=[,.!?;])|(?<=[,.!?;])(?=[，。！？；])")
SPACE_BEFORE_CJK_PUNCT_RE = re.compile(r"[ \t　]+(?=[，。！？；：、）)】」』%])")
# 两个连字符常见误用为破折号
DASH_RE = re.compile(r"(?<![-\w])--(?![-\w])")


class ProofTimeout(Exception):
    """单篇扫描超过时间预算。"""


class Proofreader:
    """中文文本校对器。

    :param whitelist: 用户白名单词（专有名词 / 方言词 / 刻意新写法），
                      命中区间不报错；也可传 (start, end) 区间。
    :param deadline: Unix 时间戳，扫描超过该时刻抛 :class:`ProofTimeout`。
    """

    def __init__(self, segmenter: Optional[Segmenter] = None,
                 ner: Optional[NERExtractor] = None,
                 whitelist: Optional[Iterable] = None,
                 deadline: Optional[float] = None,
                 check_missing: bool = True):
        # 默认复用全局单例分词器：校对时补充的证据词（安装/再次/犯错……）
        # 对下游分词/情感/摘要同样生效，保证「改后文本回流」时切词一致。
        if segmenter is None:
            from . import get_segmenter
            segmenter = get_segmenter()
        self.segmenter = self._with_proof_words(segmenter)
        self.ner = ner or NERExtractor(self.segmenter)
        self.dictionary = self.segmenter.dictionary
        self.whitelist_words: set[str] = set()
        self.whitelist_spans: list[tuple[int, int]] = []
        for item in whitelist or []:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                self.whitelist_spans.append((int(item[0]), int(item[1])))
            elif isinstance(item, str):
                self.whitelist_words.add(item)
        for w in self.whitelist_words:
            # 白名单词也加入词典，避免被编辑邻域判成弱串
            self.dictionary.setdefault(w, 500)
        self.deadline = deadline
        self.check_missing = check_missing
        self._rules = self._compile_rules()
        self._wrong_word_re = self._compile_wrong_words()

    # -- 初始化辅助 -------------------------------------------------------
    @staticmethod
    def _with_proof_words(seg: Segmenter) -> Segmenter:
        if getattr(seg, "_proof_words_loaded", False):
            return seg
        for w, f in pd.proof_dictionary().items():
            seg.dictionary.setdefault(w, f)
        seg.max_word_len = max(
            max((len(w) for w in seg.dictionary), default=4), 4)
        seg.total_freq = max(sum(seg.dictionary.values()), 1)
        import math
        seg._log_total = math.log(seg.total_freq)
        seg._oov_log = math.log(0.4 / seg.total_freq)
        seg._proof_words_loaded = True
        return seg

    def _compile_rules(self):
        compiled = []
        # 「的/地/得」三条规则的超长字符类由词表动态生成
        adv_alt = "|".join(sorted(pd.ADVERBS, key=len, reverse=True))
        adj_alt = "|".join(sorted(pd.DE_ADJECTIVES, key=len, reverse=True))
        noun_words_alt = "|".join(
            sorted(pd.DE_DINGYU_NOUNS, key=len, reverse=True))
        verb_cls = "".join(sorted(pd.DE_VERB_CHARS))
        noun_cls = "".join(sorted(pd.DE_NOUN_CHARS))
        ding_cls = "".join(sorted(pd.DE_DINGYU_CHARS))
        generated = {
            "@de_adv@": (
                # 状语句：X 的 后面必须紧跟动词；若紧跟名词（勇敢的孩子）
                # 则「的」是正确写法，不能报。
                rf"(?:{adv_alt})(?P<x>的)(?=[{verb_cls}])",
            ),
            "@de_v_noun@": (rf"(?P<x>得)(?=[{noun_cls}])",),
            "@de_noun@": (
                rf"(?:{adj_alt})(?P<x>地)(?=(?:{noun_words_alt}))",
                rf"(?:{adj_alt})(?P<x3>地)(?=[{ding_cls}])",
            ),
        }
        for rid, wrong, right, patterns, excludes, conf in CONFUSION_RULES:
            expanded = []
            for p in patterns:
                if p in generated:
                    expanded.extend(generated[p])
                else:
                    expanded.append(p)
            regs = [re.compile(p) for p in expanded]
            exregs = [re.compile(p) for p in excludes]
            compiled.append((rid, wrong, right, regs, exregs, conf))
        return compiled

    def _compile_wrong_words(self):
        entries = [(w, r, c) for w, r, c in WRONG_WORDS if c > 0 and w != r]
        # 长词优先匹配
        entries.sort(key=lambda x: -len(x[0]))
        return entries

    # -- 对外主入口 -------------------------------------------------------
    def proofread(self, text: str) -> list[dict]:
        findings: list[dict] = []
        if not text:
            return findings
        self._tick()

        exempt = self._exempt_spans(text)

        # 1. 显式上下文规则
        findings.extend(self._check_confusion_rules(text, exempt))
        # 2. 错词 / 错成语表
        findings.extend(self._check_wrong_words(text, exempt))
        # 2b. 显式漏字规则（高精度固定搭配，可由参数关闭）
        if self.check_missing:
            findings.extend(self._check_missing_patterns(text, exempt))
        # 3. 分词弱串编辑邻域（同音 / 形近替换、多字）
        findings.extend(self._check_edit_neighborhood(text, exempt))
        # 3b. 相邻同字重复（的的/了了/是是……，AA 合法叠词除外）
        findings.extend(self._check_adjacent_dup(text, exempt))
        # 4. 成对标点
        findings.extend(self._check_paired_punct(text))
        # 5. 标点误用
        findings.extend(self._check_punct_misuse(text, exempt))

        findings = self._dedupe(findings)
        findings.sort(key=lambda f: (f["start"], f["end"]))
        return findings

    def _tick(self) -> None:
        if self.deadline is not None and time.time() > self.deadline:
            raise ProofTimeout("校对超时")

    # -- 豁免区间 ---------------------------------------------------------
    def _exempt_spans(self, text: str) -> list[tuple[int, int]]:
        spans = list(self.whitelist_spans)
        for m in URL_RE.finditer(text):
            spans.append((m.start(), m.end()))
        for m in EMAIL_RE.finditer(text):
            spans.append((m.start(), m.end()))
        for m in LATIN_RUN_RE.finditer(text):
            spans.append((m.start(), m.end()))
        # NER：地名 / 机构（词典或后缀匹配，可靠性高）整体豁免；
        # 人名只豁免词典确知的专名（如「毛泽东」），规则猜测的
        # 「姓氏+名」误报率高（例如把「甘败」猜成人名），不豁免，
        # 以免压住真的错别字。数字/时间的标点仍需检查，同样不豁免。
        try:
            for ent in self.ner.recognize(text):
                if ent["type"] in ("LOCATION", "ORGANIZATION"):
                    spans.append((ent["start"], ent["end"]))
                elif ent["type"] == "PERSON" and ent.get("text") in PERSONS:
                    spans.append((ent["start"], ent["end"]))
        except Exception:  # noqa: BLE001
            pass
        # 白名单词
        for w in self.whitelist_words:
            if len(w) < 2:
                continue
            start = 0
            while True:
                pos = text.find(w, start)
                if pos < 0:
                    break
                spans.append((pos, pos + len(w)))
                start = pos + 1
        return spans

    @staticmethod
    def _exempted(start: int, end: int, exempt: list[tuple[int, int]]) -> bool:
        for a, b in exempt:
            # 与豁免区间相交（标点类错误的核心字符落在豁免区才豁免）
            if start < b and a < end:
                return True
        return False

    # -- 1. 显式易混字规则 ------------------------------------------------
    def _check_confusion_rules(self, text, exempt) -> list[dict]:
        out = []
        for rid, wrong, right, regs, exregs, conf in self._rules:
            for reg in regs:
                for m in reg.finditer(text):
                    xs = xe = None
                    for gname in ("x", "x3"):
                        try:
                            xs, xe = m.span(gname)
                            break
                        except IndexError:
                            continue
                    if xs is None or text[xs:xe] != wrong:
                        continue
                    if self._exempted(xs, xe, exempt):
                        continue
                    context = text[max(0, xs - 6):min(len(text), xe + 6)]
                    if any(ex.search(context) for ex in exregs):
                        continue
                    rule_conf = conf
                    if m.lastgroup == "x3":
                        rule_conf = 0.6  # 形容词+地+单字名词首字，弱证据
                    out.append(self._finding(
                        text, xs, xe, right,
                        category="homophone" if right in HOMOPHONE_MAP.get(wrong, [])
                        else "shape",
                        confidence=rule_conf, rule=rid,
                        reason=f"「{wrong}」应为「{right}」（上下文规则）"))
        return out

    # -- 2. 错词 / 错成语 -------------------------------------------------
    def _check_wrong_words(self, text, exempt) -> list[dict]:
        out = []
        n = len(text)
        for wrong, right, conf in self._wrong_word_re:
            self._tick()
            start = 0
            lw = len(wrong)
            while True:
                pos = text.find(wrong, start)
                if pos < 0:
                    break
                start = pos + lw
                end = pos + lw
                if self._exempted(pos, end, exempt):
                    continue
                # 左右是 CJK 字符时不影响（成语常嵌入句中），照常报
                out.append(self._finding(
                    text, pos, end, right, category="word",
                    confidence=conf, rule="wrong_word",
                    reason=f"「{wrong}」应为「{right}」"))
        return out

    # -- 2b. 显式漏字规则 -------------------------------------------------
    def _check_missing_patterns(self, text, exempt) -> list[dict]:
        out = []
        for rid, reg, ch, conf in MISSING_PATTERNS:
            for m in reg.finditer(text):
                try:
                    bs, be = m.span("before")
                except IndexError:
                    continue
                if self._exempted(bs, be, exempt):
                    continue
                # 插入点前一个字已经是目标字（状语本身以「地」结尾，
                # 如「慢慢地走」中「慢慢」后紧跟「地」），并非缺字。
                if bs < be and text[be - 1] == ch:
                    continue
                # 插入点处已经是该字，同样不可能缺。
                if be < len(text) and text[be] == ch:
                    continue
                # AA 式叠词本身可直接作状语（静静坐、好好说），不缺「地」
                before = m.group("before")
                if len(before) == 2 and before[0] == before[1]:
                    continue
                # 「大声/高声/轻声/悄声」等可直接作状语，降为中置信度
                rule_conf = conf
                if before in ("大声", "高声", "轻声", "悄声", "低声"):
                    rule_conf = 0.6
                out.append(self._finding(
                    text, be, be, ch, category="missing",
                    confidence=rule_conf, rule=rid,
                    reason=f"此处疑似漏了「{ch}」（{m.group()}）",
                    insert_at=be, insert_text=ch))
        return out

    # -- 3. 编辑邻域 ------------------------------------------------------
    def _check_edit_neighborhood(self, text, exempt) -> list[dict]:
        """对分词得到的「单字弱串」做 1- 编辑候选验证。

        弱串：连续若干个只切成单字的 token（登录词边界完好的片段不动）。
        对每个 2~4 字窗口尝试：替换（同音/形近表）、删除、插入，
        若改后窗口全部能切成多字登录词且词频证据显著增强，则上报。
        """
        out = []
        words = self.segmenter.cut(text)
        # token 对齐回字符偏移（cut 会去掉空白，需重新对齐）
        tokens = self._align_tokens(text, words)

        # 找出「弱串」：连续单字 CJK token，且其中至少有一个字不是
        # 高频功能/实义单字（要/再/不/了/我……即使被切成单字也是正常
        # 句法，对它们做替换会把「不要再」误改成「不一再」）。
        weak_runs: list[list[tuple[int, int, str]]] = []
        run: list[tuple[int, int, str]] = []
        for a, b, w in tokens:
            if len(w) == 1 and CJK_RE.match(w):
                run.append((a, b, w))
            else:
                if self._is_weak_run(run):
                    weak_runs.append(run)
                run = []
        if self._is_weak_run(run):
            weak_runs.append(run)

        seen_spots: set[tuple[int, int]] = set()
        for run in weak_runs:
            self._tick()
            run_start = run[0][0]
            run_end = run[-1][1]
            if self._exempted(run_start, run_end, exempt):
                continue
            finding = self._best_edit_for_run(text, run)
            if finding and (finding["start"], finding["end"]) not in seen_spots:
                seen_spots.add((finding["start"], finding["end"]))
                out.append(finding)
        return out

    # 高频成词功能字：这些字作为单字词出现完全正常（在/再/要/不/了…），
    # 它们的误用已有显式上下文规则负责，编辑邻域不再对其做同音替换，
    # 避免把「不要再」改成「不一再」这类「切法两可」的误报。
    _COMMON_SINGLE_FUNCT = set(
        "的了是在有和也就都而又及与或但并把被让给对从到向为以地得着过"
        "吗呢吧啊呀哦嗯哈不没我你他她它们这那个些什么怎谁上下中里外前后"
        "要不要想能会可以该应敢肯须必"
        "一二三四五六七八九十百千万几两第次年月日时分秒")

    def _is_weak_run(self, run) -> bool:
        if len(run) < 2:
            return False
        # 至少有一个字既不在词典、也不是常见功能单字，弱串才成立。
        for _, _, ch in run:
            if ch not in self.dictionary and ch not in self._COMMON_SINGLE_FUNCT:
                return True
        return False

    @staticmethod
    def _align_tokens(text: str, words: list[str]) -> list[tuple[int, int, str]]:
        """把分词结果非贪婪对齐回原文（跳过空白/标点差异）。"""
        tokens = []
        cursor = 0
        for w in words:
            pos = text.find(w, cursor)
            if pos < 0:
                continue
            tokens.append((pos, pos + len(w), w))
            cursor = pos + len(w)
        return tokens

    @staticmethod
    def _align_tokens_strict(text: str, words: list[str]) -> list[tuple[int, int, str]]:
        """严格对齐：每个 token 从光标位置开始就是它本身（用于 fixed 窗口）。"""
        tokens = []
        cursor = 0
        n = len(text)
        for w in words:
            # 跳过空白（cut 会丢弃空白）
            while cursor < n and text[cursor].isspace():
                cursor += 1
            if text[cursor:cursor + len(w)] == w:
                tokens.append((cursor, cursor + len(w), w))
                cursor += len(w)
            else:
                # 切词结果应能连续覆盖；覆盖不了就放弃该词的位置信息
                pos = text.find(w, cursor)
                if pos < 0:
                    pos = cursor
                tokens.append((pos, pos + len(w), w))
                cursor = pos + len(w)
        return tokens

    def _best_edit_for_run(self, text, run):
        best = None
        best_score = 0.0
        n = len(run)
        for length in range(min(4, n), 1, -1):
            for i in range(0, n - length + 1):
                window = run[i:i + length]
                ws, we = window[0][0], window[-1][1]
                original = text[ws:we]
                if not CJK_RE.search(original):
                    continue
                # —— 替换候选 ——
                for k, (a, b, ch) in enumerate(window):
                    # 常见功能单字的误用（在/再、要/不/了…）由显式
                    # 上下文规则负责，邻域替换会产生「切法两可」误报。
                    if ch in self._COMMON_SINGLE_FUNCT:
                        continue
                    cands = list(HOMOPHONE_MAP.get(ch, []))
                    shape_first = [c for c in SHAPE_MAP.get(ch, [])
                                   if c not in cands]
                    for cand, kind in [(c, "homophone") for c in cands] + \
                                     [(c, "shape") for c in shape_first]:
                        fixed = original[:k] + cand + original[k + 1:]
                        # 改动点在 fixed 中的绝对偏移（供覆盖检查）
                        self._edit_index = a - ws + k
                        score = self._repair_score(original, fixed,
                                                   replacement=(ch, cand))
                        if score and score > best_score:
                            best_score = score
                            best = self._finding(
                                text, a, b, cand, category=kind,
                                confidence=min(0.85, 0.45 + score / 18),
                                rule=f"edit_{kind}",
                                reason=f"「{ch}」改为「{cand}」后可切分为"
                                       f"常见词：{self._segment_preview(fixed)}")
                # —— 删除候选（多字）——
                for k, (a, b, ch) in enumerate(window):
                    fixed = original[:k] + original[k + 1:]
                    if self._is_legal_redup(original, ch, k):
                        continue
                    self._edit_index = a - ws + k
                    score = self._repair_score(original, fixed, deletion=ch)
                    # 删除天然更激进，阈值更高
                    if score and score > max(best_score, 6.0):
                        best_score = score
                        best = self._finding(
                            text, a, b, "", category="duplicate",
                            confidence=min(0.8, 0.4 + score / 20),
                            rule="edit_delete",
                            reason=f"删去「{ch}」后语句切分正常："
                                   f"{self._segment_preview(fixed)}")
                # —— 插入候选（疑似漏字）已改为显式规则
                # :meth:`_check_missing_patterns`，避免纯邻域猜测误报。
        return best

    def _breaks_valid_bigram(self, original: str, idx: int) -> bool:
        """替换位置与相邻字在原文中是否已构成登录的双字搭配。"""
        for a, b in ((idx - 1, idx + 1), (idx, idx + 2)):
            if a < 0 or b > len(original):
                continue
            bigram = original[a:b]
            if len(bigram) == 2 and bigram in self.dictionary:
                return True
        return False

    def _is_legal_redup(self, original: str, ch: str, k: int) -> bool:
        """重复字是否属于合法叠词。"""
        if original in LEGAL_REDUPLICATION or original in LEGAL_DUPLICATION_PHRASES:
            return True
        # AA 式：删除点两侧是同一字且该 AA 常见（慢慢、好好……）
        if 0 <= k - 1 < len(original) and k < len(original) and \
                original[k - 1] == original[k]:
            aa = ch * 2
            if aa in LEGAL_REDUPLICATION:
                return True
        return False

    def _segment_preview(self, s: str) -> str:
        toks = self.segmenter.cut(s)
        return "/".join(toks)

    def _repair_score(self, original: str, fixed: str,
                      replacement=None, deletion=None) -> float:
        """替换/删除后分词质量提升分；0 表示不构成有效修复。

        规则：改后窗口必须全部切成**多字登录词**，且至少一个多字词
        覆盖改动位置，避免把「本身成立、只是切法不同」的片段误判成错
        （例如「一再」与「不要再」都成立）。
        """
        if not fixed or fixed == original:
            return 0.0
        self._edit_index = getattr(self, "_edit_index", 0)
        new_tokens = self.segmenter.cut(fixed)
        if not new_tokens:
            return 0.0

        # 替换 / 删除：所有 CJK 字符都应被多字登录词覆盖，
        # 且至少有一个多字词在 fixed 中**包含改动位置**。
        # 用严格的连续对齐（不回退到后续文本查找）。
        aligned = self._align_tokens_strict(fixed, new_tokens)
        covers_edit = False
        for tstart, tend, t in aligned:
            if CJK_RE.search(t) and len(t) < 2:
                return 0.0
            if CJK_RE.search(t) and t not in self.dictionary:
                return 0.0
            if len(t) >= 2 and t in self.dictionary and \
                    tstart <= self._edit_index < tend:
                covers_edit = True
        if not covers_edit:
            return 0.0
        # 原搭配保护：原文中与改动位置相邻的双字若本身是登录词
        # （如「不要」「要再」），替换会破坏一个成立的搭配，
        # 而改后只是另一种切法——这种歧义交显式规则处理，邻域不报。
        if replacement and self._breaks_valid_bigram(
                original, self._edit_index):
            return 0.0
        score = 0.0
        for t in new_tokens:
            if t in self.dictionary:
                # 词频取对数，避免高频词压倒一切
                score += 2.0 + min(6.0, self.dictionary[t] ** 0.5 / 8)
        # 覆盖长度越长越可信
        score += len(fixed) * 0.5

        if replacement:
            old, new = replacement
            if new in HOMOPHONE_MAP.get(old, []):
                score += 1.0
            if new in SHAPE_MAP.get(old, []):
                score += 0.8
        if deletion is not None:
            score -= 1.0  # 多字判定从严
        return score

    # -- 3b. 相邻同字重复 -------------------------------------------------
    # 功能字相邻重复几乎都是多字错误；普通实字的 AA（慢慢/好好）合法。
    _DUP_FUNCT_CHARS = "的了是在有和也就都而又及与或但并把被让给对从到向为以地得着过吗呢吧啊呀哦嗯哈不没"

    def _check_adjacent_dup(self, text, exempt) -> list[dict]:
        out = []
        for m in re.finditer(rf"([{self._DUP_FUNCT_CHARS}])\1", text):
            pos = m.start()
            ch = m.group(1)
            if self._exempted(pos, m.end(), exempt):
                continue
            if ch * 2 in LEGAL_REDUPLICATION or ch * 2 in LEGAL_DUPLICATION_PHRASES:
                continue
            out.append(self._finding(
                text, pos, m.end(), ch, category="duplicate",
                confidence=0.9, rule="adjacent_dup",
                reason=f"「{ch}」连续重复，疑为多字"))
        return out

    # -- 4. 成对标点 ------------------------------------------------------
    def _check_paired_punct(self, text) -> list[dict]:
        out = []
        # 按行处理：引号允许跨行则更宽松，这里按物理行配对，符合常见稿件
        for line_no, line in self._iter_lines(text):
            base = line["base"]
            out.extend(self._check_brackets_line(line["text"], base))
            out.extend(self._check_quotes_line(line["text"], base))
        return out

    @staticmethod
    def _iter_lines(text: str):
        start = 0
        no = 0
        for m in re.finditer(r"\n|$", text):
            yield no, {"text": text[start:m.start()], "base": start}
            start = m.end()
            no += 1
            if m.group() == "" and m.start() == len(text):
                break

    def _check_brackets_line(self, line: str, base: int) -> list[dict]:
        out = []
        stack: list[tuple[str, int]] = []  # (开符号, 位置)
        for i, ch in enumerate(line):
            if ch in BRACKET_PAIRS:
                stack.append((ch, base + i))
            elif ch in CLOSERS and ch not in QUOTE_CLOSERS:
                if stack and BRACKET_PAIRS.get(stack[-1][0]) == ch:
                    stack.pop()
                else:
                    # 有闭无开：在文首补开符号
                    opener = self._matching_opener(ch)
                    if opener:
                        out.append(self._finding(
                            line, base + i, base + i + 1,
                            opener + ch, category="punct_pair",
                            confidence=0.9, rule="bracket_close_only",
                            reason=f"「{ch}」缺少对应的「{opener}」"))
        for opener, pos in stack:
            closer = BRACKET_PAIRS[opener]
            line_end = base + len(line)
            out.append(self._finding(
                line, pos, pos + 1, opener,
                category="punct_pair", confidence=0.9,
                rule="bracket_unclosed",
                reason=f"「{opener}」缺少闭合的「{closer}」",
                insert_at=line_end, insert_text=closer))
        return out

    @staticmethod
    def _matching_opener(closer: str) -> Optional[str]:
        for o, c in BRACKET_PAIRS.items():
            if c == closer:
                return o
        return None

    def _check_quotes_line(self, line: str, base: int) -> list[dict]:
        out = []
        stack: list[tuple[str, int]] = []
        straight_open = True
        for i, ch in enumerate(line):
            if ch in QUOTE_OPENERS:
                if ch == "\"":
                    if straight_open:
                        stack.append((ch, base + i))
                    straight_open = not straight_open
                else:
                    stack.append((ch, base + i))
            elif ch in QUOTE_CLOSERS:
                if ch == "\"":
                    if not straight_open and stack and stack[-1][0] == "\"":
                        stack.pop()
                    straight_open = not straight_open
                else:
                    if stack and PAIR_PUNCT.get(stack[-1][0]) == ch:
                        stack.pop()
                    else:
                        opener = self._matching_opener(ch)
                        if opener:
                            out.append(self._finding(
                                line, base + i, base + i + 1,
                                opener + ch, category="punct_pair",
                                confidence=0.85, rule="quote_close_only",
                                reason=f"「{ch}」缺少对应的「{opener}」"))
        for opener, pos in stack:
            closer = PAIR_PUNCT.get(opener, opener)
            line_end = base + len(line)
            out.append(self._finding(
                line, pos, pos + 1, opener,
                category="punct_pair", confidence=0.75,
                rule="quote_unclosed",
                reason=f"「{opener}」缺少闭合的「{closer}」",
                insert_at=line_end, insert_text=closer))
        return out

    # -- 5. 标点误用 ------------------------------------------------------
    def _check_punct_misuse(self, text, exempt) -> list[dict]:
        out = []
        n = len(text)
        # 先算重复标点区间，区间内（含紧邻区间后）的半角标点不再逐条报
        repeat_spans = [m.span() for m in REPEAT_PUNCT_RE.finditer(text)]
        # 半角句读夹在中文之间
        for m in HALF_PUNCT_RE.finditer(text):
            pos = m.start()
            if any(a <= pos <= b for a, b in repeat_spans):
                continue
            left_cjk = pos > 0 and bool(CJK_RE.match(text[pos - 1]))
            right = text[pos + 1:pos + 2]
            right_cjk = bool(CJK_RE.match(right)) if right else False
            # 小数点 / 省略号 / URL 等情况排除
            if m.group() == "." and pos > 0 and pos + 1 < n \
                    and text[pos - 1].isdigit() and text[pos + 1].isdigit():
                continue
            if left_cjk or (right_cjk and m.group() in "!?,;:"):
                if self._exempted(pos, pos + 1, exempt):
                    continue
                full = FULL_EQUIV[m.group()]
                out.append(self._finding(
                    text, pos, pos + 1, full, category="punct_misuse",
                    confidence=0.8, rule="half_punct",
                    reason=f"中文语境中宜用全角「{full}」"))
        # 重复 / 连续半角标点：合并为一条建议
        for m in REPEAT_PUNCT_RE.finditer(text):
            pos, end = m.span()
            if self._exempted(pos, end, exempt):
                continue
            keep = m.group(1)
            keep_full = FULL_EQUIV.get(keep, keep)
            out.append(self._finding(
                text, pos, end, keep_full, category="punct_misuse",
                confidence=0.75, rule="repeat_punct",
                reason=f"标点「{keep}」重复，保留一个"
                       + ("（并转全角）" if keep != keep_full else "")))
        # -- 破折号
        for m in DASH_RE.finditer(text):
            pos, end = m.span()
            out.append(self._finding(
                text, pos, end, "——", category="punct_misuse",
                confidence=0.6, rule="dash",
                reason="中文破折号为「——」"))
        # 中文标点前多余空白
        for m in SPACE_BEFORE_CJK_PUNCT_RE.finditer(text):
            pos, end = m.span()
            if self._exempted(pos, end, exempt):
                continue
            out.append(self._finding(
                text, pos, end, "", category="punct_misuse",
                confidence=0.55, rule="space_before_punct",
                reason="标点前不需要空格"))
        return out

    # -- 结果构造与去重 ---------------------------------------------------
    def _finding(self, text: str, start: int, end: int, replacement: str,
                 category: str, confidence: float, rule: str, reason: str,
                 insert_at: Optional[int] = None,
                 insert_text: Optional[str] = None) -> dict:
        original = text[start:end]
        ctx_l = max(0, start - 8)
        ctx_r = min(len(text), (insert_at if insert_at is not None else end) + 8)
        fid = self._stable_id(category, rule, start, end, original)
        f = {
            "id": fid,
            "category": category,
            "category_name": CATEGORY_NAMES.get(category, category),
            "severity": self._severity(category, confidence),
            "severity_name": "",
            "confidence": round(confidence, 3),
            "rule": rule,
            "start": start,
            "end": end,
            "original": original,
            "replacement": replacement,
            "reason": reason,
            "context": text[ctx_l:ctx_r],
        }
        f["severity_name"] = SEVERITY_NAMES.get(f["severity"], "")
        if insert_at is not None:
            f["insert_at"] = insert_at
            f["insert_text"] = insert_text or ""
        return f

    @staticmethod
    def _severity(category: str, confidence: float) -> str:
        if category == "missing" or confidence < 0.6:
            return "low"
        if confidence < 0.8:
            return "medium"
        return "high"

    @staticmethod
    def _stable_id(category, rule, start, end, original) -> str:
        raw = f"{category}|{rule}|{start}|{end}|{original}"
        return "f_" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]

    def _dedupe(self, findings: list[dict]) -> list[dict]:
        """同位置只保留证据最强的一条；同 id 去重。标点/字符区间重叠按规则优先级。"""
        priority = {
            "word": 5, "homophone": 4, "shape": 4, "duplicate": 3,
            "missing": 2, "punct_pair": 1, "punct_misuse": 1,
        }
        chosen: dict[tuple[int, int], dict] = {}
        ids_seen: set[str] = set()
        result = []
        # 先按优先级/置信度排序
        findings.sort(key=lambda f: (-priority.get(f["category"], 0),
                                     -f["confidence"]))
        occupied: list[tuple[int, int, str]] = []
        for f in findings:
            if f["id"] in ids_seen:
                continue
            overlap = False
            for a, b, cat in occupied:
                # 标点类与字符类允许重叠（如引号与引号内错字）；
                # 同类重叠才抑制
                if f["start"] < b and a < f["end"] and \
                        cat.startswith("punct") == f["category"].startswith("punct"):
                    overlap = True
                    break
            if overlap:
                continue
            ids_seen.add(f["id"])
            occupied.append((f["start"], f["end"], f["category"]))
            result.append(f)
        # 去掉 _dedupe 里没用的中间变量
        _ = chosen
        return result


def proofread_text(text: str, whitelist: Optional[Iterable] = None,
                   timeout: Optional[float] = None,
                   check_missing: bool = True) -> list[dict]:
    """便捷函数：扫描单篇文本。

    :param timeout: 秒级预算（基于当前时间计算截止时刻）。
    """
    deadline = time.time() + timeout if timeout else None
    return Proofreader(whitelist=whitelist, deadline=deadline,
                       check_missing=check_missing).proofread(text)
