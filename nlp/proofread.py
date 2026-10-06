"""中文文本校对引擎。

针对语料库「入库前体检」场景：自动扫描文本，把疑似写错的位置逐处标出
并给出候选改法。覆盖四类问题：

1. **错别字（同音 / 形近）**：内置混淆字表（在/再、己/已、做/作 …），
   用分词词典做「替换打分」——把某字换成混淆字后能组成频率明显更高的
   词典词，则记为疑似错误。该机制保证「已经是对的字」不会被误报。
2. **多字 / 漏字**：虚词叠用（「的的地」）检测；高频成语 / 固定搭配缺字检测。
3. **标点误用**：栈式配对扫描成对引号 / 括号（不闭合、多余闭合），
   以及中文语境夹用半角标点。
4. **的地得误用**：基于搭配规则的专项检测。

防误报三重保护：
- NER 实体区间（专有名词）不参与字词类检查；
- 用户保护词（方言、刻意的新写法）命中的区间跳过；
- 忽略列表（rule + 原文 + 改法 三元组）精确过滤。

引擎只产出「疑似问题 + 候选改法」，是否采纳由调用方决定；
:meth:`Proofreader.apply_fixes` 把确认的修改按偏移降序写回文本，
多处修改互不冲突，区间已漂移的修改会被安全地拒绝。
"""

from __future__ import annotations

import math
import re
from typing import Callable, Iterable, Optional

from .lexicon import load_dictionary
from .text import text_hash

# ---------------------------------------------------------------------------
# 规则类型
# ---------------------------------------------------------------------------

RULE_NAMES = {
    "confusion": "易混字",
    "de_particle": "的地得",
    "reduplication": "多字叠用",
    "missing_char": "漏字",
    "punct_unclosed": "标点未闭合",
    "punct_redundant": "多余闭合符号",
    "punct_halfwidth": "半角标点误用",
}

# ---------------------------------------------------------------------------
# 混淆字表：每组内的字互为易混字（同音或形近）
# ---------------------------------------------------------------------------

CONFUSION_GROUPS = [
    # 同音易混
    "在再", "做作", "坐座", "像象", "账帐", "度渡", "长常", "分份",
    "工功", "气汽", "即既", "记纪", "竟竞", "决绝", "克刻", "连联",
    "名明", "是事", "受授", "题提", "形型", "需须", "义意", "于与",
    "鱼渔", "园圆", "至致", "状壮", "反返", "汇会", "建健", "节结",
    "令另", "曲屈", "斯撕", "田甜", "唯维", "迎盈", "源原", "张章",
    "申伸", "坚艰", "倍备", "费废", "含函", "毫豪", "优忧", "因应",
    # 形近易混
    "己已巳", "末未", "候侯", "蓝篮", "密蜜", "辨辩辫", "采彩",
    "历厉", "拨拔", "治冶", "晴睛", "喝渴", "折拆", "戎戒", "钓钩",
    "倦券", "洒酒", "鸟乌", "免兔", "人入", "土士", "刀刁", "坏环",
]

# 校对补充词表：保证无 jieba 时高频案例仍可检出。
# 频率给得较高，使「替换后成词」的证据足够强。
PROOF_LEXICON = {
    "再见": 5000, "再次": 5000, "再说": 5000, "再来": 5000, "再三": 3000,
    "再一次": 3000, "再也不": 2000, "再见吧": 500,
    "已经": 8000, "已往": 300, "已知": 800, "已婚": 500, "已故": 400,
    "自己": 8000, "知己": 1500, "律己": 300, "己方": 300,
    "作业": 5000, "做法": 4000, "做事": 4000, "做为": 100, "作为": 8000,
    "作用": 6000, "作品": 4000, "作者": 4000, "工作": 9000,
    "事情": 6000, "事实": 5000, "事业": 4000, "事件": 4000, "事故": 3000,
    "于是": 6000, "关于": 8000, "对于": 7000, "由于": 6000, "至于": 3000,
    "与会": 800, "参与": 5000, "与其": 2000,
    "形象": 4000, "好像": 6000, "想象": 4000, "象征": 3000, "现象": 5000,
    "印象": 4000, "对象": 5000, "抽象": 2000,
    "账户": 3000, "账号": 3000, "账单": 2000, "转账": 2000, "记账": 1500,
    "帐篷": 800, "蚊帐": 300,
    "欢度": 1000, "度假": 1500, "温度": 5000, "制度": 5000, "程度": 4000,
    "过渡": 2000, "渡河": 500, "渡船": 300,
    "经常": 6000, "非常": 8000, "平常": 3000, "日常": 3000, "常常": 3000,
    "长短": 1500, "长久": 1000,
    "身份": 3000, "份额": 1500, "充分": 4000, "部分": 6000, "月份": 3000,
    "功夫": 2000, "功劳": 800, "成功": 6000, "功能": 5000, "工资": 3000,
    "汽车": 4000, "汽油": 2000, "汽水": 800, "蒸汽": 1000, "空气": 4000,
    "即使": 4000, "即将": 2000, "既然": 3000, "既定": 500,
    "记录": 4000, "记者": 3000, "记忆": 2000, "纪律": 2000, "纪念": 2000,
    "竟然": 3000, "究竟": 2000, "竞争": 3000, "竞选": 1000,
    "决定": 6000, "坚决": 1500, "绝对": 3000, "拒绝": 3000, "觉悟": 1000,
    "刻苦": 1500, "立刻": 3000, "克服": 2000, "时刻": 2000,
    "联系": 5000, "联合": 3000, "连忙": 1500, "连接": 2000, "连续": 3000,
    "名字": 4000, "明白": 5000, "明天": 6000, "明显": 4000, "著名": 3000,
    "接受": 4000, "教授": 2000, "授权": 1000, "受害": 800, "受伤": 1500,
    "提高": 5000, "问题": 8000, "提醒": 2000, "题目": 3000, "提供": 5000,
    "形状": 2000, "形式": 4000, "模型": 2000, "类型": 3000, "典型": 2000,
    "需要": 8000, "必须": 5000, "须知": 500, "需求": 3000,
    "意义": 4000, "意思": 4000, "意见": 3000, "愿意": 3000, "注意": 4000,
    "导致": 3000, "至于": 3000, "至今": 2000, "至少": 3000, "甚至": 3000,
    "状态": 3000, "状况": 2000, "壮大": 1000, "强壮": 1000,
    "反应": 3000, "反映": 2000, "返回": 2000, "反复": 1500, "相反": 3000,
    "机会": 5000, "社会": 6000, "会议": 3000, "汇合": 500, "词汇": 2000,
    "健康": 4000, "建设": 4000, "建议": 3000, "键盘": 1000, "健美": 300,
    "结果": 5000, "结合": 4000, "结束": 4000, "节日": 2000, "节省": 1000,
    "另外": 4000, "命令": 2000, "令人": 2000,
    "委屈": 1500, "弯曲": 800, "歌曲": 3000, "曲折": 800,
    "撕开": 500, "嘶哑": 300, "斯文": 300,
    "维护": 3000, "思维": 3000, "唯一": 3000, "纤维": 800,
    "原来": 5000, "原因": 4000, "资源": 3000, "能源": 3000, "源泉": 500,
    "文章": 4000, "紧张": 3000, "张开": 800, "章程": 500, "印章": 500,
    "申请": 2000, "伸手": 500, "延伸": 1000, "精神": 4000,
    "艰苦": 1500, "坚持": 4000, "坚决": 1500, "艰难": 1500,
    "准备": 5000, "设备": 3000, "加倍": 1000, "倍数": 500,
    "免费": 2000, "浪费": 2000, "费用": 2000, "废品": 500,
    "含义": 1500, "包含": 3000, "函数": 1500, "寒假": 800,
    "毫米": 800, "丝毫": 1000, "自豪": 1500, "豪华": 800,
    "优秀": 3000, "忧虑": 800, "忧伤": 500, "忧愁": 300,
    "未来": 5000, "未必": 1000, "末尾": 500, "周末": 3000, "期末": 800,
    "时候": 6000, "气候": 2000, "问候": 1000, "诸侯": 300, "王侯": 200,
    "篮球": 2000, "篮子": 500, "蓝天": 1500, "蓝色": 2000,
    "秘密": 2000, "密码": 1500, "蜜蜂": 800, "甜蜜": 1000, "茂密": 500,
    "分辨": 1500, "辨别": 1000, "辩论": 1500, "答辩": 800, "辫子": 300,
    "采用": 3000, "采访": 1500, "彩色": 1500, "精彩": 3000, "喝彩": 500,
    "历史": 5000, "厉害": 2000, "严厉": 1500, "经历": 3000, "日历": 800,
    "拨打": 800, "拨款": 500, "拔河": 300, "挺拔": 500,
    "治理": 2000, "政治": 4000, "冶炼": 300, "陶冶": 200,
    "眼睛": 3000, "晴天": 1000, "晴朗": 500, "画龙点睛": 500,
    "喝水": 1500, "口渴": 500, "渴望": 1000, "吃喝": 800,
    "打折": 800, "折叠": 500, "拆开": 500, "拆迁": 300,
    "钓鱼": 1000, "鱼钩": 200, "钩子": 200,
    "环境": 5000, "破坏": 3000, "圆环": 300, "耳环": 300,
}

# 「的→地」：前面的副词 / 形容词 + 的 + 动词字 → 应为「地」
DE_TO_DI = {
    "高兴", "开心", "认真", "仔细", "轻轻", "慢慢", "悄悄", "静静",
    "拼命", "兴奋", "紧张", "激动", "伤心", "难过", "生气", "愤怒",
    "平静", "耐心", "努力", "疯狂", "快速", "缓慢", "大声", "小声",
    "深情", "不停", "不断", "专心", "专注", "熟练", "笨拙", "温柔",
    "严厉", "亲切", "热情", "积极", "主动", "默默", "悄悄", "偷偷",
    "缓缓", "渐渐", "牢牢", "紧紧", "深深", "重重", "狠狠", "稳稳",
}
DE_VERB_CHARS = set(
    "说讲问喊道叫哭笑走跑跳唱读写看听闻想做办站坐躺睡爱恨打骂吃喝"
    "买卖开关来去进出拉动推摇摆挥举抬放接等停留奔飞游泳爬滚指点头"
    "摇头回答离开靠近前进后退转动工作学习生活思考"
)
# 「的→得」：的 + 程度补语 → 应为「得」
DE_COMPLEMENTS = ("很", "极", "透", "慌", "厉害", "要命", "不行", "不得了")
ADJ_SINGLE = set("好坏冷热累饿渴急气忙闲快慢高低远近新旧美丑善恶")

# 叠字检测
REDUP_FUNCTION_CHARS = set("的地得了着过在是和不都也很就才又再呢吗吧啊于与及或把被让给对从到向为所之其")
AA_OK = {
    "人人", "天天", "年年", "月月", "日日", "个个", "家家", "户户",
    "处处", "事事", "时时", "往往", "渐渐", "慢慢", "悄悄", "轻轻",
    "静静", "仅仅", "刚刚", "常常", "每每", "步步", "声声", "句句",
    "字字", "条条", "件件", "样样", "种种", "代代", "层层", "看看",
    "想想", "试试", "走走", "聊聊", "等等", "找找", "读读", "写写",
    "听听", "说说", "做做", "学学", "问问", "算算", "用用", "玩玩",
    "睡睡", "吃吃", "喝喝", "跑跑", "跳跳", "笑笑", "谈谈", "数数",
    "点点", "双双", "对对", "满满", "深深", "浅浅", "红红", "绿绿",
    "蓝蓝", "白白", "黑黑", "黄黄", "甜甜", "苦苦", "辣辣", "酸酸",
    "香香", "软软", "硬硬", "松松", "紧紧", "平平", "安安", "好好",
    "多多", "少少", "长长", "短短", "高高", "低低", "大大", "小小",
    "胖胖", "瘦瘦", "美美", "新新", "旧旧", "明明", "茫茫", "滚滚",
    "悠悠", "匆匆", "纷纷", "历历", "碌碌", "赫赫", "耿耿", "翩翩",
    "津津", "炯炯", "娓娓", "滔滔", "绵绵", "潺潺", "潇潇", "瑟瑟",
    "萧萧", "嗷嗷", "啧啧", "唧唧", "喳喳", "吱吱", "哇哇", "呜呜",
    "哈哈", "嘿嘿", "嘻嘻", "呵呵", "宝宝", "妈妈", "爸爸", "爷爷",
    "奶奶", "哥哥", "姐姐", "弟弟", "妹妹", "叔叔", "阿姨", "星星",
    "宝宝", "乖乖", "婆婆", "公公", "姥姥", "舅舅", "姑姑",
}

# 高频成语 / 固定搭配（用于漏字检测）
IDIOMS = [
    "再接再厉", "一如既往", "莫名其妙", "不知所措", "络绎不绝", "迫不及待",
    "脍炙人口", "按部就班", "别出心裁", "不计其数", "川流不息", "得心应手",
    "废寝忘食", "鬼斧神工", "海阔天空", "画龙点睛", "焕然一新", "惊心动魄",
    "精益求精", "开门见山", "刻骨铭心", "琳琅满目", "流连忘返", "美轮美奂",
    "名列前茅", "目中无人", "宁缺毋滥", "平易近人", "千方百计", "锲而不舍",
    "轻而易举", "全神贯注", "人山人海", "日新月异", "如鱼得水", "生机勃勃",
    "实事求是", "滔滔不绝", "天长地久", "万里无云", "万无一失", "无微不至",
    "兴高采烈", "一帆风顺", "一丝不苟", "因地制宜", "迎刃而解", "源远流长",
    "志同道合", "自相矛盾", "足智多谋", "举足轻重", "举世闻名", "来之不易",
    "理直气壮", "恋恋不舍", "默默无闻", "目不转睛", "耐人寻味", "破釜沉舟",
    "千姿百态", "勤能补拙", "深思熟虑", "生龙活虎", "手舞足蹈", "水到渠成",
    "忐忑不安", "同心协力", "突飞猛进", "万紫千红", "闻名遐迩", "无可奈何",
    "无价之宝", "五光十色", "喜气洋洋", "小心翼翼", "心旷神怡", "欣欣向荣",
    "雪中送炭", "言而有信", "一心一意", "异口同声", "引人入胜", "应有尽有",
    "语重心长", "载歌载舞", "斩钉截铁", "争分夺秒", "知难而进", "孜孜不倦",
    "自言自语", "不卑不亢", "不骄不躁", "有条不紊", "错落有致", "大相径庭",
    "得心应手", "耳濡目染", "防患未然", "各抒己见", "和蔼可亲", "恍然大悟",
    "疾风知劲草", "欲速则不达", "百闻不如一见", "事实胜于雄辩",
]

# 成对符号（开 -> 闭）
_OPEN_TO_CLOSE = {
    "“": "”", "‘": "’", "《": "》", "〈": "〉", "（": "）",
    "【": "】", "「": "」", "『": "』", "〔": "〕", "[": "]", "{": "}",
}
_CLOSE_TO_OPEN = {v: k for k, v in _OPEN_TO_CLOSE.items()}

# 中文语境中误用的半角标点 -> 应替换为
_HALFWIDTH_MAP = {",": "，", ";": "；", "!": "！", "?": "？", ":": "：", ".": "。"}

_CJK_RE = re.compile(r"[一-鿿]")
# 替换打分时的上下文窗口：(左取字数, 右取字数)
_WINDOWS = ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (2, 1), (1, 2), (2, 2))


def _is_cjk(ch: str) -> bool:
    return "一" <= ch <= "鿿"


def _in_ranges(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    """区间 [start, end) 是否与任一保护区间相交。"""
    for s, e in ranges:
        if start < e and s < end:
            return True
    return False


class Proofreader:
    """中文校对引擎：扫描疑似错误并给出候选改法。"""

    def __init__(self, dictionary: Optional[dict] = None,
                 ner_provider: Optional[Callable] = None,
                 min_word_freq: int = 3):
        """
        :param dictionary: 打分词典（词 -> 频率），默认用分词词典 + 校对补充词表
        :param ner_provider: 返回 NER 单例的可调用对象（惰性注入，避免循环依赖）
        :param min_word_freq: 候选词进入打分的最低频率（过滤生僻组合）
        """
        base = dictionary if dictionary is not None else load_dictionary()
        self.dictionary = dict(PROOF_LEXICON)
        for word, freq in base.items():
            if freq > self.dictionary.get(word, 0):
                self.dictionary[word] = freq
        self.ner_provider = ner_provider
        self.min_word_freq = min_word_freq

        # 易混字 -> 所在组
        self._confusion_map: dict[str, str] = {}
        for group in CONFUSION_GROUPS:
            for ch in group:
                self._confusion_map[ch] = group

        # 成语缺字索引：{模式长度: {缺字模式: [(成语, 缺字, 缺字位置)]}}
        self._idiom_index: dict[int, dict[str, list]] = {}
        for idiom in IDIOMS:
            for k in range(len(idiom)):
                pat = idiom[:k] + idiom[k + 1:]
                self._idiom_index.setdefault(len(pat), {}).setdefault(pat, []).append(
                    (idiom, idiom[k], k))

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def scan(self, text: str, *,
             protected_words: Optional[Iterable[str]] = None,
             ignores: Optional[Iterable[tuple]] = None,
             use_ner: bool = True) -> list[dict]:
        """扫描文本，返回疑似问题列表（按位置排序）。

        :param protected_words: 用户保护词（方言 / 新写法），命中区间不检查
        :param ignores: 忽略项 ``(rule, original, replacement[, context])``；
            带 context 的项只忽略相同上下文中的同一处，不带则全局忽略
        :param use_ner: 是否用 NER 实体区间做保护
        """
        if not text:
            return []
        issues: list[dict] = []
        protected = self._protected_ranges(text, protected_words, use_ner)

        self._scan_confusion(text, issues, protected)
        self._scan_de_particle(text, issues, protected)
        self._scan_reduplication(text, issues, protected)
        self._scan_missing_char(text, issues, protected)
        self._scan_punctuation(text, issues)

        issues.sort(key=lambda it: (it["start"], it["end"]))
        for it in issues:
            s, e = it["start"], it["end"]
            it["context"] = text[max(0, s - 8):min(len(text), e + 8)]

        # 忽略列表过滤：(rule, original, 首选改法[, context])
        if ignores:
            norm = [tuple(ig) for ig in ignores if ig]
            issues = [it for it in issues
                      if not self._is_ignored(it, norm)]

        for n, it in enumerate(issues, 1):
            it["id"] = f"i{n}"
        return issues

    @staticmethod
    def _is_ignored(issue: dict, ignores: list[tuple]) -> bool:
        top_repl = (issue["candidates"][0]["fix"]["replacement"]
                    if issue["candidates"] else "")
        for ig in ignores:
            rule, orig, repl = ig[0], ig[1], ig[2]
            ctx = ig[3] if len(ig) > 3 else ""
            if (issue["rule"], issue["original"], top_repl) != (rule, orig, repl):
                continue
            if not ctx or ctx == issue["context"]:
                return True
        return False

    def apply_fixes(self, text: str, items: list[dict]) -> dict:
        """把确认的修改写回文本。

        每项形如 ``{"start", "end", "original", "replacement"}``：
        - 逐项校验 ``text[start:end] == original``，已漂移的项进入 conflicts；
        - 区间互相重叠的项，后者进入 conflicts；
        - 其余按偏移降序应用，多处修改互不冲突。

        返回 ``{"text", "applied", "conflicts", "text_hash"}``。
        """
        valid: list[dict] = []
        conflicts: list[dict] = []
        for it in items:
            s, e = int(it.get("start", -1)), int(it.get("end", -1))
            if not (0 <= s <= e <= len(text)):
                conflicts.append({**it, "error": "偏移越界"})
                continue
            if text[s:e] != it.get("original", text[s:e]):
                conflicts.append({**it, "error": "原文已变化",
                                  "actual": text[s:e]})
                continue
            valid.append({"start": s, "end": e,
                          "original": text[s:e],
                          "replacement": it.get("replacement", "")})

        # 重叠检测：按起点排序后，与上一项相交的剔除
        valid.sort(key=lambda x: (x["start"], x["end"]))
        kept: list[dict] = []
        last_end = -1
        for it in valid:
            if it["start"] < last_end:
                conflicts.append({**it, "error": "与其它修改区间重叠"})
                continue
            kept.append(it)
            last_end = it["end"]

        new_text = text
        for it in sorted(kept, key=lambda x: x["start"], reverse=True):
            new_text = (new_text[:it["start"]] + it["replacement"]
                        + new_text[it["end"]:])
        return {"text": new_text, "applied": kept, "conflicts": conflicts,
                "text_hash": text_hash(new_text)}

    # ------------------------------------------------------------------
    # 保护区间
    # ------------------------------------------------------------------
    def _protected_ranges(self, text: str,
                          protected_words: Optional[Iterable[str]],
                          use_ner: bool) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        if use_ner and self.ner_provider is not None:
            try:
                for ent in self.ner_provider().recognize(text):
                    ranges.append((ent["start"], ent["end"]))
            except Exception:  # noqa: BLE001 - NER 失败不应阻断校对
                pass
        for word in protected_words or []:
            if not word:
                continue
            start = 0
            while True:
                pos = text.find(word, start)
                if pos < 0:
                    break
                ranges.append((pos, pos + len(word)))
                start = pos + 1
        return ranges

    # ------------------------------------------------------------------
    # 易混字：双向证据打分
    # ------------------------------------------------------------------
    # 对每个候选字同时计算：
    #   支持证据 support —— 替换后能组成词典词的最强窗口（log 词频）
    #   反对证据 oppose  —— 原字本已组成词典词、替换后反而被破坏的最强窗口
    # support 明显超过 oppose（余量 _CONF_MARGIN）才判定为错；
    # 两者相当（如「自已经」中 自己/已经 都是强词）则标为「疑似、需人工判断」，
    # 给出低置信度，不参与一键应用——避免「改过去又改回来」的震荡。
    _CONF_MARGIN = 0.3

    def _scan_confusion(self, text: str, issues: list[dict],
                        protected: list[tuple[int, int]]) -> None:
        n = len(text)
        for i, ch in enumerate(text):
            group = self._confusion_map.get(ch)
            if not group or _in_ranges(i, i + 1, protected):
                continue
            best = None  # (net, alt, word, new_word, f_old, f_new, support, oppose)
            for alt in group:
                if alt == ch:
                    continue
                support = 0.0
                oppose = 0.0
                support_hit = None  # (word, new_word, f_old, f_new)
                for left, right in _WINDOWS:
                    s, e = i - left, i + 1 + right
                    if s < 0 or e > n:
                        continue
                    word = text[s:e]
                    if len(word) < 2 or not all(_is_cjk(c) for c in word):
                        continue
                    new_word = word[:left] + alt + word[left + 1:]
                    f_old = self.dictionary.get(word, 0)
                    f_new = self.dictionary.get(new_word, 0)
                    # 支持：替换后成词，且明显强于原组合
                    if (f_new >= self.min_word_freq
                            and f_new >= max(self.min_word_freq, f_old * 4)):
                        ev = math.log10(f_new + 1) + 0.1 * len(new_word)
                        if ev > support:
                            support = ev
                            support_hit = (word, new_word, f_old, f_new)
                    # 反对：原组合本身是词，替换后不再是词
                    if (f_old >= self.min_word_freq
                            and f_new * 4 < f_old):
                        oppose = max(oppose,
                                     math.log10(f_old + 1) + 0.1 * len(word))
                if support <= 0:
                    continue
                net = support - oppose
                if best is None or net > best[0]:
                    best = (net, alt, support_hit, support, oppose)
            if best is None:
                continue
            net, alt, hit, support, oppose = best
            word, new_word, f_old, f_new = hit
            if net > self._CONF_MARGIN:
                conf = min(0.97, 0.45 + 0.13 * support)
                if f_old == 0:
                    reason = f"「{word}」不是常见词，应为「{new_word}」"
                else:
                    reason = f"「{new_word}」比「{word}」更常见"
                issues.append(self._issue(
                    "confusion", i, i + 1, text,
                    candidates=[{
                        "text": alt, "reason": reason,
                        "confidence": round(conf, 2),
                        "fix": {"start": i, "end": i + 1, "original": ch,
                                "replacement": alt},
                    }],
                    message=f"「{ch}」疑似应为「{alt}」",
                ))
            elif support >= 1.5 and oppose >= 1.5:
                # 两种写法都常见：标出供人工判断，不给高置信度
                issues.append(self._issue(
                    "confusion", i, i + 1, text,
                    candidates=[{
                        "text": alt,
                        "reason": f"「{word}」与「{new_word}」均常见，需结合语义判断",
                        "confidence": 0.5,
                        "fix": {"start": i, "end": i + 1, "original": ch,
                                "replacement": alt},
                    }],
                    message=f"「{ch}」写法存疑（「{word}」/「{new_word}」均常见）",
                ))

    # ------------------------------------------------------------------
    # 的地得
    # ------------------------------------------------------------------
    def _scan_de_particle(self, text: str, issues: list[dict],
                          protected: list[tuple[int, int]]) -> None:
        n = len(text)
        for i, ch in enumerate(text):
            if ch != "的" or _in_ranges(i, i + 1, protected):
                continue
            prev2 = text[max(0, i - 2):i]
            nxt = text[i + 1] if i + 1 < n else ""
            # 副词/形容词 + 的 + 动词 → 地
            if prev2 in DE_TO_DI and nxt in DE_VERB_CHARS:
                issues.append(self._issue(
                    "de_particle", i, i + 1, text,
                    candidates=[{
                        "text": "地",
                        "reason": f"「{prev2}」修饰动词「{nxt}」，应用「地」",
                        "confidence": 0.82,
                        "fix": {"start": i, "end": i + 1, "original": "的",
                                "replacement": "地"},
                    }],
                    message=f"「{prev2}的{nxt}」中「的」疑似应为「地」",
                ))
                continue
            # 形容词 + 的 + 程度补语 → 得
            rest = text[i + 1:i + 4]
            if any(rest.startswith(c) for c in DE_COMPLEMENTS):
                prev1 = text[i - 1] if i > 0 else ""
                if prev2 in self.dictionary or prev1 in ADJ_SINGLE:
                    issues.append(self._issue(
                        "de_particle", i, i + 1, text,
                        candidates=[{
                            "text": "得",
                            "reason": "程度补语前应用「得」",
                            "confidence": 0.78,
                            "fix": {"start": i, "end": i + 1, "original": "的",
                                    "replacement": "得"},
                        }],
                        message=f"「的{rest[:1]}」中「的」疑似应为「得」",
                    ))

    # ------------------------------------------------------------------
    # 多字（叠用）
    # ------------------------------------------------------------------
    def _scan_reduplication(self, text: str, issues: list[dict],
                            protected: list[tuple[int, int]]) -> None:
        n = len(text)
        for i in range(n - 1):
            ch = text[i]
            if ch != text[i + 1] or not _is_cjk(ch):
                continue
            # 三连及以上由前一次迭代覆盖不到，这里跳过中间位置
            if i > 0 and text[i - 1] == ch:
                continue
            if i + 2 < n and text[i + 2] == ch:
                continue
            pair = ch + ch
            if pair in AA_OK or _in_ranges(i, i + 2, protected):
                continue
            if ch in REDUP_FUNCTION_CHARS:
                conf, reason = 0.95, f"虚词「{ch}」叠用，疑为多写一字"
            else:
                conf, reason = 0.6, f"「{pair}」若非刻意重复，疑为多写一字"
            issues.append(self._issue(
                "reduplication", i, i + 2, text,
                candidates=[{
                    "text": ch, "reason": reason, "confidence": conf,
                    "fix": {"start": i, "end": i + 2, "original": pair,
                            "replacement": ch},
                }],
                message=f"「{pair}」疑似多写了一个「{ch}」",
            ))

    # ------------------------------------------------------------------
    # 漏字（成语 / 固定搭配）
    # ------------------------------------------------------------------
    def _scan_missing_char(self, text: str, issues: list[dict],
                           protected: list[tuple[int, int]]) -> None:
        n = len(text)
        for i in range(n):
            for pat_len, index in self._idiom_index.items():
                pat = text[i:i + pat_len]
                if len(pat) < pat_len:
                    continue
                hits = index.get(pat)
                if not hits:
                    continue
                candidates = []
                for idiom, ch, k in hits[:3]:
                    full = len(idiom)
                    # 模式只是文中完整成语的一部分（去首 / 去尾字所致）→ 非缺字
                    if text[i:i + full] == idiom:
                        continue
                    if k == 0 and i > 0 and text[i - 1:i - 1 + full] == idiom:
                        continue
                    pos = i + k
                    if _in_ranges(pos, pos + 1, protected):
                        continue
                    candidates.append({
                        "text": ch,
                        "reason": f"补全为「{idiom}」",
                        "confidence": 0.85,
                        "fix": {"start": pos, "end": pos, "original": "",
                                "replacement": ch},
                    })
                if not candidates:
                    continue
                pos0 = i + hits[0][2]
                issues.append(self._issue(
                    "missing_char", pos0, pos0, text,
                    candidates=candidates,
                    message=f"「{pat}」疑似「{hits[0][0]}」，漏了「{hits[0][1]}」",
                ))

    # ------------------------------------------------------------------
    # 标点
    # ------------------------------------------------------------------
    def _scan_punctuation(self, text: str, issues: list[dict]) -> None:
        # 1) 成对符号配对（栈式扫描）
        stack: list[tuple[str, int]] = []
        for i, ch in enumerate(text):
            if ch in _OPEN_TO_CLOSE:
                stack.append((ch, i))
            elif ch in _CLOSE_TO_OPEN:
                if stack and stack[-1][0] == _CLOSE_TO_OPEN[ch]:
                    stack.pop()
                else:
                    issues.append(self._issue(
                        "punct_redundant", i, i + 1, text,
                        candidates=[{
                            "text": f"删除「{ch}」",
                            "reason": "没有与之匹配的开启符号",
                            "confidence": 0.85,
                            "fix": {"start": i, "end": i + 1, "original": ch,
                                    "replacement": ""},
                        }],
                        message=f"闭合符号「{ch}」没有匹配的开启符号",
                    ))
        for ch, i in stack:
            close = _OPEN_TO_CLOSE[ch]
            issues.append(self._issue(
                "punct_unclosed", i, i + 1, text,
                candidates=[
                    {
                        "text": f"文末补「{close}」",
                        "reason": "成对符号应闭合",
                        "confidence": 0.8,
                        "fix": {"start": len(text), "end": len(text),
                                "original": "", "replacement": close},
                    },
                    {
                        "text": f"删除「{ch}」",
                        "reason": "若此处本不需要该符号",
                        "confidence": 0.6,
                        "fix": {"start": i, "end": i + 1, "original": ch,
                                "replacement": ""},
                    },
                ],
                message=f"「{ch}」未闭合，缺少对应的「{close}」",
            ))

        # 2) 中文语境夹用半角标点（句末无后续中文字也算）
        for m in re.finditer(r"[一-鿿]([,;!?:.])(?=[一-鿿]|$)", text):
            half = m.group(1)
            full = _HALFWIDTH_MAP[half]
            pos = m.start(1)
            issues.append(self._issue(
                "punct_halfwidth", pos, pos + 1, text,
                candidates=[{
                    "text": full,
                    "reason": "中文语句中应使用全角标点",
                    "confidence": 0.85,
                    "fix": {"start": pos, "end": pos + 1, "original": half,
                            "replacement": full},
                }],
                message=f"半角「{half}」疑似应为全角「{full}」",
            ))

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def _issue(rule: str, start: int, end: int, text: str, *,
               candidates: list[dict], message: str) -> dict:
        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        return {
            "id": "",  # 由 scan() 统一编号
            "rule": rule,
            "rule_name": RULE_NAMES[rule],
            "start": start,
            "end": end,
            "original": text[start:end],
            "candidates": candidates,
            "message": message,
            "context": "",
        }
