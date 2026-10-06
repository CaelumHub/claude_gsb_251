"""文本校对与修改回流的单元测试。

运行：``python -m unittest discover -s tests -v``
覆盖：
- 同音字 / 形近字 / 错词成语 / 多字 / 漏字 / 成对标点 / 标点误用的检出；
- 专名、英文、URL、方言、白名单、正确写法不被误判；
- 修改应用的版本冲突、重叠冲突、原文一致性校验；
- 偏移重映射（多处修改彼此不冲突）；
- 批量扫描中单篇失败不拖垮整批。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.proofreader import Proofreader, ProofTimeout, proofread_text
from nlp.revision import (EditConflict, EditSet, apply_edits,
                          remap_finding, remap_findings)
from pipeline.proof_service import ProofService
from storage import StoreRegistry


def categories(findings):
    return {f["category"] for f in findings}


def replacements(findings):
    return {(f["start"], f["replacement"]) for f in findings}


class TestProofreaderRecall(unittest.TestCase):
    def test_zai_zai(self):
        fs = proofread_text("我明天在去，你在家等我。")
        self.assertTrue(any(f["original"] == "在" and f["replacement"] == "再"
                            for f in fs))

    def test_zai_ci(self):
        fs = proofread_text("他在次犯错，一在提醒他。")
        joined = "".join(f["original"] for f in fs) + \
                 "".join(f["replacement"] for f in fs)
        self.assertIn("再次", joined)

    def test_ji_yi(self):
        fs = proofread_text("我己经完成了自已的任务。")
        rep = " ".join(f["replacement"] for f in fs)
        self.assertIn("已", rep)
        self.assertTrue(any("自己" in f["replacement"] for f in fs))

    def test_dedi_de(self):
        fs = proofread_text("他高兴的跳起来，跑的很快。")
        rep = "".join(f["replacement"] for f in fs)
        self.assertIn("地", rep)
        self.assertIn("得", rep)

    def test_wrong_idiom(self):
        fs = proofread_text("他迫不急待，甘败下风。")
        rep = " ".join(f["replacement"] for f in fs)
        self.assertIn("迫不及待", rep)
        self.assertIn("甘拜下风", rep)

    def test_install(self):
        fs = proofread_text("他按装软件。")
        self.assertTrue(any(f["replacement"] == "安装" for f in fs))

    def test_duplicate(self):
        fs = proofread_text("他慢慢的的走了。")
        self.assertTrue(any(f["category"] == "duplicate" for f in fs))

    def test_missing_de(self):
        fs = proofread_text("他高兴说不出话来。")
        self.assertTrue(any(f["category"] == "missing"
                            and f["replacement"] == "地" for f in fs))

    def test_missing_dei(self):
        fs = proofread_text("他跑很快。")
        self.assertTrue(any(f["replacement"] == "得" for f in fs))

    def test_unclosed_quote(self):
        fs = proofread_text("他说：“你好。")
        self.assertTrue(any(f["category"] == "punct_pair" for f in fs))

    def test_unclosed_bracket(self):
        fs = proofread_text("这是括号（不闭合的句子。")
        self.assertTrue(any(f["category"] == "punct_pair" for f in fs))

    def test_closed_pairs_ok(self):
        fs = proofread_text("他说：“你好。”（见注释【一】）")
        self.assertFalse(any(f["category"] == "punct_pair" for f in fs))

    def test_half_punct(self):
        fs = proofread_text("他说,今天很重要.")
        self.assertTrue(any(f["category"] == "punct_misuse" for f in fs))

    def test_repeat_punct(self):
        fs = proofread_text("非常,,好用..")
        self.assertTrue(any(f["original"] == ",," for f in fs))


class TestProofreaderPrecision(unittest.TestCase):
    """正确写法 / 专名 / 方言 / 英文混合不允许误报。"""

    CLEAN = [
        "我在家看书，他再来时请等一下。",
        "不要再迟到了，再次提醒大家。",
        "他再三解释，我再一次被说服了。",
        "他激动得不能自已，赞叹不已。",
        "他高兴地跳了起来，跑得很快，是个勇敢的孩子。",
        "美丽的花园里，孩子们在开心地玩耍，玩得不亦乐乎。",
        "他以身作则，再接再厉，甘拜下风的对手心服口服。",
        "腾讯和阿里巴巴在杭州、深圳设有研发中心。",
        "李克强在北京大学会见了世界卫生组织总干事。",
        "Python 3.11 性能提升 25%，价格 99.9 元。",
        "访问 https://a.com/path,x 或邮件 a@b.com。",
        "他说：“你好。”我回答：「再见。」",
        "他慢慢地走，好好地想，静静坐着。",
        "海内存知己，克己奉公，舍己为人，安分守己。",
        "即使下雨也去，既然决定了就不要犹豫。",
        "他克服困难，刻苦训练，取得了三千克的成绩。",
        "他跑了第一名，跑得很快，笑得开心。",
    ]

    def test_clean_texts(self):
        for t in self.CLEAN:
            fs = proofread_text(t, timeout=5.0)
            self.assertEqual(fs, [], f"干净文本被误报：{t} -> "
                                    f"{[(f['original'], f['replacement']) for f in fs]}")

    def test_whitelist_dialect(self):
        fs = proofread_text("这旮旯儿的事儿俺们自个儿搞定。",
                            whitelist=["旮旯儿", "自个儿"])
        # 白名单词不应触发错词/同音报告
        self.assertFalse(any("旮旯" in f["original"] or
                             "自个儿" in f["original"] for f in fs))

    def test_internet_newwords(self):
        fs = proofread_text("内卷、躺平、佛系，老铁们纷纷点赞。")
        self.assertFalse(any(f["severity"] == "high" for f in fs))

    def test_proper_noun_protected(self):
        fs = proofread_text("马云在北京创办阿里巴巴。")
        self.assertFalse(any(f["category"] in ("homophone", "shape", "word")
                             for f in fs))


class TestTimeout(unittest.TestCase):
    def test_timeout_raises(self):
        # 截止时刻已过，任何扫描都应立即抛超时
        p = Proofreader(deadline=time.time() - 1)
        with self.assertRaises(ProofTimeout):
            p.proofread("他在次犯错，迫不急待。" * 50)


class TestRevision(unittest.TestCase):
    def setUp(self):
        self.text = "我明天在去，他己经到了。"

    def test_apply_multiple(self):
        edits = [
            {"id": "a", "start": 3, "end": 4, "replacement": "再",
             "original": "在"},
            {"id": "b", "start": 7, "end": 8, "replacement": "已",
             "original": "己"},
        ]
        new, applied, ver = apply_edits(self.text, edits,
                                        base_version=1, expected_version=1)
        self.assertEqual(new, "我明天再去，他已经到了。")
        self.assertEqual(ver, 2)
        self.assertEqual(len(applied), 2)

    def test_stale_version(self):
        with self.assertRaises(EditConflict):
            apply_edits(self.text,
                        [{"start": 3, "end": 4, "replacement": "再"}],
                        base_version=1, expected_version=3)

    def test_overlap_rejected(self):
        with self.assertRaises(EditConflict):
            apply_edits(self.text,
                        [{"start": 3, "end": 6, "replacement": "xx"},
                         {"start": 4, "end": 7, "replacement": "yy"}])

    def test_original_mismatch(self):
        with self.assertRaises(EditConflict):
            apply_edits(self.text,
                        [{"start": 3, "end": 4, "replacement": "再",
                          "original": "X"}])

    def test_insertion_and_remap(self):
        new, applied, _ = apply_edits(
            "他高兴说", [{"id": "i", "insert_at": 3, "insert_text": "地"}],
            base_version=1)
        self.assertEqual(new, "他高兴地说")
        # 后面的 finding 偏移整体 +1
        later = {"id": "x", "start": 5, "end": 6, "replacement": "。",
                 "original": "了"}
        nf = remap_finding(later, applied)
        self.assertEqual(nf["start"], 6)

    def test_overlapping_finding_removed(self):
        _, applied, _ = apply_edits(
            self.text, [{"id": "a", "start": 3, "end": 4,
                         "replacement": "再", "original": "在"}],
            base_version=1)
        # 与已应用区间重叠的 finding 被丢弃
        overlap = {"id": "c", "start": 3, "end": 4, "replacement": "再"}
        self.assertIsNone(remap_finding(overlap, applied))

    def test_many_nonconflicting_edits(self):
        text = "在次按装迫不急待再接再励甘败下风"
        edits = [
            {"id": "1", "start": 0, "end": 2, "replacement": "再次",
             "original": "在次"},
            {"id": "2", "start": 2, "end": 4, "replacement": "安装",
             "original": "按装"},
            {"id": "3", "start": 4, "end": 8, "replacement": "迫不及待",
             "original": "迫不急待"},
            {"id": "4", "start": 8, "end": 12, "replacement": "再接再厉",
             "original": "再接再励"},
            {"id": "5", "start": 12, "end": 16, "replacement": "甘拜下风",
             "original": "甘败下风"},
        ]
        new, applied, _ = apply_edits(text, edits, base_version=1)
        self.assertEqual(new, "再次安装迫不及待再接再厉甘拜下风")
        self.assertEqual(len(applied), 5)


class TestProofServiceBatch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.reg = StoreRegistry(self.tmp)
        self.svc = ProofService(self.reg)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _add(self, text, name="t"):
        return self.reg.task("corpus").insert(
            {"name": name, "text": text, "version": 1})

    def test_batch_isolation_and_persistence(self):
        good = self._add("他在次犯错。")
        clean = self._add("文本完全正常，没有错别字。")
        job = self.svc.start_batch([good, clean], max_workers=2,
                                   doc_timeout=5)
        # 等待完成
        for _ in range(100):
            if job.status in ("done", "partial", "failed"):
                break
            time.sleep(0.05)
        snap = job.snapshot()
        self.assertEqual(snap["succeeded"], 2)
        self.assertEqual(snap["failed"], 0)
        # 结果已持久化
        scan = self.svc.latest_scan(good)
        self.assertTrue(scan["findings"])
        scan2 = self.svc.latest_scan(clean)
        self.assertEqual(scan2["findings"], [])

    def test_missing_doc_does_not_block(self):
        # 一个不存在的 id + 一个正常 id：坏的标记失败，好的照样完成
        good = self._add("他己经到了。")
        job = self.svc.start_batch(["corpus_not_exist", good],
                                   max_workers=2, doc_timeout=5)
        for _ in range(100):
            if job.status in ("done", "partial", "failed"):
                break
            time.sleep(0.05)
        snap = job.snapshot()
        self.assertEqual(snap["succeeded"], 1)
        self.assertEqual(snap["failed"], 1)
        # 好的那篇确实扫描出了问题
        scan = self.svc.latest_scan(good)
        self.assertTrue(any(f["replacement"] == "已" for f in
                            scan["findings"]))

    def test_rescan_after_revision(self):
        cid = self._add("他在次犯错，高兴说不出话。")
        first = self.svc.scan_text("他在次犯错，高兴说不出话。")
        self.assertTrue(first)
        # 应用第一处
        target = [f for f in first if f["original"] == "在次"][0]
        new, applied, ver = apply_edits(
            "他在次犯错，高兴说不出话。", [target], base_version=1)
        remaining = [f for f in first if f["id"] != target["id"]]
        out = self.svc.rescan_after_revision(
            cid, new, ver, applied, remaining)
        # 已应用的「在次」不再出现；漏字建议仍在（偏移已平移）
        self.assertFalse(any(f["original"] == "在次"
                             for f in out["findings"]))
        self.assertTrue(any(f["category"] == "missing"
                            for f in out["findings"]))


class TestPipelineStage(unittest.TestCase):
    def test_proofread_stage_auto_apply(self):
        from pipeline import PipelineEngine
        engine = PipelineEngine().register_builtin()
        cfg = {"name": "p", "stages": [
            {"name": "proofread"}, {"name": "segment"}]}
        out = engine.build(cfg).run({"text": "他在次犯错。"})
        # 高置信错误已自动修正，原文保留
        self.assertIn("再次", out["text"])
        self.assertIn("在次", out["raw_text"])
        # 下游分词基于改后文本
        self.assertIn("再次", out["words"])


if __name__ == "__main__":
    unittest.main()
