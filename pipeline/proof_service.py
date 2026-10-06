"""批量校对扫描服务。

诉求：「语料几篇到几百篇、长短不一，扫描要跟上批量入库的节奏，
某一篇扫不动不能卡住整批」。

实现：
- 内存任务表 + ThreadPoolExecutor 分块并发；
- 每篇独立 :class:`~nlp.proofreader.Proofreader`、独立时间预算，
  超时抛 :class:`~nlp.proofreader.ProofTimeout`，该篇标记 failed，
  其余继续；
- 进度（done/total）与每篇结果实时可查询，前端可轮询；
- 结果持久化到 ``proofread`` 分片存储，任务本身只保留状态摘要，
  服务重启后历史结果不丢（任务状态丢失可重新扫描，扫描幂等）。
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from nlp.proofreader import Proofreader, ProofTimeout
from nlp.revision import remap_findings


# 单篇扫描的默认时间预算（秒）。几百篇的批次里，个别异常长文不应拖垮整批。
DEFAULT_DOC_TIMEOUT = 8.0
DEFAULT_MAX_WORKERS = 4
DEFAULT_CHUNK_SIZE = 16


class ProofJob:
    def __init__(self, job_id: str, corpus_ids: list[str]):
        self.job_id = job_id
        self.corpus_ids = list(corpus_ids)
        self.total = len(corpus_ids)
        self.done = 0
        self.succeeded = 0
        self.failed = 0
        self.status = "pending"          # pending/running/done/partial/failed
        self.results: dict[str, dict] = {}   # corpus_id -> 结果摘要
        self.errors: dict[str, str] = {}
        self.started = time.time()
        self.finished: Optional[float] = None
        self._lock = threading.Lock()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "job_id": self.job_id,
                "status": self.status,
                "total": self.total,
                "done": self.done,
                "succeeded": self.succeeded,
                "failed": self.failed,
                "started": self.started,
                "finished": self.finished,
                "items": [
                    dict(corpus_id=cid, **self.results.get(cid, {}),
                         error=self.errors.get(cid))
                    if cid in self.results or cid in self.errors else
                    {"corpus_id": cid, "status": "pending"}
                    for cid in self.corpus_ids
                ],
            }


class ProofService:
    """校对扫描的应用服务：单篇扫描 + 批量任务编排 + 结果持久化。"""

    def __init__(self, registry=None):
        self.registry = registry
        self._jobs: dict[str, ProofJob] = {}
        self._lock = threading.Lock()

    def set_registry(self, registry) -> None:
        self.registry = registry

    # -- 单篇扫描（同步） -------------------------------------------------
    def scan_text(self, text: str, whitelist=None,
                  timeout: float = DEFAULT_DOC_TIMEOUT,
                  check_missing: bool = True) -> list[dict]:
        deadline = time.time() + timeout
        return Proofreader(whitelist=whitelist, deadline=deadline,
                           check_missing=check_missing).proofread(text)

    # -- 持久化 -----------------------------------------------------------
    def _store(self) -> Optional[object]:
        return self.registry.task("proofread") if self.registry else None

    def _persist(self, corpus_id: str, text: str, findings: list[dict],
                 version: int, whitelist=None) -> None:
        store = self._store()
        if store is None:
            return
        record = {
            "corpus_id": corpus_id,
            "text": text,
            "text_version": version,
            "findings": findings,
            "whitelist": list(whitelist or []),
            "created_at": time.time(),
        }
        store.insert(record)

    def latest_scan(self, corpus_id: str) -> Optional[dict]:
        store = self._store()
        if store is None:
            return None
        records = store.query(where=[("corpus_id", "eq", corpus_id)],
                              order_by="created_at", order="desc", limit=1)
        return records[0] if records else None

    # -- 批量任务 ---------------------------------------------------------
    def start_batch(self, corpus_ids: list[str],
                    max_workers: int = DEFAULT_MAX_WORKERS,
                    chunk_size: int = DEFAULT_CHUNK_SIZE,
                    doc_timeout: float = DEFAULT_DOC_TIMEOUT,
                    whitelist=None, check_missing: bool = True) -> ProofJob:
        job_id = "pj_" + uuid.uuid4().hex[:12]
        job = ProofJob(job_id, corpus_ids)
        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_batch,
            args=(job, corpus_ids, max_workers, chunk_size, doc_timeout,
                  whitelist, check_missing),
            daemon=True,
        )
        thread.start()
        return job

    def get_job(self, job_id: str) -> Optional[ProofJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def _run_batch(self, job: ProofJob, corpus_ids: list[str],
                   max_workers: int, chunk_size: int, doc_timeout: float,
                   whitelist, check_missing: bool) -> None:
        job.status = "running"
        store = self.registry.task("corpus") if self.registry else None
        try:
            for start in range(0, len(corpus_ids), chunk_size):
                chunk_ids = corpus_ids[start:start + chunk_size]
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = {
                        ex.submit(self._scan_one, cid, store, doc_timeout,
                                  whitelist, check_missing): cid
                        for cid in chunk_ids
                    }
                    for fut in as_completed(futures):
                        cid = futures[fut]
                        try:
                            summary = fut.result()
                            with job._lock:
                                job.results[cid] = summary
                                job.succeeded += 1
                        except ProofTimeout:
                            with job._lock:
                                job.errors[cid] = "扫描超时，已跳过（不影响其它语料）"
                                job.failed += 1
                        except Exception as exc:  # noqa: BLE001
                            with job._lock:
                                job.errors[cid] = f"{type(exc).__name__}: {exc}"
                                job.failed += 1
                        with job._lock:
                            job.done += 1
        finally:
            job.finished = time.time()
            job.status = "done" if job.failed == 0 else \
                ("partial" if job.succeeded else "failed")

    def _scan_one(self, corpus_id: str, store, doc_timeout: float,
                  whitelist, check_missing: bool) -> dict:
        """扫描单篇；任何异常都向上抛，由批量编排隔离，不影响其它文档。"""
        if store is None:
            raise RuntimeError("存储未就绪")
        record = store.get(corpus_id)
        if not record:
            raise RuntimeError(f"语料 {corpus_id} 不存在")
        text = record.get("text", "")
        version = int(record.get("version", 1))
        # 语料级白名单（方言/专名/新写法）优先，再叠加本次请求的全局白名单
        doc_wl = list(record.get("whitelist") or [])
        if whitelist:
            for w in whitelist:
                if w not in doc_wl:
                    doc_wl.append(w)
        findings = self.scan_text(text, whitelist=doc_wl,
                                  timeout=doc_timeout,
                                  check_missing=check_missing)
        self._persist(corpus_id, text, findings, version, doc_wl)
        counts = {"high": 0, "medium": 0, "low": 0}
        cat_counts: dict[str, int] = {}
        for f in findings:
            counts[f["severity"]] = counts.get(f["severity"], 0) + 1
            cat_counts[f["category"]] = cat_counts.get(f["category"], 0) + 1
        return {
            "status": "ok",
            "finding_count": len(findings),
            "severity_counts": counts,
            "category_counts": cat_counts,
            "text_version": version,
            "name": record.get("name", ""),
        }

    # -- 修改后重新扫描（供应用修改后调用） -------------------------------
    def rescan_after_revision(self, corpus_id: str, text: str, version: int,
                              applied, remaining_findings: list[dict],
                              whitelist=None,
                              doc_timeout: float = DEFAULT_DOC_TIMEOUT) -> dict:
        """应用修改后增量刷新：重映射未处理 finding + 只重扫改动附近文本。

        为保证「多处修改彼此不冲突、结果对得上」：
        - 已经忽略的 finding 保留（状态由调用方维护）；
        - 已应用的 finding 从列表移除；
        - 其余 finding 用 :func:`remap_findings` 平移偏移；
        - 再对新文本做一次轻量全扫，与重映射结果合并去重，得到新扫描集。
        """
        remapped = remap_findings(remaining_findings, applied)
        fresh = self.scan_text(text, whitelist=whitelist,
                               timeout=doc_timeout)
        merged = self._merge_findings(remapped, fresh)
        self._persist(corpus_id, text, merged, version, whitelist)
        return {"findings": merged, "text_version": version}

    @staticmethod
    def _merge_findings(mapped: list[dict], fresh: list[dict]) -> list[dict]:
        """合并重映射旧 finding 与新扫描 finding（按 rule+偏移去重）。"""
        seen = {(f["rule"], f["start"], f["end"], f["original"])
                for f in mapped}
        out = list(mapped)
        for f in fresh:
            key = (f["rule"], f["start"], f["end"], f["original"])
            if key not in seen:
                seen.add(key)
                out.append(f)
        out.sort(key=lambda f: (f["start"], f["end"]))
        return out
