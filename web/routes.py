"""Flask API 路由。

把所有 NLP 能力、存储与流水线编排暴露为 REST 接口，
前端 10 个页面通过 ``fetch`` 调用这些接口。
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Optional

from flask import Blueprint, current_app, jsonify, request

from nlp import (get_constituency_parser, get_embeddings, get_keywords, get_ner,
                 get_parser, get_segmenter, get_sentiment, get_summarizer,
                 get_tagger, get_translator, ENTITY_TYPE_NAMES, TAG_NAMES,
                 DEP_REL_NAMES, PHRASE_NAMES, POLARITY_NAMES)
from nlp.lexicon import STOPWORDS
from nlp.revision import (AppliedEdit, EditConflict, apply_edits,
                          remap_findings)
from storage import StoreRegistry


api = Blueprint("api", __name__, url_prefix="/api")


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _registry() -> StoreRegistry:
    return current_app.config["STORE_REGISTRY"]


def _engine():
    return current_app.config["PIPELINE_ENGINE"]


def _proofs():
    return current_app.config["PROOF_SERVICE"]


def _models_dir() -> str:
    import os
    path = os.path.join(current_app.config["DATA_ROOT"], "models")
    os.makedirs(path, exist_ok=True)
    return path


def _store_result(task: str, text: str, result: dict,
                  corpus_id: Optional[str] = None,
                  text_version: Optional[int] = None) -> str:
    record = {"text": text, "result": result, "created_at": time.time()}
    if corpus_id:
        record["corpus_id"] = corpus_id
    if text_version is not None:
        record["text_version"] = text_version
        # 供结果列表按版本对齐：同语料同版本的结果才代表「当前文本」
        record["source"] = {"corpus_id": corpus_id,
                            "text_version": text_version}
    return _registry().task(task).insert(record)


def _payload() -> dict:
    data = request.get_json(silent=True) or {}
    return data


def _resolve_text(data: dict) -> tuple[str, Optional[str], Optional[int]]:
    """从请求中取文本：优先 text，其次 corpus_id。

    返回 (text, corpus_id, text_version)；语料来源时附带当前版本号，
    下游任务结果据此记录「基于哪个版本的文本」，保证改前改后对得上。
    """
    if data.get("text"):
        return data["text"], data.get("corpus_id"), data.get("text_version")
    corpus_id = data.get("corpus_id")
    if corpus_id:
        record = _registry().task("corpus").get(corpus_id)
        if record:
            return record.get("text", ""), corpus_id, \
                int(record.get("version", 1))
        return "", corpus_id, None
    return "", None, None


def _clean(text: str, remove_stopwords: bool = True) -> dict:
    text = re.sub(r"\s+", " ", text).strip()
    seg = get_segmenter()
    words = seg.cut(text)
    if remove_stopwords:
        kept = [w for w in words if w not in STOPWORDS]
    else:
        kept = words
    removed = len(words) - len(kept)
    return {
        "text": text,
        "cleaned": " ".join(kept),
        "tokens": kept,
        "original_tokens": words,
        "removed_stopwords": removed,
    }


# ---------------------------------------------------------------------------
# 状态
# ---------------------------------------------------------------------------

@api.get("/status")
def status():
    return jsonify({
        "ok": True,
        "version": "1.0.0",
        "tasks": _registry().tasks(),
        "time": time.time(),
    })


@api.get("/meta")
def meta():
    """给前端提供标签集合与可配置参数。"""
    return jsonify({
        "tag_names": TAG_NAMES,
        "dep_rel_names": DEP_REL_NAMES,
        "phrase_names": PHRASE_NAMES,
        "entity_type_names": ENTITY_TYPE_NAMES,
        "polarity_names": POLARITY_NAMES,
        "directions": [{"id": "zh2en", "name": "中文 → 英文"},
                       {"id": "en2zh", "name": "英文 → 中文"}],
    })


# ---------------------------------------------------------------------------
# 语料库管理
# ---------------------------------------------------------------------------

@api.get("/corpus")
def list_corpus():
    records = _registry().task("corpus").all()
    items = [{
        "id": r.get("id"),
        "name": r.get("name", "未命名"),
        "length": len(r.get("text", "")),
        "version": int(r.get("version", 1)),
        "ignored_count": len(r.get("ignored_findings", [])),
        "last_proof_at": r.get("last_proof_at"),
        "created_at": r.get("created_at"),
        "preview": r.get("text", "")[:80],
    } for r in records if not r.get("_deleted")]
    items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return jsonify({"corpora": items})


@api.post("/corpus")
def create_corpus():
    data = _payload()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "语料内容不能为空"}), 400
    record = {
        "name": data.get("name") or f"语料_{int(time.time())}",
        "text": text,
        "version": 1,
        "whitelist": data.get("whitelist") or [],
        "created_at": time.time(),
    }
    rid = _registry().task("corpus").insert(record)
    _schedule_auto_scan(rid)
    return jsonify({"id": rid, "ok": True})


@api.post("/corpus/upload")
def upload_corpus():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "未接收到文件"}), 400
    raw = file.read()
    text = None
    for enc in ("utf-8", "gbk", "gb18030", "utf-16"):
        try:
            text = raw.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if text is None:
        return jsonify({"error": "无法解码文件内容"}), 400
    name = data_name = file.filename or "上传文件"
    record = {"name": name, "text": text.strip(), "version": 1,
              "created_at": time.time()}
    rid = _registry().task("corpus").insert(record)
    _schedule_auto_scan(rid)
    return jsonify({"id": rid, "name": name, "length": len(text), "ok": True})


def _schedule_auto_scan(cid: str) -> None:
    """入库后后台自动扫描一次（不阻塞响应；失败静默，可在页面手动重扫）。"""
    import threading

    app = current_app._get_current_object()

    def _job():
        with app.app_context():
            try:
                record = _registry().task("corpus").get(cid)
                if not record:
                    return
                findings = _proofs().scan_text(
                    record.get("text", ""),
                    whitelist=record.get("whitelist"),
                    timeout=10.0)
                _proofs()._persist(cid, record.get("text", ""), findings,
                                   int(record.get("version", 1)),
                                   record.get("whitelist"))
            except Exception:  # noqa: BLE001
                pass

    threading.Thread(target=_job, daemon=True).start()


@api.get("/corpus/<cid>")
def get_corpus(cid: str):
    record = _registry().task("corpus").get(cid)
    if not record:
        return jsonify({"error": "语料不存在"}), 404
    return jsonify(record)


@api.delete("/corpus/<cid>")
def delete_corpus(cid: str):
    ok = _registry().task("corpus").delete(cid)
    return jsonify({"ok": ok})


@api.post("/corpus/<cid>/clean")
def clean_corpus(cid: str):
    record = _registry().task("corpus").get(cid)
    if not record:
        return jsonify({"error": "语料不存在"}), 404
    data = _payload()
    result = _clean(record.get("text", ""), data.get("remove_stopwords", True))
    _store_result("clean", record.get("text", ""), result, corpus_id=cid)
    return jsonify(result)


# ---------------------------------------------------------------------------
# 文本校对（错别字 / 标点扫描、应用、忽略、撤销、批量任务）
# ---------------------------------------------------------------------------

def _finding_stats(findings: list[dict]) -> dict:
    sev = {"high": 0, "medium": 0, "low": 0}
    cats: dict[str, int] = {}
    for f in findings:
        sev[f["severity"]] = sev.get(f["severity"], 0) + 1
        cats[f["category"]] = cats.get(f["category"], 0) + 1
    return {"total": len(findings), "severity": sev, "categories": cats}


@api.post("/corpus/<cid>/whitelist")
def update_whitelist(cid: str):
    """保存语料级白名单（方言/专名/刻意新写法），扫描时豁免。"""
    record = _registry().task("corpus").get(cid)
    if not record:
        return jsonify({"error": "语料不存在"}), 404
    words = _payload().get("whitelist")
    if words is None:
        return jsonify({"error": "缺少 whitelist"}), 400
    words = [str(w).strip() for w in words if str(w).strip()]
    _registry().task("corpus").update(cid, {"whitelist": words})
    return jsonify({"ok": True, "whitelist": words})


@api.post("/proofread/scan")
def proofread_scan():
    """扫描任意文本（不落库），或扫描某篇语料（同时持久化扫描结果）。"""
    data = _payload()
    timeout = float(data.get("doc_timeout", 8.0))
    whitelist = data.get("whitelist") or []
    check_missing = bool(data.get("check_missing", True))
    cid = data.get("corpus_id")
    if cid:
        record = _registry().task("corpus").get(cid)
        if not record:
            return jsonify({"error": "语料不存在"}), 404
        text = record.get("text", "")
        version = int(record.get("version", 1))
        findings = _proofs().scan_text(text, whitelist=whitelist,
                                       timeout=timeout,
                                       check_missing=check_missing)
        _proofs()._persist(cid, text, findings, version, whitelist)
        return jsonify({"corpus_id": cid, "text_version": version,
                        "findings": findings, "stats": _finding_stats(findings)})
    text = data.get("text") or ""
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    findings = _proofs().scan_text(text, whitelist=whitelist,
                                   timeout=timeout,
                                   check_missing=check_missing)
    return jsonify({"findings": findings, "stats": _finding_stats(findings)})


@api.get("/proofread/<cid>")
def proofread_status(cid: str):
    """某篇语料最近一次扫描结果 + 当前文本版本。"""
    record = _registry().task("corpus").get(cid)
    if not record:
        return jsonify({"error": "语料不存在"}), 404
    scan = _proofs().latest_scan(cid)
    ignored = record.get("ignored_findings", [])
    return jsonify({
        "corpus_id": cid,
        "text": record.get("text", ""),
        "text_version": int(record.get("version", 1)),
        "name": record.get("name", ""),
        "findings": (scan or {}).get("findings", []),
        "scan_text_version": (scan or {}).get("text_version"),
        "ignored_ids": ignored,
        "whitelist": record.get("whitelist", []),
        "stats": _finding_stats((scan or {}).get("findings", [])),
        "history": record.get("proof_history", []),
    })


@api.post("/proofread/batch")
def proofread_batch():
    """批量扫描多篇语料，立即返回 job_id，前端轮询进度。"""
    data = _payload()
    registry = _registry()
    corpus_ids = data.get("corpus_ids") or []
    if not corpus_ids:
        # 默认扫描全部
        corpus_ids = [r["id"] for r in registry.task("corpus").all()
                      if not r.get("_deleted")]
    if not corpus_ids:
        return jsonify({"error": "没有可扫描的语料"}), 400
    job = _proofs().start_batch(
        corpus_ids,
        max_workers=int(data.get("max_workers", 4)),
        chunk_size=int(data.get("chunk_size", 16)),
        doc_timeout=float(data.get("doc_timeout", 8.0)),
        whitelist=data.get("whitelist") or [],
        check_missing=bool(data.get("check_missing", True)))
    return jsonify({"job_id": job.job_id, "total": job.total, "ok": True})


@api.get("/proofread/job/<job_id>")
def proofread_job(job_id: str):
    job = _proofs().get_job(job_id)
    if not job:
        return jsonify({"error": "扫描任务不存在或已过期"}), 404
    return jsonify(job.snapshot())


def _serialize_applied(applied: list[AppliedEdit]) -> list[dict]:
    return [{
        "finding_id": a.finding_id, "old_start": a.old_start,
        "old_end": a.old_end, "new_start": a.new_start,
        "new_end": a.new_end, "original": a.original,
        "replacement": a.replacement, "rule": a.rule,
        "category": a.category,
    } for a in applied]


@api.post("/corpus/<cid>/proofread/apply")
def proofread_apply(cid: str):
    """应用一组校对修改，文本立即回流（版本 +1），并增量刷新扫描结果。"""
    record = _registry().task("corpus").get(cid)
    if not record:
        return jsonify({"error": "语料不存在"}), 404
    data = _payload()
    edits = data.get("edits")
    if not edits:
        return jsonify({"error": "没有要应用的修改"}), 400
    base_version = int(data.get("base_version", record.get("version", 1)))
    current_version = int(record.get("version", 1))
    if base_version != current_version:
        return jsonify({
            "error": f"文本版本已过期（当前 v{current_version}，"
                     f"你基于 v{base_version}），请刷新后重试",
            "stale": True, "current_version": current_version,
        }), 409
    text = record.get("text", "")
    try:
        new_text, applied, new_version = apply_edits(
            text, edits, base_version=base_version,
            expected_version=current_version)
    except EditConflict as exc:
        return jsonify({"error": str(exc), "conflict": True}), 409

    # 未处理的 finding 重映射，再增量扫描合并
    scan = _proofs().latest_scan(cid) or {}
    old_findings = scan.get("findings", [])
    applied_ids = {a.finding_id for a in applied}
    remaining = [f for f in old_findings if f["id"] not in applied_ids]
    refreshed = _proofs().rescan_after_revision(
        cid, new_text, new_version, applied, remaining,
        whitelist=record.get("whitelist"))

    # 历史记录（保留最近 20 次，便于撤销/审计）
    history = list(record.get("proof_history", []))
    history.append({
        "version": new_version,
        "at": time.time(),
        "changes": _serialize_applied(applied),
        "old_text": text,
    })
    history = history[-20:]
    _registry().task("corpus").update(cid, {
        "text": new_text, "version": new_version,
        "proof_history": history,
        "last_proof_at": time.time(),
    })
    return jsonify({
        "ok": True, "text": new_text, "text_version": new_version,
        "applied": _serialize_applied(applied),
        "findings": refreshed["findings"],
        "stats": _finding_stats(refreshed["findings"]),
        "ignored_ids": record.get("ignored_findings", []),
    })


@api.post("/corpus/<cid>/proofread/ignore")
def proofread_ignore(cid: str):
    """忽略/恢复一处或多处 finding（记录到语料，重扫不再提示）。"""
    record = _registry().task("corpus").get(cid)
    if not record:
        return jsonify({"error": "语料不存在"}), 404
    data = _payload()
    ids = set(data.get("ids") or [])
    mode = data.get("mode", "ignore")  # ignore | restore
    ignored = set(record.get("ignored_findings", []))
    if mode == "restore":
        ignored -= ids
    else:
        ignored |= ids
    _registry().task("corpus").update(cid, {"ignored_findings": sorted(ignored)})
    return jsonify({"ok": True, "ignored_ids": sorted(ignored)})


@api.post("/corpus/<cid>/proofread/undo")
def proofread_undo(cid: str):
    """撤销最近一次校对应用（回滚文本与版本）。"""
    record = _registry().task("corpus").get(cid)
    if not record:
        return jsonify({"error": "语料不存在"}), 404
    history = list(record.get("proof_history", []))
    if not history:
        return jsonify({"error": "没有可撤销的校对记录"}), 400
    last = history.pop()
    old_text = last["old_text"]
    version = int(last["version"]) - 1
    _registry().task("corpus").update(cid, {
        "text": old_text, "version": max(1, version),
        "proof_history": history,
    })
    # 重新扫描回滚后的文本
    findings = _proofs().scan_text(old_text,
                                   whitelist=record.get("whitelist"))
    _proofs()._persist(cid, old_text, findings, max(1, version),
                       record.get("whitelist"))
    return jsonify({"ok": True, "text": old_text,
                    "text_version": max(1, version),
                    "findings": findings,
                    "stats": _finding_stats(findings)})


# ---------------------------------------------------------------------------
# 分词与词性标注
# ---------------------------------------------------------------------------

@api.post("/segment")
def segment():
    data = _payload()
    text, cid, ver = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    seg = get_segmenter()
    words = seg.cut(text)
    result = {"words": words, "count": len(words)}
    rid = _store_result("segment", text, result, corpus_id=cid,
                        text_version=ver)
    result["id"] = rid
    result["text_version"] = ver
    return jsonify(result)


@api.post("/pos")
def pos_tag():
    data = _payload()
    text, cid, ver = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    tagger = get_tagger()
    tokens = [[w, t] for w, t in tagger.tag(text)]
    result = {"tokens": tokens, "tag_names": TAG_NAMES}
    rid = _store_result("pos", text, result, corpus_id=cid, text_version=ver)
    result["id"] = rid
    result["text_version"] = ver
    return jsonify(result)


# ---------------------------------------------------------------------------
# 句法分析
# ---------------------------------------------------------------------------

@api.post("/parse")
def parse():
    data = _payload()
    text, cid, ver = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    dep = get_parser().parse(text)
    const = get_constituency_parser().parse(text)
    result = {
        "dependency": dep,
        "constituency": const,
        "dep_rel_names": DEP_REL_NAMES,
        "phrase_names": PHRASE_NAMES,
    }
    rid = _store_result("parse", text, result, corpus_id=cid, text_version=ver)
    result["id"] = rid
    result["text_version"] = ver
    return jsonify(result)


# ---------------------------------------------------------------------------
# 命名实体识别与标注
# ---------------------------------------------------------------------------

@api.post("/ner")
def ner():
    data = _payload()
    text, cid, ver = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    entities = get_ner().recognize(text)
    result = {"entities": entities, "entity_type_names": ENTITY_TYPE_NAMES}
    rid = _store_result("ner", text, result, corpus_id=cid, text_version=ver)
    result["id"] = rid
    result["text_version"] = ver
    return jsonify(result)


@api.post("/ner/annotate")
def ner_annotate():
    data = _payload()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    record = {
        "text": text,
        "entities": data.get("entities", []),
        "note": data.get("note", ""),
        "created_at": time.time(),
    }
    rid = _registry().task("annotation").insert(record)
    return jsonify({"id": rid, "ok": True})


@api.get("/ner/annotations")
def ner_annotations():
    records = _registry().task("annotation").all()
    return jsonify({"annotations": records})


# ---------------------------------------------------------------------------
# 情感分析
# ---------------------------------------------------------------------------

@api.post("/sentiment")
def sentiment():
    data = _payload()
    text, cid, ver = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    result = get_sentiment().analyze(text)
    result["polarity_name"] = POLARITY_NAMES.get(result["polarity"], "")
    rid = _store_result("sentiment", text, result, corpus_id=cid,
                        text_version=ver)
    result["id"] = rid
    result["text_version"] = ver
    return jsonify(result)


# ---------------------------------------------------------------------------
# 文本摘要
# ---------------------------------------------------------------------------

@api.post("/summary")
def summary():
    data = _payload()
    text, cid, ver = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    result = get_summarizer().summarize(
        text, ratio=data.get("ratio", 0.3),
        max_sentences=data.get("max_sentences"))
    rid = _store_result("summary", text, result, corpus_id=cid,
                        text_version=ver)
    result["id"] = rid
    result["text_version"] = ver
    return jsonify(result)


# ---------------------------------------------------------------------------
# 机器翻译（模拟）
# ---------------------------------------------------------------------------

@api.post("/translate")
def translate():
    data = _payload()
    text, cid, ver = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    result = get_translator().translate(text, direction=data.get("direction", "zh2en"))
    rid = _store_result("translate", text, result, corpus_id=cid,
                        text_version=ver)
    result["id"] = rid
    result["text_version"] = ver
    return jsonify(result)


# ---------------------------------------------------------------------------
# 关键词提取
# ---------------------------------------------------------------------------

@api.post("/keywords")
def keywords():
    data = _payload()
    text, cid, ver = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    result = get_keywords().extract(text, top_k=data.get("top_k", 10),
                                    method=data.get("method", "hybrid"))
    rid = _store_result("keywords", text, result, corpus_id=cid,
                        text_version=ver)
    result["id"] = rid
    result["text_version"] = ver
    return jsonify(result)


# ---------------------------------------------------------------------------
# 词向量
# ---------------------------------------------------------------------------

def _embedding_path() -> str:
    import os
    return os.path.join(_models_dir(), "embeddings.json")


@api.post("/embeddings/train")
def train_embeddings():
    data = _payload()
    corpus_ids = data.get("corpus_ids")
    store = _registry().task("corpus")
    if corpus_ids:
        texts = [store.get(c)["text"] for c in corpus_ids if store.get(c)]
    else:
        texts = [r["text"] for r in store.all() if not r.get("_deleted")]
    if not texts:
        return jsonify({"error": "没有可用语料，请先上传语料"}), 400

    emb = get_embeddings()
    emb.train(texts, vocab_size=data.get("vocab_size", 200),
              dim=data.get("dim", 20), window=data.get("window", 5),
              min_count=data.get("min_count", 1))

    payload = {
        "vocab": emb.vocab,
        "vectors": emb.vectors,
        "dim": emb.dim,
        "trained_at": time.time(),
        "corpus_count": len(texts),
    }
    with open(_embedding_path(), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return jsonify(emb.stats())


@api.get("/embeddings/vectors")
def embeddings_vectors():
    emb = get_embeddings()
    if not emb.vectors:
        _load_embeddings()
        emb = get_embeddings()
    if not emb.vectors:
        return jsonify({"error": "尚未训练词向量"}), 404
    n_clusters = int(request.args.get("clusters", 5))
    proj = emb.project_2d()
    clusters = emb.cluster(n_clusters)
    return jsonify({
        "points": [{"word": w, "x": round(p[0], 4), "y": round(p[1], 4),
                    "cluster": clusters.get(w, 0)} for w, p in proj.items()],
        "stats": emb.stats(),
    })


@api.get("/embeddings/neighbors")
def embeddings_neighbors():
    word = request.args.get("word", "")
    k = int(request.args.get("k", 10))
    emb = get_embeddings()
    if not emb.vectors:
        _load_embeddings()
        emb = get_embeddings()
    return jsonify({"word": word, "neighbors": emb.nearest(word, k)})


def _load_embeddings():
    import os
    path = _embedding_path()
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        emb = get_embeddings()
        emb.vocab = data.get("vocab", [])
        emb.vectors = data.get("vectors", {})
        emb.dim = data.get("dim", 0)
    except (json.JSONDecodeError, OSError):
        pass


# ---------------------------------------------------------------------------
# 流水线配置与执行
# ---------------------------------------------------------------------------

@api.get("/pipeline/stages")
def pipeline_stages():
    return jsonify({"stages": _engine().list_stages()})


@api.post("/pipeline")
def save_pipeline():
    data = _payload()
    config = data.get("config") or data
    if not config.get("stages"):
        return jsonify({"error": "流水线至少需要一个阶段"}), 400
    name = config.get("name") or f"流水线_{int(time.time())}"
    record = {"name": name, "config": config, "created_at": time.time()}
    rid = _registry().task("pipeline_config").insert(record)
    return jsonify({"id": rid, "name": name, "ok": True})


@api.get("/pipeline")
def list_pipelines():
    records = _registry().task("pipeline_config").all()
    items = [{"id": r["id"], "name": r.get("name"), "config": r.get("config"),
              "created_at": r.get("created_at")}
             for r in records if not r.get("_deleted")]
    items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return jsonify({"pipelines": items})


@api.get("/pipeline/<pid>")
def get_pipeline(pid: str):
    record = _registry().task("pipeline_config").get(pid)
    if not record:
        return jsonify({"error": "流水线不存在"}), 404
    return jsonify(record)


@api.post("/pipeline/preview")
def pipeline_preview():
    """对单条文本跑流水线（不持久化），供配置页预览。"""
    data = _payload()
    text = (data.get("text") or "").strip()
    config = data.get("config")
    if not text or not config:
        return jsonify({"error": "缺少文本或配置"}), 400
    try:
        result = _engine().build(config).run({"text": text})
        return jsonify({"ok": True, "output": result})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 400


@api.post("/pipeline/<pid>/run")
def run_pipeline(pid: str):
    record = _registry().task("pipeline_config").get(pid)
    if not record:
        return jsonify({"error": "流水线不存在"}), 404
    config = record.get("config")
    data = _payload()

    run_id = uuid.uuid4().hex[:12]
    started = time.time()

    if data.get("batch"):
        # 批量：对语料库中的多篇文档执行
        corpus_ids = data.get("corpus_ids") or []
        store = _registry().task("corpus")
        docs = []
        if corpus_ids:
            docs = [store.get(c)["text"] for c in corpus_ids if store.get(c)]
        else:
            docs = [r["text"] for r in store.all() if not r.get("_deleted")]
        if not docs:
            return jsonify({"error": "没有可处理的文档"}), 400

        progress_state = {"done": 0, "total": len(docs)}

        def _progress(done, total):
            progress_state["done"] = done
            progress_state["total"] = total

        results = _engine().run_batch(
            config, docs, shared=data.get("shared"),
            max_workers=data.get("max_workers", 4),
            chunk_size=data.get("chunk_size", 16),
            progress=_progress)
        succeeded = sum(1 for r in results if r and r["ok"])
        failed = len(results) - succeeded
        run_record = {
            "run_id": run_id, "pipeline_id": pid, "batch": True,
            "doc_count": len(docs), "succeeded": succeeded, "failed": failed,
            "started": started, "finished": time.time(),
            "results": results,
        }
        rid = _registry().task("pipeline_run").insert(run_record)
        return jsonify({"run_id": run_id, "id": rid, "succeeded": succeeded,
                        "failed": failed, "doc_count": len(docs)})
    else:
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "缺少文本"}), 400
        try:
            output = _engine().build(config).run({"text": text})
            run_record = {
                "run_id": run_id, "pipeline_id": pid, "batch": False,
                "text": text, "output": output,
                "started": started, "finished": time.time(),
            }
            rid = _registry().task("pipeline_run").insert(run_record)
            return jsonify({"run_id": run_id, "id": rid, "ok": True,
                            "output": output})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(exc)}), 400


@api.get("/pipeline/run/<run_id>")
def get_pipeline_run(run_id: str):
    records = _registry().task("pipeline_run").query(
        where=[("run_id", "eq", run_id)])
    if not records:
        return jsonify({"error": "执行记录不存在"}), 404
    return jsonify(records[0])


# ---------------------------------------------------------------------------
# 结果查询（分片合并与查询）
# ---------------------------------------------------------------------------

@api.get("/results")
def list_result_tasks():
    registry = _registry()
    tasks = []
    for name in registry.tasks():
        if name in ("corpus", "pipeline_config", "annotation"):
            continue
        stats = registry.task(name).stats()
        tasks.append(stats)
    return jsonify({"tasks": tasks})


@api.get("/results/<task>")
def query_results(task: str):
    registry = _registry()
    if task not in registry.tasks():
        return jsonify({"error": "任务不存在"}), 404
    store = registry.task(task)
    where = []
    for key in ("type", "corpus_id"):
        val = request.args.get(key)
        if val:
            where.append((key, "eq", val))
    order_by = request.args.get("order_by")
    order = request.args.get("order", "desc")
    limit = request.args.get("limit", type=int)
    offset = request.args.get("offset", 0, type=int)
    records = store.query(where=where or None, order_by=order_by,
                          order=order, limit=limit, offset=offset)
    return jsonify({
        "task": task,
        "count": len(records),
        "stats": store.stats(),
        "records": records,
    })


@api.post("/results/<task>/compact")
def compact_results(task: str):
    registry = _registry()
    if task not in registry.tasks():
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(registry.task(task).compact())


@api.get("/results/<task>/merge")
def merge_results(task: str):
    registry = _registry()
    if task not in registry.tasks():
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(registry.task(task).merge())


@api.post("/results/compact_all")
def compact_all():
    return jsonify({"compacted": _registry().compact_all()})
