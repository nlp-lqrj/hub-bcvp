# -*- coding: utf-8 -*-


import re
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional


_EVAL_SENTENCES: Optional[List[str]] = None


def set_eval_sentences(sentences: List[str]) -> None:
    global _EVAL_SENTENCES
    _EVAL_SENTENCES = sentences


def _decode_legacy(sentence: str, label_ids: List[int]) -> Dict[str, List[str]]:
    # 复刻旧版评测脚本：将标签序列截断到句子长度，再拼成数字字符串用于正则匹配
    labels_str = "".join([str(x) for x in label_ids[: len(sentence)]])
    results = defaultdict(list)

    for m in re.finditer(r"(04+)", labels_str):
        s, e = m.span()
        results["LOCATION"].append(sentence[s:e])
    for m in re.finditer(r"(15+)", labels_str):
        s, e = m.span()
        results["ORGANIZATION"].append(sentence[s:e])
    for m in re.finditer(r"(26+)", labels_str):
        s, e = m.span()
        results["PERSON"].append(sentence[s:e])
    for m in re.finditer(r"(37+)", labels_str):
        s, e = m.span()
        results["TIME"].append(sentence[s:e])

    return results


def _update_stats(stats_dict, true_entities, pred_entities) -> None:
    for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
        stats_dict[key]["正确识别"] += len([ent for ent in pred_entities[key] if ent in true_entities[key]])
        stats_dict[key]["样本实体数"] += len(true_entities[key])
        stats_dict[key]["识别出实体数"] += len(pred_entities[key])


def _compute_from_stats(stats_dict) -> Dict[str, float]:
    f1_scores = []
    out = {}
    for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
        precision = stats_dict[key]["正确识别"] / (1e-5 + stats_dict[key]["识别出实体数"])
        recall = stats_dict[key]["正确识别"] / (1e-5 + stats_dict[key]["样本实体数"])
        f1 = (2 * precision * recall) / (precision + recall + 1e-5)
        f1_scores.append(f1)
        out[f"{key.lower()}_precision"] = float(precision)
        out[f"{key.lower()}_recall"] = float(recall)
        out[f"{key.lower()}_f1"] = float(f1)

    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    out["macro_f1"] = macro_f1

    correct_pred = sum(stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"])
    total_pred = sum(stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"])
    true_enti = sum(stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"])
    micro_precision = correct_pred / (total_pred + 1e-5)
    micro_recall = correct_pred / (true_enti + 1e-5)
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)

    out["micro_f1"] = float(micro_f1)
    out["micro_precision"] = float(micro_precision)
    out["micro_recall"] = float(micro_recall)
    return out


def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    """Trainer compute_metrics.

    Expects:
      - eval_pred.predictions: (N, T, num_labels) logits OR tuple(logits, ...)
      - eval_pred.label_ids:   (N, T) with -100 for ignored positions
    We reconstruct per-sentence label sequences by selecting positions where label != -100.
    """
    if _EVAL_SENTENCES is None:
        raise RuntimeError("Eval sentences not set. Call set_eval_sentences(dev_dataset.sentences) before training.")

    # 兼容训练器的评测输出对象与 (预测结果, 标签编号) 形式的元组
    predictions = getattr(eval_pred, "predictions", None)
    label_ids = getattr(eval_pred, "label_ids", None)
    if predictions is None or label_ids is None:
        # 视为 (预测结果, 标签编号) 形式的元组
        predictions, label_ids = eval_pred

    if isinstance(predictions, tuple):
        logits = predictions[0]
    else:
        logits = predictions

    preds = np.argmax(logits, axis=-1)

    stats_dict = {
        "LOCATION": defaultdict(int),
        "TIME": defaultdict(int),
        "PERSON": defaultdict(int),
        "ORGANIZATION": defaultdict(int),
    }

    n = len(label_ids)
    if len(_EVAL_SENTENCES) != n:
        raise RuntimeError(f"Sentence count mismatch: {len(_EVAL_SENTENCES)} sentences vs {n} label sequences.")

    for i in range(n):
        sent = _EVAL_SENTENCES[i]
        lab_seq = label_ids[i]
        pred_seq = preds[i]

        # 仅保留对齐到原始字符的位置（忽略 -100 的位置）
        true_char_labels = []
        pred_char_labels = []
        for lb, pr in zip(lab_seq, pred_seq):
            if lb == -100:
                continue
            true_char_labels.append(int(lb))
            pred_char_labels.append(int(pr))

        true_entities = _decode_legacy(sent, true_char_labels)
        pred_entities = _decode_legacy(sent, pred_char_labels)
        _update_stats(stats_dict, true_entities, pred_entities)

    return _compute_from_stats(stats_dict)
