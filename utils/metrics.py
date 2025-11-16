# utils/metrics.py
"""
Detection and classification metrics utilities.

Functions:
- iou(boxA, boxB)
- match_detections(pred_boxes, gt_boxes, iou_thresh=0.5) -> matches, unmatched lists
- precision_recall_f1_from_matches(...)
- average_precision_for_class(...) -> AP for 1 class (mAP over IoU=0.5)
- mean_average_precision(predictions, ground_truths, iou_thresh=0.5)
- classification_metrics(y_true, y_pred) -> accuracy, precision, recall, f1
"""

from typing import List, Tuple, Dict
import numpy as np
from collections import defaultdict

def iou(boxA: Tuple[float,float,float,float], boxB: Tuple[float,float,float,float]) -> float:
    """
    Compute IoU for two boxes in (x1,y1,x2,y2)
    """
    if boxA is None or boxB is None:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    if interArea <= 0:
        return 0.0
    boxAArea = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    denom = float(boxAArea + boxBArea - interArea)
    if denom <= 0:
        return 0.0
    return interArea / denom

def match_detections(preds: List[Dict], gts: List[Dict], iou_thresh: float = 0.5):
    """
    Match predictions to ground truth for a single frame.
    preds and gts are lists of dicts like {'bbox':(x1,y1,x2,y2), 'label':class_id, 'score':float}
    Returns lists:
      - matches: list of tuples (pred_idx, gt_idx, iou)
      - unmatched_preds: list of pred_idx
      - unmatched_gts: list of gt_idx
    Uses greedy matching by highest IoU.
    """
    matches = []
    if not preds or not gts:
        return matches, list(range(len(preds))), list(range(len(gts)))
    iou_matrix = np.zeros((len(preds), len(gts)), dtype=float)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            if p.get('label') != g.get('label'):
                # optionally allow cross-class matching? Here we require same label
                iou_matrix[i, j] = 0.0
            else:
                iou_matrix[i, j] = iou(p.get('bbox'), g.get('bbox'))
    # greedy matching
    used_preds = set()
    used_gts = set()
    while True:
        idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        max_iou = iou_matrix[idx]
        if max_iou <= 0:
            break
        pi, gj = int(idx[0]), int(idx[1])
        matches.append((pi, gj, float(max_iou)))
        used_preds.add(pi)
        used_gts.add(gj)
        iou_matrix[pi, :] = -1
        iou_matrix[:, gj] = -1
    unmatched_preds = [i for i in range(len(preds)) if i not in used_preds]
    unmatched_gts = [j for j in range(len(gts)) if j not in used_gts]
    return matches, unmatched_preds, unmatched_gts

def precision_recall_f1_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Simple classifier metrics. y_true, y_pred lists of same length.
    """
    if not y_true:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    from collections import Counter
    n = len(y_true)
    tp = fp = tn = fn = 0
    # For multiclass compute macro-averaged precision/recall/f1 using pairwise approach
    labels = set(y_true) | set(y_pred)
    precisions = []
    recalls = []
    f1s = []
    for label in labels:
        yt = [1 if y==label else 0 for y in y_true]
        yp = [1 if y==label else 0 for y in y_pred]
        _tp = sum(1 for a,b in zip(yt,yp) if a==1 and b==1)
        _fp = sum(1 for a,b in zip(yt,yp) if a==0 and b==1)
        _fn = sum(1 for a,b in zip(yt,yp) if a==1 and b==0)
        p,r,f = precision_recall_f1_from_counts(_tp, _fp, _fn)
        precisions.append(p); recalls.append(r); f1s.append(f)
    accuracy = sum(1 for a,b in zip(y_true,y_pred) if a==b) / float(n)
    return {'accuracy': float(accuracy),
            'precision_macro': float(sum(precisions)/len(precisions)) if precisions else 0.0,
            'recall_macro': float(sum(recalls)/len(recalls)) if recalls else 0.0,
            'f1_macro': float(sum(f1s)/len(f1s)) if f1s else 0.0 }

def average_precision_for_class(pred_boxes: List[Dict], gt_boxes: List[Dict], iou_thresh: float = 0.5):
    """
    Compute AP for one class using predicted boxes with 'score' and GT boxes.
    Inputs:
      - pred_boxes: list of dicts {'bbox':..., 'score':...}
      - gt_boxes: list of dicts {'bbox':...}
    Returns: average precision (0..1)
    Implementation: simple 11-point interpolation or exact AP by PR curve (here we do PR curve)
    """
    if gt_boxes is None:
        gt_boxes = []
    if not pred_boxes:
        return 0.0
    # sort preds by score desc
    preds = sorted(pred_boxes, key=lambda x: x.get('score', 0.0), reverse=True)
    detected = [False] * len(gt_boxes)
    tps = []
    fps = []
    for p in preds:
        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(gt_boxes):
            if detected[j]:
                continue
            val = iou(p.get('bbox'), g.get('bbox'))
            if val > best_iou:
                best_iou = val
                best_j = j
        if best_iou >= iou_thresh and best_j >= 0:
            tps.append(1)
            fps.append(0)
            detected[best_j] = True
        else:
            tps.append(0)
            fps.append(1)
    # cumulative
    tps_cum = np.cumsum(tps)
    fps_cum = np.cumsum(fps)
    precisions = tps_cum / (tps_cum + fps_cum + 1e-9)
    recalls = tps_cum / (len(gt_boxes) + 1e-9)
    # compute AP via trapezoidal rule on recall-precision curve
    # ensure monotonic precision
    for i in range(len(precisions)-2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i+1])
    # integrate
    recall_vals = np.concatenate(([0.0], recalls, [1.0]))
    precision_vals = np.concatenate(([0.0], precisions, [0.0]))
    ap = 0.0
    for i in range(len(recall_vals)-1):
        ap += (recall_vals[i+1] - recall_vals[i]) * precision_vals[i+1]
    return float(ap)

def mean_average_precision(all_predictions: Dict[int, List[Dict]], all_ground_truths: Dict[int, List[Dict]], iou_thresh: float = 0.5):
    """
    Compute mAP across multiple frames or multiple images.
    Inputs:
      - all_predictions: dict image_id -> list of preds [{'bbox','label','score'},...]
      - all_ground_truths: dict image_id -> list of gts [{'bbox','label'},...]
    Returns: mAP (0..1) averaged over classes present
    NOTE: This simple implementation computes AP per class by aggregating predictions across images.
    """
    # collect per-class lists
    per_class_preds = defaultdict(list)
    per_class_gts = defaultdict(list)
    for img_id, preds in all_predictions.items():
        for p in preds or []:
            lab = p.get('label', 0)
            per_class_preds[lab].append({'bbox': p.get('bbox'), 'score': p.get('score', 0.0)})
    for img_id, gts in all_ground_truths.items():
        for g in gts or []:
            lab = g.get('label', 0)
            per_class_gts[lab].append({'bbox': g.get('bbox')})
    ap_vals = []
    for lab in set(list(per_class_preds.keys()) + list(per_class_gts.keys())):
        ap = average_precision_for_class(per_class_preds.get(lab, []), per_class_gts.get(lab, []), iou_thresh=iou_thresh)
        ap_vals.append(ap)
    if not ap_vals:
        return 0.0
    return float(np.mean(ap_vals))