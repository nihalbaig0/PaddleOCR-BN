import os
import re
import sys
import shapely
from shapely.geometry import Polygon
import numpy as np
from collections import defaultdict
import operator
import editdistance
from typing import Tuple, List, Dict
import unicodedata

def strQ2B(ustring: str) -> str:
    """Convert full-width characters to half-width ones."""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif inside_code >= 65281 and inside_code <= 65374:
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

def is_bangla(text: str) -> bool:
    """Check if the text contains Bangla characters."""
    bangla_range = range(0x0980, 0x09FF)  # Unicode range for Bangla
    return any(ord(char) in bangla_range for char in text)

def normalize_bangla(text: str) -> str:
    """Normalize Bangla text by handling common variations."""
    # Remove Zero Width Joiner and Non-Joiner
    text = text.replace('\u200C', '').replace('\u200D', '')
    # Normalize Unicode compositions
    text = unicodedata.normalize('NFKC', text)
    return text

def polygon_from_str(polygon_points: List[float]) -> Polygon:
    """Create a shapely polygon object from points."""
    polygon_points = np.array(polygon_points).reshape(4, 2)
    polygon = Polygon(polygon_points).convex_hull
    return polygon

def polygon_iou(poly1: Polygon, poly2: Polygon) -> float:
    """Calculate intersection over union between two polygons."""
    if not poly1.intersects(poly2):
        return 0
    try:
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.area + poly2.area - inter_area
        return float(inter_area) / union_area
    except shapely.geos.TopologicalError:
        print("shapely.geos.TopologicalError occurred, iou set to 0")
        return 0

def calculate_text_similarity(gt_str: str, dt_str: str) -> int:
    """Calculate edit distance with language-specific handling."""
    gt_str = normalize_bangla(gt_str) if is_bangla(gt_str) else gt_str
    dt_str = normalize_bangla(dt_str) if is_bangla(dt_str) else dt_str
    return editdistance.eval(gt_str, dt_str)

def e2e_eval(gt_dir: str, res_dir: str, ignore_blank: bool = False, language: str = 'mixed') -> Dict[str, float]:
    """
    End-to-end evaluation for OCR results.
    
    Args:
        gt_dir: Ground truth directory
        res_dir: Results directory
        ignore_blank: Whether to ignore blank spaces
        language: 'bangla', 'english', or 'mixed'
    """
    print(f"Starting evaluation for {language} OCR...")
    iou_thresh = 0.5
    val_names = os.listdir(gt_dir)
    stats = {
        'num_gt_chars': 0,
        'gt_count': 0,
        'dt_count': 0,
        'hit': 0,
        'ed_sum': 0,
        'bangla_count': 0,
        'english_count': 0
    }

    for val_name in val_names:
        # Read ground truth
        with open(os.path.join(gt_dir, val_name), encoding="utf-8") as f:
            gt_lines = [o.strip() for o in f.readlines()]
        
        gts = []
        ignore_masks = []
        for line in gt_lines:
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            gts.append(parts[:8] + [parts[9] if len(parts) == 10 else ""])
            ignore_masks.append(parts[8])

        # Read detection results
        val_path = os.path.join(res_dir, val_name)
        if not os.path.exists(val_path):
            dt_lines = []
        else:
            with open(val_path, encoding="utf-8") as f:
                dt_lines = [o.strip() for o in f.readlines()]
        
        dts = []
        for line in dt_lines:
            parts = line.strip().split("\t")
            dts.append(parts[:8] + [parts[8] if len(parts) == 9 else ""])

        # Match detections with ground truth
        dt_match = [False] * len(dts)
        gt_match = [False] * len(gts)
        all_ious = defaultdict(tuple)

        # Calculate IOUs
        for index_gt, gt in enumerate(gts):
            gt_poly = polygon_from_str([float(x) for x in gt[0:8]])
            for index_dt, dt in enumerate(dts):
                dt_poly = polygon_from_str([float(x) for x in dt[0:8]])
                iou = polygon_iou(dt_poly, gt_poly)
                if iou >= iou_thresh:
                    all_ious[(index_gt, index_dt)] = iou

        # Sort and match pairs
        for index_gt, index_dt in sorted(all_ious.items(), key=operator.itemgetter(1), reverse=True):
            if not gt_match[index_gt[0]] and not dt_match[index_dt[0]]:
                gt_match[index_gt[0]] = True
                dt_match[index_dt[0]] = True
                
                gt_str = strQ2B(gts[index_gt[0]][8])
                dt_str = strQ2B(dts[index_dt[0]][8])
                
                if ignore_blank:
                    gt_str = gt_str.replace(" ", "")
                    dt_str = dt_str.replace(" ", "")

                if ignore_masks[index_gt[0]] == "0":
                    ed = calculate_text_similarity(gt_str, dt_str)
                    stats['ed_sum'] += ed
                    stats['num_gt_chars'] += len(gt_str)
                    
                    if gt_str == dt_str:
                        stats['hit'] += 1
                    
                    stats['gt_count'] += 1
                    stats['dt_count'] += 1
                    
                    if is_bangla(gt_str):
                        stats['bangla_count'] += 1
                    else:
                        stats['english_count'] += 1

        # Handle unmatched detections and ground truths
        for i, matched in enumerate(dt_match):
            if not matched:
                stats['ed_sum'] += len(dts[i][8])
                stats['dt_count'] += 1

        for i, matched in enumerate(gt_match):
            if not matched and ignore_masks[i] == "0":
                stats['ed_sum'] += len(gts[i][8])
                stats['num_gt_chars'] += len(gts[i][8])
                stats['gt_count'] += 1

    # Calculate metrics
    eps = 1e-9
    metrics = {}
    metrics['precision'] = stats['hit'] / (stats['dt_count'] + eps) * 100
    metrics['recall'] = stats['hit'] / (stats['gt_count'] + eps) * 100
    metrics['f_measure'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'] + eps)
    metrics['character_accuracy'] = (1 - stats['ed_sum'] / (stats['num_gt_chars'] + eps)) * 100
    metrics['avg_edit_dist_per_img'] = stats['ed_sum'] / len(val_names)
    metrics['avg_edit_dist_per_field'] = stats['ed_sum'] / (stats['gt_count'] + eps)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Total Images Processed: {len(val_names)}")
    print(f"Bangla Text Fields: {stats['bangla_count']}")
    print(f"English Text Fields: {stats['english_count']}")
    print(f"Character Accuracy: {metrics['character_accuracy']:.2f}%")
    print(f"Average Edit Distance per Field: {metrics['avg_edit_dist_per_field']:.2f}")
    print(f"Average Edit Distance per Image: {metrics['avg_edit_dist_per_img']:.2f}")
    print(f"Precision: {metrics['precision']:.2f}%")
    print(f"Recall: {metrics['recall']:.2f}%")
    print(f"F-measure: {metrics['f_measure']:.2f}%")

    return metrics

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 ocr_e2e_eval.py <gt_dir> <res_dir> <language>")
        print("language options: bangla, english, mixed")
        sys.exit(1)

    gt_folder = sys.argv[1]
    pred_folder = sys.argv[2]
    language = sys.argv[3].lower()
    
    if language not in ['bangla', 'english', 'mixed']:
        print("Error: language must be 'bangla', 'english', or 'mixed'")
        sys.exit(1)
        
    e2e_eval(gt_folder, pred_folder, language=language)
