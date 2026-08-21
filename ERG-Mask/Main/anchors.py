from __future__ import annotations

from typing import Optional

import numpy as np


def cas_iou(box: np.ndarray, clusters: np.ndarray) -> np.ndarray:

    box = np.asarray(box, dtype=np.float64)
    clusters = np.asarray(clusters, dtype=np.float64)

    inter_wh = np.minimum(clusters, box)
    intersection = inter_wh[:, 0] * inter_wh[:, 1]

    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    union = box_area + cluster_area - intersection

    return intersection / np.maximum(union, 1e-12)


def avg_iou(boxes: np.ndarray, clusters: np.ndarray) -> float:

    boxes = np.asarray(boxes, dtype=np.float64)
    return float(
        np.mean(
            [
                np.max(cas_iou(boxes[i], clusters))
                for i in range(boxes.shape[0])
            ]
        )
    )


def kmeans(
    boxes: np.ndarray,
    k: int = 9,
    seed: Optional[int] = 0,
    max_iter: int = 1000,
) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64)

    if boxes.ndim != 2 or boxes.shape[1] != 2:
        raise ValueError("boxes must have shape (N, 2).")
    if boxes.shape[0] < k:
        raise ValueError("The number of boxes must be >= k.")

    rng = np.random.default_rng(seed)
    clusters = boxes[
        rng.choice(boxes.shape[0], size=k, replace=False)
    ].copy()

    last_assignment = np.full(boxes.shape[0], -1, dtype=np.int64)

    for _ in range(max_iter):
        distance = np.stack(
            [1.0 - cas_iou(box, clusters) for box in boxes],
            axis=0,
        )
        assignment = np.argmin(distance, axis=1)

        if np.array_equal(last_assignment, assignment):
            break

        for j in range(k):
            members = boxes[assignment == j]
            if members.shape[0] > 0:
                clusters[j] = np.median(members, axis=0)

        last_assignment = assignment

    return clusters


def scale_anchors(
    anchors_wh: np.ndarray,
    image_size: int = 512,
) -> np.ndarray:

    anchors_wh = np.asarray(anchors_wh, dtype=np.float64)
    return np.rint(anchors_wh * image_size).astype(np.int64)
