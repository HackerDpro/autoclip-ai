"""
Face-aware cropping module for AutoClip AI.

Analyzes faces ACROSS THE FULL DURATION of a video segment (not just a
single snapshot) and builds a time-varying crop path, so the 9:16 frame
actually follows speakers as they move — true tracking, not a one-time
static decision.

Three strategies, decided per-clip based on the dominant pattern observed
across all sampled frames:
  - SINGLE speaker  -> dynamic crop_x(t) that follows that face over time.
  - TWO speakers on opposite sides -> vertical split-screen, each half
    independently tracking its own speaker over time.
  - NO reliable face -> static center-crop fallback.

Uses OpenCV's bundled Haar Cascade detector: free, CPU-only, no downloads.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import logger, OUTPUT_WIDTH, OUTPUT_HEIGHT

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

SAMPLE_FPS = 0.5  # one sample every 2 seconds — still smooth after
                  # interpolation, but keeps the ffmpeg expression short
                  # enough to avoid filter graph parsing failures on
                  # longer clips (empirically, 2.0 fps broke on 60s+ clips)
FACE_MIN_SIZE_RATIO = 0.05
DUAL_FACE_SEPARATION_RATIO = 0.25
SMOOTHING_WINDOW = 5  # moving-average window (in samples) to prevent jitter


class CropMode(Enum):
    CENTER = "center"
    SINGLE = "single"
    SPLIT = "split"


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def center_x(self) -> int:
        return self.x + self.w // 2


@dataclass
class CropTimeline:
    """
    A time-varying crop decision. `times` are seconds relative to the clip
    start (0 = clip start). `positions` are the corresponding crop center-x
    values in SOURCE video pixel coordinates at each timestamp.
    """
    mode: CropMode
    times: List[float]
    single_x: List[int]                    # used if mode == SINGLE
    left_x: List[int]                       # used if mode == SPLIT
    right_x: List[int]                      # used if mode == SPLIT


def _sample_frames_with_timestamps(
    video_path: Path, start: float, end: float, sample_fps: float
) -> List[Tuple[float, np.ndarray]]:
    """
    Sample frames at a fixed rate across [start, end], returning
    (relative_time_in_clip, frame) pairs.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    duration = end - start
    num_samples = max(2, int(duration * sample_fps))
    step = duration / num_samples

    results: List[Tuple[float, np.ndarray]] = []

    for i in range(num_samples):
        rel_time = i * step
        timestamp_ms = (start + rel_time) * 1000
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        success, frame = cap.read()
        if success and frame is not None:
            results.append((rel_time, frame))

    cap.release()
    return results


def _detect_faces(frame: np.ndarray) -> List[FaceBox]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    min_size = int(frame.shape[1] * FACE_MIN_SIZE_RATIO)

    detections = _FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size)
    )
    return [FaceBox(x=x, y=y, w=w, h=h) for (x, y, w, h) in detections]


def _smooth(values: List[int], window: int) -> List[int]:
    """Simple moving average to prevent the crop from jittering frame-to-frame."""
    if len(values) <= window:
        return values

    arr = np.array(values, dtype=np.float64)
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")[: len(arr)]
    return [int(v) for v in smoothed]


def _fill_gaps(times: List[float], values: List[Optional[int]]) -> List[int]:
    """
    Forward/backward-fill missing detections (frames where no face was
    found) so the timeline has no holes — holding the last known position
    is better than snapping back to center on a momentary miss.
    """
    filled: List[int] = []
    last_known: Optional[int] = None

    for v in values:
        if v is not None:
            last_known = v
        filled.append(last_known if last_known is not None else 0)

    # Backward-fill any leading Nones (before the first detection)
    first_known = next((v for v in filled if v != 0), None)
    if first_known is not None:
        for i, v in enumerate(filled):
            if v == 0 and values[i] is None:
                filled[i] = first_known
            else:
                break

    return filled


def analyze_segment_faces(video_path: Path, start: float, end: float) -> CropTimeline:
    """
    Sample faces across the FULL duration of a segment at SAMPLE_FPS, decide
    the overall crop strategy (single/split/center), and build a smoothed,
    time-varying crop position timeline for true face tracking.
    """
    try:
        samples = _sample_frames_with_timestamps(video_path, start, end, SAMPLE_FPS)
    except Exception as e:
        logger.warning(f"Frame sampling failed, falling back to center crop: {e}")
        return CropTimeline(mode=CropMode.CENTER, times=[0.0], single_x=[0], left_x=[], right_x=[])

    if not samples:
        logger.warning("No frames sampled, falling back to center crop.")
        return CropTimeline(mode=CropMode.CENTER, times=[0.0], single_x=[0], left_x=[], right_x=[])

    frame_width = samples[0][1].shape[1]

    times: List[float] = []
    single_raw: List[Optional[int]] = []
    left_raw: List[Optional[int]] = []
    right_raw: List[Optional[int]] = []

    single_votes = 0
    split_votes = 0
    no_face_votes = 0

    for rel_time, frame in samples:
        faces = _detect_faces(frame)
        times.append(rel_time)

        if len(faces) == 0:
            single_raw.append(None)
            left_raw.append(None)
            right_raw.append(None)
            no_face_votes += 1
            continue

        if len(faces) == 1:
            single_raw.append(faces[0].center_x)
            left_raw.append(None)
            right_raw.append(None)
            single_votes += 1
            continue

        if len(faces) >= 3:
            # 3+ faces: we have no reliable way to know who's actually
            # speaking (no active-speaker detection yet). Guessing wrong
            # confidently is worse than falling back safely — treat this
            # frame as a "no reliable pattern" vote rather than picking
            # an arbitrary pair.
            single_raw.append(None)
            left_raw.append(None)
            right_raw.append(None)
            no_face_votes += 1
            continue

        # Exactly 2 faces detected
        left_face, right_face = sorted(faces, key=lambda f: f.center_x)
        separation = (right_face.center_x - left_face.center_x) / frame_width

        if separation >= DUAL_FACE_SEPARATION_RATIO:
            left_raw.append(left_face.center_x)
            right_raw.append(right_face.center_x)
            single_raw.append(None)
            split_votes += 1
        else:
            avg_center = (left_face.center_x + right_face.center_x) // 2
            single_raw.append(avg_center)
            left_raw.append(None)
            right_raw.append(None)
            single_votes += 1

    total = len(samples)

    if split_votes >= total * 0.35:
        left_filled = _fill_gaps(times, left_raw)
        right_filled = _fill_gaps(times, right_raw)
        left_smoothed = _smooth(left_filled, SMOOTHING_WINDOW)
        right_smoothed = _smooth(right_filled, SMOOTHING_WINDOW)

        logger.info(
            f"Face tracking: SPLIT-SCREEN mode "
            f"({split_votes}/{total} frames showed 2 separated faces)"
        )
        return CropTimeline(
            mode=CropMode.SPLIT, times=times,
            single_x=[], left_x=left_smoothed, right_x=right_smoothed,
        )

    if single_votes >= total * 0.35:
        single_filled = _fill_gaps(times, single_raw)
        single_smoothed = _smooth(single_filled, SMOOTHING_WINDOW)

        movement = max(single_smoothed) - min(single_smoothed) if single_smoothed else 0
        logger.info(
            f"Face tracking: SINGLE speaker mode "
            f"({single_votes}/{total} frames), tracked movement range: {movement}px"
        )
        return CropTimeline(
            mode=CropMode.SINGLE, times=times,
            single_x=single_smoothed, left_x=[], right_x=[],
        )

    logger.info(
        f"Face tracking: no reliable pattern "
        f"({no_face_votes}/{total} no-face frames) -> CENTER fallback"
    )
    return CropTimeline(mode=CropMode.CENTER, times=[0.0], single_x=[0], left_x=[], right_x=[])


def _build_crop_expression(times: List[float], positions: List[int], crop_w: int, src_width: int) -> str:
    """
    Build an ffmpeg 'crop' filter x-expression that interpolates linearly
    between known (time, position) points, with every value explicitly
    clamped to valid bounds to prevent ffmpeg filter configuration errors.

    Downsamples to a maximum number of points if needed — ffmpeg's
    expression parser has practical complexity limits, and deeply nested
    if(between(...)) chains from too many points can fail to parse
    entirely (observed empirically on 60s+ clips at high sample density).
    """
    MAX_EXPRESSION_POINTS = 25

    max_x = max(0, src_width - crop_w)
    clamped = [max(0, min(p - crop_w // 2, max_x)) for p in positions]

    if len(times) > MAX_EXPRESSION_POINTS:
        # Downsample evenly to stay under the safe complexity limit.
        indices = np.linspace(0, len(times) - 1, MAX_EXPRESSION_POINTS).astype(int)
        times = [times[i] for i in indices]
        clamped = [clamped[i] for i in indices]

    if len(times) == 1 or len(set(clamped)) == 1:
        return str(clamped[0])

    expr = str(clamped[-1])
    for i in range(len(times) - 1, 0, -1):
        t0, t1 = times[i - 1], times[i]
        x0, x1 = clamped[i - 1], clamped[i]

        if t1 <= t0:
            continue

        interp = f"({x0}+({x1}-{x0})*(t-{t0})/({t1}-{t0}))"
        interp_clamped = f"max(0,min({interp},{max_x}))"
        expr = f"if(between(t,{t0},{t1}),{interp_clamped},{expr})"

    return expr


def build_video_filter(timeline: CropTimeline, src_width: int, src_height: int) -> Tuple[str, bool]:
    """
    Build the ffmpeg filter for a given crop timeline.

    Returns:
        (filter_string, is_complex) — is_complex=True means this must be
        used with -filter_complex + explicit -map, not a simple -vf.
    """
    if timeline.mode == CropMode.CENTER:
        return f"scale=-2:{OUTPUT_HEIGHT},crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}", False

    scale_factor = OUTPUT_HEIGHT / src_height
    scaled_width = int(src_width * scale_factor)

    if timeline.mode == CropMode.SINGLE:
        scaled_positions = [int(p * scale_factor) for p in timeline.single_x]
        x_expr = _build_crop_expression(timeline.times, scaled_positions, OUTPUT_WIDTH, scaled_width)
        return f"scale=-2:{OUTPUT_HEIGHT},crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:'{x_expr}':0", False

    if timeline.mode == CropMode.SPLIT:
        half_height = OUTPUT_HEIGHT // 2
        target_aspect = OUTPUT_WIDTH / half_height

        if src_width / src_height >= target_aspect:
            crop_h = src_height
            crop_w = int(src_height * target_aspect)
        else:
            crop_w = src_width
            crop_h = int(src_width / target_aspect)

        crop_y = max(0, (src_height - crop_h) // 2)

        left_x_expr = _build_crop_expression(timeline.times, timeline.left_x, crop_w, src_width)
        right_x_expr = _build_crop_expression(timeline.times, timeline.right_x, crop_w, src_width)

        filter_str = (
            f"[0:v]crop={crop_w}:{crop_h}:'{left_x_expr}':{crop_y},"
            f"scale={OUTPUT_WIDTH}:{half_height}[top];"
            f"[0:v]crop={crop_w}:{crop_h}:'{right_x_expr}':{crop_y},"
            f"scale={OUTPUT_WIDTH}:{half_height}[bottom];"
            f"[top][bottom]vstack=inputs=2[vcrop]"
        )
        return filter_str, True

    raise ValueError(f"Unknown crop mode: {timeline.mode}")