"""
holistic_features.py

Shared by the extraction, training and live scripts so all three compute
features IDENTICALLY. One frame = 232 numbers:

  [  0: 22]  upper-body pose: nose, eyes, ears, mouth, shoulders, elbows (x, y),
             relative to the shoulder midpoint, in units of shoulder width
  [ 22: 85]  left  hand SHAPE: 21 landmarks x (x,y,z), relative to its wrist
  [ 85:127]  left  hand LOCATION: 21 landmarks x (x,y), relative to the
             shoulder midpoint, in shoulder widths  <- where the hand is
  [127:190]  right hand SHAPE
  [190:232]  right hand LOCATION

Location is body-relative, so it does not depend on how far you sit from the
camera or where you are in the frame. Missing hand/pose = zeros.
"""

import numpy as np

SEQUENCE_LENGTH = 30
FRAME_FEATURES = 232
MODEL_FEATURES = 464  # frame features + per-frame velocity

POSE_IDX = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14]
# position j of the mirrored pose = position POSE_SWAP[j] of the original
POSE_SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9]

POSE = slice(0, 22)
L_SHAPE, L_LOC = slice(22, 85), slice(85, 127)
R_SHAPE, R_LOC = slice(127, 190), slice(190, 232)


def _hand_features(hand_landmarks, center, width):
    if hand_landmarks is None:
        return np.zeros(63, np.float32), np.zeros(42, np.float32)
    lm = np.array([[l.x, l.y, l.z] for l in hand_landmarks.landmark], dtype=np.float32)
    rel = lm - lm[0]
    m = np.max(np.linalg.norm(rel, axis=1))
    if m > 0:
        rel = rel / m
    loc = (lm[:, :2] - center) / width
    return rel.flatten(), loc.flatten().astype(np.float32)


def extract_frame_features(results):
    """MediaPipe Holistic results -> (232,) feature vector."""
    out = np.zeros(FRAME_FEATURES, dtype=np.float32)
    pose = results.pose_landmarks
    if pose is None:
        return out
    pts = np.array([[l.x, l.y] for l in pose.landmark], dtype=np.float32)
    vis = [pose.landmark[11].visibility, pose.landmark[12].visibility]
    center = (pts[11] + pts[12]) / 2.0
    width = float(np.linalg.norm(pts[11] - pts[12]))
    if min(vis) < 0.3 or width < 1e-3:
        return out
    out[POSE] = ((pts[POSE_IDX] - center) / width).flatten()
    ls, ll = _hand_features(results.left_hand_landmarks, center, width)
    rs, rl = _hand_features(results.right_hand_landmarks, center, width)
    out[L_SHAPE], out[L_LOC], out[R_SHAPE], out[R_LOC] = ls, ll, rs, rl
    return out


def hands_present(seq):
    """(T,) bool: at least one hand visible in each frame."""
    return np.any(seq[:, L_SHAPE] != 0, axis=1) | np.any(seq[:, R_SHAPE] != 0, axis=1)


def add_motion_features(seq):
    """(T, 232) -> (T, 464): features + 10x frame-to-frame velocity.
    Velocity is zero wherever the thing being tracked is missing in either frame."""
    seq = np.asarray(seq, dtype=np.float32)
    vel = np.zeros_like(seq)
    groups = [
        (POSE, POSE),
        (slice(22, 127), L_SHAPE),   # left shape+loc, presence from left shape
        (slice(127, 232), R_SHAPE),  # right shape+loc, presence from right shape
    ]
    for block, pres_block in groups:
        present = np.any(seq[:, pres_block] != 0, axis=1)
        both = present[1:] & present[:-1]
        vel[1:, block] = (seq[1:, block] - seq[:-1, block]) * both[:, None]
    return np.concatenate([seq, vel * 10.0], axis=1).astype(np.float32)


def trim_and_resample(raw, min_active=6):
    """Keep first..last frame with a visible hand (+2 margin), resample to 30."""
    idx = np.where(hands_present(raw))[0]
    if len(idx) < min_active:
        return None
    start, end = max(0, idx[0] - 2), min(len(raw) - 1, idx[-1] + 2)
    seg = raw[start:end + 1]
    pick = np.linspace(0, len(seg) - 1, SEQUENCE_LENGTH).astype(int)
    return seg[pick]
