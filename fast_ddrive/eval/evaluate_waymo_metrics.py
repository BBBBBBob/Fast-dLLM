#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Waymo E2E planning with ADE and Rater Feedback Score (RFS).

Usage:
  python evaluate_waymo.py \
    --pred_json ./predictions_bench_format.json \
    --gt_tfrecords /path/to/waymo/val*.tfrecord* \
    --output_dir ./eval_results

Or if you have a pre-saved GT dict:
  python evaluate_waymo.py \
    --pred_json ./predictions_bench_format.json \
    --gt_dict_pkl ./gt_dict.pkl \
    --output_dir ./eval_results
"""

import os
import json
import argparse
import pickle
import re
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import sys as _sys
from pathlib import Path as _Path
_eval_dir = str(_Path(__file__).resolve().parent)
if _eval_dir not in _sys.path:
    _sys.path.insert(0, _eval_dir)

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Import RFS evaluation from waymo_rfs_utils (was kewei_scripts/utils.py)
from waymo_rfs_utils import get_rater_feedback_score

# Waymo protos
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2


# Navigation command mapping (from the original code)
command = {
    0: "UNDEFINED",
    1: "TURN_LEFT",
    2: "TURN_RIGHT",
    3: "STRAIGHT",
    4: "U_TURN"
}


NUM = r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?'

PAIR_RE = re.compile(
    rf'\\?\[\s*({NUM})\s*\\?,\s*({NUM})\s*\\?\]',
    flags=re.IGNORECASE
)


PAIR_RE_PLAIN = re.compile(
    rf'\[\s*({NUM})\s*,\s*({NUM})\s*\]',
    flags=re.IGNORECASE
)

# Fallback: match bare number pairs like "1.23, 4.56" on separate lines or comma-separated
BARE_PAIR_RE = re.compile(
    rf'({NUM})\s*,\s*({NUM})',
    flags=re.IGNORECASE
)


def extract_trajectory(blob: str) -> List[List[float]]:
    """Extract trajectory from model response string (lenient matching)."""
    # Try to focus on text after "trajectory" keyword if present
    if "trajectory" in blob.lower():
        idx = blob.lower().index("trajectory")
        blob = blob[idx:]

    inner = re.sub(r'<\|mdm_start\|>|<\|mdm_end\|>', '', blob)

    inner = (inner
             .replace(r'\"', '"')
             .replace(r'\[', '[')
             .replace(r'\]', ']')
             .replace(r'\,', ',')
             .replace(r'\+', '+'))

    # Try escaped bracket pairs first
    pairs = PAIR_RE.findall(inner)

    # Try plain bracket pairs
    if not pairs:
        pairs = PAIR_RE_PLAIN.findall(inner)

    # Try bare number pairs as last resort
    if not pairs:
        pairs = BARE_PAIR_RE.findall(inner)

    if not pairs:
        return []

    return [[float(x), float(y)] for x, y in pairs]


def _finite_diff_velocity(p, t):
    """Compute velocity using finite differences."""
    p = np.asarray(p, float)
    t = np.asarray(t, float)
    n = len(p)
    v = np.zeros(n, float)
    if n >= 2:
        v[0] = (p[1]-p[0])/(t[1]-t[0])
        v[-1] = (p[-1]-p[-2])/(t[-1]-t[-2])
    if n >= 3:
        v[1:-1] = (p[2:]-p[:-2])/(t[2:]-t[:-2])
    return v


def _finite_diff_accel(p, t):
    """Compute acceleration using finite differences."""
    p = np.asarray(p, float)
    t = np.asarray(t, float)
    n = len(p)
    a = np.zeros(n, float)
    if n < 3:
        return a

    a[0] = 2*(((p[1]-p[0])/(t[1]-t[0])) - ((p[2]-p[1])/(t[2]-t[1]))) / ((t[1]-t[0]) + (t[2]-t[1]))
    a[-1] = 2*(((p[-1]-p[-2])/(t[-1]-t[-2])) - ((p[-2]-p[-3])/(t[-2]-t[-3]))) / ((t[-1]-t[-2]) + (t[-2]-t[-3]))
    for i in range(1, n-1):
        dt1 = t[i]-t[i-1]
        dt2 = t[i+1]-t[i]
        a[i] = 2*(((p[i+1]-p[i])/dt2) - ((p[i]-p[i-1])/dt1)) / (dt1+dt2)
    return a


def _jmt_coeffs(p0, v0, a0, p1, v1, a1, T):
    """Compute Jerk Minimization Trajectory (JMT) coefficients."""
    A0 = p0
    A1 = v0
    A2 = a0/2.0
    T2, T3, T4, T5 = T**2, T**3, T**4, T**5
    M = np.array([
        [  T3,    T4,     T5],
        [3*T2,  4*T3,   5*T4],
        [6*T,  12*T2,  20*T3]
    ], float)
    b = np.array([
        p1 - (A0 + A1*T + A2*T2),
        v1 - (A1 + 2*A2*T),
        a1 - (2*A2)
    ], float)
    A3, A4, A5 = np.linalg.solve(M, b)
    return np.array([A0, A1, A2, A3, A4, A5], float)


def _eval_quintic(coeffs, tau):
    """Evaluate quintic polynomial at time tau."""
    a0, a1, a2, a3, a4, a5 = coeffs
    return (((a5*tau + a4)*tau + a3)*tau + a2)*tau**2 + a1*tau + a0


def jmt_interpolate_xy_with_start(p_start, traj_1to5, t_new):
    """
    Interpolate trajectory using JMT from start point through 5 waypoints.

    Args:
        p_start: Starting point [x, y]
        traj_1to5: 5 future waypoints, shape (5, 2)
        t_new: New time points to interpolate at

    Returns:
        Interpolated trajectory at t_new
    """
    P = np.vstack([np.asarray(p_start, float)[None, :], np.asarray(traj_1to5, float)])  # (6, 2)
    t = np.arange(0.0, 6.0)  # [0,1,2,3,4,5]

    vx = _finite_diff_velocity(P[:, 0], t)
    vy = _finite_diff_velocity(P[:, 1], t)
    ax = _finite_diff_accel(P[:, 0], t)
    ay = _finite_diff_accel(P[:, 1], t)

    coeffs_x, coeffs_y, seg_starts = [], [], []
    for i in range(len(t)-1):
        T = t[i+1] - t[i]
        cx = _jmt_coeffs(P[i, 0], vx[i], ax[i], P[i+1, 0], vx[i+1], ax[i+1], T)
        cy = _jmt_coeffs(P[i, 1], vy[i], ay[i], P[i+1, 1], vy[i+1], ay[i+1], T)
        coeffs_x.append(cx)
        coeffs_y.append(cy)
        seg_starts.append(t[i])
    seg_starts = np.asarray(seg_starts)

    t_new = np.asarray(t_new, float)
    t_new = np.clip(t_new, 0.0, 5.0)

    X = np.empty_like(t_new)
    Y = np.empty_like(t_new)
    for k, tk in enumerate(t_new):
        i = min(np.searchsorted(seg_starts, tk, side='right')-1, len(seg_starts)-1)
        i = max(i, 0)
        tau = tk - seg_starts[i]
        X[k] = _eval_quintic(coeffs_x[i], tau)
        Y[k] = _eval_quintic(coeffs_y[i], tau)
    return np.stack([X, Y], axis=1)


def load_waymo_e2e_data(tfrecord_pattern: str) -> Dict[str, Any]:
    """
    Load ground truth data from Waymo E2E TFRecords.

    Args:
        tfrecord_pattern: Glob pattern for TFRecord files

    Returns:
        Dictionary mapping frame_id to E2EDFrame proto
    """
    filenames = tf.io.matching_files(tfrecord_pattern)
    if tf.size(filenames) == 0:
        raise FileNotFoundError(f"No TFRecords matched {tfrecord_pattern}")

    gt_map = {}
    for f in filenames.numpy().tolist():
        if isinstance(f, bytes):
            f = f.decode('utf-8')
        ds = tf.data.TFRecordDataset(f, compression_type='')
        it = ds.as_numpy_iterator()

        for raw in tqdm(it, desc=f"Reading {os.path.basename(f)}"):
            data = wod_e2ed_pb2.E2EDFrame()
            data.ParseFromString(raw)

            # Only keep frames with valid preference trajectories
            if len(data.preference_trajectories) == 0 or \
                    data.preference_trajectories[0].preference_score == -1:
                continue

            gt_map[data.frame.context.name] = data

    print(f"Loaded {len(gt_map)} GT frames from TFRecords")
    return gt_map


def load_predictions(json_path: str) -> Dict[str, np.ndarray]:
    """
    Load predictions from JSON file and convert to 4Hz 20-point trajectories.

    Uses lenient matching: tries multiple fields to find trajectory data,
    pads short trajectories, and skips only truly unparseable entries.

    Args:
        json_path: Path to predictions JSON file

    Returns:
        Dictionary mapping sample_id to trajectory array (20, 2)
    """
    with open(json_path, "r") as f:
        obj = json.load(f)

    # Handle both list and dict formats
    if isinstance(obj, list):
        preds = obj
    elif isinstance(obj, dict):
        preds = obj.get("predictions", obj)
    else:
        raise ValueError(f"Unrecognized predictions.json format: {type(obj)}")

    t_4hz = np.linspace(0.25, 5.0, 20)
    pred_map = {}
    n_ok = n_skip = n_zero = 0

    for item in tqdm(preds, desc="Processing predictions"):
        # Skip non-dict entries
        if not isinstance(item, dict):
            n_skip += 1
            continue

        sample_id = item.get("sample_id", "")
        if not sample_id:
            n_skip += 1
            continue

        traj = None

        # Strategy 1: use pre-parsed pred_trajectory if available
        traj = item.get("pred_trajectory")

        # Strategy 2: extract from conversations[-1]['value']
        if not traj:
            try:
                convs = item.get("conversations")
                if convs and isinstance(convs, list) and len(convs) > 0:
                    response = convs[-1]
                    if isinstance(response, dict):
                        response = response.get("value", "")
                    if isinstance(response, str) and response:
                        traj = extract_trajectory(response)
            except Exception:
                traj = None

        # Strategy 3: extract from model_output_raw
        if not traj:
            raw = item.get("model_output_raw", "")
            if raw and isinstance(raw, str):
                try:
                    traj = extract_trajectory(raw)
                except Exception:
                    traj = None

        # Strategy 4: extract from any string field that might contain trajectory
        if not traj:
            for key in ("output", "response", "prediction", "text"):
                raw = item.get(key, "")
                if raw and isinstance(raw, str):
                    try:
                        traj = extract_trajectory(raw)
                        if traj:
                            break
                    except Exception:
                        continue

        # If no trajectory found, fall back to zero trajectory
        if not traj:
            pred_map[sample_id] = np.zeros((20, 2), dtype=float)
            n_zero += 1
            continue

        # Take first 5 waypoints, pad if fewer than 5
        traj5 = traj[:5]
        if len(traj5) < 1:
            pred_map[sample_id] = np.zeros((20, 2), dtype=float)
            n_zero += 1
            continue

        while len(traj5) < 5:
            traj5.append(traj5[-1])

        # Interpolate to 4Hz
        try:
            traj_4hz = jmt_interpolate_xy_with_start((0.0, 0.0), traj5, t_4hz)
            pred_map[sample_id] = traj_4hz
            n_ok += 1
        except Exception as e:
            print(f"Warning: JMT interpolation failed for {sample_id}: {e}")
            pred_map[sample_id] = np.zeros((20, 2), dtype=float)
            n_zero += 1

    print(f"Loaded {len(pred_map)} predictions (parsed: {n_ok}, zero-fallback: {n_zero}, skipped: {n_skip})")
    return pred_map


def average_distance_per_step(predictions, observed_traj, mask, time) -> np.ndarray:
    """
    Compute Average Displacement Error (ADE) per proposal.

    Args:
        predictions: [P, T, 2] or [T, 2]
        observed_traj: [T, 2]
        mask: [T] bool
        time: Number of timesteps to evaluate

    Returns:
        [P] ADE for each proposal
    """
    if predictions.ndim == 2:
        predictions = predictions[None]  # -> [1, T, 2]

    observed_traj = observed_traj[None]
    mask = mask[None]

    predictions = predictions[:, :time, :]
    observed_traj = observed_traj[:, :time, :]
    mask = mask[:, :time]

    dist_per_step = np.linalg.norm(predictions - observed_traj, axis=-1)  # [P, T]
    dist_per_traj = (dist_per_step * mask[None]).sum(axis=-1)  # [P]
    valid_steps = max(int(mask.sum()), 1)
    return dist_per_traj / float(valid_steps)


def main():
    ap = argparse.ArgumentParser(
        description="Waymo E2E ADE + RFS evaluation (consumes predictions.json from batch_inference.py)."
    )
    ap.add_argument("--pred_json", required=True,
                    help="Prediction JSON written by batch_inference.py.")
    ap.add_argument("--gt", required=True,
                    help="Ground truth source. Either a TFRecord glob "
                         "(e.g. '/path/to/val*.tfrecord*') or a pre-computed .pkl produced by a previous run.")
    ap.add_argument("--output_dir", default="./eval_results",
                    help="Directory to write waymo_eval_results.json + waymo_eval_detailed.json.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.gt.endswith(".pkl") and os.path.exists(args.gt):
        print(f"Loading GT dict from {args.gt}")
        with open(args.gt, "rb") as f:
            gt_dict = pickle.load(f)
        print(f"Loaded {len(gt_dict)} GT frames")
    else:
        gt_dict = load_waymo_e2e_data(args.gt)

    # Load predictions
    prediction_dict = load_predictions(args.pred_json)

    # Prepare data structures
    gt_traj_dict = {}
    ade_3s_list = []
    ade_5s_list = []

    rater_specified_trajectories = []
    rater_scores = []
    initial_speed = []
    prediction_trajectories = []
    prediction_probabilities = []
    frame_name_list = []
    gt_trajectories = []

    print("\nComputing metrics...")
    for frame_name in tqdm(gt_dict, desc="Processing frames"):
        if frame_name not in prediction_dict:
            print(f'Warning: No prediction for {frame_name}, skipping...')
            continue

        data = gt_dict[frame_name]
        gt_traj = np.stack([data.future_states.pos_x, data.future_states.pos_y], axis=1)
        pred_traj = prediction_dict[frame_name]
        mask = np.ones(gt_traj.shape[0], dtype=np.bool_)
        gt_traj_dict[frame_name] = gt_traj

        # Compute ADE at 3s and 5s
        ade_3s = average_distance_per_step(pred_traj[None], gt_traj, mask, 12)[0]
        ade_5s = average_distance_per_step(pred_traj[None], gt_traj, mask, 20)[0]
        ade_3s_list.append(ade_3s)
        ade_5s_list.append(ade_5s)

        # Prepare RFS data
        rater_specified_trajs_and_scores_i = data.preference_trajectories
        current_rater_trajs = []
        current_rater_scores = []
        for j in range(len(rater_specified_trajs_and_scores_i)):
            current_rater_trajs.append(
                np.stack(
                    [
                        rater_specified_trajs_and_scores_i[j].pos_x,
                        rater_specified_trajs_and_scores_i[j].pos_y,
                    ],
                    axis=-1,
                )
            )
            current_rater_scores.append(rater_specified_trajs_and_scores_i[j].preference_score)
        current_rater_scores = np.array(current_rater_scores)

        # Initial speed calculation
        vel_x = data.past_states.vel_x[-1]
        vel_y = data.past_states.vel_y[-1]
        initial_speed.append(np.sqrt(vel_x ** 2 + vel_y ** 2))

        # Add prediction (single proposal)
        prediction_traj = prediction_dict[frame_name]
        prediction_trajectories.append(prediction_traj[None])
        prediction_probabilities.append(np.ones(1))

        rater_specified_trajectories.append(current_rater_trajs)
        rater_scores.append(current_rater_scores)
        frame_name_list.append(frame_name)
        gt_trajectories.append(gt_traj_dict[frame_name])

    if not ade_3s_list:
        print("Error: No valid predictions found!")
        return

    # Convert to numpy arrays for RFS computation
    initial_speed = np.stack(initial_speed)
    prediction_trajectories = np.stack(prediction_trajectories)
    prediction_probabilities = np.stack(prediction_probabilities)

    print("\nComputing Rater Feedback Score...")
    rater_feedback_metrics = get_rater_feedback_score(
        prediction_trajectories,
        prediction_probabilities,
        rater_specified_trajectories,
        rater_scores,
        initial_speed,
        frequency=4,
        length_seconds=5,
        output_trust_region_visualization=False,
    )
    rfs_score = rater_feedback_metrics['rater_feedback_score']
    rfs_score = rfs_score.tolist()

    # Compute aggregate metrics
    results = {
        "ADE_3s": float(np.mean(ade_3s_list)),
        "ADE_5s": float(np.mean(ade_5s_list)),
        "RFS": float(np.mean(rfs_score)),
        "num_samples": len(ade_3s_list),
        "ADE_3s_std": float(np.std(ade_3s_list)),
        "ADE_5s_std": float(np.std(ade_5s_list)),
        "RFS_std": float(np.std(rfs_score)),
    }

    # Print results
    print("\n" + "="*60)
    print("WAYMO E2E EVALUATION RESULTS")
    print("="*60)
    print(f"Number of samples: {results['num_samples']}")
    print(f"ADE 3s: {results['ADE_3s']:.4f} ± {results['ADE_3s_std']:.4f}")
    print(f"ADE 5s: {results['ADE_5s']:.4f} ± {results['ADE_5s_std']:.4f}")
    print(f"RFS:    {results['RFS']:.4f} ± {results['RFS_std']:.4f}")
    print("="*60)

    # Save results to JSON
    results_file = os.path.join(args.output_dir, "waymo_eval_results.json")
    print(f"\nSaving results to: {results_file}")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Save detailed per-sample results
    detailed_results = []
    for i, (name, ade3, ade5, rfs) in enumerate(zip(frame_name_list, ade_3s_list, ade_5s_list, rfs_score)):
        detailed_results.append({
            "frame_name": name,
            "ade_3s": float(ade3),
            "ade_5s": float(ade5),
            "rfs": float(rfs)
        })

    detailed_file = os.path.join(args.output_dir, "waymo_eval_detailed.json")
    print(f"Saving detailed results to: {detailed_file}")
    with open(detailed_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
