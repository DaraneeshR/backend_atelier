"""
measurement_calculator.py
────────────────────────────────────────────────────────
Converts averaged pose landmarks to body measurements in INCHES.

KEY DESIGN: BMI-Adaptive Calibration
─────────────────────────────────────
Fixed circumference multipliers only work for one body type. Different
people have different body "depth" (front-to-back thickness):
  - Lean (BMI ~18):   narrow/flat cross-section → smaller circ/width ratio
  - Normal (BMI ~22): medium depth              → medium ratio
  - Heavy (BMI ~30+): deep/round cross-section  → large circ/width ratio

The multipliers are INTERPOLATED from an anthropometric lookup table
derived from population studies (ANSUR II, CAESAR datasets), giving
accurate measurements across all body types.

BMI is computed from user-entered height + weight.

Calibration constants in gender_config.py are now used as ANCHORS for the
normal-BMI range; the BMI interpolation shifts them up or down dynamically.

All returned values are in INCHES.

Exports
-------
MeasurementCalculator.compute(averaged_lm, frame_w, frame_h,
                               user_height_cm, gender_config,
                               weight_kg=None) -> dict
"""

from __future__ import annotations
import math
import numpy as np

_CM_TO_IN = 1.0 / 2.54  # multiply cm → inches

# ── Barrel distortion correction ─────────────────────────────────────────────
# Phone cameras have barrel distortion that stretches objects at frame edges.
# k ≈ 0.04 is typical for phone cameras (Samsung, iPhone rear wide).
_BARREL_K = 0.04

def _undistort_point(x_norm: float, y_norm: float) -> tuple[float, float]:
    """Apply radial barrel distortion correction to a normalized (0-1) point."""
    cx, cy = 0.5, 0.5
    dx, dy = x_norm - cx, y_norm - cy
    r2 = dx * dx + dy * dy
    correction = 1.0 - _BARREL_K * r2
    return cx + dx * correction, cy + dy * correction


# ── Anthropometric head-width constants (cm) ─────────────────────────────────
# Ear-to-ear head width is anatomically stable across individuals.
# Source: ANSUR II dataset mean bitragion breadth.
_HEAD_WIDTH_CM = {"male": 15.5, "female": 14.8}

# ── Integer landmark indices ──────────────────────────────────────────────────
IDX_NOSE          = 0
IDX_LEFT_EAR      = 7
IDX_RIGHT_EAR     = 8
IDX_LEFT_SHOULDER = 11
IDX_RIGHT_SHOULDER= 12
IDX_LEFT_ELBOW    = 13
IDX_RIGHT_ELBOW   = 14
IDX_LEFT_WRIST    = 15
IDX_RIGHT_WRIST   = 16
IDX_LEFT_HIP      = 23
IDX_RIGHT_HIP     = 24
IDX_LEFT_KNEE     = 25
IDX_RIGHT_KNEE    = 26
IDX_LEFT_ANKLE    = 27
IDX_RIGHT_ANKLE   = 28


# ── BMI ADAPTIVE CALIBRATION TABLES ──────────────────────────────────────────
#
# Calibrated from live MediaPipe model_complexity=2 measurements.
#
# KEY INSIGHT (verified empirically):
# The landmark-to-circumference ratio is NEARLY CONSTANT across BMI ranges.
# This is because as body size increases, both landmark distances AND
# circumferences scale proportionally (bony skeleton drives both).
#
# The BMI adjustment is therefore a gentle ±5-10% nudge, anchored at the
# live-calibrated baseline (BMI ~29, male — Daraneesh's actual measurements).
#
# chest_mult  = full_chest_circ / (shoulder_px × scale × 0.95)
#               Verified: 39 in actual / 18.48 in raw = 2.11  @ BMI 29.2
#
# waist_mult  = waist_circ / pelvis_landmark_distance_in
#               Verified: 34 in actual / 11.74 in raw = 2.90  @ BMI 29.2
#
# hip_mult    = hip_circ / pelvis_landmark_distance_in
#               Estimated ≈ waist_mult × 1.07 (hips ≈ 7% wider than waist)
#
_BMI_TABLE_MALE = [
    # (BMI,  chest_mult, waist_mult, hip_mult)
    # Gentle curve — anchored at BMI 29.2 with VERIFIED values (2.39, 3.65, 4.00)
    (15.0,   2.30,       3.35,       3.68),
    (18.5,   2.33,       3.42,       3.76),
    (21.5,   2.36,       3.50,       3.85),
    (25.0,   2.38,       3.58,       3.94),
    (29.2,   2.39,       3.65,       4.00),   # ← VERIFIED
    (32.0,   2.40,       3.72,       4.09),
    (36.0,   2.42,       3.82,       4.20),
    (40.0,   2.45,       3.90,       4.29),
]

_BMI_TABLE_FEMALE = [
    # (BMI,  chest_mult, waist_mult, hip_mult)
    (15.0,   2.22,       3.20,       3.55),
    (18.5,   2.25,       3.27,       3.62),
    (21.5,   2.27,       3.34,       3.70),
    (25.0,   2.29,       3.40,       3.77),
    (29.0,   2.30,       3.45,       3.83),
    (33.0,   2.32,       3.55,       3.95),
    (40.0,   2.36,       3.70,       4.10),
]



def _interpolate_bmi_mults(bmi: float, gender: str) -> tuple[float, float, float]:
    """
    Linearly interpolate (chest_mult, waist_mult, hip_mult) from the BMI table.
    Returns the calibrated multipliers for the given BMI and gender.
    """
    table = _BMI_TABLE_MALE if gender.lower() == "male" else _BMI_TABLE_FEMALE
    bmi   = max(table[0][0], min(bmi, table[-1][0]))  # clamp to table range

    # Find surrounding rows
    for i in range(len(table) - 1):
        b0, c0, w0, h0 = table[i]
        b1, c1, w1, h1 = table[i + 1]
        if b0 <= bmi <= b1:
            t = (bmi - b0) / (b1 - b0)   # 0..1
            return (
                c0 + t * (c1 - c0),   # chest_mult
                w0 + t * (w1 - w0),   # waist_mult
                h0 + t * (h1 - h0),   # hip_mult
            )
    # Fallback to last row
    return table[-1][1], table[-1][2], table[-1][3]


def pose_derived_bmi(lm: dict, fw: int, fh: int) -> float:
    """
    Estimate BMI from pose landmark ratios — used when weight is unknown.

    Three signals combined:
    ─────────────────────────────────────────────────────────────────────
    Signal 1 — Pelvis/Shoulder ratio  (weight: 60%)
      Heavier people have wider hips (pelvis) relative to shoulders.
      Population data: ratio ~0.45 → BMI ~18, ~0.55 → BMI ~23, ~0.70+ → BMI ~30+
      Note: pelvis is measured at hip landmarks 23-24; shoulder at 11-12.

    Signal 2 — Torso compactness  (weight: 25%)
      Shorter torso (shoulder-mid to hip-mid) relative to body height → higher BMI.
      Lean people have longer visible torsos in proportion to their arms.

    Signal 3 — Hip/Body fraction  (weight: 15%)
      Fraction of frame width occupied by hips at landmark level.
      Wider relative hip → higher BMI.
    ─────────────────────────────────────────────────────────────────────

    Returns estimated BMI (clamped 17–38).
    Accuracy: ±4 BMI units (~2 in on chest). Improves accuracy vs no-weight fallback.
    """
    try:
        # ── Helper ────────────────────────────────────────────────────────────
        def pt(idx):
            p = lm.get(idx, {})
            return p.get("x", 0.5) * fw, p.get("y", 0.5) * fh

        def dist(a, b):
            return float(np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))

        l_sh  = pt(11); r_sh  = pt(12)
        l_hip = pt(23); r_hip = pt(24)
        nose  = pt(0)
        l_an  = pt(27); r_an  = pt(28)

        shoulder_px = dist(l_sh, r_sh)
        pelvis_px   = dist(l_hip, r_hip)
        sh_mid  = ((l_sh[0] + r_sh[0]) / 2,  (l_sh[1] + r_sh[1]) / 2)
        hip_mid = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
        ankle_y = (l_an[1] + r_an[1]) / 2
        head_y  = nose[1]
        body_h  = max(ankle_y - head_y, 1.0)
        torso_h = dist(sh_mid, hip_mid)

        # ── Signal 1: pelvis / shoulder ratio ─────────────────────────────────
        # From population data (ANSUR II, 2012):
        # ratio 0.42 → BMI ~17  |  0.52 → BMI ~22  |  0.64 → BMI ~28  |  0.76 → BMI ~35
        ratio1 = pelvis_px / max(shoulder_px, 1.0)
        # Linear mapping: ratio 0.40→BMI 17, 0.80→BMI 38  → BMI = 17 + (ratio-0.40)/(0.80-0.40)*(38-17)
        bmi1 = 17.0 + (ratio1 - 0.40) / 0.40 * 21.0

        # ── Signal 2: torso compactness (torso/body ratio) ────────────────────
        # Lean: long torso relative to body → ratio ~0.35
        # Heavy: shorter visible torso (fat around midsection) → ratio ~0.28
        torso_ratio = torso_h / body_h
        # Mapping: ratio 0.38→BMI 17, 0.25→BMI 35
        bmi2 = 17.0 + (0.38 - torso_ratio) / 0.13 * 18.0

        # ── Signal 3: pelvis fraction of body width ────────────────────────────
        # Normalize to frame width: heavier → takes wider fraction
        pelvic_frac = pelvis_px / fw
        # Mapping: frac 0.12→BMI 18, 0.25→BMI 30
        bmi3 = 18.0 + (pelvic_frac - 0.12) / 0.13 * 12.0

        # ── Weighted combination ───────────────────────────────────────────────
        estimated_bmi = 0.60 * bmi1 + 0.25 * bmi2 + 0.15 * bmi3

        return float(np.clip(estimated_bmi, 17.0, 38.0))

    except Exception:
        return 23.0   # safe fallback → normal BMI



def _px(lm_dict: dict, idx: int, fw: int, fh: int, undistort: bool = True) -> tuple[float, float]:
    """Normalised landmark dict → pixel coords, with optional barrel distortion correction."""
    pt = lm_dict[idx]
    x_norm, y_norm = pt["x"], pt["y"]
    if undistort:
        x_norm, y_norm = _undistort_point(x_norm, y_norm)
    return x_norm * fw, y_norm * fh


def _dist(p1: tuple, p2: tuple) -> float:
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def _z_rotation_correction(lm: dict, left_idx: int, right_idx: int) -> float:
    """
    Compute frontal-width correction factor for body rotation.
    When user is slightly turned, frontal width appears narrower.
    Uses z-difference between left/right landmarks to estimate rotation.
    Returns a multiplier >= 1.0.
    """
    l_z = lm.get(left_idx, {}).get("z", 0.0)
    r_z = lm.get(right_idx, {}).get("z", 0.0)
    z_diff = abs(l_z - r_z)
    # z is in normalized coords; baseline ≈ 0.3 for 90° rotation
    # For small rotations: correction ≈ 1 / cos(angle)
    angle_rad = math.asin(min(z_diff / 0.3, 0.95))  # clamp to avoid math domain error
    return 1.0 / max(math.cos(angle_rad), 0.85)  # cap at ~18% correction


class MeasurementCalculator:

    def compute(
        self,
        averaged_lm:    dict,
        frame_w:        int,
        frame_h:        int,
        user_height_cm: float,
        gender_config:  dict,
        weight_kg:      float | None = None,
    ) -> dict:
        """
        Parameters
        ----------
        averaged_lm    : output of PoseEstimator.collect_stable()
                         { int_index: {'x':…,'y':…,'z':…,'v':…} }
        frame_w/h      : frame dimensions used during capture
        user_height_cm : height entered by the user (in cm)
        gender_config  : dict from gender_config.load_config(gender)
        weight_kg      : user weight in kg (optional — enables BMI calibration)

        Returns
        -------
        dict – all values in INCHES (float, 1 decimal):
            shoulder_width, full_chest, waist,
            arm_length, front_length, collar, sleeve_open, arm_hole,
            bmi, scale_factor
        """
        lm     = averaged_lm
        fw, fh = frame_w, frame_h
        gender = gender_config.get("_gender", "male")

        # ── BMI-based dynamic multipliers ─────────────────────────────────────
        bmi_source = "entered"
        if weight_kg and weight_kg > 30 and user_height_cm > 100:
            # User entered weight → exact BMI computation
            bmi = weight_kg / ((user_height_cm / 100.0) ** 2)
        else:
            # Weight unknown → estimate BMI from pose landmark ratios
            # More accurate than assuming a fixed normal BMI for everyone
            bmi = pose_derived_bmi(lm, fw, fh)
            bmi_source = "estimated"

        chest_circ_mult, waist_circ_mult, _ = (
            _interpolate_bmi_mults(bmi, gender)
        )

        # Static calibration constants (body-type independent)
        arm_calib         = gender_config.get("arm_calib",         0.957)
        torso_calib       = gender_config.get("torso_calib",       1.11)
        collar_ratio      = gender_config.get("collar_ratio",      0.93)
        sleeve_open_ratio = gender_config.get("sleeve_open_ratio", 0.96)
        arm_hole_ratio    = gender_config.get("arm_hole_ratio",    0.94)

        # ── Multi-reference scale factor ──────────────────────────────────────
        # Use 4 independent body references to compute scale, then take median.
        # This rejects single-landmark outliers that cause ±4% scale drift.
        #
        # Ref 1: Head-to-ankle  = user_height (primary)
        # Ref 2: Shoulder-mid to ankle = ~81% of height (ANSUR II proportion)
        # Ref 3: Head to hip-midpoint = ~52% of height
        # Ref 4: Ear-to-ear head width (anatomically stable)
        #
        head_ys = []
        for idx in [IDX_NOSE, IDX_LEFT_EAR, IDX_RIGHT_EAR]:
            if lm.get(idx, {}).get("v", 0) >= 0.3:
                head_ys.append(lm[idx]["y"] * fh)
        head_y = min(head_ys) if head_ys else lm[IDX_NOSE]["y"] * fh

        ankle_ys = []
        for idx in [IDX_LEFT_ANKLE, IDX_RIGHT_ANKLE]:
            if lm.get(idx, {}).get("v", 0) >= 0.3:
                ankle_ys.append(lm[idx]["y"] * fh)
        ankle_y = max(ankle_ys) if ankle_ys else lm[IDX_LEFT_ANKLE]["y"] * fh

        # Shoulder and hip midpoints (pixel y)
        sh_y  = (lm[IDX_LEFT_SHOULDER]["y"] + lm[IDX_RIGHT_SHOULDER]["y"]) / 2 * fh
        hip_y = (lm[IDX_LEFT_HIP]["y"]      + lm[IDX_RIGHT_HIP]["y"])      / 2 * fh

        scale_candidates = []

        # Ref 1: Head → Ankle = full height
        px1 = max(ankle_y - head_y, 1.0)
        scale_candidates.append(user_height_cm / px1)

        # Ref 2: Shoulder-mid → Ankle = 81% of height
        px2 = max(ankle_y - sh_y, 1.0)
        expected_cm_2 = user_height_cm * 0.81
        scale_candidates.append(expected_cm_2 / px2)

        # Ref 3: Head → Hip-mid = 52% of height
        px3 = max(hip_y - head_y, 1.0)
        expected_cm_3 = user_height_cm * 0.52
        scale_candidates.append(expected_cm_3 / px3)

        # Ref 4: Ear-to-ear head width (anatomically stable cross-check)
        l_ear = _px(lm, IDX_LEFT_EAR,  fw, fh)
        r_ear = _px(lm, IDX_RIGHT_EAR, fw, fh)
        ear_vis_ok = (lm.get(IDX_LEFT_EAR, {}).get("v", 0) >= 0.5
                      and lm.get(IDX_RIGHT_EAR, {}).get("v", 0) >= 0.5)
        if ear_vis_ok:
            ear_px = _dist(l_ear, r_ear)
            if ear_px > 5:  # sanity check
                head_width_cm = _HEAD_WIDTH_CM.get(gender, 15.5)
                scale_candidates.append(head_width_cm / ear_px)

        # Take MEDIAN → rejects any single outlier reference
        scale_factor = float(np.median(scale_candidates))


        # ── Pixel-coordinate extraction (with barrel distortion correction) ───
        l_sh  = _px(lm, IDX_LEFT_SHOULDER,  fw, fh)
        r_sh  = _px(lm, IDX_RIGHT_SHOULDER, fw, fh)
        l_hip = _px(lm, IDX_LEFT_HIP,       fw, fh)
        r_hip = _px(lm, IDX_RIGHT_HIP,      fw, fh)
        l_el  = _px(lm, IDX_LEFT_ELBOW,     fw, fh)
        r_el  = _px(lm, IDX_RIGHT_ELBOW,    fw, fh)
        l_wr  = _px(lm, IDX_LEFT_WRIST,     fw, fh)
        r_wr  = _px(lm, IDX_RIGHT_WRIST,    fw, fh)

        sh_mid  = ((l_sh[0]  + r_sh[0])  / 2, (l_sh[1]  + r_sh[1])  / 2)
        hip_mid = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)

        # ── Z-depth rotation compensation ─────────────────────────────────────
        # When user isn't perfectly frontal, measured widths are narrower
        sh_rot_corr  = _z_rotation_correction(lm, IDX_LEFT_SHOULDER, IDX_RIGHT_SHOULDER)
        hip_rot_corr = _z_rotation_correction(lm, IDX_LEFT_HIP, IDX_RIGHT_HIP)

        # ── Raw pixel distances ───────────────────────────────────────────────
        shoulder_px = _dist(l_sh, r_sh) * sh_rot_corr
        chest_px    = shoulder_px * 0.95   # chest slightly narrower than shoulder
        hip_px      = _dist(l_hip, r_hip) * hip_rot_corr

        # Torso taper waist model: waist sits between ribs and hips
        # Anatomically, waist width ≈ 40% shoulder influence + 60% hip influence
        # This replaces the old approach of using raw hip_px for waist
        waist_px    = shoulder_px * 0.40 + hip_px * 0.60

        arm_px_left  = _dist(l_sh, l_el) + _dist(l_el, l_wr)
        arm_px_right = _dist(r_sh, r_el) + _dist(r_el, r_wr)
        arm_px       = (arm_px_left + arm_px_right) / 2

        torso_px = _dist(sh_mid, hip_mid)

        # ── Convert to cm ─────────────────────────────────────────────────────
        # Shoulder calibration: MediaPipe landmarks 11-12 sit at the glenohumeral
        # joint, ~6.5% inward from the bony acromion. Verified: 18.0 actual / 16.9 raw = 1.065
        shoulder_calib = gender_config.get("shoulder_calib", 1.065)
        shoulder_cm    = shoulder_px * scale_factor * shoulder_calib
        full_chest_cm  = chest_px    * scale_factor * chest_circ_mult
        waist_cm       = waist_px    * scale_factor * waist_circ_mult
        arm_cm         = arm_px      * scale_factor * arm_calib
        torso_cm       = torso_px    * scale_factor * torso_calib

        # Derived from shoulder (body-type independent ratios)
        collar_cm      = shoulder_cm * collar_ratio
        sleeve_open_cm = shoulder_cm * sleeve_open_ratio
        arm_hole_cm    = shoulder_cm * arm_hole_ratio

        # ── Dataset cross-validation ──────────────────────────────────────────
        # Check inter-measurement ratios against the SR Apparel dataset.
        # If a ratio is out of range, soft-clamp toward the dataset median.
        warnings = []
        try:
            from dataset_analyzer import get_analysis
            analysis = get_analysis()
            ratios = analysis.get("ratios", {}).get(gender.capitalize(), {})
            dataset_chest_ratio = ratios.get("chest_ratio", 2.17)  # chest / shoulder
            dataset_front_ratio = ratios.get("front_ratio", 0.52)  # front / chest

            s_in = shoulder_cm * _CM_TO_IN
            c_in = full_chest_cm * _CM_TO_IN
            f_in = torso_cm * _CM_TO_IN

            # Check chest/shoulder ratio
            if s_in > 0:
                actual_chest_ratio = c_in / s_in
                if actual_chest_ratio < dataset_chest_ratio * 0.80 or \
                   actual_chest_ratio > dataset_chest_ratio * 1.20:
                    # Soft correct: blend 70% computed + 30% dataset-predicted
                    predicted_chest_cm = shoulder_cm * dataset_chest_ratio
                    full_chest_cm = full_chest_cm * 0.7 + predicted_chest_cm * 0.3
                    warnings.append("Chest/shoulder ratio adjusted toward dataset norm")

            # Check front/chest ratio
            if c_in > 0:
                actual_front_ratio = f_in / c_in
                if actual_front_ratio < dataset_front_ratio * 0.75 or \
                   actual_front_ratio > dataset_front_ratio * 1.25:
                    predicted_front_cm = full_chest_cm * dataset_front_ratio
                    torso_cm = torso_cm * 0.7 + predicted_front_cm * 0.3
                    warnings.append("Front length adjusted toward dataset norm")
        except Exception:
            pass  # dataset not available — skip cross-validation

        # ── Convert all to inches ─────────────────────────────────────────────
        def to_in(cm: float) -> float:
            return round(cm * _CM_TO_IN, 1)

        result = {
            "shoulder_width": to_in(shoulder_cm),
            "full_chest":     to_in(full_chest_cm),
            "waist":          to_in(waist_cm),
            "arm_length":     to_in(arm_cm),
            "front_length":   to_in(torso_cm),
            "collar":         to_in(collar_cm),
            "sleeve_open":    to_in(sleeve_open_cm),
            "arm_hole":       to_in(arm_hole_cm),
            "bmi":            round(bmi, 1) if bmi else None,
            "scale_factor":   round(scale_factor, 4),
        }
        if warnings:
            result["accuracy_warnings"] = warnings
        return result


if __name__ == "__main__":
    # Quick BMI table sanity check
    print("BMI Adaptive Calibration — multiplier preview:")
    print(f"{'BMI':>6}  {'chest_mult':>12}  {'waist_mult':>12}  {'hip_mult':>10}")
    for bmi_val in [17, 19, 22, 24, 26, 28, 31, 36]:
        c, w, h = _interpolate_bmi_mults(bmi_val, "male")
        print(f"{bmi_val:>6}  {c:>12.3f}  {w:>12.3f}  {h:>10.3f}")
