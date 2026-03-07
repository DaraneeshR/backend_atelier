"""
gender_config.py
────────────────────────────────────────────────────────
Provides gender-specific anthropometric constants consumed by:
  - measurement_calculator.py  (chest/waist proportional ratios + calibration)
  - adjustment_engine.py       (ease values, fit modifiers — all in INCHES)
  - size_classifier.py         (size thresholds)

All ease and fit modifier values are in INCHES.
Calibration multipliers are tuned against real measurements.
"""

from __future__ import annotations

# ── Structural fallback constants ────────────────────────────────────────────
_FALLBACK: dict[str, dict] = {
    "Male": {
        # ── Circumference / calibration multipliers ────────────────────────
        # Calibrated from Run2 live data (model_complexity=2, 25-frame avg, IQR)
        # Each value is back-calculated: actual / raw_output_before_mult
        #
        # chest = shoulder_px × 0.95 × scale × chest_circ_mult
        # actual 39.0 in / (17.2×0.95) in raw = 2.387
        "chest_circ_mult":   2.39,
        # arm = arm_chain_px × scale × arm_calib
        # actual 25.5 in / 26.65 in raw = 0.957
        "arm_calib":         0.957,
        # torso = torso_px × scale × torso_calib
        # actual 28.0 in / 25.21 in raw = 1.111
        "torso_calib":       1.11,
        # waist = pelvis_px × scale × waist_circ_mult
        # actual 34.0 in / 9.31 in raw = 3.652
        "waist_circ_mult":   3.65,

        # ── Derived-feature ratios (from shoulder_width) ─────────────────
        # collar actual 16 / shoulder 17.2 = 0.930
        "collar_ratio":      0.93,
        # sleeve_open actual 16.5 / shoulder 17.2 = 0.959
        "sleeve_open_ratio": 0.96,
        # arm_hole stays relative to shoulder
        "arm_hole_ratio":    0.94,

        # ── Industry ease (INCHES) ─────────────────────────────────────────
        "ease_chest":        4.0,
        "ease_waist":        1.5,
        "ease_sleeve_full":  0.75,
        "ease_sleeve_half":  0.5,

        # ── Fit preference modifiers (INCHES) ─────────────────────────────
        "fit_slim":         -0.75,
        "fit_regular":       0.0,
        "fit_loose":        +2.0,
    },
    "Female": {
        # Female calibration (proportionally scaled from corrected Male values)
        "chest_circ_mult":   2.30,
        "arm_calib":         0.957,
        "torso_calib":       1.11,
        "waist_circ_mult":   3.45,

        "collar_ratio":      0.83,
        "sleeve_open_ratio": 0.875,
        "arm_hole_ratio":    0.90,

        "ease_chest":        3.0,
        "ease_waist":        1.25,
        "ease_sleeve_full":  0.75,
        "ease_sleeve_half":  0.5,

        "fit_slim":         -0.75,
        "fit_regular":       0.0,
        "fit_loose":        +1.5,
    },
}


def load_config(gender: str) -> dict:
    """
    Return the full config dict for the given gender.

    Parameters
    ----------
    gender : str   'Male' or 'Female' (case-insensitive)

    Returns
    -------
    dict with keys:
        chest_circ_mult, arm_calib, torso_calib,
        waist_circ_mult,
        collar_ratio, sleeve_open_ratio,
        ease_chest, ease_waist,
        ease_sleeve_full, ease_sleeve_half,
        fit_slim, fit_regular, fit_loose
    """
    gender = gender.strip().capitalize()
    if gender not in ("Male", "Female"):
        raise ValueError(f"gender must be 'Male' or 'Female', got: {gender!r}")

    cfg = dict(_FALLBACK[gender])
    return cfg


# ── CLI check ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for g in ("Male", "Female"):
        cfg = load_config(g)
        print(f"\n{g} config:")
        for k, v in cfg.items():
            print(f"  {k:25s}: {v}")
