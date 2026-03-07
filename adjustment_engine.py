"""
adjustment_engine.py
────────────────────────────────────────────────────────
Applies garment ease and fit modifiers to raw body measurements
to produce final garment dimensions. All values are in INCHES.

Formula
-------
  garment_measurement = body_measurement_in + gender_ease_in + fit_modifier_in

Measurements produced
---------------------
  garment_chest, garment_waist,
  garment_sleeve, garment_front_length,
  garment_collar, garment_sleeve_open
"""

from __future__ import annotations

FIT_KEYS = {
    "Slim":    "fit_slim",
    "Regular": "fit_regular",
    "Loose":   "fit_loose",
}


class AdjustmentEngine:

    def compute(
        self,
        body_measurements: dict,
        gender_config:     dict,
        fit_preference:    str,   # 'Slim' | 'Regular' | 'Loose'
        sleeve_type:       str,   # 'Half Sleeve' | 'Full Sleeve'
    ) -> dict:
        """
        Parameters
        ----------
        body_measurements   : output of MeasurementCalculator.compute()
                              (all values in inches)
        gender_config       : output of gender_config.load_config(gender)
        fit_preference      : 'Slim', 'Regular', or 'Loose'
        sleeve_type         : 'Half Sleeve' or 'Full Sleeve'

        Returns
        -------
        dict with final garment measurements (all float, inches):
            garment_chest, garment_waist,
            garment_sleeve, garment_front_length,
            garment_collar, garment_sleeve_open,
            fit_modifier_applied
        """
        fit_key      = FIT_KEYS.get(fit_preference, "fit_regular")
        fit_modifier = gender_config.get(fit_key, 0.0)

        ease_chest = gender_config.get("ease_chest", 4.0)
        ease_waist = gender_config.get("ease_waist", 1.5)

        garment_chest = body_measurements["full_chest"] + ease_chest + fit_modifier
        garment_waist = body_measurements["waist"]      + ease_waist + fit_modifier

        # ── Sleeve length ─────────────────────────────────────────────────────
        arm_length = body_measurements.get("arm_length", 0.0)

        if "full" in sleeve_type.lower():
            ease_sv        = gender_config.get("ease_sleeve_full", 0.75)
            garment_sleeve = arm_length + ease_sv
        else:  # Half Sleeve
            ease_sv        = gender_config.get("ease_sleeve_half", 0.5)
            garment_sleeve = (arm_length * 0.5) + ease_sv

        # ── Passthrough measurements (no ease) ────────────────────────────────
        garment_front_length = body_measurements.get("front_length",   0.0)
        garment_collar       = body_measurements.get("collar",          0.0)
        garment_sleeve_open  = body_measurements.get("sleeve_open",    0.0)
        garment_arm_hole     = body_measurements.get("arm_hole",       0.0)
        garment_shoulder     = body_measurements.get("shoulder_width", 0.0)

        return {
            "garment_chest":        round(garment_chest,        1),
            "garment_waist":        round(garment_waist,        1),
            "garment_sleeve":       round(garment_sleeve,       1),
            "garment_front_length": round(garment_front_length, 1),
            "garment_collar":       round(garment_collar,       1),
            "garment_sleeve_open":  round(garment_sleeve_open,  1),
            "garment_arm_hole":     round(garment_arm_hole,     1),
            "garment_shoulder":     round(garment_shoulder,     1),
            "fit_modifier_applied": round(fit_modifier,         1),
        }


# ── CLI check ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from gender_config import load_config

    body = {
        "shoulder_width":  18.0,
        "full_chest":      39.0,
        "waist":           33.0,
        "hip":             38.0,
        "arm_length":      25.5,
        "front_length":    28.0,
        "collar":          16.0,
        "sleeve_open":     16.5,
    }

    for gender in ("Male", "Female"):
        cfg = load_config(gender)
        for fit in ("Slim", "Regular", "Loose"):
            for sleeve in ("Full Sleeve", "Half Sleeve"):
                eng = AdjustmentEngine()
                result = eng.compute(body, cfg, fit, sleeve)
                print(f"{gender} | {fit} | {sleeve:12s} | "
                      f"chest={result['garment_chest']}in  "
                      f"sleeve={result['garment_sleeve']}in")
