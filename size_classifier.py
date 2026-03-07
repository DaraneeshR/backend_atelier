"""
size_classifier.py
────────────────────────────────────────────────────────
Maps garment chest measurement (INCHES) to a size label (XS–XXL)
using gender-specific thresholds.

Falls back to hardcoded thresholds if the dataset is unavailable.

Fix notes (2026-03-05):
  - XS added to SIZE_LABELS so the full dataset size range is checked.
  - XS added to _FALLBACK_THRESHOLDS so the fallback path covers XS.
  - _load_thresholds() now applies a scale-sanity guard: if the dataset
    S lower bound is below 30 in (indicating the CSV 'chest' column is
    in a different scale — e.g. half-chest ~22-32 in — rather than the
    full circumference ~36-55 in that the pipeline produces), the method
    silently falls back to the hardcoded thresholds which are correctly
    defined in full-circumference inches.
"""

from __future__ import annotations

SIZE_LABELS = ["XS", "S", "M", "L", "XL", "XXL"]

# Hardcoded fallback thresholds (INCHES, garment chest circumference)
# Based on size chart: S(32-34), M(36-38), L(40-42), XL(44-46), XXL(48-50)
_FALLBACK_THRESHOLDS: dict[str, dict[str, tuple[float, float]]] = {
    "Male": {
        "XS":  (0.0,  36.0),   # up to 36 in
        "S":   (36.0, 40.0),   # 36–40 in
        "M":   (40.0, 44.0),   # 40–44 in
        "L":   (44.0, 48.0),   # 44–48 in
        "XL":  (48.0, 52.0),   # 48–52 in
        "XXL": (52.0, 999.0),  # 52+ in
    },
    "Female": {
        "XS":  (0.0,  34.0),
        "S":   (34.0, 38.0),
        "M":   (38.0, 42.0),
        "L":   (42.0, 46.0),
        "XL":  (46.0, 50.0),
        "XXL": (50.0, 999.0),
    },
}


class SizeClassifier:
    """
    Classifies a garment chest measurement into a size label.
    Thresholds are loaded from the real dataset on first use.
    """

    def __init__(self):
        self._thresholds: dict | None = None

    def _load_thresholds(self) -> dict:
        if self._thresholds is not None:
            return self._thresholds
        try:
            from dataset_analyzer import get_analysis
            dataset_thr = get_analysis()["thresholds"]

            # Scale-sanity guard:
            # The SR Apparel Shop CSV 'chest' column uses half-chest values
            # (~22–32 in range), while the pipeline's garment_chest is a
            # full circumference (~36–55 in).  When the dataset S lower
            # bound is below 30 in, the scales are incompatible and using
            # dataset thresholds would always produce XXL.  In that case
            # we discard the dataset values and use the hardcoded fallback
            # which is correctly expressed as full-circumference inches.
            male_thr = dataset_thr.get("Male", {})
            s_lower  = male_thr.get("S", (0.0, 999.0))[0]
            if s_lower < 30.0:
                # Dataset scale is incompatible with pipeline output unit
                self._thresholds = _FALLBACK_THRESHOLDS
            else:
                self._thresholds = dataset_thr
        except Exception:
            self._thresholds = _FALLBACK_THRESHOLDS
        return self._thresholds

    def classify(self, garment_chest_in: float, gender: str) -> tuple[str, str]:
        """
        Classify garment chest measurement (inches) into size label.

        Parameters
        ----------
        garment_chest_in : float  — final garment chest in INCHES (after ease + fit)
        gender           : str    — 'Male' or 'Female'

        Returns
        -------
        (size_label, size_range_str)
        e.g. ('M', '35.5 – 38.5 in')
        """
        gender = gender.strip().capitalize()
        thresholds = self._load_thresholds()
        gender_thr = thresholds.get(gender, _FALLBACK_THRESHOLDS.get(gender, {}))

        for label in SIZE_LABELS:
            lo, hi = gender_thr.get(label, (0, 999))
            if lo <= garment_chest_in < hi:
                return label, f"{lo:.1f} – {hi:.1f} in"

        # If above XXL upper bound
        last = SIZE_LABELS[-1]
        lo, hi = gender_thr.get(last, (48.0, 999))
        return last, f">= {lo:.1f} in"

    def get_all_thresholds(self, gender: str) -> dict:
        """Return full threshold table for display in analytics."""
        gender = gender.strip().capitalize()
        return self._load_thresholds().get(gender, _FALLBACK_THRESHOLDS.get(gender, {}))


# ── CLI check ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sc = SizeClassifier()
    print("Size Classifier: testing with inch thresholds")
    for gender in ("Male", "Female"):
        print(f"\n{gender}:")
        for chest_val in [30, 34, 37, 40, 43, 46]:
            label, rng = sc.classify(chest_val, gender)
            print(f"  chest={chest_val} in -> {label}  ({rng})")
