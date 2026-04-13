from kortex.calibrate import AffectBaseline, calibrate_affect, update_baseline
from kortex.models import AffectSignal


class TestAffectBaseline:
    def test_update_baseline_uses_ema(self):
        baseline = AffectBaseline(baseline_frustration=0.2, ema_alpha=0.5)
        affect = AffectSignal(frustration=0.6)
        updated = update_baseline(baseline, affect)
        assert updated.baseline_frustration == 0.4
        assert updated.sample_count == 1

    def test_calibration_returns_raw_before_min_samples(self):
        baseline = AffectBaseline(sample_count=5)
        affect = AffectSignal(frustration=0.6, dominant_emotion="frustration")
        calibrated = calibrate_affect(affect, baseline, minimum_samples=20)
        assert calibrated.frustration == 0.6
        assert calibrated.dominant_emotion == "frustration"

    def test_calibration_subtracts_baseline_after_threshold(self):
        baseline = AffectBaseline(
            baseline_frustration=0.4,
            baseline_warmth=0.1,
            sample_count=20,
        )
        affect = AffectSignal(
            frustration=0.7,
            warmth=0.2,
            dominant_emotion="frustration",
        )
        calibrated = calibrate_affect(affect, baseline, minimum_samples=20)
        assert calibrated.frustration == 0.3
        assert calibrated.warmth == 0.1
