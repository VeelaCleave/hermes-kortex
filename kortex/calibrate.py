"""Per-user affect calibration helpers for KORTEX."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import AffectSignal
from .time_utils import now_epoch, _ema


@dataclass
class AffectBaseline:
    user_id: str = "__default__"
    baseline_frustration: float = 0.0
    baseline_warmth: float = 0.0
    baseline_humor: float = 0.0
    baseline_hostility: float = 0.0
    baseline_gratitude: float = 0.0
    baseline_anxiety: float = 0.0
    baseline_excitement: float = 0.0
    baseline_trust_signal: float = 0.0
    sample_count: int = 0
    ema_alpha: float = 0.1
    created_at: float = field(default_factory=now_epoch)
    updated_at: float = field(default_factory=now_epoch)


def update_baseline(baseline: AffectBaseline, affect: AffectSignal) -> AffectBaseline:
    alpha = baseline.ema_alpha
    return AffectBaseline(
        user_id=baseline.user_id,
        baseline_frustration=_ema(
            baseline.baseline_frustration, affect.frustration, alpha
        ),
        baseline_warmth=_ema(baseline.baseline_warmth, affect.warmth, alpha),
        baseline_humor=_ema(baseline.baseline_humor, affect.humor, alpha),
        baseline_hostility=_ema(baseline.baseline_hostility, affect.hostility, alpha),
        baseline_gratitude=_ema(baseline.baseline_gratitude, affect.gratitude, alpha),
        baseline_anxiety=_ema(baseline.baseline_anxiety, affect.anxiety, alpha),
        baseline_excitement=_ema(
            baseline.baseline_excitement, affect.excitement, alpha
        ),
        baseline_trust_signal=_ema(
            baseline.baseline_trust_signal, affect.trust_signal, alpha
        ),
        sample_count=baseline.sample_count + 1,
        ema_alpha=alpha,
        created_at=baseline.created_at,
        updated_at=now_epoch(),
    )


def calibrate_affect(
    affect: AffectSignal,
    baseline: AffectBaseline,
    minimum_samples: int = 20,
) -> AffectSignal:
    if baseline.sample_count < minimum_samples:
        return affect

    frustration = max(0.0, affect.frustration - baseline.baseline_frustration)
    warmth = max(0.0, affect.warmth - baseline.baseline_warmth)
    humor = max(0.0, affect.humor - baseline.baseline_humor)
    hostility = max(0.0, affect.hostility - baseline.baseline_hostility)
    gratitude = max(0.0, affect.gratitude - baseline.baseline_gratitude)
    anxiety = max(0.0, affect.anxiety - baseline.baseline_anxiety)
    excitement = max(0.0, affect.excitement - baseline.baseline_excitement)
    trust_signal = max(0.0, affect.trust_signal - baseline.baseline_trust_signal)
    valence = (warmth + gratitude + excitement * 0.7) - (
        frustration + hostility + anxiety * 0.5
    )
    arousal = max(
        frustration,
        hostility,
        excitement,
        anxiety,
        warmth * 0.6,
        gratitude * 0.5,
        humor * 0.3,
    )
    dimensions = {
        "frustration": frustration,
        "warmth": warmth,
        "humor": humor,
        "hostility": hostility,
        "gratitude": gratitude,
        "anxiety": anxiety,
        "excitement": excitement,
        "trust": trust_signal,
    }
    dominant = (
        max(dimensions, key=dimensions.get) if any(dimensions.values()) else "neutral"
    )

    return AffectSignal(
        frustration=round(frustration, 3),
        warmth=round(warmth, 3),
        humor=round(humor, 3),
        hostility=round(hostility, 3),
        gratitude=round(gratitude, 3),
        anxiety=round(anxiety, 3),
        excitement=round(excitement, 3),
        trust_signal=round(trust_signal, 3),
        valence=round(valence, 3),
        arousal=round(min(arousal, 1.0), 3),
        dominant_emotion=dominant,
        is_sarcastic=affect.is_sarcastic,
    )


