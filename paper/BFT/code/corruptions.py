"""Test-time signal corruptions, Section IV-F of the paper.

Fig. 5 (classification) and Fig. 6 (regression) evaluate every method on target
data that has been corrupted before it reaches the model. This module is the
corruption bank those two experiments draw from, shared by the classification
and the regression test scripts so that both apply the identical perturbation.

Seven corruptions are defined, each at three severities:

    temporal_segment_noise  additive Gaussian noise inside one random time window
    channel_noise           additive Gaussian noise on a random subset of channels
    baseline_drift          a slow sinusoid added to every channel
    band_limited_noise      Gaussian noise restricted to one EEG rhythm band
    temporal_mask           attenuation of one random time window
    channel_dropout         attenuation of a random subset of channels
    mixed_artifact          all six of the above applied in sequence

Every corruption is scaled by the per-channel standard deviation of the trial it
is applied to, so a severity level means the same thing across datasets whose
amplitude units differ. The paper reports the temporal and the spatial Gaussian
cases as "temporal noise" and "spatial noise"; they are
``temporal_segment_noise`` and ``channel_noise`` here.

The corruption is applied to the raw trials, before Euclidean Alignment, because
the point of the experiment is that the whole test-time pipeline sees degraded
input, alignment included.

Input and output are float32 arrays of shape ``[trials, channels, time]``.
Nothing here is fit on the data, so the same call reproduces exactly given the
same seed.
"""

from __future__ import annotations

import copy

import numpy as np


ARTIFACT_NAMES = [
    "temporal_segment_noise",
    "channel_noise",
    "baseline_drift",
    "band_limited_noise",
    "temporal_mask",
    "channel_dropout",
    "mixed_artifact",
]


def severity_config(severity: int) -> dict[str, float]:
    """Corruption strengths for severity 1, 2 and 3.

    ``noise_scale``, ``drift_scale`` are fractions of the per-channel standard
    deviation. ``channel_fraction`` and ``time_fraction`` are the fractions of
    channels and of time samples a corruption touches. ``dropout_scale`` is the
    factor a dropped channel or masked window is multiplied by, so it decreases
    with severity and reaches complete removal at severity 3.
    """
    if severity not in {1, 2, 3}:
        raise ValueError(f"severity must be 1, 2, or 3, got {severity}")
    return {
        "noise_scale": {1: 0.10, 2: 0.25, 3: 0.50}[severity],
        "channel_fraction": {1: 0.10, 2: 0.20, 3: 0.35}[severity],
        "time_fraction": {1: 0.10, 2: 0.20, 3: 0.35}[severity],
        "dropout_scale": {1: 0.50, 2: 0.20, 3: 0.0}[severity],
        "drift_scale": {1: 0.10, 2: 0.20, 3: 0.35}[severity],
    }


def channel_std(x: np.ndarray) -> np.ndarray:
    """Per-trial, per-channel standard deviation, floored away from zero."""
    return np.maximum(np.std(x, axis=2, keepdims=True), 1e-6)


def random_time_window(
    rng: np.random.Generator, length: int, fraction: float
) -> tuple[int, int]:
    """Half-open index range covering ``fraction`` of the trial, placed at random."""
    width = max(1, min(length, int(round(length * fraction))))
    start = int(rng.integers(0, max(1, length - width + 1)))
    return start, start + width


def temporal_segment_noise(
    x: np.ndarray, rng: np.random.Generator, severity: int, sample_rate: int
) -> np.ndarray:
    """Temporal Gaussian noise: one random window is corrupted on all channels."""
    del sample_rate
    cfg = severity_config(severity)
    out = x.copy()
    start, end = random_time_window(rng, x.shape[2], cfg["time_fraction"])
    noise = rng.standard_normal(out[:, :, start:end].shape)
    out[:, :, start:end] += noise * channel_std(x) * cfg["noise_scale"]
    return out


def channel_noise(
    x: np.ndarray, rng: np.random.Generator, severity: int, sample_rate: int
) -> np.ndarray:
    """Spatial Gaussian noise: a random subset of channels is corrupted throughout.

    The subset is redrawn per trial, which models electrodes degrading at
    different times rather than one electrode failing for the whole session.
    """
    del sample_rate
    cfg = severity_config(severity)
    out = x.copy()
    n_noisy = max(1, int(round(x.shape[1] * cfg["channel_fraction"])))
    for sample_id in range(x.shape[0]):
        channels = rng.choice(x.shape[1], size=n_noisy, replace=False)
        std = np.maximum(np.std(x[sample_id, channels], axis=1, keepdims=True), 1e-6)
        noise = rng.standard_normal(out[sample_id, channels].shape)
        out[sample_id, channels] += noise * std * cfg["noise_scale"]
    return out


def channel_dropout(
    x: np.ndarray, rng: np.random.Generator, severity: int, sample_rate: int
) -> np.ndarray:
    """Electrode failure: a random subset of channels is attenuated or removed."""
    del sample_rate
    cfg = severity_config(severity)
    out = x.copy()
    n_drop = max(1, int(round(x.shape[1] * cfg["channel_fraction"])))
    for sample_id in range(x.shape[0]):
        channels = rng.choice(x.shape[1], size=n_drop, replace=False)
        out[sample_id, channels] *= cfg["dropout_scale"]
    return out


def temporal_mask(
    x: np.ndarray, rng: np.random.Generator, severity: int, sample_rate: int
) -> np.ndarray:
    """Signal loss: one random time window is attenuated or removed."""
    del sample_rate
    cfg = severity_config(severity)
    out = x.copy()
    start, end = random_time_window(rng, x.shape[2], cfg["time_fraction"])
    out[:, :, start:end] *= cfg["dropout_scale"]
    return out


def band_limited_noise(
    x: np.ndarray, rng: np.random.Generator, severity: int, sample_rate: int
) -> np.ndarray:
    """Band-limited interference inside one randomly chosen EEG rhythm band.

    White noise is generated, its spectrum is zeroed outside theta, alpha, beta
    or low gamma, and the result is renormalized to unit standard deviation
    before scaling, so the severity means the same thing whichever band is drawn.
    """
    cfg = severity_config(severity)
    out = x.copy()
    length = x.shape[2]
    freqs = np.fft.rfftfreq(length, d=1.0 / float(sample_rate))
    bands = [(4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, min(45.0, sample_rate / 2.0))]
    low, high = bands[int(rng.integers(0, len(bands)))]
    mask = (freqs >= low) & (freqs <= high)
    spectrum = np.fft.rfft(rng.standard_normal(x.shape), axis=2)
    spectrum *= mask.reshape(1, 1, -1)
    noise = np.fft.irfft(spectrum, n=length, axis=2)
    noise_std = np.maximum(np.std(noise, axis=2, keepdims=True), 1e-6)
    out += noise / noise_std * channel_std(x) * cfg["noise_scale"]
    return out


def baseline_drift(
    x: np.ndarray, rng: np.random.Generator, severity: int, sample_rate: int
) -> np.ndarray:
    """Slow drift: a 0.05 to 0.5 Hz sinusoid with a per-channel random phase."""
    cfg = severity_config(severity)
    out = x.copy()
    time = np.arange(x.shape[2], dtype=np.float32) / float(sample_rate)
    frequency = float(rng.uniform(0.05, 0.5))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(x.shape[0], x.shape[1], 1))
    drift = np.sin(2.0 * np.pi * frequency * time.reshape(1, 1, -1) + phase)
    out += drift * channel_std(x) * cfg["drift_scale"]
    return out


def mixed_artifact(
    x: np.ndarray, rng: np.random.Generator, severity: int, sample_rate: int
) -> np.ndarray:
    """All six corruptions in sequence, the worst case of Section IV-F.

    Each component is given a generator restarted from the same state, so the
    windows and channels a component picks do not depend on how many random
    draws the components before it happened to consume. Without this the mixed
    condition would not be comparable to its parts.
    """
    initial_state = copy.deepcopy(rng.bit_generator.state)

    def component_rng() -> np.random.Generator:
        child = np.random.default_rng()
        child.bit_generator.state = copy.deepcopy(initial_state)
        return child

    out = baseline_drift(x, component_rng(), severity, sample_rate)
    out = band_limited_noise(out, component_rng(), severity, sample_rate)
    out = temporal_segment_noise(out, component_rng(), severity, sample_rate)
    out = channel_noise(out, component_rng(), severity, sample_rate)
    out = temporal_mask(out, component_rng(), severity, sample_rate)
    out = channel_dropout(out, component_rng(), severity, sample_rate)
    return out


ARTIFACT_FUNCTIONS = {
    "temporal_segment_noise": temporal_segment_noise,
    "channel_noise": channel_noise,
    "baseline_drift": baseline_drift,
    "band_limited_noise": band_limited_noise,
    "temporal_mask": temporal_mask,
    "channel_dropout": channel_dropout,
    "mixed_artifact": mixed_artifact,
}


def apply_artifact(
    x: np.ndarray, artifact: str, severity: int, sample_rate: int, seed: int
) -> np.ndarray:
    """Corrupt ``x`` of shape [trials, channels, time] and return a new array.

    ``artifact`` is one of ``ARTIFACT_NAMES``, or ``'clean'`` for the uncorrupted
    control condition, which returns a float32 copy unchanged.
    """
    x_float = np.asarray(x, dtype=np.float32)
    if artifact == "clean":
        return x_float.copy()
    if artifact not in ARTIFACT_FUNCTIONS:
        raise ValueError(
            f"unknown artifact {artifact!r}; expected 'clean' or one of {ARTIFACT_NAMES}"
        )
    rng = np.random.default_rng(seed)
    output = ARTIFACT_FUNCTIONS[artifact](x_float, rng, severity, sample_rate)
    return np.asarray(output, dtype=np.float32)
