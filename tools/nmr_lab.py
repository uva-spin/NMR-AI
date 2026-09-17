"""Standalone physical generator, calibrated inputs, and tutorial networks."""
from dataclasses import asdict
import numpy as np
import torch
from torch import nn
from circuit import Circuit, detector_voltage, nominal_circuit
from lineshape import pake_susceptibility

from learning_data import BINS, FREQUENCY, FREQUENCY_HZ, integration_weights, make_features
ARCHITECTURES = ("mlp", "dnn", "dnn_extra", "compact", "physics_multiscale")


def thermal_polarization(frequency_hz, temperature_k=1.5):
    if not np.isfinite([frequency_hz, temperature_k]).all() or min(frequency_hz, temperature_k) <= 0:
        raise ValueError("Frequency and temperature must be positive")
    t = np.tanh(6.62607015e-34*frequency_hz/(2*1.380649e-23*temperature_k))
    return 4*t/(3+t*t)


def sample_configurations(count, seed=17, coverage="narrow"):
    """Controlled sensitivity study, not inferred hardware-parameter distributions.

    All variation is through physical fields. Broad excursions are four times
    narrow excursions. Fixed/varied choices and their provenance are in the notes.
    """
    if count < 1 or coverage not in ("narrow", "broad"):
        raise ValueError("Require positive count and narrow/broad coverage")
    rng = np.random.default_rng(seed)
    spread = 1 if coverage == "broad" else 0.25
    reference = asdict(nominal_circuit())
    configurations = []
    for _ in range(count):
        c = reference.copy()
        for key, fractional_range in (
            ("drive_v", .1), ("tune_capacitance_f", .03),
            ("coil_inductance_h", .03), ("coil_resistance_ohm", .1),
            ("damping_resistance_ohm", .05), ("cable_length_m", .01),
            ("cable_resistance_ohm_per_m", .1), ("filling_factor", .1),
            ("susceptibility_scale_cgs", .1)):
            c[key] *= 1 + rng.uniform(-fractional_range, fractional_range)*spread
        c["detector_phase_rad"] = float(rng.uniform(-.12, .12)*spread)
        configurations.append({
            "circuit": c, "center_mhz": float(32.68+rng.uniform(-.025, .025)*spread),
            "split_mhz": float(.08+rng.uniform(-.01, .01)*spread),
            "g": float(.08+rng.uniform(-.02, .02)*spread),
            "eta": float(.03+rng.uniform(-.02, .02)*spread),
            "noise_rms_v": 1e-9, "reference_averages": 16, "te_temperature_k": 1.5,
        })
    return configurations


def validate_configurations(configurations):
    if not isinstance(configurations, list) or len(configurations) < 3:
        raise ValueError("Supply at least three configurations")
    for config in configurations:
        Circuit(**config["circuit"])
        values = [config[k] for k in ("center_mhz", "split_mhz", "g", "eta", "noise_rms_v",
                                     "reference_averages", "te_temperature_k")]
        if not np.isfinite(values).all():
            raise ValueError("Configuration fields must be finite")
        if min(config["center_mhz"], config["split_mhz"], config["g"], config["te_temperature_k"]) <= 0:
            raise ValueError("Frequency, widths, and temperature must be positive")
        if not 0 <= config["eta"] <= 1 or config["noise_rms_v"] < 0:
            raise ValueError("Invalid EFG asymmetry or noise RMS")
        if not isinstance(config["reference_averages"], int) or config["reference_averages"] < 1:
            raise ValueError("Reference averages must be a positive integer")


def clean_response(p, config, center_shift_mhz=0):
    c = Circuit(**config["circuit"])
    x = (FREQUENCY-config["center_mhz"]-center_shift_mhz)/config["split_mhz"]
    chi = c.susceptibility_scale_cgs*pake_susceptibility(x, p, config["eta"], config["g"])[0]
    baseline = detector_voltage(FREQUENCY_HZ, c)
    return detector_voltage(FREQUENCY_HZ, c, chi)-baseline, baseline


def te_calibration(config):
    """Ideal independent TE reference; temperature/calibration uncertainty excluded."""
    p_te = thermal_polarization(config["center_mhz"]*1e6, config["te_temperature_k"])
    signal, _ = clean_response(p_te, config)
    area = float(signal @ integration_weights(FREQUENCY_HZ))
    if not np.isfinite(area) or abs(area) < 1e-20:
        raise ValueError("TE response has no usable area calibration at this detector phase")
    return p_te/area


def noise_factor(covariance_v2):
    covariance = np.asarray(covariance_v2, dtype=float)
    if covariance.shape != (BINS, BINS) or not np.isfinite(covariance).all():
        raise ValueError("Noise covariance must be a finite 500x500 matrix in V²")
    scale = max(float(np.max(np.abs(covariance))), 1e-30)
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=scale*1e-12):
        raise ValueError("Noise covariance must be symmetric")
    eigenvalues, vectors = np.linalg.eigh(covariance)
    if eigenvalues.min() < -scale*1e-10:
        raise ValueError("Noise covariance must be positive semidefinite")
    return vectors*np.sqrt(np.maximum(eigenvalues, 0))


def simulate(p, config, rng, center_jitter=0.0, covariance_factor=None):
    if not np.isfinite(center_jitter) or center_jitter < 0:
        raise ValueError("Center jitter must be finite and nonnegative")
    shift = rng.uniform(-center_jitter, center_jitter) if center_jitter else 0.0
    nuclear, baseline = clean_response(p, config, shift)
    noise = (rng.normal(size=BINS)*config["noise_rms_v"] if covariance_factor is None
             else covariance_factor @ rng.normal(size=BINS))
    return {"signal": nuclear+baseline+noise, "lineshape": nuclear,
            "baseline": baseline, "noise": noise}


def group_split(groups, seed):
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("At least three sampled configurations are needed for grouped evaluation")
    np.random.default_rng(seed).shuffle(unique)
    n_test = max(1, int(0.1 * len(unique)))
    n_val = max(1, int(0.1 * len(unique)))
    partitions = [unique[:len(unique) - n_test - n_val], unique[-n_test - n_val:-n_test], unique[-n_test:]]
    return [np.flatnonzero(np.isin(groups, partition)) for partition in partitions]


class BasicPolarizationMLP(nn.Module):
    """One hidden dense layer over the same two channels used by the CNNs."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(2*BINS, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)


class DeepPolarizationDNN(nn.Module):
    """Three hidden dense layers learn successive global spectral features."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(2*BINS, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


class ExtraLayerPolarizationDNN(nn.Module):
    """Four hidden dense layers for the targeted-complexity challenge."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(2*BINS, 256), nn.GELU(),
            nn.Linear(256, 192), nn.GELU(), nn.Linear(192, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


class SmallPolarizationCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(2, 16, 7, padding=3), nn.GELU(), nn.AvgPool1d(2),
                                     nn.Conv1d(16, 32, 7, padding=3), nn.GELU(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        return self.head(self.encoder(x).squeeze(-1))


class SpectralBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(6, channels), nn.GELU(),
            nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation),
            nn.GroupNorm(6, channels), nn.GELU(),
            nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation),
        )

    def forward(self, x):
        return x + self.net(x)


class PhysicsMultiscaleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.ModuleList([nn.Conv1d(2, 12, k, padding=k // 2) for k in (5, 15, 31)])
        self.encoder = nn.Sequential(nn.Conv1d(36, 48, 1), nn.GELU(), SpectralBlock(48, 1),
                                     nn.AvgPool1d(2), SpectralBlock(48, 2), nn.AvgPool1d(2),
                                     SpectralBlock(48, 4))
        self.register_buffer("frequency", torch.linspace(-1.0, 1.0, BINS))
        self.register_buffer("physics_mean", torch.zeros(1, 25))
        self.register_buffer("physics_std", torch.ones(1, 25))
        self.physics_head = nn.Linear(25, 1)
        self.correction = nn.Sequential(nn.Linear(144, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def physics_features(self, x):
        residual, raw = x[:, 1], x[:, 0]
        center = (residual.abs() * self.frequency).sum(-1) / residual.abs().sum(-1).clamp_min(1e-6)
        axis = self.frequency[None, :] - center[:, None]
        signed = [(residual * axis**power).mean(-1, keepdim=True) for power in range(11)]
        absolute = [(residual.abs() * axis**power).mean(-1, keepdim=True) for power in range(7)]
        stats = [residual.amin(-1, keepdim=True), residual.amax(-1, keepdim=True), residual.std(-1, keepdim=True),
                 raw.mean(-1, keepdim=True), raw.std(-1, keepdim=True), raw.amin(-1, keepdim=True), raw.amax(-1, keepdim=True)]
        return torch.cat(signed + absolute + stats, dim=1)

    def forward(self, x):
        encoded = self.encoder(torch.cat([layer(x) for layer in self.stem], dim=1))
        pooled = torch.cat([encoded.mean(-1), encoded.amax(-1), encoded.std(-1)], dim=1)
        physics = (self.physics_features(x) - self.physics_mean) / self.physics_std
        return self.physics_head(physics) + self.correction(pooled)

    def calibrate(self, inputs, target):
        with torch.no_grad():
            features = self.physics_features(torch.from_numpy(inputs)).numpy().astype(np.float64)
        mean, std = features.mean(0), features.std(0)
        std[std < 1e-8] = 1.0
        design = np.column_stack([(features - mean) / std, np.ones(len(features))])
        penalty = np.eye(design.shape[1]) * 1e-3
        penalty[-1, -1] = 0
        weights = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        with torch.no_grad():
            self.physics_mean.copy_(torch.from_numpy(mean[None]))
            self.physics_std.copy_(torch.from_numpy(std[None]))
            self.physics_head.weight.copy_(torch.from_numpy(weights[:-1].T))
            self.physics_head.bias.copy_(torch.from_numpy(weights[-1]))
        self.physics_head.requires_grad_(False)


def build_model(architecture):
    if architecture == "mlp":
        return BasicPolarizationMLP()
    if architecture == "dnn":
        return DeepPolarizationDNN()
    if architecture == "dnn_extra":
        return ExtraLayerPolarizationDNN()
    if architecture == "compact":
        return SmallPolarizationCNN()
    if architecture == "physics_multiscale":
        return PhysicsMultiscaleCNN()
    raise ValueError(f"Unknown architecture: {architecture}")
