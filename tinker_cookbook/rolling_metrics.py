from dataclasses import dataclass, field
from typing import NamedTuple
import math


@dataclass(frozen=True)
class AdamDiagnosticsState:
    """Immutable state tracking metric history for signal computation."""
    
    # Rolling windows (most recent last)
    grad_norm_history: tuple[float, ...] = ()
    m_norm_history: tuple[float, ...] = ()
    v_norm_history: tuple[float, ...] = ()
    loss_history: tuple[float, ...] = ()
    effective_lr_mean_history: tuple[float, ...] = ()
    
    # Config
    window_size: int = 10
    
    def with_update(self, **kwargs) -> "AdamDiagnosticsState":
        """Return new state with updated fields."""
        return AdamDiagnosticsState(**{**self.__dict__, **kwargs})

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "grad_norm_history": list(self.grad_norm_history),
            "m_norm_history": list(self.m_norm_history),
            "v_norm_history": list(self.v_norm_history),
            "loss_history": list(self.loss_history),
            "effective_lr_mean_history": list(self.effective_lr_mean_history),
            "window_size": self.window_size,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AdamDiagnosticsState":
        """Deserialize from dict (converts lists back to tuples)."""
        return cls(
            grad_norm_history=tuple(d["grad_norm_history"]),
            m_norm_history=tuple(d["m_norm_history"]),
            v_norm_history=tuple(d["v_norm_history"]),
            loss_history=tuple(d["loss_history"]),
            effective_lr_mean_history=tuple(d["effective_lr_mean_history"]),
            window_size=d["window_size"],
        )


class RawMetrics(NamedTuple):
    """Single step of raw Adam metrics from the logger."""
    grad_norm: float
    grad_mean: float
    grad_std: float
    grad_abs_mean: float
    grad_norm_max_layer: float
    grad_norm_min_layer: float
    m_norm: float
    m_abs_mean: float
    v_norm: float
    v_mean: float
    v_max: float
    effective_lr_mean: float
    effective_lr_std: float
    effective_lr_min: float
    effective_lr_max: float
    update_norm: float
    param_norm: float
    m_to_v_ratio: float
    loss: float
    lr: float  # nominal learning rate


class DerivedSignals(NamedTuple):
    """Computed diagnostic signals for a single step."""
    # Instantaneous signals (no history needed)
    # Note: flags are int (0/1) for wandb compatibility
    grad_norm_exploding: int
    grad_norm_vanishing: int
    gradient_variance_ratio: float
    layer_gradient_imbalance: float
    second_moment_imbalance: float
    second_moment_underaccumulated: int
    effective_lr_decay_ratio: float
    effective_lr_spread: float
    effective_lr_stalled_params: int
    effective_lr_runaway_params: int
    update_too_large: int
    update_too_small: int
    momentum_dominated: int
    adaptation_dominated: int
    momentum_premature_decay: int

    # History-dependent signals
    grad_norm_spike: int
    momentum_unbounded_growth: int
    momentum_oscillation: int
    second_moment_unbounded_growth: int
    loss_instability: int
    loss_zigzag: int
    update_norm_spike: int

    # Raw computed values (for logging/inspection)
    grad_norm_vs_rolling_mean: float | None
    m_norm_cv: float | None  # coefficient of variation
    loss_cv: float | None
    loss_sign_change_ratio: float | None


def _append_to_window(history: tuple[float, ...], value: float, window_size: int) -> tuple[float, ...]:
    """Append value to history, keeping only last window_size elements."""
    new_history = history + (value,)
    if len(new_history) > window_size:
        return new_history[-window_size:]
    return new_history


def _rolling_mean(history: tuple[float, ...]) -> float | None:
    if not history:
        return None
    return sum(history) / len(history)


def _rolling_std(history: tuple[float, ...]) -> float | None:
    if len(history) < 2:
        return None
    mean = sum(history) / len(history)
    variance = sum((x - mean) ** 2 for x in history) / len(history)
    return math.sqrt(variance)


def _coefficient_of_variation(history: tuple[float, ...]) -> float | None:
    """std / mean — measures relative variability."""
    if len(history) < 2:
        return None
    mean = _rolling_mean(history)
    if mean is None or abs(mean) < 1e-12:
        return None
    std = _rolling_std(history)
    if std is None:
        return None
    return std / abs(mean)


def _sign_change_ratio(history: tuple[float, ...]) -> float | None:
    """Fraction of consecutive steps where diff changes sign."""
    if len(history) < 3:
        return None
    diffs = [history[i] - history[i - 1] for i in range(1, len(history))]
    sign_changes = sum(
        1 for i in range(1, len(diffs))
        if (diffs[i] > 0) != (diffs[i - 1] > 0)
    )
    return sign_changes / (len(diffs) - 1) if len(diffs) > 1 else 0.0


def _is_monotonic_increasing(history: tuple[float, ...], min_steps: int = 10) -> bool:
    """Check if history has been monotonically increasing for at least min_steps."""
    if len(history) < min_steps:
        return False
    recent = history[-min_steps:]
    return all(recent[i] >= recent[i - 1] for i in range(1, len(recent)))


def step(
    state: AdamDiagnosticsState,
    metrics: RawMetrics,
) -> tuple[AdamDiagnosticsState, DerivedSignals]:
    """
    Pure function: compute derived signals and return updated state.
    
    Args:
        state: Previous diagnostic state
        metrics: Current step's raw metrics
        
    Returns:
        (new_state, signals) tuple
    """
    w = state.window_size
    
    # Update histories
    new_grad_norm_history = _append_to_window(state.grad_norm_history, metrics.grad_norm, w)
    new_m_norm_history = _append_to_window(state.m_norm_history, metrics.m_norm, w)
    new_v_norm_history = _append_to_window(state.v_norm_history, metrics.v_norm, w)
    new_loss_history = _append_to_window(state.loss_history, metrics.loss, w)
    new_effective_lr_mean_history = _append_to_window(
        state.effective_lr_mean_history, metrics.effective_lr_mean, w
    )
    
    # === Instantaneous signals ===

    grad_norm_exploding = int(metrics.grad_norm > 100)
    grad_norm_vanishing = int(metrics.grad_norm < 1e-6)
    
    gradient_variance_ratio = (
        metrics.grad_std / metrics.grad_abs_mean
        if metrics.grad_abs_mean > 1e-12 else float('inf')
    )
    
    layer_gradient_imbalance = (
        metrics.grad_norm_max_layer / metrics.grad_norm_min_layer
        if metrics.grad_norm_min_layer > 1e-12 else float('inf')
    )
    
    second_moment_imbalance = (
        metrics.v_max / metrics.v_mean
        if metrics.v_mean > 1e-12 else float('inf')
    )
    
    second_moment_underaccumulated = int(metrics.v_mean < 1e-12)
    
    effective_lr_decay_ratio = (
        metrics.effective_lr_mean / metrics.lr
        if metrics.lr > 1e-12 else 0.0
    )
    
    effective_lr_spread = (
        metrics.effective_lr_std / metrics.effective_lr_mean
        if metrics.effective_lr_mean > 1e-12 else float('inf')
    )
    
    effective_lr_stalled_params = int(
        metrics.effective_lr_min / metrics.effective_lr_mean < 0.001
        if metrics.effective_lr_mean > 1e-12 else False
    )

    effective_lr_runaway_params = int(
        metrics.effective_lr_max / metrics.effective_lr_mean > 100
        if metrics.effective_lr_mean > 1e-12 else False
    )
    
    update_to_param_ratio = (
        metrics.update_norm / metrics.param_norm
        if metrics.param_norm > 1e-12 else float('inf')
    )
    update_too_large = int(update_to_param_ratio > 0.1)
    update_too_small = int(update_to_param_ratio < 1e-6)

    momentum_dominated = int(metrics.m_to_v_ratio > 10)
    adaptation_dominated = int(metrics.m_to_v_ratio < 0.1)

    momentum_premature_decay = int(
        (metrics.m_norm / metrics.grad_norm < 0.01)
        if metrics.grad_norm > 1e-4 else False
    )
    
    # === History-dependent signals ===
    
    # Gradient spike detection
    grad_rolling_mean = _rolling_mean(state.grad_norm_history)  # use previous history
    grad_norm_vs_rolling_mean = (
        metrics.grad_norm / grad_rolling_mean
        if grad_rolling_mean and grad_rolling_mean > 1e-12 else None
    )
    grad_norm_spike = int(
        grad_norm_vs_rolling_mean is not None and grad_norm_vs_rolling_mean > 10
    )

    # Momentum oscillation
    m_norm_cv = _coefficient_of_variation(new_m_norm_history)
    momentum_oscillation = int(m_norm_cv is not None and m_norm_cv > 0.5)

    # Unbounded growth detection
    momentum_unbounded_growth = int(_is_monotonic_increasing(new_m_norm_history))
    second_moment_unbounded_growth = int(_is_monotonic_increasing(new_v_norm_history))

    # Loss instability
    loss_cv = _coefficient_of_variation(new_loss_history)
    loss_instability = int(loss_cv is not None and loss_cv > 0.5)

    # Loss zigzag
    loss_sign_change_ratio = _sign_change_ratio(new_loss_history)
    loss_zigzag = int(loss_sign_change_ratio is not None and loss_sign_change_ratio > 0.4)

    # Update norm spike (compare to rolling mean of update_norm, but we're not tracking it)
    # For now, use grad_norm spike as proxy; or we could add update_norm_history
    update_norm_spike = grad_norm_spike  # simplified; extend state if needed
    
    # Build new state
    new_state = AdamDiagnosticsState(
        grad_norm_history=new_grad_norm_history,
        m_norm_history=new_m_norm_history,
        v_norm_history=new_v_norm_history,
        loss_history=new_loss_history,
        effective_lr_mean_history=new_effective_lr_mean_history,
        window_size=w,
    )
    
    signals = DerivedSignals(
        # Instantaneous
        grad_norm_exploding=grad_norm_exploding,
        grad_norm_vanishing=grad_norm_vanishing,
        gradient_variance_ratio=gradient_variance_ratio,
        layer_gradient_imbalance=layer_gradient_imbalance,
        second_moment_imbalance=second_moment_imbalance,
        second_moment_underaccumulated=second_moment_underaccumulated,
        effective_lr_decay_ratio=effective_lr_decay_ratio,
        effective_lr_spread=effective_lr_spread,
        effective_lr_stalled_params=effective_lr_stalled_params,
        effective_lr_runaway_params=effective_lr_runaway_params,
        update_too_large=update_too_large,
        update_too_small=update_too_small,
        momentum_dominated=momentum_dominated,
        adaptation_dominated=adaptation_dominated,
        momentum_premature_decay=momentum_premature_decay,
        # History-dependent
        grad_norm_spike=grad_norm_spike,
        momentum_unbounded_growth=momentum_unbounded_growth,
        momentum_oscillation=momentum_oscillation,
        second_moment_unbounded_growth=second_moment_unbounded_growth,
        loss_instability=loss_instability,
        loss_zigzag=loss_zigzag,
        update_norm_spike=update_norm_spike,
        # Raw values
        grad_norm_vs_rolling_mean=grad_norm_vs_rolling_mean,
        m_norm_cv=m_norm_cv,
        loss_cv=loss_cv,
        loss_sign_change_ratio=loss_sign_change_ratio,
    )
    
    return new_state, signals


def init_state(window_size: int = 10) -> AdamDiagnosticsState:
    """Create initial empty state."""
    return AdamDiagnosticsState(window_size=window_size)


# === Usage example ===

def run_diagnostics(metrics_stream):
    """
    Example: fold over a stream of metrics.
    
    metrics_stream: Iterable[RawMetrics]
    """
    state = init_state(window_size=10)
    
    for metrics in metrics_stream:
        state, signals = step(state, metrics)
        yield signals