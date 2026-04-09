"""
STPA - Jump Process Engine (Compound Poisson Process)
=======================================================
Models sudden, discrete financial shocks that hit a borrower's
health score unexpectedly — job loss, medical emergency, divorce, etc.

The Compound Poisson Process:
    J(t) = Σ Y_i  for i = 1..N(t)

Where:
    N(t)  = Poisson process with arrival rate λ (shocks per year)
    Y_i   = shock magnitude (drawn from empirical distribution)

Shock Types:
    - MINOR:    Small income dip, unexpected bill       (−3 to −8 pts)
    - MODERATE: Job loss, medical expense               (−10 to −20 pts)
    - SEVERE:   Bankruptcy trigger, major life event    (−25 to −40 pts)

Recovery:
    - Shocks can also be positive (raise, windfall, debt payoff)
    - Modeled as upward jumps with lower arrival rate
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from enum import Enum


DT = 1 / 12  # Monthly step


class ShockType(Enum):
    MINOR    = "minor"
    MODERATE = "moderate"
    SEVERE   = "severe"
    POSITIVE = "positive"


@dataclass
class ShockProfile:
    """Defines the shock distribution for a borrower segment."""
    lambda_negative: float = 1.5   # Negative shocks per year
    lambda_positive: float = 0.5   # Positive shocks per year

    # Shock magnitude distributions (Gaussian params: mean, std)
    minor_mean:    float = -5.0
    minor_std:     float = 2.0
    moderate_mean: float = -15.0
    moderate_std:  float = 4.0
    severe_mean:   float = -30.0
    severe_std:    float = 6.0
    positive_mean: float = 8.0
    positive_std:  float = 3.0

    # Probability of each negative shock type
    minor_prob:    float = 0.60
    moderate_prob: float = 0.30
    severe_prob:   float = 0.10


# Preset profiles for different borrower risk tiers
SHOCK_PROFILES = {
    "low_risk":    ShockProfile(lambda_negative=0.8,  lambda_positive=0.6),
    "medium_risk": ShockProfile(lambda_negative=1.5,  lambda_positive=0.4),
    "high_risk":   ShockProfile(lambda_negative=2.5,  lambda_positive=0.2),
    "subprime":    ShockProfile(lambda_negative=4.0,  lambda_positive=0.1,
                                severe_prob=0.25, moderate_prob=0.40, minor_prob=0.35),
}


class JumpEngine:
    """
    Compound Poisson jump process engine.

    Injects stochastic shocks onto a base financial health path.

    Usage:
        engine = JumpEngine()
        shocked_path = engine.apply_shocks(base_path, ShockProfile())
        shock_log = engine.get_shock_log()
    """

    def __init__(self):
        self._shock_log: List[Dict] = []

    def _sample_shock_magnitude(self, profile: ShockProfile) -> Tuple[float, ShockType]:
        """Sample a negative shock magnitude based on shock profile."""
        shock_type_draw = np.random.random()
        if shock_type_draw < profile.minor_prob:
            return np.random.normal(profile.minor_mean, profile.minor_std), ShockType.MINOR
        elif shock_type_draw < profile.minor_prob + profile.moderate_prob:
            return np.random.normal(profile.moderate_mean, profile.moderate_std), ShockType.MODERATE
        else:
            return np.random.normal(profile.severe_mean, profile.severe_std), ShockType.SEVERE

    def apply_shocks(
        self,
        base_path: np.ndarray,
        profile: ShockProfile,
        record_shocks: bool = True
    ) -> np.ndarray:
        """
        Apply Compound Poisson shocks to a base health score path.

        Args:
            base_path: Array of health scores (n_steps,)
            profile: ShockProfile defining shock distribution
            record_shocks: Whether to log shock events

        Returns:
            Shocked path of same shape
        """
        path = base_path.copy()
        n_steps = len(path)
        self._shock_log = []

        for t in range(n_steps):
            # Negative shocks
            n_neg = np.random.poisson(profile.lambda_negative * DT)
            for _ in range(n_neg):
                magnitude, shock_type = self._sample_shock_magnitude(profile)
                path[t:] += magnitude  # Persistent effect
                path = np.clip(path, 0, 100)
                if record_shocks:
                    self._shock_log.append({
                        "month": t,
                        "type": shock_type.value,
                        "magnitude": round(magnitude, 2),
                        "direction": "negative"
                    })

            # Positive shocks
            n_pos = np.random.poisson(profile.lambda_positive * DT)
            for _ in range(n_pos):
                magnitude = abs(np.random.normal(profile.positive_mean, profile.positive_std))
                path[t:] += magnitude
                path = np.clip(path, 0, 100)
                if record_shocks:
                    self._shock_log.append({
                        "month": t,
                        "type": ShockType.POSITIVE.value,
                        "magnitude": round(magnitude, 2),
                        "direction": "positive"
                    })

        return path

    def apply_shocks_batch(
        self,
        base_paths: np.ndarray,
        profile: ShockProfile
    ) -> np.ndarray:
        """
        Vectorized batch shock application across many simulated paths.

        Args:
            base_paths: Array of shape (n_simulations, n_steps)
            profile: ShockProfile

        Returns:
            Shocked paths of same shape
        """
        paths = base_paths.copy()
        n_sims, n_steps = paths.shape

        for t in range(n_steps):
            # Negative shocks
            n_neg = np.random.poisson(profile.lambda_negative * DT, size=n_sims)
            for i in np.where(n_neg > 0)[0]:
                for _ in range(n_neg[i]):
                    mag, _ = self._sample_shock_magnitude(profile)
                    paths[i, t:] = np.clip(paths[i, t:] + mag, 0, 100)

            # Positive shocks
            n_pos = np.random.poisson(profile.lambda_positive * DT, size=n_sims)
            for i in np.where(n_pos > 0)[0]:
                for _ in range(n_pos[i]):
                    mag = abs(np.random.normal(profile.positive_mean, profile.positive_std))
                    paths[i, t:] = np.clip(paths[i, t:] + mag, 0, 100)

        return paths

    def get_shock_log(self) -> List[Dict]:
        """Return the log of all shocks applied in the last simulation."""
        return self._shock_log

    def expected_total_shock(self, profile: ShockProfile, horizon_months: int = 24) -> float:
        """Analytical expected total shock impact over a horizon."""
        T = horizon_months / 12
        avg_neg = (
            profile.minor_prob * profile.minor_mean
            + profile.moderate_prob * profile.moderate_mean
            + profile.severe_prob * profile.severe_mean
        )
        avg_pos = profile.positive_mean
        return (profile.lambda_negative * avg_neg + profile.lambda_positive * avg_pos) * T
