# rank_method_selector.py
import math
import random
import numpy as np
from dataclasses import dataclass

@dataclass
class ArmStats:
    n: int = 0
    mean: float = 0.0
    # Welford online variance (optional; here we keep a simple running M2 for posterior std)
    M2: float = 0.0

    def update(self, r: float):
        self.n += 1
        delta = r - self.mean
        self.mean += delta / self.n
        delta2 = r - self.mean
        self.M2 += delta * delta2

    @property
    def var(self):
        return self.M2 / (self.n - 1) if self.n > 1 else 1.0  # conservative when n<=1

class GaussianTSSelector:
    """
    Non-contextual Gaussian Thompson Sampling over 3 arms:
      0: Similarity (wS=1, wU=0)
      1: Prediction (wS=0, wU=1)
      2: Fusion     (wS=0.5, wU=0.5)   # change prior if you like

    Reward should be larger-is-better (e.g., baseline_utility - candidate_utility for RMSE),
    or negative iterations-to-pass.
    """

    def __init__(self, seed=42, prior_mean=0.0, prior_var=1.0, noise_var=1.0):
        self.rng = random.Random(seed)
        self.K = 3
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.noise_var = noise_var  # observation noise
        self.arms = [ArmStats() for _ in range(self.K)]

    def _sample_posterior_mean(self, arm_idx: int):
        stats = self.arms[arm_idx]
        # Conjugate Gaussian with known noise variance → posterior variance shrinks as n grows.
        # Here we use a simple approx: post_var ≈ prior_var / (1 + n*prior_var/noise_var)
        n = stats.n
        post_mean = (self.prior_mean if n == 0 else stats.mean)
        post_var  = self.prior_var / (1.0 + n * self.prior_var / self.noise_var)
        # Draw one sample for TS
        return np.random.normal(loc=post_mean, scale=math.sqrt(max(1e-9, post_var)))

    def select_arm(self):
        samples = [self._sample_posterior_mean(k) for k in range(self.K)]
        return int(np.argmax(samples))

    def update(self, arm_idx: int, reward: float):
        self.arms[arm_idx].update(reward)

    def best_arm_by_posterior_mean(self):
        # Tie-break by larger mean, then by counts
        means = [a.mean if a.n > 0 else self.prior_mean for a in self.arms]
        return int(np.argmax(means))

    @staticmethod
    def arm_to_weights(arm_idx: int):
        if arm_idx == 0:   # Similarity
            return 1.0, 0.0
        if arm_idx == 1:   # Prediction
            return 0.0, 1.0
        return 0.5, 0.5    # Fusion (default)
