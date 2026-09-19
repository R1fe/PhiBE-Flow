"""Acrobot dynamics without importing the machine-learning runtime."""

from __future__ import annotations

import numpy as np


class AcrobotSystem:
    """Two-link Acrobot dynamics with optional stochastic perturbations."""

    def __init__(
        self,
        m1: float = 1.0,
        m2: float = 1.0,
        l1: float = 1.0,
        l2: float = 1.0,
        lc1: float = 0.5,
        lc2: float = 0.5,
        I1: float = 1.0,
        I2: float = 1.0,
        g: float = 9.8,
        noise: float = 0.5,
    ) -> None:
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.lc1 = lc1
        self.lc2 = lc2
        self.I1 = I1
        self.I2 = I2
        self.g = g
        self.noise = noise

    def dynamics(self, _: float, state: np.ndarray) -> list[float]:
        theta1, theta2, theta1_dot, theta2_dot = state
        sin_theta2 = np.sin(theta2)
        cos_theta2 = np.cos(theta2)

        d1 = self.I1 + self.I2 + self.m1 * self.lc1**2 + self.m2 * (
            self.l1**2 + self.lc2**2 + 2 * self.l1 * self.lc2 * cos_theta2
        )
        d2 = self.I2 + self.m2 * (self.lc2**2 + self.l1 * self.lc2 * cos_theta2)

        c1 = (
            -self.m2 * self.l1 * self.lc2 * sin_theta2 * theta2_dot**2
            - 2 * self.m2 * self.l1 * self.lc2 * sin_theta2 * theta1_dot * theta2_dot
            + (self.m2 * self.l1 + self.m1 * self.lc1)
            * self.g
            * np.cos(theta1 - np.pi / 2)
            + self.m2 * self.lc2 * self.g * np.cos(theta1 + theta2 - np.pi / 2)
        )
        c2 = self.m2 * self.lc2 * self.g * np.cos(theta1 + theta2 - np.pi / 2)

        theta2_ddot = (
            (d2 / d1) * c1
            - self.m2 * self.l1 * self.lc2 * theta1_dot**2 * sin_theta2
            - c2
        ) / (self.m2 * self.lc2**2 + self.I2 - d2**2 / d1)
        theta1_ddot = -(d2 * theta2_ddot + c1) / d1

        if self.noise:
            theta1_dot += self.noise * np.random.randn()
            theta2_dot += self.noise * np.random.randn()
            theta1_ddot += self.noise * np.random.randn()
            theta2_ddot += self.noise * np.random.randn()

        return [theta1_dot, theta2_dot, theta1_ddot, theta2_ddot]
