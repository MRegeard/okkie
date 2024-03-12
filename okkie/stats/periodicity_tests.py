import numpy as np

__all__ = ["alpha_m", "beta_m", "ZmTest", "HTest"]


def alpha_m(n, m):
    return (1 / n) * np.sum(np.cos(n * m))


def beta_m(n, m):
    return (1 / n) * np.sum(np.sin(n * m))


class ZmTest:
    def __init__(self, phases, m):
        self.phases = np.asarray(phases) * 2 * np.pi
        self.m = m

    def alpha(self, m):
        return alpha_m(len(self.phases), m)

    def beta(self, m):
        return beta_m(len(self.phases), m)

    @property
    def z_stat_array(self):
        return (
            2
            * len(self.phases)
            * np.asarray(
                [self.alpha(i) ** 2 + self.beta(i) ** 2 for i in range(1, self.m + 1)]
            )
        )

    @property
    def z_stat(self):
        return np.sum(self.z_stat_array)


class HTest:
    def __init__(self, phases, m=20, c=4):
        self.phases = np.asarray(phases) * 2 * np.pi
        self.m = m
        self.c = c

    @property
    def h_stat_array(self):
        zm = ZmTest(self.phases, self.m)
        return zm.z_stat_array - self.c * np.arange(0, self.m)

    @property
    def h_stat(self):
        return np.max(self.h_stat_array)
