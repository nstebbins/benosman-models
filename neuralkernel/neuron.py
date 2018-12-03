import numpy as np

from .constants import V_THRESHOLD, TAU_F, TAU_M


class Neuron:

    def __init__(self, name, t):
        self.name = name
        self.t = t
        self.v = np.zeros(t.shape)
        self.ge = np.zeros(t.shape)
        self.gf = np.zeros(t.shape)
        self.gate = np.zeros(t.shape)

    # TODO: modify contract (use time, not indices and set, not list)
    def update_v(self, spike_idx_set):
        """update voltage with set of spike indices"""
        for spike_idx in spike_idx_set:
            self.v[spike_idx] = V_THRESHOLD

    def simulate(self, idx):
        """simulate neuron at index"""
        if self.v[idx - 1] >= V_THRESHOLD:
            v_prev = 0
            ge_prev = 0
            gf_prev = 0
            gate_prev = 0
        else:
            v_prev = self.v[idx - 1]
            ge_prev = self.ge[idx - 1]
            gf_prev = self.gf[idx - 1]
            gate_prev = self.gate[idx - 1]
        dt = self.t[idx] - self.t[idx - 1]
        self.ge[idx] += ge_prev
        self.gate[idx] = max([self.gate[idx], gate_prev])  # TODO: simplify
        self.gf[idx] += gf_prev + dt * (-gf_prev / TAU_F)
        self.v[idx] += v_prev + dt * ((ge_prev + gf_prev * gate_prev) / TAU_M)
