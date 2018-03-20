import numpy as np

from neural.constants import *


class Neuron(object):

    def __init__(self, name, t):
        self.name = name
        self.t = t
        self.v = np.zeros(np.shape(t))

        self.g_e = np.zeros(np.shape(t))
        self.g_f = np.zeros(np.shape(t))
        self.gate = np.zeros(np.shape(t))

    def next_v(self, i):  # compute voltage at pos i

        # constants (time in mS; V in mV)
        dt = self.t[1] - self.t[0]

        if self.v[i - 1] >= V_t:
            v_p = V_reset
            ge_p = 0
            gf_p = 0
            gate_p = 0
        else:
            v_p = self.v[i - 1]
            ge_p = self.g_e[i - 1]
            gf_p = self.g_f[i - 1]
            gate_p = self.gate[i - 1]

        self.g_e[i] = self.g_e[i] + ge_p
        self.gate[i] = max([self.gate[i], gate_p])
        self.g_f[i] = self.g_f[i] + gf_p + dt * (-gf_p / tau_f)
        self.v[i] = self.v[i] + v_p + dt * ((ge_p + gf_p * gate_p) / tau_m)
