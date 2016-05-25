
# constants (time constants in mS)
TO_MS = 1000; T_TO_POS = 10
T_min = 10; T_cod = 100; T_max = T_min + T_cod # time range
T_syn = 1; T_neu = 0.1 # std. delays (slightly modified T_neu)

tau_m = 100 * TO_MS; tau_f = 20
V_t = 10; V_reset = 0 # voltage model params
w_e = V_t; w_i = -V_t # std. voltage weights
g_mult = V_t * tau_m / tau_f
w_acc = V_t * tau_m / T_max; w_bar_acc = V_t * tau_m / T_cod

# coefficient constants
alpha0 = 1 # unit gain
alpha1 = 1e-4 * TO_MS * 10 # 10 * dt gain

''' old coeff values (effective for testing)
alpha0 = 0.4
alpha1 = 0.6
'''
