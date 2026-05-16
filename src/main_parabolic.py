import numpy as np
from scipy.integrate import simpson
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. Universal Physical Constants
# ==========================================
hbar = 1.054571817e-34
m_e = 9.10938356e-31
e = 1.602176634e-19
eps_0 = 8.85418782e-12    
eps_rel = 12.9            
c_light = 3e8             

# ==========================================
# 2. GaAs Material Parameters
# ==========================================
m_star = 0.067 * m_e
alpha = 0.068
hw_LO_meV = 36.25
hw_LO = hw_LO_meV * 1e-3 * e
w_LO = hw_LO / hbar
R_p = np.sqrt(hbar / (2 * m_star * w_LO))

# The corrected phonon interaction prefactor (hw_LO squared)
prefactor = (4 * np.pi * alpha * (hw_LO**2) * R_p) / ((2 * np.pi)**3)

# ==========================================
# 3. Parabolic Bare Energy Function
# ==========================================
def bare_energy(l0, B, state, lambda_var):
    # Added the factor of 2 in the denominator to match the paper's l0 convention
    w_0 = hbar / (2 * m_star * l0**2)
    w_c = (e * B) / m_star
    Omega = np.sqrt(w_0**2 + (w_c/2)**2)
    
    T_kin = (hbar**2 * lambda_var**2) / (2 * m_star)
    V_trap = (m_star * Omega**2) / (2 * lambda_var**2)
    
    if state == '0':
        return T_kin + V_trap
    elif state == '+1':
        return 2*T_kin + 2*V_trap + 0.5 * hbar * w_c
    elif state == '-1':
        return 2*T_kin + 2*V_trap - 0.5 * hbar * w_c

# ==========================================
# 4. Deterministic Variational Optimizer
# ==========================================
def optimize_state(l0, B, state):
    def target_function(vars):
        s, a = vars
        L = s * (1.0 / l0) 
        
        E_b = bare_energy(l0, B, state, L)
        
        # Fixed-Grid Phonon Integration (Eliminates all jagged noise)
        q_arr = np.linspace(0, 10 * L, 150)
        theta_arr = np.linspace(0, np.pi, 50)
        
        Q, THETA = np.meshgrid(q_arr, theta_arr)
        q_p = Q * np.sin(THETA)
        
        exponent = - ((1 - a)**2 * q_p**2) / (2 * L**2)
        if state == '0':
            rho2 = np.exp(exponent)
        else:
            bracket = 1 - ((1 - a)**2 * q_p**2) / (4 * L**2)
            rho2 = (bracket**2) * np.exp(exponent)
            
        denominator = (a**2 * hbar**2 * Q**2) / (2 * m_star) + hw_LO
        integrand = (2 * np.pi * np.sin(THETA)) * rho2 / denominator
        
        # 2D Simpson's rule integration
        inner_int = simpson(integrand, x=q_arr, axis=1)
        final_int = simpson(inner_int, x=theta_arr)
            
        E_ph = -prefactor * final_int
        
        return (E_b + E_ph) / (e * 1e-3) # Normalize to meV for optimizer

    bounds = [(0.1, 10.0), (0.01, 0.99)]
    guess = [1.0, 0.5]
    
    res = minimize(target_function, guess, bounds=bounds, method='L-BFGS-B')
    
    E_min_joules = res.fun * (e * 1e-3)
    opt_lambda = res.x[0] * (1.0 / l0)
    opt_a = res.x[1]
    
    return E_min_joules, opt_lambda, opt_a

# ==========================================
# 5. Main Execution: Scanning L0 (Figs 1 & 2)
# ==========================================
print("Running High-Resolution L0 Scan (Figures 1 & 2)...")
start_time = time.time()

l0_array_angstroms = np.linspace(30, 100, 30) # Increased to 30 for smoothness
l0_array = l0_array_angstroms * 1e-10

results = {'l0': l0_array_angstroms, 'dE_0T': [], 'dE_plus_20T': [], 'dE_minus_20T': [], 
           'T_0T': [], 'T_plus_20T': [], 'T_minus_20T': []}

for l0 in l0_array:
    E0_0, _, _ = optimize_state(l0, 0, '0')
    E1_0, _, _ = optimize_state(l0, 0, '+1')
    dE_0 = E1_0 - E0_0
    
    results['dE_0T'].append(dE_0 / (e * 1e-3))
    results['T_0T'].append((2 * np.pi * hbar) / dE_0 * 1e15)
    
    E0_20, _, _ = optimize_state(l0, 20, '0')
    E1_plus_20, _, _ = optimize_state(l0, 20, '+1')
    E1_minus_20, _, _ = optimize_state(l0, 20, '-1')
    
    dE_plus = E1_plus_20 - E0_20
    dE_minus = E1_minus_20 - E0_20
    
    results['dE_plus_20T'].append(dE_plus / (e * 1e-3))
    results['dE_minus_20T'].append(dE_minus / (e * 1e-3))
    results['T_plus_20T'].append((2 * np.pi * hbar) / dE_plus * 1e15)
    results['T_minus_20T'].append((2 * np.pi * hbar) / dE_minus * 1e15)

# ==========================================
# 6. Main Execution: Scanning B-field (Fig 3)
# ==========================================
print("Running Magnetic Field Scan (Figure 3)...")

l0_fixed = 40e-10
B_array = np.linspace(0, 60, 30)

results_fig3 = {'B': B_array, 'tau_plus': [], 'tau_minus': []}

for B in B_array:
    E0_joules, lam0, _ = optimize_state(l0_fixed, B, '0')
    E1_plus_joules, lam1_plus, _ = optimize_state(l0_fixed, B, '+1')
    E1_minus_joules, lam1_minus, _ = optimize_state(l0_fixed, B, '-1')
    
    dE_plus_joules = E1_plus_joules - E0_joules
    dE_minus_joules = E1_minus_joules - E0_joules
    
    r2_plus = (8 * (lam0**2) * (lam1_plus**4)) / ((lam0**2 + lam1_plus**2)**4)
    r2_minus = (8 * (lam0**2) * (lam1_minus**4)) / ((lam0**2 + lam1_minus**2)**4)
    
    coeff = (e**2 * np.sqrt(eps_rel)) / (3 * np.pi * eps_0 * (hbar**4) * (c_light**3))
    
    gamma_plus = coeff * (dE_plus_joules**3) * r2_plus
    gamma_minus = coeff * (dE_minus_joules**3) * r2_minus
    
    results_fig3['tau_plus'].append((1.0 / gamma_plus) * 1e9)
    results_fig3['tau_minus'].append((1.0 / gamma_minus) * 1e9)

print(f"All simulations finished in {(time.time() - start_time):.2f} seconds.")

# ==========================================
# 7. Plotting the Perfected 1x3 Layout
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Figure 1: Energy
ax1.plot(results['l0'], results['dE_0T'], 'k-', label='B = 0 T')
ax1.plot(results['l0'], results['dE_plus_20T'], 'k--', label='B = 20 T ($\\Delta E_+$)')
ax1.plot(results['l0'], results['dE_minus_20T'], 'k:', label='B = 20 T ($\\Delta E_-$)')
ax1.set_xlabel('Confinement length $l_0$ (Å)', fontsize=12)
ax1.set_ylabel('$\\Delta E$ (meV)', fontsize=12)
ax1.set_title('Fig 1: Energy Spacing', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Figure 2: Period (Y-axis capped so the separation is highly visible)
ax2.plot(results['l0'], results['T_0T'], 'k-', label='B = 0 T')
ax2.plot(results['l0'], results['T_plus_20T'], 'k--', label='B = 20 T ($T_+$)')
ax2.plot(results['l0'], results['T_minus_20T'], 'k:', label='B = 20 T ($T_-$)')
ax2.set_xlabel('Confinement length $l_0$ (Å)', fontsize=12)
ax2.set_ylabel('T (fs)', fontsize=12)
ax2.set_title('Fig 2: Oscillation Period', fontsize=14)
ax2.set_ylim(0, 800) # Prevents the T- spike from crushing the graph
ax2.legend()
ax2.grid(True, alpha=0.3)

# Figure 3: Decoherence (Axes capped to show early-field separation)
ax3.plot(results_fig3['B'], results_fig3['tau_plus'], 'k--', label='$\\tau_+$ (+1 branch)')
ax3.plot(results_fig3['B'], results_fig3['tau_minus'], 'k-', label='$\\tau_-$ (-1 branch)')
ax3.set_xlabel('Magnetic Field $B$ (T)', fontsize=12)
ax3.set_ylabel('Decoherence Time $\\tau$ (ns)', fontsize=12)
ax3.set_title('Fig 3: Decoherence Time', fontsize=14)
ax3.set_xlim(0, 30)   # Focuses on the 0 to 30 Tesla range
ax3.set_ylim(0, 5000) # Caps the massive exponential tail
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()