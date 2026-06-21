# decoherence.py
# Day 5: Lindblad decoherence using scipy RK45 adaptive integration.
# Replaces Euler stepping which was too inaccurate for quantum oscillations.

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp

# ── Grid setup ────────────────────────────────────────────────────────────────
GRID_SIZE = 7
G = nx.grid_2d_graph(GRID_SIZE, GRID_SIZE)
G = nx.convert_node_labels_to_integers(G)
N = G.number_of_nodes()
L = nx.laplacian_matrix(G).toarray().astype(float)

center_node    = N // 2
node_positions = np.array([(i, j) for i in range(GRID_SIZE)
                            for j in range(GRID_SIZE)])
center_pos     = node_positions[center_node]
sq_distances   = np.sum((node_positions - center_pos) ** 2,
                         axis=1).astype(float)

print(f"Grid: {GRID_SIZE}×{GRID_SIZE} = {N} nodes")
print(f"Max squared distance from center: {sq_distances.max():.1f}")
print()

# ── Lindblad RHS as a flat real vector ───────────────────────────────────────
# scipy.solve_ivp works with real vectors, so we flatten the complex N×N
# density matrix into a real vector of length 2*N*N (real and imag parts).

def rho_to_vec(rho):
    return np.concatenate([rho.real.ravel(), rho.imag.ravel()])

def vec_to_rho(v):
    half = N * N
    return v[:half].reshape(N, N) + 1j * v[half:].reshape(N, N)

def lindblad_rhs(t, v, gamma):
    rho = vec_to_rho(v)

    # Quantum part: -i [L, rho]
    d_rho = -1j * (L @ rho - rho @ L)

    # Dephasing part: destroys off-diagonal coherences at rate gamma
    if gamma > 0:
        rho_diag = np.diag(np.diag(rho))
        d_rho += -gamma * (rho - rho_diag)

    return rho_to_vec(d_rho)

# ── Initial state ─────────────────────────────────────────────────────────────
psi0 = np.zeros(N, dtype=complex)
psi0[center_node] = 1.0
rho0 = np.outer(psi0, psi0.conj())
v0   = rho_to_vec(rho0)

# ── MSD from density matrix ───────────────────────────────────────────────────
def msd_from_rho(rho):
    diag = np.real(np.diag(rho)).copy()
    diag = np.clip(diag, 0, None)
    if diag.sum() > 0:
        diag /= diag.sum()
    return float(np.dot(sq_distances, diag))

# ── Simulate one gamma value ──────────────────────────────────────────────────
T_MAX   = 4.0
T_EVAL  = np.linspace(0.05, T_MAX, 120)

def simulate(gamma):
    sol = solve_ivp(
        lindblad_rhs,
        [0, T_MAX],
        v0,
        method="RK45",
        t_eval=T_EVAL,
        args=(gamma,),
        rtol=1e-7,
        atol=1e-9,
        max_step=0.02        # upper bound on step size — keeps accuracy
    )
    msd_arr = np.array([msd_from_rho(vec_to_rho(sol.y[:, k]))
                        for k in range(sol.y.shape[1])])
    return sol.t, msd_arr

# ── Run for all gamma values ──────────────────────────────────────────────────
gammas = [0, 0.3, 1.0, 4.0, 20.0]
colors = ["#1D9E75", "#5BA88A", "#D4A843", "#D87A30", "#7F77DD"]
labels = [f"γ = {g}" for g in gammas]
labels[0]  = "γ = 0  (pure quantum)"
labels[-1] = f"γ = {gammas[-1]}  (near classical)"

results = {}
alphas  = {}

print("Running simulations (RK45 adaptive integrator)...")
for gamma in gammas:
    print(f"  gamma = {gamma:5.1f} ...", end=" ", flush=True)
    t_arr, msd_arr = simulate(gamma)
    results[gamma] = (t_arr, msd_arr)

    # Fit alpha over t = 0.3 to 3.0
    mask  = (t_arr >= 0.3) & (t_arr <= 3.0) & (msd_arr > 1e-6)
    if mask.sum() > 5:
        alpha, _ = np.polyfit(np.log(t_arr[mask]),
                               np.log(msd_arr[mask]), 1)
    else:
        alpha = float("nan")
    alphas[gamma] = alpha
    print(f"alpha = {alpha:.3f}")

print()
print("Gamma      Alpha")
print("──────────────────────────────────────────────")
for g in gammas:
    note = ""
    if g == 0:           note = "  ← pure quantum  (expect ~1.5–1.9)"
    if g == gammas[-1]:  note = "  ← near classical  (expect ~1.0)"
    print(f"  {g:5.1f}      {alphas[g]:.3f}{note}")

# ── Sanity check: MSD at a few time points ─────────────────────────────────────
print()
print("MSD values at selected times:")
print(f"{'t':>6}  {'γ=0':>8}  {'γ=1':>8}  {'γ=20':>8}")
for t_check in [0.5, 1.0, 2.0, 3.0]:
    row = []
    for g in [0, 1.0, 20.0]:
        t_arr, msd_arr = results[g]
        idx = np.argmin(np.abs(t_arr - t_check))
        row.append(msd_arr[idx])
    print(f"  {t_check:4.1f}  {row[0]:8.3f}  {row[1]:8.3f}  {row[2]:8.3f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 9))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("Day 5 — Decoherence: Tuning Between Quantum and Classical",
             fontsize=13, fontweight="bold")

# Panel 1: MSD vs time
ax1 = fig.add_subplot(gs[0, :])
for i, gamma in enumerate(gammas):
    t_arr, msd_arr = results[gamma]
    mask = (t_arr > 0) & (msd_arr > 0)
    ax1.loglog(t_arr[mask], msd_arr[mask],
               color=colors[i], linewidth=2,
               label=f"{labels[i]}   α = {alphas[gamma]:.3f}")

t_shade = np.linspace(0.3, 3.0, 100)
ax1.fill_between(t_shade,
    0.5 * t_shade ** 1.2,
    0.5 * t_shade ** 1.8,
    alpha=0.13, color="#D85A30",
    label="Cancer cell data range  (α = 1.2–1.8)")

ax1.set_xlabel("Time", fontsize=11)
ax1.set_ylabel("MSD  ⟨r²⟩", fontsize=11)
ax1.set_title("MSD vs time for increasing decoherence γ", fontsize=11)
ax1.legend(fontsize=9, loc="upper left")
ax1.spines[["top", "right"]].set_visible(False)

# Panel 2: alpha vs gamma
ax2 = fig.add_subplot(gs[1, 0])
g_arr = np.array(gammas)
a_arr = np.array([alphas[g] for g in gammas])
ax2.plot(g_arr, a_arr, 'o-', color="#1D9E75", linewidth=2, markersize=8)
ax2.axhline(1.0, color="#7F77DD", linestyle="--",
            linewidth=1.5, label="Classical limit  α = 1.0")
ax2.axhspan(1.2, 1.8, alpha=0.13, color="#D85A30",
            label="Cancer cell range")
ax2.set_xlabel("Decoherence rate γ", fontsize=10)
ax2.set_ylabel("Spreading exponent α", fontsize=10)
ax2.set_title("α decreases monotonically with γ", fontsize=10)
ax2.legend(fontsize=9)
ax2.spines[["top", "right"]].set_visible(False)

# Panel 3: off-diagonal coherences at t=1.5
ax3 = fig.add_subplot(gs[1, 1])
t_show = 1.5
for gamma, col in zip([0, 1.0, 20.0], ["#1D9E75", "#D4A843", "#7F77DD"]):
    t_arr, _ = results[gamma]
    idx = np.argmin(np.abs(t_arr - t_show))

    # Re-extract rho at that time point
    sol = solve_ivp(
        lindblad_rhs, [0, t_show], v0,
        method="RK45", args=(gamma,),
        rtol=1e-7, atol=1e-9, max_step=0.02,
        dense_output=True
    )
    rho_t = vec_to_rho(sol.sol(t_show))
    row   = np.abs(rho_t[center_node, :])
    ax3.plot(row, color=col, linewidth=1.5,
             label=f"γ = {gamma}   α = {alphas[gamma]:.2f}")

ax3.set_xlabel("Node index", fontsize=10)
ax3.set_ylabel(f"|ρ(center, node)|  at t={t_show}", fontsize=10)
ax3.set_title("Off-diagonal coherences decay with γ", fontsize=10)
ax3.legend(fontsize=9)
ax3.spines[["top", "right"]].set_visible(False)

plt.savefig("figures/day5_decoherence.pdf", bbox_inches="tight")
print("\nFigure saved to figures/day5_decoherence.pdf")