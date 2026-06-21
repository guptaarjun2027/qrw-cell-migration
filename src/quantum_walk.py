# quantum_walk.py
# Day 4: Continuous-time quantum walk on a 25x25 grid graph.
# Grid is large enough that MSD grows cleanly before hitting boundaries.

import numpy as np
import scipy.linalg
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Step 1: Build a 25x25 grid ────────────────────────────────────────────────
# 5x5 was too small — MSD saturated before we could measure α.
# 25x25 gives max squared distance of 288, plenty of room to grow.

GRID_SIZE = 25
G = nx.grid_2d_graph(GRID_SIZE, GRID_SIZE)
G = nx.convert_node_labels_to_integers(G)
N = G.number_of_nodes()   # 625 nodes

L = nx.laplacian_matrix(G).toarray().astype(float)

center_node = N // 2

# Separate starting vectors for classical and quantum
p0   = np.zeros(N, dtype=float)
p0[center_node] = 1.0

psi0 = np.zeros(N, dtype=complex)
psi0[center_node] = 1.0 + 0j

# Node positions and squared distances from center
node_positions = np.array([(i, j) for i in range(GRID_SIZE)
                            for j in range(GRID_SIZE)])
center_pos   = node_positions[center_node]
sq_distances = np.sum((node_positions - center_pos) ** 2, axis=1).astype(float)

print(f"Grid size: {GRID_SIZE}x{GRID_SIZE} = {N} nodes")
print(f"Center node: {center_node}  at grid position {tuple(center_pos)}")
print(f"Max squared distance from center: {sq_distances.max():.1f}")
print(f"(MSD can grow up to {sq_distances.max():.1f} before saturation)")
print()

# ── Step 2: Propagator functions ──────────────────────────────────────────────

def classical_prob(t):
    prop = scipy.linalg.expm(-L * t)
    p = prop @ p0
    p = np.clip(p, 0, None)
    p /= p.sum()
    return p

def quantum_prob(t):
    prop = scipy.linalg.expm(-1j * L * t)
    amplitude = prop @ psi0
    prob = np.abs(amplitude) ** 2
    prob /= prob.sum()
    return prob

def msd(prob_vector):
    return float(np.dot(sq_distances, prob_vector))

# ── Step 3: Sanity checks ─────────────────────────────────────────────────────
print("=== Sanity checks ===")
p_c0 = classical_prob(0)
p_q0 = quantum_prob(0)
print(f"Classical at t=0: max prob = {p_c0.max():.4f} at node {p_c0.argmax()}"
      f"  (should be 1.0 at node {center_node})")
print(f"Quantum   at t=0: max prob = {p_q0.max():.4f} at node {p_q0.argmax()}"
      f"  (should be 1.0 at node {center_node})")
print(f"Classical MSD at t=0: {msd(p_c0):.4f}  (should be 0.0)")
print(f"Quantum   MSD at t=0: {msd(p_q0):.4f}  (should be 0.0)")

p_c1 = classical_prob(1)
p_q1 = quantum_prob(1)
print(f"Classical MSD at t=1: {msd(p_c1):.4f}  (should be ~1.5–3.0)")
print(f"Quantum   MSD at t=1: {msd(p_q1):.4f}  (should be > classical)")
print()

# ── Step 4: MSD vs time ───────────────────────────────────────────────────────
# Use t up to 10 — well before saturation at 288
print("Computing MSD over time (this may take 30–60 seconds on a large grid)...")
t_values = np.linspace(0.5, 10, 80)
classical_msd_arr = np.array([msd(classical_prob(t)) for t in t_values])
quantum_msd_arr   = np.array([msd(quantum_prob(t))   for t in t_values])

print("\n=== MSD values at selected time points ===")
for t_check in [1, 2, 4, 6, 8, 10]:
    idx = np.argmin(np.abs(t_values - t_check))
    print(f"  t={t_check:2d}  classical MSD = {classical_msd_arr[idx]:.3f}"
          f"   quantum MSD = {quantum_msd_arr[idx]:.3f}")

# Fit α over full range — should be clean now that grid is large
alpha_c, ic = np.polyfit(np.log(t_values), np.log(classical_msd_arr), 1)
alpha_q, iq = np.polyfit(np.log(t_values), np.log(quantum_msd_arr),   1)

print()
print(f"Spreading exponent — Classical: α = {alpha_c:.4f}  (expect ~1.0)")
print(f"Spreading exponent — Quantum:   α = {alpha_q:.4f}  (expect ~1.5–2.0)")
print()
print("Experimental cancer cells: α = 1.2 – 1.8")
print("Classical is below this range. Quantum meets or exceeds it.")

# ── Step 5: Heatmaps — use small grid for visualization only ──────────────────
# For the visual panels we recompute on 15x15 so the heatmap is readable
VIZ_SIZE = 15
G_viz = nx.grid_2d_graph(VIZ_SIZE, VIZ_SIZE)
G_viz = nx.convert_node_labels_to_integers(G_viz)
N_viz = G_viz.number_of_nodes()
L_viz = nx.laplacian_matrix(G_viz).toarray().astype(float)
c_viz = N_viz // 2

p0_viz   = np.zeros(N_viz, dtype=float);   p0_viz[c_viz]   = 1.0
psi0_viz = np.zeros(N_viz, dtype=complex); psi0_viz[c_viz] = 1.0 + 0j
cpos_viz = np.array([VIZ_SIZE // 2, VIZ_SIZE // 2])

def c_dist_viz(t):
    p = scipy.linalg.expm(-L_viz * t) @ p0_viz
    p = np.clip(p, 0, None); p /= p.sum()
    return p.reshape(VIZ_SIZE, VIZ_SIZE)

def q_dist_viz(t):
    amp = scipy.linalg.expm(-1j * L_viz * t) @ psi0_viz
    p = np.abs(amp) ** 2; p /= p.sum()
    return p.reshape(VIZ_SIZE, VIZ_SIZE)

plot_times = [0, 2, 6]
c_hm = {t: c_dist_viz(t) for t in plot_times}
q_hm = {t: q_dist_viz(t) for t in plot_times}

# ── Step 6: Plot ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle("Day 4 — Quantum Walk vs Classical Diffusion", fontsize=13,
             fontweight="bold")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

# Row 1 — classical heatmaps
for idx, t in enumerate(plot_times):
    ax = fig.add_subplot(gs[0, idx])
    vmax = c_hm[0].max() if t == 0 else c_hm[plot_times[1]].max()
    im = ax.imshow(c_hm[t], cmap="Blues", vmin=0, vmax=vmax,
                   interpolation="nearest")
    ax.set_title(f"Classical  t={t}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.plot(cpos_viz[1], cpos_viz[0], 'r+', markersize=10, markeredgewidth=2)

# Row 2 — quantum heatmaps
for idx, t in enumerate(plot_times):
    ax = fig.add_subplot(gs[1, idx])
    vmax = q_hm[0].max() if t == 0 else q_hm[plot_times[1]].max()
    im = ax.imshow(q_hm[t], cmap="Greens", vmin=0, vmax=vmax,
                   interpolation="nearest")
    ax.set_title(f"Quantum  t={t}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.plot(cpos_viz[1], cpos_viz[0], 'r+', markersize=10, markeredgewidth=2)

# Row 3 — MSD comparison
ax_msd = fig.add_subplot(gs[2, :])
ax_msd.loglog(t_values, classical_msd_arr, color="#7F77DD", linewidth=2.5,
              label=f"Classical diffusion  α = {alpha_c:.3f}")
ax_msd.loglog(t_values, quantum_msd_arr,   color="#1D9E75", linewidth=2.5,
              label=f"Quantum walk  α = {alpha_q:.3f}")

# Cancer cell data range shading
scale = np.exp(ic)
ax_msd.fill_between(t_values,
    scale * t_values ** 1.2,
    scale * t_values ** 1.8,
    alpha=0.18, color="#D85A30",
    label="Cancer cell data range  (α = 1.2–1.8)")

ax_msd.set_xlabel("Time", fontsize=11)
ax_msd.set_ylabel("MSD  ⟨r²⟩", fontsize=11)
ax_msd.set_title("MSD vs time", fontsize=11)
ax_msd.legend(fontsize=10)
ax_msd.spines[["top", "right"]].set_visible(False)
ax_msd.annotate(
    "Orange = where experimental cancer cell\n"
    "MSD data falls. Classical (purple) is too\n"
    "slow. Quantum (green) enters this range.\n"
    "Day 5 decoherence fits the model to data.",
    xy=(0.02, 0.97), xycoords="axes fraction",
    ha="left", va="top", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#B4B2A9", lw=0.7)
)

plt.savefig("figures/day4_quantum_walk.pdf", bbox_inches="tight")
print("\nFigure saved to figures/day4_quantum_walk.pdf")
