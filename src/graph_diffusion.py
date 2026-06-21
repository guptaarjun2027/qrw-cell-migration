# graph_diffusion.py
# Day 3: Build a 5x5 grid graph, compute its Laplacian matrix,
# and run classical diffusion using the matrix exponential exp(-Lt).
# This is the foundation for the quantum walk on Day 4 — the only
# change will be exp(-iLt) instead of exp(-Lt).

import numpy as np
import scipy.linalg
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Step 1: Build the grid graph ───────────────────────────────────────────────
# A 5x5 grid: 25 nodes, each connected to its up/down/left/right neighbors.
# This is our first model of tissue — uniform, simple, and easy to visualize.

GRID_SIZE = 5
G = nx.grid_2d_graph(GRID_SIZE, GRID_SIZE)   # nodes are (row, col) tuples

# Relabel nodes from (row,col) tuples to integers 0..24 for matrix indexing
G = nx.convert_node_labels_to_integers(G)
N = G.number_of_nodes()   # 25 nodes

print(f"Graph created: {N} nodes, {G.number_of_edges()} edges")

# ── Step 2: Extract the Laplacian matrix ───────────────────────────────────────
# L = D - A  where D = degree matrix, A = adjacency matrix
# scipy returns a sparse matrix — we convert to a dense numpy array

L = nx.laplacian_matrix(G).toarray().astype(float)

print(f"\nLaplacian matrix shape: {L.shape}")
print(f"\nTop-left 5x5 corner of the Laplacian (so you can inspect it):")
print(np.array2string(L[:5, :5], precision=0, suppress_small=True))
print("\nWhat you should see:")
print("  Diagonal entries = number of connections at that node (2, 3, or 4)")
print("  Off-diagonal entries = -1 where nodes are connected, 0 elsewhere")

# ── Step 3: Define the starting state ─────────────────────────────────────────
# All probability starts at the center node (node 12 in a 5x5 grid)
# This represents a single cancer cell starting at the center of the tissue

center_node = N // 2   # node 12
p0 = np.zeros(N)
p0[center_node] = 1.0  # all probability at center

print(f"\nStarting state: all probability at node {center_node} (center of grid)")

# ── Step 4: Run classical diffusion at several time points ────────────────────
# p(t) = exp(-L * t) @ p0
# We compute this at t = 0, 2, 5, 10, 20 to watch the spreading

time_points = [0, 2, 5, 10, 20]
distributions = {}

for t in time_points:
    propagator = scipy.linalg.expm(-L * t)   # matrix exponential
    p_t = propagator @ p0                     # apply to initial state
    p_t = np.abs(p_t)                        # numerical noise can give tiny negatives
    p_t /= p_t.sum()                         # renormalize to sum to 1
    distributions[t] = p_t.reshape(GRID_SIZE, GRID_SIZE)

# ── Step 5: Compute MSD vs time for classical diffusion ───────────────────────
# MSD(t) = sum over all nodes of (distance² × probability at that node)
# Distance is measured from the center node

# Get (row, col) position of each node
node_positions = np.array([(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)])
center_pos = node_positions[center_node]   # (2, 2)

# Squared distance from center for each node
sq_distances = np.sum((node_positions - center_pos) ** 2, axis=1)

# Compute MSD at fine time resolution for the log-log plot
t_values = np.linspace(0.1, 20, 200)
msd_values = []

for t in t_values:
    propagator = scipy.linalg.expm(-L * t)
    p_t = np.abs(propagator @ p0)
    p_t /= p_t.sum()
    msd = np.dot(sq_distances, p_t)   # weighted average of squared distances
    msd_values.append(msd)

msd_values = np.array(msd_values)

# Fit spreading exponent α on log-log scale
log_t   = np.log(t_values)
log_msd = np.log(msd_values)
alpha, intercept = np.polyfit(log_t, log_msd, 1)

print(f"\nSpreading exponent α = {alpha:.4f}  (classical prediction: 1.0000)")

# ── Step 6: Plot everything ────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8))
fig.suptitle("Day 3 — Classical Diffusion on a 5×5 Grid Graph", 
             fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# --- Row 1: probability heatmaps at t = 0, 5, 20 ----------------------------
plot_times = [0, 5, 20]
for idx, t in enumerate(plot_times):
    ax = fig.add_subplot(gs[0, idx])
    im = ax.imshow(distributions[t], cmap="Blues", vmin=0,
                   vmax=distributions[0].max(), interpolation="nearest")
    ax.set_title(f"t = {t}", fontsize=11)
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("Grid column", fontsize=9)
    ax.set_ylabel("Grid row", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Probability")

    # mark center node
    ax.plot(center_pos[1], center_pos[0], 'r+', markersize=12, 
            markeredgewidth=2, label="Start" if idx == 0 else "")
    if idx == 0:
        ax.legend(fontsize=8, loc="upper right")

# --- Row 2 left: MSD vs time (log-log) ---------------------------------------
ax_msd = fig.add_subplot(gs[1, :2])
ax_msd.loglog(t_values, msd_values, color="#7F77DD", linewidth=2,
              label="Classical diffusion MSD")
fit_line = np.exp(intercept) * t_values ** alpha
ax_msd.loglog(t_values, fit_line, color="#D85A30", linewidth=2,
              linestyle="--", label=f"Power-law fit: α = {alpha:.3f}")
ax_msd.set_xlabel("Time", fontsize=11)
ax_msd.set_ylabel("MSD  ⟨r²⟩", fontsize=11)
ax_msd.set_title("MSD vs time on graph (log-log)", fontsize=11)
ax_msd.legend(fontsize=10)
ax_msd.spines[["top", "right"]].set_visible(False)
ax_msd.annotate(
    f"α ≈ {alpha:.2f} → normal diffusion on graph\n"
    f"(edge effects cause slight deviation from 1.00\n"
    f"because the grid has finite size)",
    xy=(0.97, 0.06), xycoords="axes fraction",
    ha="right", va="bottom", fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#B4B2A9", lw=0.7)
)

# --- Row 2 right: draw the graph itself -------------------------------------
ax_graph = fig.add_subplot(gs[1, 2])
pos = {node: (node_positions[node][1], GRID_SIZE - 1 - node_positions[node][0])
       for node in G.nodes()}
node_colors = ["#D85A30" if n == center_node else "#7F77DD" for n in G.nodes()]
nx.draw(G, pos=pos, ax=ax_graph,
        node_color=node_colors, node_size=120,
        edge_color="#BBBBBB", width=0.8, with_labels=False)
ax_graph.set_title("The 5×5 grid graph\n(red = start node)", fontsize=11)

plt.savefig("figures/day3_graph_diffusion.pdf", bbox_inches="tight")
print("\nFigure saved to figures/day3_graph_diffusion.pdf")
print("\nDay 3 complete. Tomorrow you change exp(-L*t) to exp(-1j*L*t)")
print("and watch the entire behavior transform.")