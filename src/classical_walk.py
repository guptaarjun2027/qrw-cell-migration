# classical_walk.py
# Implements classical random walk and diffusion on graph networks.
# Used as the baseline comparison for the quantum walk model.
# Day 2 script: simulates 500 independent 1D walkers over 1000 steps,
# computes mean-square displacement (MSD) vs time, and plots:
#   (1) the position distribution at t=1000 (should be Gaussian)
#   (2) MSD vs time on a log-log scale (should give a straight line, slope ~ 1)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Parameters ────────────────────────────────────────────────────────────────

NUM_WALKERS = 500       # number of independent walkers
NUM_STEPS   = 1000      # total time steps per walker
SEED        = 42        # random seed for reproducibility

rng = np.random.default_rng(SEED)

# ── Simulate all walkers at once ───────────────────────────────────────────────
# steps[i, t] = +1 or -1 for walker i at time step t
steps = rng.choice([-1, 1], size=(NUM_WALKERS, NUM_STEPS))

# positions[i, t] = cumulative position of walker i after t steps
# shape: (NUM_WALKERS, NUM_STEPS + 1)  — column 0 is the starting position 0
positions = np.zeros((NUM_WALKERS, NUM_STEPS + 1), dtype=int)
positions[:, 1:] = np.cumsum(steps, axis=1)

# ── Compute MSD vs time ────────────────────────────────────────────────────────
# MSD(t) = mean over all walkers of (position at t)^2
# shape: (NUM_STEPS + 1,)
msd = np.mean(positions ** 2, axis=0)   # average over walkers
time = np.arange(NUM_STEPS + 1)         # t = 0, 1, 2, ..., 1000

# ── Fit spreading exponent α on log-log scale ──────────────────────────────────
# We fit log(MSD) = α * log(t) + const  over t = 1 .. NUM_STEPS
# Classical diffusion predicts α = 1 exactly
fit_start = 1   # skip t=0 (log(0) is undefined)
log_t   = np.log(time[fit_start:])
log_msd = np.log(msd[fit_start:])
alpha, intercept = np.polyfit(log_t, log_msd, 1)
fit_line = np.exp(intercept) * time[fit_start:] ** alpha

# ── Plot ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 5))
fig.suptitle("Classical Random Walk — Day 2 Results", fontsize=14, fontweight="bold", y=1.01)
gs = gridspec.GridSpec(1, 2, wspace=0.35)

# --- Panel 1: position distribution at final time step -----------------------
ax1 = fig.add_subplot(gs[0])
final_positions = positions[:, -1]   # position of each walker at t = NUM_STEPS

ax1.hist(final_positions, bins=40, color="#7F77DD", edgecolor="white",
         linewidth=0.4, density=True, label="Simulated distribution")

# overlay theoretical Gaussian:  N(0, NUM_STEPS)
x_gauss = np.linspace(final_positions.min(), final_positions.max(), 300)
sigma = np.sqrt(NUM_STEPS)
gauss = np.exp(-x_gauss**2 / (2 * NUM_STEPS)) / (sigma * np.sqrt(2 * np.pi))
ax1.plot(x_gauss, gauss, color="#D85A30", linewidth=2, label=f"Gaussian  σ = √t = {sigma:.0f}")

ax1.set_xlabel("Position", fontsize=12)
ax1.set_ylabel("Probability density", fontsize=12)
ax1.set_title(f"Position distribution at t = {NUM_STEPS}", fontsize=12)
ax1.legend(fontsize=10)
ax1.spines[["top", "right"]].set_visible(False)

# --- Panel 2: MSD vs time on log-log scale -----------------------------------
ax2 = fig.add_subplot(gs[1])

ax2.loglog(time[fit_start:], msd[fit_start:],
           color="#7F77DD", linewidth=1.5, alpha=0.85, label="Simulated MSD")
ax2.loglog(time[fit_start:], fit_line,
           color="#D85A30", linewidth=2, linestyle="--",
           label=f"Power-law fit:  α = {alpha:.3f}")

ax2.set_xlabel("Time (steps)", fontsize=12)
ax2.set_ylabel("Mean-square displacement  ⟨x²⟩", fontsize=12)
ax2.set_title("MSD vs time  (log-log scale)", fontsize=12)
ax2.legend(fontsize=10)
ax2.spines[["top", "right"]].set_visible(False)

# annotation explaining what α means
ax2.annotate(
    f"α ≈ {alpha:.2f}  →  normal diffusion\n(classical prediction: α = 1.00)",
    xy=(0.97, 0.06), xycoords="axes fraction",
    ha="right", va="bottom", fontsize=9,
    color="#444441",
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#B4B2A9", lw=0.7)
)

plt.tight_layout()

# ── Save figure ────────────────────────────────────────────────────────────────
output_path = "figures/day2_classical_walk.pdf"
plt.savefig(output_path, bbox_inches="tight")
print(f"Figure saved to {output_path}")

# ── Print summary stats ────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────────")
print(f"  Walkers           : {NUM_WALKERS}")
print(f"  Steps             : {NUM_STEPS}")
print(f"  Final MSD         : {msd[-1]:.1f}  (theoretical: {NUM_STEPS:.1f})")
print(f"  Spreading exponent: α = {alpha:.4f}  (classical prediction: 1.0000)")
print(f"  Std of positions  : {np.std(final_positions):.2f}  (theoretical: {np.sqrt(NUM_STEPS):.2f})")
print("─────────────────────────────────────────────────────────────")

plt.show()