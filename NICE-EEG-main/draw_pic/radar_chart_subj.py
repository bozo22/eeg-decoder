import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------- global style ----------------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.linewidth": 0.8,
})

# ---------------- data ----------------
data = {
    "Nice-GA (Baseline)": [15.2, 13.9, 14.7, 17.6, 9.0, 16.4, 14.9, 20.3, 14.1, 19.6],
    "SuperNICE (ours)":       [18.5, 19.5, 20.0, 23.5, 14.0, 16.0, 20.0, 24.5, 17.0, 22.5],
}

subjects = [f"S{i}" for i in range(1, 11)]
angles = np.linspace(0, 2*np.pi, len(subjects), endpoint=False).tolist()
angles += angles[:1]                       # close the loop

# ---------------- figure & polar axis -------------
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for model, acc in data.items():
    vals = acc + acc[:1]
    ax.plot(angles, vals, lw=2.4, marker="o", ms=4, label=model)
    ax.fill(angles, vals, alpha=0.08)

# ---------- axis / grid styling ----------
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), subjects, fontsize=12, weight="medium")

min_lim  = 5
outer_lim = np.ceil(max(map(max, data.values()))/5)*5
ax.set_ylim(min_lim, outer_lim+.01)

major = np.arange(10, outer_lim+.1, 5)
minor = major[:-1] + 2.5
ax.set_yticks(major); ax.set_yticks(minor, minor=True)
ax.set_yticklabels([f"{t:.0f} %" for t in major], fontsize=10, weight="bold")
ax.tick_params(axis='y', which='minor', labelsize=0)

ax.yaxis.grid(True,  which='major', lw=1.5, ls='-',  alpha=.65)
ax.yaxis.grid(True,  which='minor', lw=1.1, ls=':',   alpha=.40)
ax.xaxis.grid(True,  ls='--', lw=.7)
ax.spines["polar"].set_visible(False)

# ---------- figure-level title & legend ----------
# fig.suptitle("Top-1 Retrieval Accuracy per Subject", y=0.97,
#              fontsize=16, weight="bold")

fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98),
           ncol=len(data), frameon=False, fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.95])     # leave space for title & legend
plt.show()