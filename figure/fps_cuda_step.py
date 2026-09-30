"""Mental-model figure for the CUDA FPS.

Top: what each GPU holds (its block of X and h; a full copy of C).
Middle: one FPS step. Each GPU proposes its farthest row (largest h), the
largest overall becomes x_new, x_new is sent back to every GPU, and every
row lowers h_r if x_new is closer.
Bottom: inside one GPU there is one warp (32 threads) per row; the warp splits
the D coordinates across its threads to compute the one distance d_r.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon

HERE = Path(__file__).resolve().parent

TEAL, COPPER, INK, MUTED = "#31717E", "#C94E3F", "black", "#8A8483"
ROW, PANEL, HBAR, WEDGE = "#CFCACA", "#F4F2F2", "#B9B3B3", "#E6E2E2"
T_A, T_B = "#7FB3B3", "#BFD9D9"
mpl.rcParams.update({"font.size": 16, "figure.facecolor": "white"})

FIG_W, FIG_H = 12.0, 11.7
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_aspect('equal')
ax.axis('off')


def cell(x, y, w, h, color, edge='white', lw=0.5, z=3):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor=edge,
                           linewidth=lw, zorder=z))


def bar(x0, y, n, w, h, colors):
    for k in range(n):
        cell(x0 + k * w, y, w, h, colors[k])


def stack(x, y_top, n, w, h, gap, colors):
    ys = []
    for k in range(n):
        y = y_top - h - k * (h + gap)
        cell(x, y, w, h, colors[k])
        ys.append(y)
    return ys


def text(x, y, s, **kw):
    kw.setdefault('color', INK)
    kw.setdefault('ha', 'center')
    kw.setdefault('va', 'center')
    ax.text(x, y, s, zorder=8, **kw)


def arrow(p0, p1, color=INK, lw=1.3, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=13,
                                 color=color, lw=lw, zorder=6, shrinkA=0,
                                 shrinkB=0, connectionstyle=f'arc3,rad={rad}'))


def band(y0, h, title):
    ax.add_patch(Rectangle((0.3, y0), FIG_W - 0.6, h, facecolor=PANEL,
                           edgecolor='none', zorder=1))
    text(0.45, y0 + h - 0.22, title, ha='left', fontsize=19, fontweight='bold')


def gpu_box(gx, gy, gw, gh, g, n_rows=5, h_vals=None, show_c=True):
    ax.add_patch(Rectangle((gx, gy), gw, gh, facecolor='white', edgecolor=INK,
                           lw=0.9, zorder=2))
    text(gx + 0.1, gy + gh - 0.13, f'GPU {g}', ha='left', fontsize=13,
         color=INK)
    rh, rg = 0.12, 0.05
    ys = stack(gx + 0.15, gy + gh - 0.3, n_rows, 0.95, rh, rg, [ROW] * n_rows)
    for k, y in enumerate(ys):
        if h_vals is None:
            cell(gx + 1.17, y, 0.18, rh, HBAR)
        else:
            top = h_vals[k] == max(h_vals)
            cell(gx + 1.17, y, h_vals[k], rh, COPPER if top else HBAR)
    if show_c:
        stack(gx + 1.65, gy + gh - 0.3, 4, 0.7, rh, rg, [TEAL] * 4)
    return ys


# =============================================================== 1. what each GPU holds
B1_Y, B1_H = 9.35, 2.15
band(B1_Y, B1_H, 'what each GPU holds')
GW, GH = 2.55, 1.3
for g in range(4):
    gx = 0.55 + g * 2.8
    gy = B1_Y + 0.42
    gpu_box(gx, gy, GW, GH, g)
    text(gx + 0.62, gy + 0.11, r'rows of $X$', fontsize=12, color=INK)
    text(gx + 1.26, gy + 0.11, r'$h$', fontsize=12.5, color=INK)
    text(gx + 2.0, gy + 0.11, r'$C$', fontsize=12.5, color=TEAL)
text(3.35, B1_Y + 0.17, r'$X$, $h$: split across GPUs', fontsize=14.5, color=INK)
text(8.95, B1_Y + 0.17, r'$C$ (selected rows): full copy on every GPU',
     fontsize=14.5, color=INK)

# =============================================================== 2. one FPS step
B2_Y, B2_H = 5.75, 3.3
band(B2_Y, B2_H, 'one FPS step')
rng = np.random.default_rng(5)
GW2, GH2 = 2.55, 1.3
gy2 = B2_Y + 0.55
node_y, node_w = B2_Y + B2_H - 0.62, 1.4
node_x = FIG_W / 2 - node_w / 2
cell(node_x, node_y, node_w, 0.2, COPPER)
text(node_x + node_w + 0.15, node_y + 0.1,
     r'$x_{\mathrm{new}}$ = the farthest of the four', ha='left', fontsize=15)
for g in range(4):
    gx = 0.55 + g * 2.8
    hv = rng.uniform(0.2, 0.6, 5)
    hv[rng.integers(5)] = 0.95
    gpu_box(gx, gy2, GW2, GH2, g, h_vals=hv, show_c=False)
    # up: this GPU's farthest row (largest h) is proposed
    arrow((gx + 0.75, gy2 + GH2), (node_x + 0.25 + g * 0.3, node_y),
          color=MUTED, lw=1.1)
    # down: x_new comes back to every GPU
    arrow((node_x + 0.25 + g * 0.3 + 0.12, node_y),
          (gx + GW2 - 0.55, gy2 + GH2), color=COPPER, lw=1.4)
text(0.55, node_y - 0.05, '1.  each GPU proposes\n     its largest $h$', ha='left',
     va='top', fontsize=14.5, color=MUTED)
text(FIG_W - 0.55, node_y - 0.05, r'2.  $x_{\mathrm{new}}$ is sent back' '\n     to every GPU',
     ha='right', va='top', fontsize=14.5, color=COPPER)
text(FIG_W / 2, B2_Y + 0.25,
     r'3.  on every GPU, every row:   $h_r \leftarrow \min\!\left(h_r,\ \|x_r - x_{\mathrm{new}}\|^2\right)$',
     fontsize=16)

# =============================================================== 3. inside one GPU
B3_Y, B3_H = 0.3, 5.2
band(B3_Y, B3_H, 'inside one GPU: one warp (32 threads) per row')
# the GPU box with its rows and warps
gx3, gy3, gw3, gh3 = 0.55, B3_Y + 0.35, 2.55, 4.05
ax.add_patch(Rectangle((gx3, gy3), gw3, gh3, facecolor='white', edgecolor=INK,
                       lw=0.9, zorder=2))
text(gx3 + 0.1, gy3 + gh3 - 0.15, 'GPU 1', ha='left', fontsize=13, color=INK)
N_ROWS, rh3, rg3 = 10, 0.17, 0.16
ys3 = stack(gx3 + 0.85, gy3 + gh3 - 0.42, N_ROWS, 1.2, rh3, rg3, [ROW] * N_ROWS)
for k, y in enumerate(ys3):
    text(gx3 + 0.75, y + rh3 / 2, f'warp {k}', ha='right', fontsize=12,
         color=INK)
    cell(gx3 + 2.12, y, 0.22, rh3, HBAR)
text(gx3 + 1.45, gy3 + 0.14, r'$\vdots$', fontsize=15.5, color=INK)
text(gx3 + 2.23, gy3 + 0.14, r'$h$', fontsize=12.5, color=INK)
R_Z = 3
yz = ys3[R_Z]
ax.add_patch(Rectangle((gx3 + 0.8, yz - 0.04), 1.3, rh3 + 0.08,
                       facecolor='none', edgecolor=INK, lw=1.1, zorder=5))

# zoom into that warp
PX0, PY0, PW, PH = 3.9, B3_Y + 0.25, 7.65, 4.25
ax.add_patch(Polygon([[gx3 + 2.1, yz + rh3 + 0.04], [gx3 + 2.1, yz - 0.04],
                      [PX0, PY0], [PX0, PY0 + PH]], closed=True,
                     facecolor=WEDGE, edgecolor='none', zorder=1))
ax.add_patch(Rectangle((PX0, PY0), PW, PH, facecolor='white', edgecolor=INK,
                       lw=0.9, zorder=2))
text(PX0 + 0.12, PY0 + PH - 0.16, 'warp 3: its row $x_r$ against $x_{\\mathrm{new}}$',
     ha='left', fontsize=14, color=INK)

D, PER, CW, CH = 64, 2, 0.088, 0.28
NT = D // PER
xw = D * CW
x0 = PX0 + (PW - xw) / 2
xc_mid = x0 + xw / 2
y1 = PY0 + PH - 0.8
bar(x0, y1, D, CW, CH, [TEAL] * D)
text(x0 - 0.12, y1 + CH / 2, r'$x_r$', ha='right', fontsize=17)
text(x0 + xw, y1 + CH + 0.12, r'$D$ coordinates', ha='right', fontsize=12.5,
     color=INK)
y1b = y1 - 0.4
bar(x0, y1b, D, CW, CH, [COPPER] * D)
text(x0 - 0.12, y1b + CH / 2, r'$x_{\mathrm{new}}$', ha='right', fontsize=17)
yb = y1b - 0.1
for t in range(NT):
    xa, xb = x0 + t * PER * CW + 0.012, x0 + (t + 1) * PER * CW - 0.012
    ax.plot([xa, xa, xb, xb], [yb, yb - 0.06, yb - 0.06, yb], color=INK,
            lw=0.7, zorder=4)
text(x0 + 0.5 * PER * CW, yb - 0.2, 'thread 0', ha='left', fontsize=12,
     color=INK)
text(x0 + (NT - 0.5) * PER * CW, yb - 0.2, 'thread 31', ha='right',
     fontsize=12, color=INK)
text(xc_mid, yb - 0.2, r'$\cdots$', fontsize=15, color=INK)
y2 = yb - 0.85
tw = PER * CW
bar(x0, y2, NT, tw, CH, [T_A if t % 2 == 0 else T_B for t in range(NT)])
for t in range(NT):
    xc_t = x0 + (t + 0.5) * tw
    arrow((xc_t, yb - 0.33), (xc_t, y2 + CH + 0.02), color=MUTED, lw=0.6)
text(x0 - 0.12, y2 + CH / 2, 'thread $t$:', ha='right', fontsize=13, color=INK)
text(xc_mid, y2 - 0.32,
     r'$\sum_{k\,\in\,t} (x_{r,k} - x_{\mathrm{new},k})^2$', fontsize=15.5)
y3 = y2 - 1.3
ax.add_patch(Polygon([[x0, y2 - 0.62], [x0 + xw, y2 - 0.62],
                      [xc_mid + 0.25, y3 + CH], [xc_mid - 0.25, y3 + CH]],
                     closed=True, facecolor='#E3DEDE', edgecolor='none', zorder=2))
text(xc_mid, (y2 - 0.62 + y3 + CH) / 2, 'add the 32 partial sums', fontsize=14)
cell(xc_mid - 0.25, y3, 0.5, CH, COPPER)
text(xc_mid + 0.4, y3 + CH / 2, r'$d_r$', ha='left', fontsize=17)
arrow((xc_mid, y3 - 0.03), (xc_mid, y3 - 0.36))
text(xc_mid, y3 - 0.52, r'$h_r \leftarrow \min(h_r,\ d_r)$', fontsize=16)

plt.savefig(HERE / "fps_cuda_step.png", dpi=200, facecolor='white')
