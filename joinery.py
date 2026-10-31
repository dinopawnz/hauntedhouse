"""The catwalk across the front of the gallery and the swing's branch, as models instead of
v12's grey boxes and cylinders. Same measurements as the code that used to build them.
  catwalk.glb       - in world coordinates (origin = the house's origin), sits in the group that lurches
  swing_branch.glb  - in world coordinates too"""
import sys, math
import numpy as np
sys.path.insert(0, "/home/claude/hh13/tools")
from sculpt import *
OUT = "/home/claude/hh13/out/"
rng = np.random.default_rng(11)

def box(c, size, R=None, seg=0.25):
    """a box split into strips along its longest side (so the grain colour has vertices to live on)"""
    c = np.asarray(c, float); hx, hy, hz = np.asarray(size, float) / 2
    n = [max(1, int(math.ceil(2 * h / seg))) for h in (hx, hy, hz)]
    vs, fs = [], []
    def face(axis, sgn):
        a, b = [k for k in range(3) if k != axis]
        na, nb = n[a], n[b]
        ha = (hx, hy, hz)[a]; hb = (hx, hy, hz)[b]; hn = (hx, hy, hz)[axis]
        off = sum(len(q) for q in vs)
        us = np.linspace(-ha, ha, na + 1); ws = np.linspace(-hb, hb, nb + 1)
        U, W = np.meshgrid(us, ws, indexing="ij")
        P = np.zeros((U.size, 3)); P[:, a] = U.ravel(); P[:, b] = W.ravel(); P[:, axis] = sgn * hn
        f = []
        for i in range(na):
            for j in range(nb):
                p0 = i * (nb + 1) + j
                f += [[p0, p0 + nb + 1, p0 + 1], [p0 + 1, p0 + nb + 1, p0 + nb + 2]]
        f = np.array(f)
        # wind outward
        P_ = P
        fn = np.cross(P_[f[:, 1]] - P_[f[:, 0]], P_[f[:, 2]] - P_[f[:, 0]])
        bad = fn[:, axis] * sgn < 0
        f[bad] = f[bad][:, ::-1]
        vs.append(P); fs.append(f + off)
    for ax in range(3):
        for sg in (-1, 1): face(ax, sg)
    v = np.vstack(vs); f = np.vstack(fs)
    if R is not None: v = v @ np.asarray(R).T
    v = v + c
    # flat shading: split vertices per face
    v2 = v[f].reshape(-1, 3); f2 = np.arange(len(v2)).reshape(-1, 3)
    return v2, f2

def wood_colour(v, base, dark, axis_len, seed):
    """grain running along the board"""
    g = 0.5 + 0.5 * np.sin(axis_len * 60 + 4 * fbm(v * np.array([2, 9, 9]), 2, seed=seed))
    k = 0.5 + 0.5 * fbm(v * 3, 3, seed=seed + 1)
    col = mixc(hexc(base), hexc(dark), 0.35 * g + 0.45 * k)
    col = mixc(col, hexc("#0c0705"), smoothstep(0.7, 0.95, 0.5 + 0.5 * fbm(v * 14, 2, seed=seed + 2)) * 0.5)   # rot and stains
    return col

PARTS = []
def part(x, y, z, w, h, d, kind="wood", rx=0.0):
    PARTS.append((V(x, y, z), V(w, h, d), kind, rx))

CW_Y, CW_Z0, CW_Z1, CW_FOOT, CW_TOP, CW_XI, CW_XO = 8.3, -11.75, -9.85, -8.0, -10.6, 6.15, 8.8

def catwalk():
    # the deck as separate planks, running the length of it, with gaps between
    w_all = CW_XI * 2 + 0.02; depth = CW_Z1 - CW_Z0
    nplank = 9; pw = depth / nplank
    for i in range(nplank):
        z = CW_Z0 + pw * (i + 0.5)
        part(0, CW_Y - 0.06 + rng.uniform(-0.006, 0.006), z, w_all, 0.06, pw - 0.012, "plank")
    for x in np.linspace(-CW_XI + 0.3, CW_XI - 0.3, 6):          # joists under the planks
        part(x, CW_Y - 0.17, (CW_Z0 + CW_Z1) / 2, 0.12, 0.16, depth - 0.05, "dark")
    part(0, CW_Y - 0.17, CW_Z1 + 0.02, CW_XI * 2, 0.3, 0.06, "dark")
    for x in (-4.6, -1.5, 1.5, 4.6):
        part(x, CW_Y - 0.65, CW_Z0 + 0.3, 0.26, 0.9, 0.6, "dark")
        part(x, CW_Y - 0.62, CW_Z0 + 0.36, 0.08, 0.95, 0.08, "dark", rx=-40)    # a diagonal brace
    for k in range(9):
        part(-CW_XI + k * CW_XI / 4, CW_Y + 0.5, CW_Z1 - 0.07, 0.12, 1.0, 0.12, "dark")
        part(-CW_XI + k * CW_XI / 4, CW_Y + 1.03, CW_Z1 - 0.07, 0.16, 0.06, 0.16, "dark")       # post caps
    for k in range(32):
        if k % 4:
            # a few balusters broken off or missing
            if rng.random() < 0.08: continue
            hh = 0.9 if rng.random() > 0.1 else rng.uniform(0.35, 0.6)
            part(-CW_XI + k * CW_XI / 16, CW_Y + 0.05 + hh / 2, CW_Z1 - 0.07, 0.045, hh, 0.045, "dark")
    part(0, CW_Y + 1.03, CW_Z1 - 0.07, CW_XI * 2 + 0.1, 0.08, 0.15, "wood")
    part(0, CW_Y + 0.08, CW_Z1 - 0.07, CW_XI * 2, 0.06, 0.08, "dark")
    run = (CW_FOOT - CW_TOP) / 6; rise = (CW_Y - 6.8) / 6; slope = math.atan2(CW_Y - 6.8, CW_FOOT - CW_TOP)
    for s in (-1, 1):
        xc = s * (CW_XI + CW_XO) / 2; wd = CW_XO - CW_XI
        for i in range(5):
            top = 6.8 + (i + 1) * rise
            part(xc, (6.8 + top) / 2 - 0.02, CW_FOOT - (i + 1) * run, wd - 0.04, top - 6.8 - 0.04, run - 0.02, "dark")   # riser block
            part(xc, top - 0.025, CW_FOOT - (i + 1) * run + 0.015, wd, 0.05, run + 0.03, "plank")                         # tread, a lip over the front
        lz0 = CW_FOOT - 5.5 * run
        part(xc, (6.8 + CW_Y) / 2 - 0.03, (lz0 + CW_Z0) / 2, wd - 0.04, CW_Y - 6.8 - 0.06, lz0 - CW_Z0, "dark")
        for k in range(4):
            part(xc, CW_Y - 0.03, CW_Z0 + (lz0 - CW_Z0) * (k + 0.5) / 4, wd, 0.05, (lz0 - CW_Z0) / 4 - 0.01, "plank")
        hx = s * (CW_XI + 0.06); hz = (CW_FOOT + CW_TOP) / 2; hy = (6.8 + CW_Y) / 2 + 0.95
        part(hx, hy, hz, 0.08, 0.08, math.hypot(CW_FOOT - CW_TOP, CW_Y - 6.8) + 0.2, "wood", rx=math.degrees(slope))
        for z, fy in ((CW_FOOT - 0.12, 6.8), (hz, (6.8 + CW_Y) / 2), (CW_TOP + 0.15, CW_Y)):
            part(hx, fy + 0.48, z, 0.11, 0.96, 0.11, "dark")
    vs, fs, cs, off = [], [], [], 0
    for c, size, kind, rx in PARTS:
        R = rot_x(math.radians(rx)) if rx else None
        v, f = box(c, size, R)
        long_axis = int(np.argmax(size))
        ax_len = (v[:, long_axis] - c[long_axis]) if R is None else (v - c) @ (R[:, long_axis])
        base, dark = {"plank": ("#3a271a", "#1e130b"), "wood": ("#33221a", "#1a110a"), "dark": ("#1f140d", "#0d0805")}[kind]
        cross = np.delete(v, long_axis, axis=1).sum(1)
        col = wood_colour(v, base, dark, cross + rng.uniform(0, 3), int(rng.integers(1, 200)))
        vs.append(v); fs.append(f + off); cs.append(col); off += len(v)
    v = np.vstack(vs); f = np.vstack(fs); col = np.vstack(cs)
    prims = [Prim(v, f, vertex_normals(v, f), col, {"name": "old_wood", "rough": 0.85}, "catwalk")]
    print("catwalk", *write_glb(OUT + "catwalk.glb", prims, "catwalk"))

def swing_branch(top=5.0):
    """the bough the swing hangs from: out of the trunk, along, a twig at the end and one going up"""
    limbs = [((40.15, 2.4), (40.5, top + 0.18), -18.0, 0.17, 0.12),
             ((40.35, top + 0.12), (42.0, top - 0.04), -18.0, 0.11, 0.07),
             ((41.95, top - 0.04), (42.45, top + 0.5), -18.0, 0.05, 0.02),
             ((40.45, top + 0.12), (40.2, top + 1.3), -18.15, 0.07, 0.025)]
    vs, fs, off = [], [], 0
    for (ax, ay), (bx, by), z, ra, rb in limbs:
        n = 18
        t = np.linspace(0, 1, n)
        P = np.column_stack([ax + (bx - ax) * t, ay + (by - ay) * t, np.full(n, z)])
        P += np.column_stack([np.zeros(n), np.zeros(n), 0.04 * np.sin(t * 5 + ax)]) * (1 - t)[:, None]   # a little crooked
        R = ra + (rb - ra) * t
        v, f = tube(P, R, sides=12)
        vs.append(v); fs.append(f + off); off += len(v)
    v = np.vstack(vs); f = np.vstack(fs)
    n_ = vertex_normals(v, f)
    bark = 0.5 + 0.5 * ridged(v * np.array([18, 4, 18]), 3, seed=4)
    col = mixc(hexc("#2e2823"), hexc("#14100d"), bark)
    col = mixc(col, hexc("#3c4a2a"), smoothstep(0.6, 0.9, 0.5 + 0.5 * fbm(v * 5, 2, seed=8)) * 0.35)   # moss
    prims = [Prim(v, f, n_, col, {"name": "bark", "rough": 0.95}, "branch")]
    print("swing_branch", *write_glb(OUT + "swing_branch.glb", prims, "swing_branch"))

if __name__ == "__main__":
    catwalk(); swing_branch()
