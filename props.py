"""Small pieces: the reaching hand, glowing eyes, the tall shadow, blood splatters, candles.
Each replaces one of v12's flat pictures, at the same size and with its origin where the
picture's centre was, so the house can swap them in place."""
import sys, math
import numpy as np
sys.path.insert(0, "/home/claude/hh13/tools")
from sculpt import *
from body import *
OUT = "/home/claude/hh13/out/"

def hand_reach():
    """a dead arm reaching up, fingers clawing. 0.9 m tall, origin at its middle, palm towards +z"""
    W = V(0, 0.05, 0)
    def f(p):
        d, nl = hand_sdf(p, W, V(0, 1, 0.12), V(0, 0, -1), size=1.75, curl=0.42, spread=0.55, finger_len=1.3, claw=0.7, knob=1.07)
        arm = rcone(p, V(0, -0.47, 0.0), W + V(0, -0.02, 0), 0.046, 0.03)
        arm = smin(arm, ellipsoid(p, V(0.008, -0.3, -0.004), (0.046, 0.12, 0.04)), 0.04)       # wasted muscle
        for k, (x0, z0) in enumerate(((-0.022, 0.026), (0.0, 0.031), (0.021, 0.025), (0.03, -0.012))):
            arm = smin(arm, capsule(p, V(x0 * 1.2, -0.3, z0 * 1.15), V(x0 * 0.7, 0.0, z0 * 0.75), 0.006), 0.012)   # tendons
        arm = smin(arm, sphere(p, W + V(0.03, -0.015, -0.01), 0.014), 0.012)   # wrist bone
        arm = smin(arm, sphere(p, W + V(-0.028, -0.012, -0.008), 0.012), 0.012)
        arm = arm - 0.0016 * ridged(p * 40, 2, seed=3)
        # a ragged flap of torn sleeve at the bottom
        sl = smax(rcone(p, V(0, -0.47, 0), V(0, -0.33, 0), 0.06, 0.054), -rcone(p, V(0, -0.5, 0), V(0, -0.3, 0), 0.052, 0.047), 0.003)
        sl = smax(sl, p[:, 1] - tatter(p[:, 1], p[:, 0], p[:, 2], -0.39, 0.05, 2.5, 11), 0.004)
        arm = smin(arm, sl, 0.004)
        return np.minimum(smin(d, arm, 0.02), nl), d, nl, sl
    g = lambda p: f(p)[0]
    v, fa, n = mesh_sdf(g, V(-0.18, -0.5, -0.14), V(0.18, 0.46, 0.18), 0.0055, smooth=1)
    _, dh, dn, ds = f(v)
    lab = ((dn < dh) & (dn < 0.006)).astype(int)
    ao = sdf_ao(g, v, n, 0.01, 4, 1.3)
    col = skin_colour(v, n, ao, base="#a3a596", dark="#5f655c", vein="#4b5670", seed=5)
    col = mixc(col, hexc("#3d0b07"), smoothstep(0.55, 0.85, 0.5 + 0.5 * fbm(v * 9, 3, seed=6)) * smoothstep(0.1, 0.3, v[:, 1]) * 0.8)   # blood on the fingers
    col = mixc(col, hexc("#2e2418"), smoothstep(-0.2, -0.45, v[:, 1]) * 0.7)            # dirt where it came up from
    col[lab == 1] = hexc("#1d1610") * (0.4 + 0.6 * ao[lab == 1])[:, None]
    cl = (ds < 0.003) & (lab == 0)
    col[cl] = cloth_colour(v[cl], ao[cl], base="#3a352d", stain="#1b1712", blood="#3a0705", blood_amt=0.5, seed=12)
    lab[cl] = 2
    prims = split_by_label(v, fa, n, col, lab, {0: {"name": "deadskin", "rough": 0.62}, 1: {"name": "nails", "rough": 0.3}, 2: {"name": "rag", "rough": 0.95, "double": True}}, "hand")
    print("hand_reach", *write_glb(OUT + "hand_reach.glb", prims, "hand_reach"))

def eyes_glow():
    """two eyes glowing in the dark, 0.4 m across, origin between them, looking +z"""
    prims = []
    for s in (-1, 1):
        c = V(s * 0.068, 0, 0)
        v, fa, n = mesh_sdf(lambda p, c=c: sphere(p, c, 0.032), c - 0.04, c + 0.04, 0.0034, smooth=1)
        rel = v - c
        ang = np.degrees(np.arccos(np.clip(rel[:, 2] / 0.032, -1, 1)))
        col = np.tile(hexc("#3a0a05"), (len(v), 1))
        col = mixc(col, hexc("#ff7a1a"), smoothstep(40, 30, ang))
        col = mixc(col, hexc("#ffd36a"), smoothstep(22, 12, ang))
        slit = smoothstep(0.006, 0.003, np.abs(rel[:, 0])) * (ang < 30)
        col = mixc(col, hexc("#050100"), slit)                                            # a slit pupil
        prims.append(Prim(v, fa, n, col, {"name": "eye_glow", "rough": 0.1, "emit": [1.0, 0.45, 0.12]}, "eye"))
    print("eyes_glow", *write_glb(OUT + "eyes_glow.glb", prims, "eyes_glow"))

def shade():
    """the tall shadow: a 3.5 m hunched, starved figure of soot whose legs fray into smoke.
    origin at its middle (1.75 m up), facing +z"""
    o = V(0, -1.75, 0)
    J = {k: v + o for k, v in dict(
        pel=V(0, 1.62, -0.04), chest=V(0, 2.42, 0.1), neck=V(0, 2.86, 0.26), head=V(0.06, 3.06, 0.4),
        shL=V(-0.3, 2.74, 0.16), elL=V(-0.4, 2.02, 0.26), wrL=V(-0.38, 1.24, 0.36),
        shR=V(0.3, 2.72, 0.14), elR=V(0.43, 2.0, 0.12), wrR=V(0.45, 1.22, 0.2),
        hipL=V(-0.12, 1.56, -0.04), knL=V(-0.15, 0.88, 0.1), hipR=V(0.12, 1.56, -0.04), knR=V(0.15, 0.9, 0.0)).items()}
    def f(p):
        d = ribcage(p, J["chest"], (0.17, 0.26, 0.12), ribs=0.006, freq=55)
        d = smin(d, ellipsoid(p, J["pel"] + V(0, 0.04, 0), (0.15, 0.11, 0.1)), 0.09)
        d = smin(d, rcone(p, J["pel"], J["chest"], 0.075, 0.12), 0.08)             # pinched waist
        d = smin(d, capsule(p, J["shL"], J["shR"], 0.05), 0.09)
        d = smin(d, ellipsoid(p, J["chest"] + V(0, 0.28, -0.1), (0.2, 0.1, 0.09)), 0.08)   # hunched back
        d = smin(d, rcone(p, J["chest"] + V(0, 0.32, 0.02), J["neck"], 0.075, 0.045), 0.06)
        d = smin(d, rcone(p, J["neck"], J["head"] + V(0, -0.06, -0.03), 0.045, 0.05), 0.04)
        # a long, tilted head, almost featureless
        q = (p - J["head"]) @ rot_z(math.radians(-18)).T
        hd = ellipsoid(q, V(0, 0, 0), (0.09, 0.15, 0.11))
        hd = smin(hd, ellipsoid(q, V(0, -0.1, 0.03), (0.06, 0.07, 0.07)), 0.04)
        for sx in (-1, 1):
            hd = smax(hd, -ellipsoid(q, V(sx * 0.038, 0.012, 0.1), (0.03, 0.018, 0.03)), 0.012)   # sunken sockets
        d = smin(d, hd, 0.03)
        for sd in "LR":
            d = smin(d, bony_limb(p, J["sh" + sd], J["el" + sd], 0.044, 0.032, 1.3, 0.02), 0.04)
            d = smin(d, bony_limb(p, J["el" + sd], J["wr" + sd], 0.032, 0.022, 1.3, 0.015), 0.015)
            hdn, _ = hand_sdf(p, J["wr" + sd], V(0, -1, 0.15), V(0, 0, -1) if sd == "L" else V(0.3, 0, -1),
                              size=2.3, curl=0.22, spread=0.3, finger_len=2.5, claw=0.0, knob=1.12)
            d = smin(d, hdn, 0.02)
            d = smin(d, bony_limb(p, J["hip" + sd], J["kn" + sd], 0.07, 0.045, 1.2, 0.03), 0.05)
            # below the knee it frays into smoke: tapering tendrils that never quite reach the floor
            rng = np.random.default_rng(3 if sd == "L" else 8)
            for t in range(4):
                a0 = J["kn" + sd]
                end = a0 + V(rng.uniform(-0.22, 0.22), -rng.uniform(0.55, 0.85), rng.uniform(-0.25, 0.2))
                mid = (a0 + end) / 2 + V(rng.uniform(-0.08, 0.08), 0, rng.uniform(-0.08, 0.08))
                d = smin(d, rcone(p, a0, mid, 0.04, 0.022), 0.04)
                d = smin(d, rcone(p, mid, end, 0.022, 0.002), 0.02)
        # boiling, smoky surface
        n1 = fbm(p * 7 + V(0, -p[:, 1].mean() * 0, 0), 3, seed=31)
        return d - 0.012 * n1 - 0.004 * ridged(p * 22, 2, seed=32)
    v, fa, n = mesh_sdf(f, V(-0.65, -1.8, -0.45), V(0.7, 1.55, 0.6), 0.016, smooth=2)
    ao = sdf_ao(f, v, n, 0.03, 4, 1.4)
    col = mixc(hexc("#070608"), hexc("#16151a"), 0.5 + 0.5 * fbm(v * 4, 2, seed=3)) * (0.5 + 0.5 * ao)[:, None]
    col = mixc(col, hexc("#2a2830"), smoothstep(0.3, 1.0, fbm(v * 9, 2, seed=5)) * 0.4)   # ashy grey where the smoke thins
    prims = [Prim(v, fa, n, col, {"name": "soot", "rough": 1.0}, "shade")]
    R = rot_z(math.radians(-18))
    for sx in (-1, 1):     # two pin-points of cold light deep in the sockets
        c = J["head"] + (R @ V(sx * 0.038, 0.012, 0.085))
        ev, ef, en = mesh_sdf(lambda p, c=c: ellipsoid(p, c, (0.012, 0.006, 0.006)), c - 0.02, c + 0.02, 0.0018, smooth=1)
        prims.append(Prim(ev, ef, en, np.tile(hexc("#e8e8e0"), (len(ev), 1)), {"name": "glow", "emit": [0.85, 0.85, 0.8], "rough": 0.2}, "eye"))
    print("shade", *write_glb(OUT + "shade.glb", prims, "shade"))

# ---------------------------------------------------------------- blood
def splat_field(P, seed, wall=False):
    """positive inside the blood. P = (N,2) points on the surface (x, z) or (x, y) for a wall"""
    rng = np.random.default_rng(seed)
    f = np.full(len(P), -1.0)
    def blob(c, r, k=1.0):
        return k * np.exp(-np.sum((P - c) ** 2, 1) / (r * r))
    s = blob(np.zeros(2), 0.32, 1.6)
    for i in range(9):
        a = rng.uniform(0, 2 * np.pi); dd = rng.uniform(0.05, 0.3)
        s = s + blob(np.array([math.cos(a), math.sin(a)]) * dd, rng.uniform(0.1, 0.2), 0.9)
    # streaks flung outwards, and droplets at their ends
    for i in range(16):
        a = rng.uniform(0, 2 * np.pi); L = rng.uniform(0.35, 0.95)
        dirn = np.array([math.cos(a), math.sin(a)])
        for t in np.linspace(0.2, 1.0, 9):
            s = s + blob(dirn * L * t, 0.035 * (1.15 - t) + 0.01, 0.95)
        s = s + blob(dirn * L * rng.uniform(1.05, 1.25), rng.uniform(0.02, 0.04), 1.2)
    for i in range(40):     # spatter
        a = rng.uniform(0, 2 * np.pi); dd = rng.uniform(0.3, 1.2)
        s = s + blob(np.array([math.cos(a), math.sin(a)]) * dd, rng.uniform(0.008, 0.02), 1.3)
    if wall:   # drips running down
        for i in range(12):
            x0 = rng.uniform(-0.35, 0.35); y0 = rng.uniform(-0.25, 0.0); L = rng.uniform(0.3, 0.9)
            for t in np.linspace(0, 1, 26):
                s = s + blob(np.array([x0 + 0.01 * math.sin(t * 9), y0 - L * t]), 0.016 * (1 - 0.5 * t), 1.0)
            s = s + blob(np.array([x0, y0 - L - 0.01]), 0.026, 1.2)
    s = s * (1 + 0.15 * vnoise(np.column_stack([P * 22, np.zeros(len(P))]), seed))
    return s - 0.55

def blood(name, wall=False, size=1.25, res=0.012, seed=1):
    xs = np.arange(-size, size + res, res)
    ys = np.arange(-size - (1.0 if wall else 0), size + res, res)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    P = np.column_stack([X.ravel(), Y.ravel()])
    F = splat_field(P, seed, wall).reshape(X.shape)
    inside = F > 0
    # keep cells with any corner inside
    cell = inside[:-1, :-1] | inside[1:, :-1] | inside[:-1, 1:] | inside[1:, 1:]
    used = np.zeros_like(inside)
    used[:-1, :-1] |= cell; used[1:, :-1] |= cell; used[:-1, 1:] |= cell; used[1:, 1:] |= cell
    idx = -np.ones(X.shape, np.int64); idx[used] = np.arange(used.sum())
    px, py, pf = X[used], Y[used], F[used]
    # pull the outside corners in onto the edge, so the outline is smooth, not stepped
    gy, gx = np.gradient(F, res)
    g = np.stack([gx[used], gy[used]], 1); gn = np.linalg.norm(g, axis=1) + 1e-9
    out = pf < 0
    shift = np.clip(pf[out] / gn[out], -res * 0.9, 0)
    px = px.copy(); py = py.copy()
    px[out] -= shift * g[out, 0] / gn[out]; py[out] -= shift * g[out, 1] / gn[out]
    hgt = 0.004 * np.clip(pf * 2.5, 0, 1) ** 0.5           # a thin, domed pool
    faces = []
    I = np.argwhere(cell)
    for (i, j) in I:
        a, b, c, d = idx[i, j], idx[i, j + 1], idx[i + 1, j], idx[i + 1, j + 1]
        faces += [[a, c, b], [b, c, d]]
    faces = np.array(faces)
    if wall:
        v = np.column_stack([px, py, hgt])              # on a wall, facing +z
    else:
        v = np.column_stack([px, hgt, -py])             # on the floor, facing +y
    faces = convex_outward(v, faces, center=v.mean(0) - (V(0, 0, 1) if wall else V(0, 1, 0)))
    n = vertex_normals(v, faces)
    dark = smoothstep(0.0, 0.6, pf)
    col = mixc(hexc("#6a0a06"), hexc("#2a0302"), dark)    # fresher red at the edges, near black where it's thick
    col = mixc(col, hexc("#8c120b"), smoothstep(0.08, 0.0, np.abs(pf)) * 0.5)
    prims = [Prim(v, faces, n, col, {"name": "blood", "rough": 0.08, "metal": 0.0}, "blood")]
    print(name, *write_glb(OUT + name + ".glb", prims, name))

def candle():
    """a fat church candle, melted down, with a living flame. 0.75 m tall box like the picture; origin at its centre"""
    def wax(p):
        d = rcone(p, V(0, -0.34, 0), V(0, -0.06, 0), 0.042, 0.04)
        d = smax(d, -0.372 - p[:, 1] + 0.0, 0.004)   # flat bottom
        d = smax(d, -(ellipsoid(p, V(0, -0.045, 0), (0.03, 0.012, 0.03))), 0.008)   # hollowed top
        rng = np.random.default_rng(4)
        for i in range(9):   # drips down the side
            a = rng.uniform(0, 2 * np.pi); L = rng.uniform(0.03, 0.14)
            top = V(0.04 * math.cos(a), -0.06, 0.04 * math.sin(a))
            d = smin(d, rcone(p, top, top + V(0, -L, 0), 0.008, 0.006), 0.006)
            d = smin(d, sphere(p, top + V(0, -L, 0), 0.008), 0.004)
        return d + 0.0008 * fbm(p * 120, 2, seed=1)
    v, fa, n = mesh_sdf(wax, V(-0.07, -0.376, -0.07), V(0.07, -0.02, 0.07), 0.0042, smooth=1)
    ao = sdf_ao(wax, v, n, 0.006, 4, 1.2)
    col = mixc(hexc("#e8dcc0"), hexc("#b49a6e"), 0.4 + 0.4 * fbm(v * 40, 2, seed=2)) * (0.5 + 0.5 * ao)[:, None]
    prims = [Prim(v, fa, n, col, {"name": "wax", "rough": 0.4, "emit": [0.12, 0.08, 0.03]}, "wax")]
    wick, wf = tube(np.array([V(0, -0.05, 0), V(0.002, -0.02, 0)]), 0.0022, sides=5)
    prims.append(Prim(wick, wf, vertex_normals(wick, wf), np.tile(hexc("#120d08"), (len(wick), 1)), {"name": "wick", "rough": 0.9}, "wick"))
    flame = lambda p: smin(sphere(p, V(0, -0.012, 0), 0.0135), rcone(p, V(0, -0.006, 0), V(0.002, 0.058, 0), 0.011, 0.0008), 0.02)
    fv, ff, fn = mesh_sdf(flame, V(-0.03, -0.04, -0.03), V(0.03, 0.08, 0.03), 0.002, smooth=1)
    fc = mixc(hexc("#ffd27a"), hexc("#ff7a20"), smoothstep(-0.01, 0.05, fv[:, 1]))
    fc = mixc(fc, hexc("#3a5cff"), smoothstep(-0.012, -0.024, fv[:, 1]) * 0.7)   # blue at the root
    prims.append(Prim(fv, ff, fn, fc, {"name": "flame", "rough": 1.0, "emit": [1.0, 0.62, 0.22], "alpha": 0.85}, "flame"))
    core = lambda p: smin(sphere(p, V(0, -0.014, 0.004), 0.0068), rcone(p, V(0, -0.012, 0.004), V(0, 0.02, 0.004), 0.005, 0.001), 0.01)
    cv_, cf, cn = mesh_sdf(core, V(-0.02, -0.03, -0.02), V(0.02, 0.05, 0.02), 0.0015, smooth=1)
    prims.append(Prim(cv_, cf, cn, np.tile(hexc("#fff4d8"), (len(cv_), 1)), {"name": "flame_core", "rough": 1.0, "emit": [1.0, 0.95, 0.8]}, "core"))
    print("candle", *write_glb(OUT + "candle.glb", prims, "candle"))

if __name__ == "__main__":
    import sys as _s
    todo = _s.argv[1:] or ["hand_reach", "eyes_glow", "shade", "candle", "blood"]
    for nm in todo:
        if nm != "blood": globals()[nm]()
    if "blood" not in todo: raise SystemExit
    blood("blood_floor", wall=False, seed=3); blood("blood_floor_b", wall=False, seed=8); blood("blood_wall", wall=True, seed=5)
