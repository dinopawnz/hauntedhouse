"""THE DOLLS - life-size porcelain Victorian dolls. Cracked glazed faces, one glass eye staring,
the other socket broken in, a tiny painted mouth just open over tiny teeth, ringlets of dirty hair.
Original design. Faces +z.
  doll_body  (origin on the floor; the neck socket is at NECK)
  doll_head  (origin = the neck pivot, so it can turn and tilt)"""
import sys, math
import numpy as np
sys.path.insert(0, "/home/claude/hh13/tools")
from sculpt import *
from body import *

NECK = V(0, 1.06, 0.0)

# ------------------------------------------------------------------ head (local to the neck)
HC = V(0, 0.14, 0.0)   # head centre

def head_sdf(p):
    d = ellipsoid(p, HC, (0.098, 0.112, 0.102))                                   # big round doll head
    d = smin(d, ellipsoid(p, HC + V(0, -0.055, 0.03), (0.075, 0.06, 0.07)), 0.04)   # chubby cheeks and chin
    for s in (-1, 1):
        d = smin(d, sphere(p, HC + V(s * 0.05, -0.045, 0.06), 0.034), 0.03)        # cheeks
        d = smax(d, -ellipsoid(p, HC + V(s * 0.038, 0.0, 0.088), (0.028, 0.026, 0.03)), 0.006)   # big eye sockets
    d = smin(d, sphere(p, HC + V(0, -0.025, 0.102), 0.011), 0.01)                  # button nose
    d = smax(d, -ellipsoid(p, HC + V(0, -0.075, 0.095), (0.012, 0.005, 0.015)), 0.003)   # tiny open mouth
    d = smin(d, rcone(p, V(0, -0.01, 0), HC + V(0, -0.08, -0.01), 0.032, 0.035), 0.02)   # neck
    # the broken piece: a chunk knocked out of the left side of the face, hollow behind
    br = ellipsoid(p, HC + V(0.055, 0.035, 0.085), (0.032, 0.03, 0.03)) + 0.004 * vnoise(p * 300, 3)
    d = smax(d, -br, 0.0015)
    return d

def head_shell(p):
    """porcelain is a shell: hollow inside so the broken hole is dark and deep"""
    d = head_sdf(p)
    inner = ellipsoid(p, HC, (0.088, 0.1, 0.09))
    return smax(d, -inner, 0.002)

def cracks(v, seed=4):
    """thin dark crack lines that branch across the glaze"""
    r = np.abs(vnoise(v * np.array([55, 42, 55]), seed)) + 0.5 * np.abs(vnoise(v * 110, seed + 1))
    near_break = smoothstep(0.08, 0.02, np.linalg.norm(v - (HC + V(0.055, 0.035, 0.085)), axis=1))
    return np.maximum(smoothstep(0.03, 0.008, r) * 0.9, smoothstep(0.06, 0.02, r) * near_break)

def ringlets(seed=3):
    rng = np.random.default_rng(seed)
    parts = []
    hc = HC + V(0, 0.01, -0.01); hr = np.array([0.104, 0.118, 0.108])
    for k in range(120):
        if len(parts) >= 34: break
        th = rng.uniform(0, 2 * np.pi); ph = rng.uniform(0.25, 1.35)
        dirn = np.array([math.sin(ph) * math.cos(th), math.cos(ph), math.sin(ph) * math.sin(th)])
        root = hc + dirn * hr
        if root[2] > -0.015 and abs(root[0]) < 0.075: continue          # nothing over the face
        out = np.array([dirn[0], 0, dirn[2]]); out /= max(np.linalg.norm(out), 1e-6)
        side = np.cross(out, [0, 1, 0])
        L = rng.uniform(0.16, 0.3); turns = rng.uniform(3, 5); rr = rng.uniform(0.012, 0.018)
        import os; CL = int(os.environ.get("CURL_LO", "0"))   # light build: 1 = 16 points x 4 sides per ringlet, 2 = 12 x 3 (default 32 x 5)
        n = {0: 32, 1: 16}.get(CL, 12)
        t = np.linspace(0, 1, n)
        ang = t * turns * 2 * np.pi + rng.uniform(0, 6.28)
        axis = root[None, :] + np.outer(t, V(0, -L, 0)) + np.outer(t ** 0.5, out * 0.035)
        coil = axis + rr * (np.cos(ang)[:, None] * out[None, :] + np.sin(ang)[:, None] * side[None, :]) * np.minimum(1, t * 6)[:, None]
        v, f = tube(coil, 0.0062 * (1 - 0.5 * t), sides={0: 5, 1: 4}.get(CL, 3))
        parts.append((v, f))
    v, f = merge(parts)
    col = mixc(hexc("#7a5b33"), hexc("#3d2c18"), 0.5 + 0.5 * fbm(v * 30, 2, seed=5))           # dirty, faded blonde-brown
    return Prim(v, f, vertex_normals(v, f), col, {"name": "hair", "rough": 0.55}, "hair")

def bow():
    """a faded ribbon bow on the side of the head"""
    c = HC + V(-0.07, 0.085, -0.02)
    def f(p):
        d = ellipsoid(p, c, (0.011, 0.01, 0.01))
        for s in (-1, 1):
            d = smin(d, ellipsoid(p, c + V(s * 0.021, 0.005, 0), (0.018, 0.013, 0.006)), 0.006)
        return d
    v, fa, n = mesh_sdf(f, c - 0.08, c + 0.08, 0.004, smooth=1)
    col = mixc(hexc("#6b2a2e"), hexc("#2b1112"), 0.5 + 0.5 * fbm(v * 40, 2, seed=8))
    return Prim(v, fa, n, col, {"name": "ribbon", "rough": 0.6}, "bow")

def head_prims():
    v, fa, n = mesh_sdf(head_shell, V(-0.12, -0.02, -0.12), V(0.12, 0.27, 0.13), 0.0046, smooth=2)
    ao = sdf_ao(head_sdf, v, n, 0.006, 5, 1.2)
    col = np.tile(hexc("#e9e2d6"), (len(v), 1))
    col = mixc(col, hexc("#c9bba3"), smoothstep(0.2, 0.9, 0.5 + 0.5 * fbm(v * 14, 3, seed=2)) * 0.6)   # yellowed, grimy glaze
    col = mixc(col, hexc("#9a8a70"), smoothstep(0.6, 0.95, 0.5 + 0.5 * fbm(v * 9, 2, seed=9)) * 0.35)   # grime
    # dark tear stains running down from both eyes
    for s, x0 in ((-1, -0.038), (1, 0.04)):
        for k, dx in enumerate((-0.006, 0.004, 0.011)):
            xx = x0 + dx * s + 0.004 * np.sin(v[:, 1] * 90 + k)
            line = smoothstep(0.0045, 0.0015, np.abs(v[:, 0] - xx))
            below = smoothstep(HC[1] - 0.015, HC[1] - 0.03, v[:, 1]) * smoothstep(HC[1] - 0.13 + k * 0.02, HC[1] - 0.06, v[:, 1])
            col = mixc(col, hexc("#3b2a1c"), line * below * (v[:, 2] > 0.02) * (0.85 - 0.2 * k))
    for s in (-1, 1):   # rosy painted cheeks, faded
        dc = np.linalg.norm(v - (HC + V(s * 0.052, -0.045, 0.075)), axis=1)
        col = mixc(col, hexc("#d48b84"), smoothstep(0.035, 0.0, dc) * 0.55)
    # painted lips: a tiny dark red bow round the open mouth
    dm = np.linalg.norm((v - (HC + V(0, -0.075, 0.098))) / np.array([1.0, 1.9, 1.0]), axis=1)
    col = mixc(col, hexc("#5e0f12"), smoothstep(0.015, 0.0095, dm))
    # painted lashes and brows
    for s in (-1, 1):
        de = np.linalg.norm((v - (HC + V(s * 0.038, 0.02, 0.105))) / np.array([1.0, 3.5, 1.0]), axis=1)
        col = mixc(col, hexc("#2a1a12"), smoothstep(0.034, 0.028, de) * smoothstep(0.022, 0.03, de) * (v[:, 1] > HC[1] + 0.012))
        db = np.linalg.norm((v - (HC + V(s * 0.04, 0.05, 0.1))) / np.array([1.0, 6.0, 1.0]), axis=1)
        col = mixc(col, hexc("#5a3a26"), smoothstep(0.03, 0.022, db) * 0.8)
    # cracks, and the broken edge, and the black inside of the shell
    cr = cracks(v)
    col = mixc(col, hexc("#2b2018"), cr * 0.9)
    inside = head_sdf(v) < -0.004
    col[inside] = hexc("#0b0806")
    col = col * (0.45 + 0.55 * ao)[:, None]
    lab = inside.astype(int)
    prims = split_by_label(v, fa, n, col, lab, {0: {"name": "porcelain", "rough": 0.22}, 1: {"name": "hollow", "rough": 0.9}}, "head")
    # one glass eye (the right one); the left socket is part of the break
    c = HC + V(-0.038, 0.0, 0.07)
    ev, ef, en = mesh_sdf(lambda p: sphere(p, c, 0.0235), c - 0.03, c + 0.03, 0.0022, smooth=1)
    ang = np.degrees(np.arccos(np.clip((ev - c)[:, 2] / 0.0235, -1, 1)))
    ecol = np.tile(hexc("#e8e4da"), (len(ev), 1))
    ecol = mixc(ecol, hexc("#4f7d9a"), smoothstep(30, 26, ang))            # pale glass-blue iris
    ecol = mixc(ecol, hexc("#253f52"), smoothstep(26, 20, ang) * smoothstep(12, 18, ang))
    ecol = mixc(ecol, hexc("#030303"), smoothstep(13, 11, ang))            # big black pupil
    prims.append(Prim(ev, ef, en, ecol, {"name": "glass_eye", "rough": 0.04, "emit": [0.03, 0.04, 0.05]}, "eye"))
    # in the broken socket: just a dark glint deep inside
    c2 = HC + V(0.04, 0.0, 0.06)
    ev, ef, en = mesh_sdf(lambda p: sphere(p, c2, 0.006), c2 - 0.01, c2 + 0.01, 0.0015, smooth=1)
    prims.append(Prim(ev, ef, en, np.tile(hexc("#ff3020"), (len(ev), 1)), {"name": "glint", "rough": 0.1, "emit": [0.5, 0.06, 0.03]}, "glint"))
    # tiny teeth in the open mouth
    tv, tf = [], []
    for i in range(6):
        x = -0.0075 + i * 0.003
        pth = np.array([V(x, HC[1] - 0.071, HC[2] + 0.094), V(x, HC[1] - 0.077, HC[2] + 0.095)])
        vv, ff = tube(pth, [0.0014, 0.0011], sides=5); tf.append(ff + sum(len(a) for a in tv)); tv.append(vv)
    tv = np.vstack(tv); tf = np.vstack(tf)
    prims.append(Prim(tv, tf, vertex_normals(tv, tf), np.tile(hexc("#e8dfc4"), (len(tv), 1)), {"name": "teeth", "rough": 0.3}, "teeth"))
    prims += [ringlets(), bow()]
    return prims

# ------------------------------------------------------------------ body
def legs(p):
    d = None
    for s in (-1, 1):
        e = bony_limb(p, V(s * 0.06, 0.62, 0), V(s * 0.065, 0.32, 0.01), 0.04, 0.034, 1.12)
        e = smin(e, bony_limb(p, V(s * 0.065, 0.32, 0.01), V(s * 0.065, 0.07, 0.0), 0.034, 0.026, 1.12), 0.01)
        d = e if d is None else np.minimum(d, e)
    return d

def shoes(p):
    d = None
    for s in (-1, 1):
        e = ellipsoid(p, V(s * 0.065, 0.035, 0.035), (0.04, 0.035, 0.075))
        e = smax(e, -p[:, 1] - 0.0, 0.005)
        e = smin(e, capsule(p, V(s * 0.03, 0.07, 0.03), V(s * 0.1, 0.07, 0.03), 0.006), 0.004)   # the strap
        d = e if d is None else np.minimum(d, e)
    return d

def dress(p):
    # bodice
    d = ellipsoid(p, V(0, 0.9, 0.005), (0.12, 0.15, 0.09))
    d = smin(d, rcone(p, V(0, 1.0, 0), NECK + V(0, -0.01, 0), 0.07, 0.036), 0.04)
    # puffed sleeves
    for s in (-1, 1):
        d = smin(d, ellipsoid(p, V(s * 0.14, 0.96, 0.0), (0.065, 0.06, 0.06)), 0.03)
    # full skirt with flounces and a lace hem
    q = p - V(0, 0.78, 0)
    ang = np.arctan2(q[:, 0], q[:, 2])
    t = np.clip(-q[:, 1] / 0.45, 0, 1)
    rad = 0.13 + 0.17 * t ** 0.8 + 0.012 * np.sin(ang * 14) * t + 0.008 * np.sin(q[:, 1] * 70) * smoothstep(0.6, 1.0, t)
    r_xz = np.sqrt(q[:, 0] ** 2 + (q[:, 2] * 1.1) ** 2)
    skirt = np.maximum(r_xz - rad, np.maximum(q[:, 1] - 0.04, -(q[:, 1] + 0.46)))
    d = smin(d, skirt, 0.04)
    # lace frill under the hem
    fr = np.maximum(np.abs(r_xz - (rad + 0.01)) - 0.006, np.abs(q[:, 1] + 0.47) - 0.025)
    d = np.minimum(d, fr + 0.004 * np.maximum(0, np.sin(ang * 60)))
    # pinafore bib and apron across the front
    ap = rbox(p, V(0, 0.65, 0.115 + 0.12 * 0.0), (0.12, 0.23, 0.01), 0.008)
    ap = smax(ap, -(p[:, 2] - (np.sqrt(np.maximum(0, rad ** 2 - q[:, 0] ** 2)) / 1.1 - 0.005)), 0.0) if False else ap
    d = d + 0.002 * fbm(p * np.array([20, 8, 20]), 3, seed=12)
    return d

def apron(p):
    q = p - V(0, 0.78, 0)
    t = np.clip(-q[:, 1] / 0.45, 0, 1)
    rad = 0.13 + 0.17 * t ** 0.8
    r_xz = np.sqrt(q[:, 0] ** 2 + (q[:, 2] * 1.1) ** 2)
    shell = r_xz - (rad + 0.02)                      # solid over the skirt (a thin shell meshes full of holes)
    front = smax(shell, -(p[:, 2] - 0.02), 0.01)
    front = smax(front, np.abs(p[:, 0]) - (0.08 + 0.08 * t), 0.01)
    front = smax(front, np.maximum(q[:, 1] - 0.03, -(q[:, 1] + 0.38)), 0.005)
    bib = rbox(p, V(0, 0.93, 0.075), (0.07, 0.09, 0.02), 0.008)
    return np.minimum(front, bib) + 0.0015 * fbm(p * 30, 2, seed=13)

def doll_hands():
    out = []
    for s in (-1, 1):
        W = V(s * 0.2, 0.6, 0.06)
        def f(p, W=W, s=s):
            d, nl = hand_sdf(p, W, V(0.1 * s, -1, 0.25), V(s, 0, 0.1), size=0.85, curl=0.25, spread=0.15, finger_len=0.9, claw=0.0)
            arm = bony_limb(p, V(s * 0.17, 0.92, 0.0), V(s * 0.19, 0.76, 0.02), 0.026, 0.022, 1.2)          # upper arm (in the sleeve)
            arm = smin(arm, bony_limb(p, V(s * 0.19, 0.76, 0.02), W, 0.022, 0.018, 1.25), 0.006)
            arm = np.minimum(arm, sphere(p, V(s * 0.19, 0.76, 0.02), 0.027))                                   # ball joint elbow
            arm = np.minimum(arm, sphere(p, W, 0.021))                                                         # ball joint wrist
            return np.minimum(d, arm)
        v, fa, n = mesh_sdf(f, V(s * 0.2 - 0.12, 0.44, -0.08), V(s * 0.2 + 0.12, 0.95, 0.2), 0.0055, smooth=1)
        ao = sdf_ao(f, v, n, 0.005, 4, 1.2)
        col = mixc(hexc("#e6ded0"), hexc("#b8a88c"), 0.5 + 0.5 * fbm(v * 15, 2, seed=3)) * (0.45 + 0.55 * ao)[:, None]
        col = mixc(col, hexc("#2b2018"), cracks(v + 0.3, 6) * 0.8)
        out.append(Prim(v, fa, n, col, {"name": "porcelain", "rough": 0.22}, "hands"))
    return out

def body_prims():
    parts = [("dress", dress), ("apron", apron), ("legs", legs), ("shoes", shoes)]
    f, v, fa, n, lab = mesh_parts(parts, (V(-0.36, -0.01, -0.33), V(0.36, 1.09, 0.33)), 0.0105)
    ao = sdf_ao(f, v, n, 0.012, 5, 1.3)
    cols = {
        0: cloth_colour(v, ao, base="#8e7d86", stain="#3b2f2c", blood="#3a0706", blood_amt=0.25, seed=21, dirt_y=0.5),   # faded mauve dress
        1: cloth_colour(v, ao, base="#d3cbbb", stain="#6a5a42", blood="#3a0706", blood_amt=0.35, seed=22, dirt_y=0.5),   # once-white pinafore
        2: mixc(hexc("#e0d8cc"), hexc("#8a7a64"), smoothstep(0.3, 0.05, v[:, 1]))[:] * (0.4 + 0.6 * ao)[:, None],       # stockings
        3: (hexc("#141114") + 0.06 * hexc("#ffffff") * (0.5 + 0.5 * fbm(v * 30, 2, seed=4))[:, None]) * (0.5 + 0.5 * ao)[:, None],
    }
    col = np.zeros((len(v), 3))
    for k, c in cols.items():
        col[lab == k] = (c if c.ndim == 2 else np.tile(c, (len(v), 1)))[lab == k]
    mats = {0: {"name": "dress", "rough": 0.88}, 1: {"name": "lace", "rough": 0.9}, 2: {"name": "stockings", "rough": 0.8}, 3: {"name": "shoes", "rough": 0.25}}
    return split_by_label(v, fa, n, col, lab, mats, "doll") + doll_hands()

if __name__ == "__main__":
    hp = head_prims()
    t1, s1 = write_glb("/home/claude/hh13/out/doll_head.glb", hp, "doll_head")
    bp = body_prims()
    t2, s2 = write_glb("/home/claude/hh13/out/doll_body.glb", bp, "doll_body")
    full = bp + [Prim(P.v + NECK, P.f, P.n, P.c, P.mat, P.name) for P in hp]
    t3, s3 = write_glb("/home/claude/hh13/out/doll.glb", full, "doll")
    print("doll_head", t1, "doll_body", t2, "doll", t3, "tris", s3 // 1024, "KB")
