"""THE THING IN THE CIRCLE - what the ritual calls up. It rises out of the attic floor:
waist-up, about 3.5 m to the horn tips, starved and charred, cracks in its hide glowing like
coals, a long goat skull of a face split by a mouth full of needles, arms long enough to reach
across the room. Original design. Faces +z. Origin = where its waist meets the floor."""
import sys, math
import numpy as np
sys.path.insert(0, "/home/claude/hh13/tools")
from sculpt import *
from body import *
OUT = "/home/claude/hh13/out/"

SHL, SHR = V(-0.56, 2.05, 0.02), V(0.56, 2.05, 0.02)
ELL, ELR = V(-1.02, 1.55, 0.45), V(0.98, 1.72, 0.55)
WRL, WRR = V(-0.82, 1.35, 1.35), V(0.72, 1.62, 1.45)
NECK = V(0, 2.38, 0.22)
HC = V(0, 2.8, 0.42)          # head centre: thrust forward, hunched
TILT = rot_x(math.radians(10))
SINK = 0.4                     # how much of the waist is below the floor (cut off, then shifted down) # face tipped down at you

def torso(p):
    d = ribcage(p, V(0, 1.6, 0.05), (0.42, 0.5, 0.3), ribs=0.012, freq=26)
    d = smin(d, rcone(p, V(0, -0.2, 0.0), V(0, 1.15, 0.0), 0.2, 0.26), 0.18)            # starved waist
    d = smax(d, -ellipsoid(p, V(0, 0.75, 0.3), (0.2, 0.28, 0.12)), 0.08)                 # sunken belly
    d = smin(d, capsule(p, SHL, SHR, 0.15), 0.2)
    d = smin(d, ellipsoid(p, V(0, 2.08, -0.16), (0.48, 0.2, 0.22)), 0.15)                # hunched back
    for s in (-1, 1):
        d = smin(d, ellipsoid(p, V(s * 0.25, 1.9, -0.22), (0.17, 0.2, 0.06)), 0.08)       # shoulder blades
        # bony spurs out of the shoulders
        d = smin(d, rcone(p, V(s * 0.55, 2.12, -0.05), V(s * 0.72, 2.45, -0.18), 0.07, 0.008), 0.05)
        d = smin(d, rcone(p, V(s * 0.45, 2.15, -0.12), V(s * 0.52, 2.38, -0.3), 0.05, 0.006), 0.04)
    for i in range(16):                                                                  # spine knuckles
        y = 0.3 + i * 0.12
        z = -0.22 - 0.1 * math.sin(i / 15 * math.pi) + (0.0 if y < 1.9 else -0.04)
        d = smin(d, sphere(p, V(0, y, z), 0.05), 0.05)
    d = smin(d, rcone(p, V(0, 2.1, 0.0), NECK, 0.2, 0.13), 0.12)
    d = smin(d, rcone(p, NECK, HC + V(0, -0.15, -0.12), 0.13, 0.11), 0.1)
    for s in (-1, 1):                                                                    # cords in the neck
        d = smin(d, capsule(p, V(s * 0.12, 2.1, 0.12), V(s * 0.06, 2.55, 0.32), 0.03), 0.04)
    d = smax(d, -(p[:, 1] - SINK), 0.05)     # ends at the floor
    return d

def arms(p):
    d = None
    for sh, el, wr in ((SHL, ELL, WRL), (SHR, ELR, WRR)):
        e = bony_limb(p, sh, el, 0.11, 0.075, 1.3, 0.05)
        e = smin(e, ellipsoid(p, sh + (el - sh) * 0.35, (0.1, 0.1, 0.1)), 0.08)
        e = smin(e, bony_limb(p, el, wr, 0.075, 0.05, 1.35, 0.04), 0.04)
        e = smin(e, ellipsoid(p, el + (wr - el) * 0.25, (0.07, 0.07, 0.07)), 0.06)
        # spines down the back of the forearm
        ax = (wr - el) / np.linalg.norm(wr - el)
        for t in (0.2, 0.45, 0.7):
            b = el + (wr - el) * t + V(0, 0.06, -0.02)
            e = smin(e, rcone(p, b, b + V(0, 0.16, -0.04) - ax * 0.08, 0.025, 0.003), 0.02)
        d = e if d is None else np.minimum(d, e)
    return d

def crack_field(p):
    return np.abs(fbm(p * 3.2, 3, seed=201))

def skin_detail(p):
    # leathery hide split by a network of burning cracks, cut in as grooves
    return 0.014 * smoothstep(0.05, 0.015, crack_field(p)) + 0.006 * fbm(p * 9, 3, seed=202)

def body_sdf(p):
    return smin(torso(p), arms(p), 0.08) + skin_detail(p)

def crack_mask(v):
    return smoothstep(0.035, 0.02, crack_field(v))

def head_sdf(p):
    """a stretched, horned skull with the skin shrunk onto it, jaw dropped open to its chest"""
    q = (p - HC) @ TILT                     # into head space: +z out of the face, +y up
    d = ellipsoid(q, V(0, 0.1, -0.06), (0.17, 0.21, 0.22))                       # cranium
    d = smin(d, ellipsoid(q, V(0, -0.02, 0.06), (0.14, 0.2, 0.13)), 0.08)            # face
    d = smin(d, capsule(q, V(-0.11, 0.09, 0.13), V(0.11, 0.09, 0.13), 0.035), 0.05)  # brow
    for s in (-1, 1):
        d = smin(d, ellipsoid(q, V(s * 0.12, -0.04, 0.08), (0.05, 0.03, 0.06)), 0.03)    # cheekbones
        d = smax(d, -ellipsoid(q, V(s * 0.105, -0.15, 0.08), (0.045, 0.07, 0.05)), 0.03)  # sunken cheeks
        d = smax(d, -ellipsoid(q, V(s * 0.06, 0.03, 0.15), (0.048, 0.052, 0.06)), 0.012)  # deep sockets
        d = smax(d, -ellipsoid(q, V(s * 0.16, 0.09, 0.02), (0.03, 0.05, 0.05)), 0.03)    # hollow temples
    d = smax(d, -ellipsoid(q, V(0, -0.07, 0.15), (0.024, 0.042, 0.05)), 0.008)       # nose hole
    d = smin(d, ellipsoid(q, V(0, -0.145, 0.085), (0.1, 0.05, 0.075)), 0.04)         # upper jaw
    # the lower jaw hangs open, far too far, down to its chest
    for s in (-1, 1):
        d = smin(d, capsule(q, V(s * 0.12, -0.1, -0.03), V(s * 0.065, -0.5, 0.1), 0.03), 0.04)
    d = smin(d, ellipsoid(q, V(0, -0.52, 0.12), (0.08, 0.05, 0.06)), 0.05)           # chin
    d = smin(d, capsule(q, V(-0.06, -0.48, 0.08), V(0.06, -0.48, 0.08), 0.03), 0.03)
    # the mouth: gaping throat between the jaws
    d = smax(d, -ellipsoid(q, V(0, -0.34, 0.08), (0.075, 0.15, 0.11)), 0.02)
    return d + 0.0035 * fbm(q * 24, 3, seed=203) + 0.35 * skin_detail(p)

def throat_mask(v):
    q = (v - HC) @ TILT
    return smoothstep(1.25, 0.95, np.linalg.norm((q - V(0, -0.34, 0.06)) / np.array([0.075, 0.15, 0.11]), axis=1))

def horns():
    """two great ridged horns: back, out, then forward like a ram's, then a spike up"""
    parts = []
    for s in (-1, 1):
        pts, rad = [], []
        for t in np.linspace(0, 1, 40):
            a = t * 3.6
            base = V(s * 0.13, 0.17, 0.0)
            c = base + V(s * (0.05 + 0.32 * t), 0.18 * math.sin(a * 0.9) + 0.25 * t, -0.25 * math.sin(a * 0.75))
            c = c + V(0, 0.55 * t ** 3, 0.25 * t ** 3)
            pts.append(c)
            rad.append(0.075 * (1 - t) ** 0.8 + 0.006)
        pts = np.array(pts) @ TILT.T + HC
        parts.append((pts, np.array(rad)))
    vs, fs, ridges = [], [], []
    for pts, rad in parts:
        # ridged: bump the radius along its length
        L = len(pts)
        import os; HL = os.environ.get("HORN_LO") is not None   # light build: 96 rings x 10 sides instead of 160 x 14
        fine = np.linspace(0, L - 1, 96 if HL else 160)
        P = np.column_stack([np.interp(fine, np.arange(L), pts[:, k]) for k in range(3)])
        rid = np.maximum(0, np.sin(fine * 3.1 + 0.6 * np.sin(fine * 0.37))) ** 3
        R = np.interp(fine, np.arange(L), rad) * (1 + 0.06 * rid)
        v, f = tube(P, R, sides=10 if HL else 14)
        fs.append(f + sum(len(q) for q in vs)); vs.append(v)
        ridges.append(np.repeat(rid, 10 if HL else 14))
    v = np.vstack(vs); f = np.vstack(fs)
    rg = np.concatenate([np.concatenate([r, [0, 0]]) for r in ridges])
    n = vertex_normals(v, f)
    col = mixc(hexc("#2b241d"), hexc("#0d0b09"), smoothstep(0.0, 1.0, rg))
    tipness = smoothstep(2.9, 3.3, v[:, 1])
    col = mixc(col, hexc("#5a4a36"), tipness * 0.5)
    return Prim(v, f, n, col, {"name": "horn", "rough": 0.45}, "horns")

def teeth():
    """needles, top and bottom, round both jaws"""
    rng = np.random.default_rng(7)
    vs, fs = [], []
    def add(a, b, r):
        m = (a + b) / 2 + rng.normal(0, 0.004, 3)
        v, f = tube(np.array([a, m, b]), [r, r * 0.6, 0.0012], sides=6)
        fs.append(f + sum(len(q) for q in vs)); vs.append(v)
    for a in np.linspace(-1.35, 1.35, 17):
        top = V(0.075 * math.sin(a), -0.17, 0.08 + 0.075 * math.cos(a))
        L = rng.uniform(0.06, 0.11) * (1 - 0.35 * abs(a) / 1.35)
        add(top, top + V(rng.normal(0, 0.008), -L, rng.normal(0, 0.008) - 0.01), 0.009)
    for a in np.linspace(-1.3, 1.3, 15):
        bot = V(0.062 * math.sin(a), -0.46, 0.09 + 0.055 * math.cos(a))
        L = rng.uniform(0.05, 0.1) * (1 - 0.35 * abs(a) / 1.3)
        add(bot, bot + V(rng.normal(0, 0.008), L, rng.normal(0, 0.008) - 0.008), 0.008)
    v = np.vstack(vs) @ TILT.T + HC; f = np.vstack(fs)
    col = mixc(np.tile(hexc("#cfc3a2"), (len(v), 1)), hexc("#5a1408"), smoothstep(0.0, 1.0, np.random.default_rng(3).random(len(v))) * 0.3)
    return Prim(v, f, vertex_normals(v, f), col, {"name": "teeth", "rough": 0.25}, "teeth")

def eyes():
    out = []
    for s_ in (-1, 1):
        c = (V(s_ * 0.06, 0.03, 0.115) @ TILT.T) + HC
        v, f, n = mesh_sdf(lambda p, c=c: sphere(p, c, 0.017), c - 0.03, c + 0.03, 0.004, smooth=1)
        out.append(Prim(v, f, n, np.tile(hexc("#ffd66a"), (len(v), 1)), {"name": "eye_fire", "rough": 0.2, "emit": [1.0, 0.6, 0.15]}, "eye"))
    # a fire deep in the throat
    c = (V(0, -0.33, -0.045) @ TILT.T) + HC
    v, f, n = mesh_sdf(lambda p: ellipsoid(p, c, (0.04, 0.08, 0.025)), c - 0.12, c + 0.12, 0.008, smooth=1)
    out.append(Prim(v, f, n, np.tile(hexc("#b8320a"), (len(v), 1)), {"name": "throat_fire", "rough": 0.6, "emit": [0.7, 0.16, 0.02]}, "throat"))
    return out

def hide_colour(v, n, ao, seed):
    col = mixc(hexc("#2a1512"), hexc("#0e0807"), 0.5 + 0.5 * fbm(v * 3, 3, seed=seed))
    col = mixc(col, hexc("#4a2219"), smoothstep(0.55, 0.85, 0.5 + 0.5 * fbm(v * 7, 2, seed=seed + 1)) * 0.5)
    return col * (0.35 + 0.65 * ao)[:, None]

def split_cracks(v, fa, n, col, name, mat):
    cm = crack_mask(v)
    glow = mixc(hexc("#ff6a14"), hexc("#ffc04a"), smoothstep(0.02, 0.005, crack_field(v)))
    col = col.copy(); col[cm > 0.5] = glow[cm > 0.5]
    lab = (cm > 0.5).astype(int)
    return split_by_label(v, fa, n, col, lab, {0: mat, 1: {"name": "ember", "rough": 0.6, "emit": [1.0, 0.38, 0.08]}}, name)

def hands():
    out = []
    for W, el in ((WRL, ELL), (WRR, ELR)):
        fwd = (W - el) / np.linalg.norm(W - el) + V(0, 0.15, 0.2)
        up = V(0, 1, -0.2)
        kw = dict(size=4.6, curl=0.3, spread=0.6, finger_len=1.9, claw=1.0, knob=1.25)
        def f(p, W=W, fwd=fwd, up=up, el=el):
            d, nl = hand_sdf(p, W, fwd, up, **kw)
            return np.minimum(smin(d, bony_limb(p, el + (W - el) * 0.75, W, 0.058, 0.05, 1.2, 0.03), 0.04), nl) + 0.3 * skin_detail(p)
        v, fa, n = mesh_sdf(f, W - 0.65, W + 0.65, 0.015, smooth=1)
        d, nl = hand_sdf(v, W, fwd, up, **kw)
        ao = sdf_ao(f, v, n, 0.02, 4, 1.3)
        col = hide_colour(v, n, ao, 211)
        lab = ((nl < d) & (nl < 0.01)).astype(int)
        col[lab == 1] = mixc(hexc("#16120e"), hexc("#3a2a1c"), smoothstep(0, 1, 0.5 + 0.5 * fbm(v[lab == 1] * 30, 2, seed=5)))[:] if (lab == 1).any() else col[lab == 1]
        hp = split_by_label(v, fa, n, col, lab, {0: {"name": "hide", "rough": 0.55}, 1: {"name": "claws", "rough": 0.25}}, "hand")
        out += hp
    return out

def build():
    v, fa, n = mesh_sdf(body_sdf, V(-1.25, SINK - 0.02, -0.6), V(1.25, 2.6, 1.6), 0.031, smooth=2)
    ao = sdf_ao(body_sdf, v, n, 0.04, 5, 1.4)
    col = hide_colour(v, n, ao, 205)
    prims = split_cracks(v, fa, n, col, "body", {"name": "hide", "rough": 0.55})
    hv, hf, hn = mesh_sdf(head_sdf, HC - V(0.28, 0.7, 0.36), HC + V(0.28, 0.4, 0.36), 0.0105, smooth=2)
    hao = sdf_ao(head_sdf, hv, hn, 0.012, 5, 1.6)
    hcol = hide_colour(hv, hn, hao, 207)
    hcol = mixc(hcol, hexc("#200302"), throat_mask(hv))
    prims += split_cracks(hv, hf, hn, hcol, "head", {"name": "hide", "rough": 0.5})
    prims += [horns(), teeth()] + eyes() + hands()
    prims = [Prim(P.v - V(0, SINK, 0), P.f, P.n, P.c, P.mat, P.name) for P in prims]
    tris, size = write_glb(OUT + "demon.glb", prims, "demon")
    for P in prims: print("  ", P.name, len(P.f))
    print("demon", tris, "tris", size // 1024, "KB")

if __name__ == "__main__":
    build()
