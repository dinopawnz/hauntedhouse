"""THE GRINNER - a gaunt, too-tall woman with an impossibly wide, too-full smile.
Original design. Head is separate (origin at the base of the neck) so it can tilt and turn.
Faces +z. Units: metres."""
import sys, math
import numpy as np
sys.path.insert(0, "/home/claude/hh13/tools")
from sculpt import *

RNG = np.random.default_rng(13)

# ---------------------------------------------------------------- the head
HEAD_C = V(0, 0.185, 0.0)

def mouth_curve(x):
    """centre line of the grin: corners pulled up and back to the cheekbones"""
    u = np.clip(np.abs(x) / 0.072, 0, 1.2)
    return 0.112 + 0.050 * u ** 2.3

def mouth_half(x):
    u = np.clip(np.abs(x) / 0.074, 0, 1)
    return 0.0035 + 0.0125 * np.sqrt(np.maximum(0, 1 - u ** 2.2))

def head_base(p):
    d = ellipsoid(p, V(0, 0.205, -0.012), (0.086, 0.104, 0.098))              # cranium
    d = smin(d, ellipsoid(p, V(0, 0.138, 0.030), (0.067, 0.086, 0.075)), 0.045)  # face
    d = smin(d, ellipsoid(p, V(0, 0.094, 0.036), (0.054, 0.042, 0.060)), 0.03)   # long, narrow jaw
    d = smin(d, ellipsoid(p, V(0, 0.072, 0.072), (0.026, 0.020, 0.02)), 0.02)    # chin
    for s in (-1, 1):
        d = smin(d, ellipsoid(p, V(s * 0.052, 0.166, 0.060), (0.022, 0.015, 0.018)), 0.02)   # cheekbones
        d = smin(d, ellipsoid(p, V(s * 0.078, 0.17, -0.01), (0.012, 0.026, 0.016)), 0.01)    # ears (mostly hidden)
    d = smin(d, capsule(p, V(-0.042, 0.207, 0.083), V(0.042, 0.207, 0.083), 0.011), 0.02)    # brow ridge
    # hollow temples and sunken cheeks (gaunt)
    for s in (-1, 1):
        d = smax(d, -ellipsoid(p, V(s * 0.085, 0.205, 0.035), (0.016, 0.03, 0.03)), 0.015)
        d = smax(d, -ellipsoid(p, V(s * 0.064, 0.126, 0.068), (0.021, 0.026, 0.018)), 0.012)
    # nose: long and thin, nostrils flared
    d = smin(d, capsule(p, V(0, 0.196, 0.094), V(0, 0.157, 0.114), 0.0085), 0.012)
    d = smin(d, ellipsoid(p, V(0, 0.151, 0.104), (0.017, 0.0095, 0.012)), 0.008)
    for s in (-1, 1):
        d = smax(d, -sphere(p, V(s * 0.0072, 0.146, 0.111), 0.0042), 0.003)
    # neck: thin, tendons standing out
    d = smin(d, rcone(p, V(0, -0.02, -0.012), V(0, 0.105, 0.0), 0.040, 0.036), 0.03)
    for s in (-1, 1):
        d = smin(d, capsule(p, V(s * 0.03, 0.0, 0.022), V(s * 0.012, 0.09, 0.048), 0.007), 0.012)
    return d

def surface_z(x, y, f=head_base):
    """march along -z to find where the face is at (x, y)"""
    z = np.full(np.shape(x), 0.25)
    for _ in range(60):
        pts = np.column_stack([np.ravel(x), np.ravel(y), np.ravel(z)])
        dd = f(pts).reshape(np.shape(x))
        z = z - dd * 0.9                     # (signed: steps back out if it overshoots)
    return z

MX = np.linspace(-0.076, 0.076, 61)
MY = mouth_curve(MX)
MZ = surface_z(MX, MY)
MH = mouth_half(MX)

def head_sdf(p):
    d = head_base(p)
    # eye sockets: deep, wide, ringed
    for s in (-1, 1):
        d = smax(d, -ellipsoid(p, V(s * 0.033, 0.181, 0.091), (0.023, 0.017, 0.02)), 0.006)
    for s in (-1, 1):   # thin lids, pulled back: too much white showing above and below
        d = smin(d, ellipsoid(p, V(s * 0.033, 0.1945, 0.083), (0.0175, 0.0042, 0.0115)), 0.004)
        d = smin(d, ellipsoid(p, V(s * 0.033, 0.1655, 0.083), (0.016, 0.0036, 0.0105)), 0.004)
    # the grin: carve the mouth along the curve
    cav = None
    for x, y, z, hh in zip(MX, MY, MZ, MH):
        e = ellipsoid(p, V(x, y, z - 0.006), (0.0075, hh, 0.028))
        cav = e if cav is None else np.minimum(cav, e)
    d = smax(d, -cav, 0.003)
    # thin lips pulled tight along the edges
    up = np.column_stack([MX, MY + MH + 0.0012, MZ - 0.002])
    lo = np.column_stack([MX, MY - MH - 0.0012, MZ - 0.002])
    lips = None
    for i in range(len(MX) - 1):
        for line in (up, lo):
            e = capsule(p, line[i], line[i + 1], 0.0036)
            lips = e if lips is None else np.minimum(lips, e)
    d = smin(d, lips, 0.004)
    # stretched-skin creases from the nose to the corners of the grin
    for s in (-1, 1):
        d = smax(d, -capsule(p, V(s * 0.018, 0.158, 0.106), V(s * 0.07, 0.15, 0.076), 0.0022), 0.004)
        d = smax(d, -capsule(p, V(s * 0.03, 0.135, 0.103), V(s * 0.078, 0.164, 0.07), 0.0018), 0.004)
    # skin texture: pores, fine wrinkles
    q = p * 70.0
    d = d + 0.0011 * fbm(q, 3, seed=3) - 0.0006 * (1 - ridged(p * 45.0, 2, seed=8))
    return d

def eye_mesh():
    """wide, wet, bloodshot eyes with pin-prick pupils staring straight ahead"""
    parts = []
    for s in (-1, 1):
        c = V(s * 0.033, 0.180, 0.077)
        f = lambda p, c=c: sphere(p, c, 0.0138)
        v, fa, n = mesh_sdf(f, c - 0.02, c + 0.02, 0.0016, smooth=1)
        rel = v - c
        ang = np.degrees(np.arccos(np.clip(rel[:, 2] / 0.0138, -1, 1)))  # 0 = straight ahead
        col = np.tile(hexc("#bdb5a0"), (len(v), 1))
        vein = 1 - ridged(v * 900, 3, seed=11 + s)
        col = mixc(col, hexc("#9a1d1d"), smoothstep(0.55, 0.95, vein) * smoothstep(10, 55, ang))
        col = mixc(col, hexc("#c99a8a"), smoothstep(60, 90, ang))              # pinker at the edge
        iris = smoothstep(15, 12, ang)
        col = mixc(col, hexc("#57543c"), iris)                                  # small dull iris
        col = mixc(col, hexc("#2a2818"), smoothstep(13, 11, ang) * smoothstep(7, 9, ang))
        col = mixc(col, hexc("#050403"), smoothstep(4.4, 3.4, ang))             # pin-prick pupil
        parts.append(Prim(v, fa, n, col, {"name": "eye", "rough": 0.08, "emit": [0.035, 0.032, 0.028]}, "eye"))
    return parts

def tooth(c, down, out, length, width, depth, tilt, rng):
    """one long, uneven tooth as a squashed sphere mesh; c = root, grows along `down`"""
    import os
    nu, nv = (10, 7) if os.environ.get("TEETH_LO") is None else (6, 4)
    th = np.linspace(0, np.pi, nv)[:, None]
    ph = np.linspace(0, 2 * np.pi, nu, endpoint=False)[None, :]
    x = np.sin(th) * np.cos(ph); y = np.cos(th) * np.ones_like(ph); z = np.sin(th) * np.sin(ph)
    taper = 0.8 + 0.2 * (y + 1) / 2                                      # a little narrower at the tip (blunt, human)
    loc = np.stack([x * width * taper, (y - 1) * length / 2, z * depth * taper], -1).reshape(-1, 3)
    # build a frame: Y = -down (tip goes down along `down`), Z = out
    Y = -np.asarray(down, float); Y /= np.linalg.norm(Y)
    Z = np.asarray(out, float); Z = Z - Y * (Z @ Y); Z /= np.linalg.norm(Z)
    X = np.cross(Y, Z)
    R = np.stack([X, Y, Z], 1) @ rot_z(tilt)
    v = loc @ R.T + c
    faces = []
    for i in range(nv - 1):
        for j in range(nu):
            a = i * nu + j; b = i * nu + (j + 1) % nu; cc = (i + 1) * nu + j; d = (i + 1) * nu + (j + 1) % nu
            faces += [[a, b, cc], [b, d, cc]]
    tip = np.clip((-(loc[:, 1])) / length, 0, 1)                        # 0 root .. 1 tip
    faces = convex_outward(v, np.array(faces))
    return v, faces, tip

def teeth_mesh():
    parts_v, parts_f, tips, offs = [], [], [], 0
    rng = np.random.default_rng(5)
    for row, sgn in (("up", -1), ("lo", 1)):
        n = 34 if row == "up" else 31
        xs = np.linspace(-0.068, 0.068, n) + rng.normal(0, 0.0008, n)
        for x in xs:
            y = mouth_curve(x); hh = mouth_half(x); z = surface_z(np.array([x]), np.array([y]))[0]
            root_y = y + (hh + 0.0005) * (1 if row == "up" else -1)
            c = V(x, root_y, z - 0.0045)
            out = V(x * 0.9, 0, 0.11) - V(0, 0, 0); out[1] = 0
            L = (hh + 0.0012) * (rng.uniform(1.0, 1.25) if row == 'up' else rng.uniform(0.85, 1.05))
            w = rng.uniform(0.0026, 0.0034) * (1.2 if abs(x) < 0.02 else 1.0) * (1.0 if row == 'up' else 0.88)
            v, f, tp = tooth(c, V(0, sgn, 0) * -1 * -1 if False else V(0, -1, 0) if row == "up" else V(0, 1, 0),
                             out, L, w, 0.0031, rng.normal(0, 0.09), rng)
            parts_v.append(v); parts_f.append(f + offs); tips.append(tp); offs += len(v)
    v = np.vstack(parts_v); f = np.vstack(parts_f); tp = np.concatenate(tips)
    n = vertex_normals(v, f)
    stain = 0.5 + 0.5 * fbm(v * 300, 2, seed=4)
    col = mixc(hexc("#efe6c6"), hexc("#b89a54"), 0.3 + 0.5 * stain)        # yellowed
    col = mixc(col, hexc("#5a4022"), smoothstep(0.35, 0.0, tp) * 0.85)    # rotten at the gum line
    return [Prim(v, f, n, col, {"name": "teeth", "rough": 0.25}, "teeth")]

def hair_mesh(scale_len=1.0, seed=2, strands=110, seg=13):
    """long, wet, black clumps hanging from the scalp, framing the face"""
    import os; strands = int(strands * float(os.environ.get("STRANDS", "1"))); seg = max(6, int(seg * float(os.environ.get("STRANDS", "1")) ** 0.5))
    rng = np.random.default_rng(seed)
    hc = V(0, 0.205, -0.012); hr = np.array([0.094, 0.112, 0.106])
    parts = []
    tries = 0
    while len(parts) < strands and tries < 3000:
        tries += 1
        th = rng.uniform(0, 2 * np.pi); ph = rng.uniform(0.0, 1.25)
        dirn = np.array([math.sin(ph) * math.cos(th), math.cos(ph), math.sin(ph) * math.sin(th)])
        root = hc + dirn * hr
        if root[2] > 0.03 and root[1] < 0.27: continue          # keep the face clear
        if root[2] > 0.06: continue
        pts = [root]
        p = root.copy()
        out = np.array([dirn[0], 0, dirn[2]]); out /= max(np.linalg.norm(out), 1e-6)
        L = rng.uniform(0.30, 0.62) * scale_len
        wave_a = rng.uniform(0.002, 0.007); wave_f = rng.uniform(1.5, 3.5); wave_p = rng.uniform(0, 6.28)
        side = np.cross(out, [0, 1, 0])
        for k in range(1, seg + 1):
            t = k / seg
            step = np.array([0, -L / seg, 0]) + out * (0.009 * (1 - t) ** 2 + 0.0004)
            step += side * wave_a * math.cos(wave_f * t * 6.28 + wave_p) * 0.5 + rng.normal(0, 0.0015, 3)
            p = p + step
            # stay outside the head
            q = (p - hc) / (hr + 0.006)
            r = np.linalg.norm(q)
            if r < 1: p = hc + q / r * (hr + 0.006)
            # stay clear of the face itself
            if p[2] > 0.04 and abs(p[0]) < 0.07 and p[1] > 0.06: p[2] = 0.04
            pts.append(p.copy())
        pts = np.array(pts)
        w = rng.uniform(0.004, 0.0085) if rng.random() < 0.7 else rng.uniform(0.0015, 0.0025)
        radii = w * (1 - 0.8 * np.linspace(0, 1, len(pts)) ** 1.4)
        v, f = tube(pts, radii, sides=4)
        parts.append((v, f))
    v, f = merge(parts)
    n = vertex_normals(v, f)
    col = np.tile(hexc("#0b0907"), (len(v), 1))
    col = mixc(col, hexc("#211a14"), 0.5 + 0.5 * fbm(v * 40, 2, seed=9))
    return [Prim(v, f, n, col, {"name": "hair", "rough": 0.32}, "hair")]

def head_prims(h=0.0038, strands=110, seg=13):
    v, f, n = mesh_sdf(head_sdf, V(-0.11, -0.03, -0.13), V(0.11, 0.33, 0.135), h, smooth=2)
    ao = sdf_ao(head_sdf, v, n, step=0.006, samples=5, k=1.4)
    skin = hexc("#b9b6a8")
    col = mixc(skin, hexc("#8e9488"), 0.5 + 0.5 * fbm(v * 25, 3, seed=1))           # ash grey, sickly
    col = mixc(col, hexc("#a8a08e"), smoothstep(0.03, 0.09, v[:, 2]) * 0.4)
    col = mixc(col, hexc("#6f6b62"), smoothstep(0.2, 0.75, fbm(v * 9, 3, seed=5)) * 0.5)        # blotches
    veins = 1 - ridged(v * 38, 3, seed=21)
    col = mixc(col, hexc("#5b5f7a"), smoothstep(0.7, 0.95, veins) * 0.55)              # veins under the skin
    # bruised hollow eyes
    for s in (-1, 1):
        de = np.linalg.norm((v - V(s * 0.033, 0.181, 0.086)) / np.array([1.3, 1.0, 1.0]), axis=1)
        col = mixc(col, hexc("#3a2a33"), smoothstep(0.034, 0.016, de) * 0.9)
    # the grin: raw red lips and corners, black-red mouth inside
    dx = np.interp(v[:, 0], MX, MY); dh = np.interp(v[:, 0], MX, MH); dz = np.interp(v[:, 0], MX, MZ)
    dist_mouth = np.abs(v[:, 1] - dx) - dh
    near_front = smoothstep(-0.03, -0.004, v[:, 2] - dz)
    col = mixc(col, hexc("#6e2a26"), smoothstep(0.006, 0.0, dist_mouth) * near_front)            # lips
    col = mixc(col, hexc("#7a3a2c"), smoothstep(0.012, 0.0, dist_mouth) * smoothstep(0.05, 0.075, np.abs(v[:, 0])))  # torn corners
    inside = smoothstep(-0.002, -0.012, v[:, 2] - dz) * smoothstep(0.004, -0.002, dist_mouth)
    col = mixc(col, hexc("#2a0807"), inside)                                                       # gums/throat
    col = mixc(col, hexc("#060202"), smoothstep(-0.012, -0.03, v[:, 2] - dz) * (np.abs(v[:, 0]) < 0.08))
    # scalp under the hair: dark, so no skin shows between the strands
    scalp = np.maximum(smoothstep(0.235, 0.26, v[:, 1]) * smoothstep(0.05, 0.0, v[:, 2] - 0.0), smoothstep(-0.0, -0.03, v[:, 2]) * smoothstep(0.12, 0.17, v[:, 1]))
    scalp = np.maximum(scalp, smoothstep(0.072, 0.085, np.abs(v[:, 0])) * smoothstep(0.13, 0.16, v[:, 1]))
    col = mixc(col, hexc("#120e0b"), scalp)
    col = col * (0.35 + 0.65 * ao)[:, None]
    # neck shading down into the body
    col = col * (0.92 + 0.08 * smoothstep(-0.02, 0.06, v[:, 1]))[:, None]
    labels = np.zeros(len(v), int)
    labels[inside > 0.5] = 1
    skin_m = {"name": "skin", "rough": 0.58}
    wet_m = {"name": "mouth", "rough": 0.12}
    prims = split_by_label(v, f, n, col, labels, {0: skin_m, 1: wet_m}, "head")
    return prims + eye_mesh() + teeth_mesh() + hair_mesh(strands=strands, seg=seg)

if __name__ == "__main__" and "head" in sys.argv[1:]:
    for name, kw in (("grinner_head", {}), ("grinner_face_lo", dict(h=0.0058, strands=60, seg=9))):
        prims = head_prims(**kw)
        tris, size = write_glb(f"/home/claude/hh13/out/{name}.glb", prims, name)
        print(name, tris, "tris", size // 1024, "KB")

# ---------------------------------------------------------------- the body
from body import *

NECK = V(0, 1.70, 0.07)          # where the head sits (its origin)
HEAD_TILT = math.radians(14)     # chin down, staring up from under the brow

J = dict(
    pelvis=V(0, 0.98, 0.0), chest=V(0, 1.40, 0.035), upper=V(0, 1.55, -0.01),
    hipL=V(-0.085, 0.95, 0.0), kneeL=V(-0.095, 0.52, 0.035), ankL=V(-0.085, 0.085, 0.0),
    hipR=V(0.085, 0.95, 0.0), kneeR=V(0.092, 0.53, 0.02), ankR=V(0.09, 0.085, -0.01),
    shL=V(-0.185, 1.585, 0.0), elL=V(-0.235, 1.17, 0.06), wrL=V(-0.235, 0.77, 0.17),
    shR=V(0.185, 1.585, 0.0), elR=V(0.228, 1.15, -0.01), wrR=V(0.235, 0.73, 0.05),
)

def torso_sdf(p):
    d = ribcage(p, J["chest"], (0.125, 0.165, 0.092), ribs=0.003)
    d = smin(d, ellipsoid(p, J["pelvis"] + V(0, 0.03, 0), (0.13, 0.09, 0.085)), 0.06)
    d = smin(d, rcone(p, J["pelvis"], J["chest"], 0.085, 0.10), 0.05)                     # starved waist
    for s in (-1, 1):
        d = smin(d, ellipsoid(p, V(s * 0.168, 1.578, 0.0), (0.042, 0.036, 0.04)), 0.04)     # bony shoulders
        d = smin(d, sphere(p, V(s * 0.19, 1.6, 0.01), 0.022), 0.02)                          # the point of the shoulder
        d = smin(d, capsule(p, V(s * 0.02, 1.56, 0.075), V(s * 0.16, 1.6, 0.03), 0.011), 0.015)  # collarbones
        d = smin(d, ellipsoid(p, V(s * 0.075, 1.48, -0.075), (0.05, 0.06, 0.012)), 0.03)     # shoulder blades
    d = smin(d, rcone(p, J["upper"], NECK + V(0, 0.01, -0.01), 0.07, 0.04), 0.04)          # hunched neck
    return d

def limbs_sdf(p):
    d = None
    for side in "LR":
        sh = J["sh" + side] + V(0, -0.01, 0)
        e = bony_limb(p, sh, J["el" + side], 0.032, 0.024, 1.3)
        e = smin(e, ellipsoid(p, sh + (J["el" + side] - sh) * 0.22, (0.036, 0.07, 0.036)), 0.03)   # wasted muscle
        e = smin(e, bony_limb(p, J["el" + side], J["wr" + side], 0.026, 0.018, 1.25), 0.012)
        e = smin(e, ellipsoid(p, J["el" + side] + (J["wr" + side] - J["el" + side]) * 0.25, (0.03, 0.08, 0.03)), 0.02)
        e = smin(e, bony_limb(p, J["hip" + side], J["knee" + side], 0.06, 0.038, 1.2), 0.04)
        e = smin(e, bony_limb(p, J["knee" + side], J["ank" + side], 0.037, 0.022, 1.25), 0.015)
        e = smin(e, ellipsoid(p, J["knee" + side] + (J["ank" + side] - J["knee" + side]) * 0.25 + V(0, 0, -0.012), (0.034, 0.08, 0.03)), 0.02)  # calf
        d = e if d is None else np.minimum(d, e)
    return d

def skin_sdf(p):
    d = smin(torso_sdf(p), limbs_sdf(p), 0.03)
    return d - 0.0015 * fbm(p * 30, 2, seed=12)

HEM = 0.34
def gown_sdf(p):
    """a clinging, rotten nightgown: bodice, flared skirt with a torn hem, short torn sleeves"""
    d = torso_sdf(p) - 0.011                                         # the bodice follows the body
    d = smax(d, -(p[:, 1] - 0.9), 0.02)
    waist = V(0, 1.06, 0.01)
    # skirt: a solid flared cone with soft folds round it
    q = p - waist
    ang = np.arctan2(q[:, 0], q[:, 2])
    t = np.clip(-(q[:, 1]) / (waist[1] - HEM + 0.15), 0, 1)
    rad = 0.135 + 0.13 * t ** 1.3 + 0.012 * np.sin(ang * 9 + 1.3) * t + 0.008 * np.sin(ang * 17) * t
    r_xz = np.sqrt(q[:, 0] ** 2 + (q[:, 2] * 1.15) ** 2)
    skirt = np.maximum(r_xz - rad, q[:, 1] - 0.08)
    d = smin(d, skirt, 0.09)
    # torn hem
    hem = tatter(p[:, 1], p[:, 0], p[:, 2], HEM, 0.05, 2.4, seed=7)
    d = smax(d, hem - p[:, 1], 0.01)
    # neckline: low and ragged, collarbones and the hollow of the throat bare
    neck_cut = 1.50 + 0.035 * fbm(p * 18, 2, seed=2) - 0.05 * smoothstep(0.0, 0.1, p[:, 2] - 0.0)
    d = smax(d, p[:, 1] - neck_cut, 0.01)
    # no sleeves: thin straps over bare, bony shoulders
    for s_ in (-1, 1):
        d = smin(d, capsule(p, V(s_ * 0.09, 1.49, 0.085), V(s_ * 0.12, 1.62, 0.0), 0.009), 0.01)
        d = smin(d, capsule(p, V(s_ * 0.12, 1.62, 0.0), V(s_ * 0.09, 1.49, -0.085), 0.009), 0.01)
    # thin fabric: shrink a touch with cloth wrinkles
    d = d + 0.003 * fbm(p * np.array([30, 6, 30]), 3, seed=31)
    return d

def hands_prims(h=0.0046):
    out = []
    for side, s in (("L", -1), ("R", 1)):
        W = J["wr" + side]; fwd = V(0.05 * s, -1, 0.3 if side == "L" else 0.12); up = V(s, 0, 0.2)
        def f(p, W=W, fwd=fwd, up=up):
            d, nl = hand_sdf(p, W, fwd, up, size=1.15, curl=0.32, spread=0.25, finger_len=1.45, claw=0.4)
            return np.minimum(d, nl)
        def parts(p, W=W, fwd=fwd, up=up):
            return hand_sdf(p, W, fwd, up, size=1.15, curl=0.32, spread=0.25, finger_len=1.45, claw=0.4)
        lo = W + V(-0.11, -0.3, -0.11); hi = W + V(0.11, 0.06, 0.11)
        v, fa, n = mesh_sdf(f, lo, hi, h, smooth=1)
        dh, dn = parts(v)
        lab = (dn < dh).astype(int)
        ao = sdf_ao(f, v, n, 0.006, 4, 1.2)
        col = skin_colour(v, n, ao, base="#a7a597", dark="#6a675f")
        col = mixc(col, hexc("#7c5f5a"), smoothstep(0.03, 0.0, np.linalg.norm(v - W, axis=1)) * 0.0)
        col[lab == 1] = mixc(hexc("#2e261c"), hexc("#0d0b08"), 0.5)[None] * (0.4 + 0.6 * ao[lab == 1])[:, None]
        out += split_by_label(v, fa, n, col, lab, {0: {"name": "skin", "rough": 0.55}, 1: {"name": "nails", "rough": 0.3}}, "hand" + side)
    return out

def feet_prims(h=0.0055):
    out = []
    for side, s in (("L", -1), ("R", 1)):
        A = J["ank" + side]
        def f(p, A=A, s=s):
            d = smin(sphere(p, A, 0.03), ellipsoid(p, A + V(0, -0.045, 0.055), (0.036, 0.022, 0.085)), 0.03)
            d = smin(d, rcone(p, A + V(0, 0.08, 0), A, 0.024, 0.026), 0.02)
            for i in range(5):     # long grey toes curling over the floor
                b = A + V(s * (-0.022 + i * 0.011), -0.06, 0.12 - i * 0.006 * abs(i - 1))
                d = smin(d, bony_limb(p, b, b + V(s * (i - 2) * 0.003, -0.014, 0.034 - i * 0.003), 0.008, 0.0055, 1.15), 0.006)
            return d
        v, fa, n = mesh_sdf(f, A + V(-0.08, -0.1, -0.06), A + V(0.08, 0.1, 0.22), h, smooth=1)
        ao = sdf_ao(f, v, n, 0.008, 4, 1.2)
        col = skin_colour(v, n, ao, base="#8e8b82", dark="#4e4a45")
        col = mixc(col, hexc("#2c241c"), smoothstep(0.03, 0.0, v[:, 1]) * 0.8)     # filthy soles
        out.append(Prim(v, fa, n, col, {"name": "skin", "rough": 0.6}, "foot" + side))
    return out

def body_prims(h=0.0125):
    parts = [("skin", skin_sdf), ("gown", gown_sdf)]
    f, v, fa, n, lab = mesh_parts(parts, (V(-0.36, 0.06, -0.3), V(0.36, 1.76, 0.32)), h)
    ao = sdf_ao(f, v, n, 0.02, 5, 1.3)
    skin = skin_colour(v, n, ao, base="#a29f92", dark="#646159")
    gown = cloth_colour(v, ao, base="#b3ab98", stain="#4f412e", blood="#3d0806", blood_amt=0.7, seed=4, dirt_y=0.6)
    # the front of the gown is soaked from the chest down
    front = smoothstep(0.0, 0.08, v[:, 2]) * smoothstep(0.7, 1.35, v[:, 1]) * smoothstep(0.25, 0.75, 0.5 + 0.5 * fbm(v * 5, 3, seed=77))
    gown = mixc(gown, hexc("#2b0504") * (0.3 + 0.7 * ao)[:, None], front * 0.9)
    col = np.where((lab == 1)[:, None], gown, skin)
    mats = {0: {"name": "skin", "rough": 0.55}, 1: {"name": "cloth", "rough": 0.85}}
    return split_by_label(v, fa, n, col, lab, mats, "body") + hands_prims() + feet_prims()

def place(prims, R, t):
    out = []
    for P in prims:
        out.append(Prim(transform(P.v, R, t), P.f, transform(P.n, R), P.c, P.mat, P.name))
    return out

if __name__ == "__main__" and "body" in sys.argv[1:]:
    body = body_prims()
    for P in body: print("  ", P.name, len(P.f))
    tris, size = write_glb("/home/claude/hh13/out/grinner_body.glb", body, "grinner_body")
    print("grinner_body", tris, "tris", size // 1024, "KB")
    head = place(head_prims(), rot_x(HEAD_TILT), NECK)
    tris, size = write_glb("/home/claude/hh13/out/grinner.glb", body + head, "grinner")
    print("grinner", tris, "tris", size // 1024, "KB")
