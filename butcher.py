"""THE BUTCHER - a hulking man in a stitched burlap sack hood with a sewn-on grin,
a filthy vest, a heavy leather apron soaked in blood, and a chainsaw held out in front.
Original design. Faces +z. Built in pieces so the house can animate the chase:
  butcher_body  (origin: on the floor between the feet; everything above the hips)
  butcher_leg   (origin: the right hip joint; mirror with sx=-1 for the left)
  chainsaw      (origin: the rear grip, where his right hand holds it; points +z)
  butcher       (all of it, for posing still)"""
import sys, math
import numpy as np
sys.path.insert(0, "/home/claude/hh13/tools")
from sculpt import *
from body import *
OUT = "/home/claude/hh13/out/"

HIP = V(0.15, 1.0, -0.01)
LIFT = 0.08                                 # body is modelled on short legs, then lifted onto the real ones
SHL, SHR = V(-0.33, 1.63, 0.0), V(0.33, 1.63, 0.0)
HEADC = V(0, 1.885, 0.05)
SAW_AT = V(0.21, 1.04, 0.3)                 # rear grip in body space
SAW_R = rot_y(math.radians(-8)) @ rot_x(math.radians(-10))   # pointing forward, a little up and across
FRONT_GRIP = V(-0.035, 0.165, 0.2)          # top of the front handle, in saw space
WRR = SAW_AT + V(0.0, 0.0, -0.06)
WRL = SAW_AT + SAW_R @ FRONT_GRIP + V(-0.06, 0.02, -0.02)

def ik(sh, wr, l1, l2, pole):
    d = wr - sh; L = np.linalg.norm(d); L = min(L, l1 + l2 - 1e-3); dirn = d / np.linalg.norm(d)
    a = (l1 * l1 - l2 * l2 + L * L) / (2 * L); h = math.sqrt(max(l1 * l1 - a * a, 0))
    pp = pole - dirn * (pole @ dirn); pp /= np.linalg.norm(pp)
    return sh + dirn * a + pp * h

ELR = ik(SHR, WRR, 0.37, 0.36, V(0.7, -0.4, -0.6))
ELL = ik(SHL, WRL, 0.37, 0.36, V(-0.8, -0.5, -0.2))

# ---------------------------------------------------------------- body
def torso(p):
    d = ellipsoid(p, V(0, 1.43, 0.0), (0.28, 0.27, 0.2))                       # barrel chest
    d = smin(d, ellipsoid(p, V(0, 1.2, 0.08), (0.27, 0.21, 0.24)), 0.12)        # gut
    d = smin(d, ellipsoid(p, V(0, 1.03, -0.01), (0.25, 0.13, 0.17)), 0.1)       # hips
    d = smin(d, capsule(p, SHL + V(0.04, 0, 0), SHR - V(0.04, 0, 0), 0.11), 0.12)
    d = smin(d, ellipsoid(p, V(0, 1.66, -0.05), (0.2, 0.1, 0.12)), 0.1)         # traps, hunched
    d = smin(d, rcone(p, V(0, 1.62, -0.01), V(0, 1.77, 0.03), 0.1, 0.085), 0.06)  # bull neck
    for s in (-1, 1):
        d = smin(d, sphere(p, V(s * HIP[0], HIP[1] + 0.02, HIP[2]), 0.1), 0.08)
    return d + 0.004 * fbm(p * np.array([9, 3, 9]), 2, seed=71)

def arms(p):
    d = None
    for sh, el, wr in ((SHL, ELL, WRL), (SHR, ELR, WRR)):
        e = rcone(p, sh, el, 0.085, 0.062)
        e = smin(e, ellipsoid(p, sh + (el - sh) * 0.42 + V(0, 0, 0.02), (0.075, 0.12, 0.075)), 0.05)      # big upper arm
        e = smin(e, rcone(p, el, wr, 0.062, 0.042), 0.04)
        e = smin(e, ellipsoid(p, el + (wr - el) * 0.3, (0.06, 0.06, 0.06)), 0.06)                         # meaty forearm
        d = e if d is None else np.minimum(d, e)
    return d - 0.002 * fbm(p * 30, 2, seed=72)

def sleeves(p):
    d = None
    for sh, el in ((SHL, ELL), (SHR, ELR)):
        e = rcone(p, sh, sh + (el - sh) * 0.45, 0.11, 0.092)
        e = smin(e, torus_any(p, sh + (el - sh) * 0.45, (el - sh), 0.088, 0.016), 0.01)    # rolled cuff
        d = e if d is None else np.minimum(d, e)
    return d + 0.003 * fbm(p * 20, 2, seed=73)

def torus_any(p, c, axis, R, r):
    a = axis / np.linalg.norm(axis)
    q = p - c
    h = q @ a
    rad = np.linalg.norm(q - h[:, None] * a, axis=1)
    return np.sqrt((rad - R) ** 2 + h * h) - r

def apron(p):
    # bib + skirt: a leather sheet a little proud of the body, front only
    shell = torso(p) - 0.03
    bib = smax(shell, -(p[:, 2] - 0.02), 0.02)
    w = 0.16 + (1.5 - p[:, 1]) * 0.55
    bib = smax(bib, np.abs(p[:, 0]) - np.clip(w, 0.16, 0.33), 0.02)
    bib = smax(bib, p[:, 1] - 1.55, 0.01)
    bib = smax(bib, -(torso(p) - 0.012), 0.01)            # just a sheet, not solid
    # skirt hanging from the waist, stiff with old blood, flared out over the thighs
    y = p[:, 1]
    t = np.clip((1.08 - y) / 0.36, 0, 1)
    rz = 0.255 + 0.05 * t; rx = 0.29 + 0.03 * t
    e = np.sqrt((p[:, 0] / rx) ** 2 + ((p[:, 2] + 0.01) / rz) ** 2) - 1
    sk = np.abs(e * 0.27) - 0.007
    sk = smax(sk, -(p[:, 2] - 0.03), 0.01)
    hem = 0.72 + 0.02 * np.sin(p[:, 0] * 23) + 0.012 * fbm(np.column_stack([p[:, 0] * 9, p[:, 2] * 9, p[:, 0] * 0]), 2, seed=74)
    sk = smax(sk, hem - y, 0.006)
    sk = smax(sk, y - 1.12, 0.01)
    d = np.minimum(bib, sk)
    # straps over the shoulders and a tie at the waist
    for s in (-1, 1):
        d = smin(d, capsule(p, V(s * 0.15, 1.55, 0.21), V(s * 0.17, 1.72, -0.02), 0.022), 0.01)
        d = smin(d, capsule(p, V(s * 0.17, 1.72, -0.02), V(s * 0.12, 1.4, -0.22), 0.022), 0.01)
    d = np.minimum(d, torus_any(p, V(0, 1.08, -0.005), V(0, 1, 0), 0.272, 0.014))
    return d + 0.0025 * fbm(p * 14, 2, seed=75)

EYES = ((-0.05, 0.03, 0.032, 0.026), (0.054, 0.012, 0.024, 0.019))     # x, y, half-width, half-height: crooked, one bigger

def mouth_y(x):
    """a grin cut cheek to cheek, rising at the corners"""
    return HEADC[1] - 0.078 + 2.6 * x ** 2 + 0.006 * np.sin(x * 70)

MOUTH_W = 0.105

def head_sack(p):
    c = HEADC
    d = ellipsoid(p, c, (0.132, 0.15, 0.138))
    d = smin(d, ellipsoid(p, c + V(0, -0.085, 0.035), (0.112, 0.08, 0.1)), 0.05)          # heavy jaw under the cloth
    # the sack's empty corners: one sticks up, one flops over
    d = smin(d, rcone(p, c + V(-0.07, 0.1, -0.01), c + V(-0.12, 0.21, -0.03), 0.06, 0.018), 0.05)
    d = smin(d, rcone(p, c + V(0.06, 0.11, -0.02), c + V(0.15, 0.08, -0.04), 0.06, 0.02), 0.05)
    d = smin(d, rcone(p, c + V(0.15, 0.08, -0.04), c + V(0.17, -0.0, -0.04), 0.02, 0.012), 0.03)
    # cinched with twine at the neck, then the sack flares out over the shoulders
    neck_y = c[1] - 0.175
    d = smin(d, rcone(p, V(0, neck_y, 0.03), V(0, neck_y - 0.1, 0.0), 0.082, 0.17), 0.035)
    hem = neck_y - 0.11 + 0.03 * fbm(np.column_stack([p[:, 0] * 12, p[:, 2] * 12, p[:, 0] * 0]), 2, seed=76)
    d = smax(d, hem - p[:, 1], 0.01)
    # hanging folds: creases that run downwards
    d = d + 0.007 * ridged(p * np.array([16, 4, 16]), 2, seed=78) + 0.004 * fbm(p * 14, 3, seed=77)
    # eye holes hacked out with a knife: ragged, deep, crooked
    for ex, ey, rx, ry in EYES:
        q = p - (c + V(ex, ey, 0.12))
        rag = 1 + 0.25 * fbm(p * 90, 2, seed=int(ex * 1000) & 255)
        e = np.sqrt((q[:, 0] / rx) ** 2 + (q[:, 1] / ry) ** 2) - rag
        e = np.maximum(e * min(rx, ry), -q[:, 2] - 0.06)
        d = smax(d, -e, 0.004)
    # the grin, a long slit sliced through the cloth
    x = p[:, 0]
    gap = 0.0055 * np.sqrt(np.maximum(0, 1 - (x / MOUTH_W) ** 2)) + 0.0008
    slit = np.maximum(np.abs(p[:, 1] - mouth_y(x)) - gap, np.abs(x) - MOUTH_W)
    slit = np.maximum(slit, -(p[:, 2] - (c[2] + 0.06)))
    d = smax(d, -slit, 0.003)
    return d

def rope(p):
    neck_y = HEADC[1] - 0.175
    ang = np.arctan2(p[:, 0], p[:, 2] - 0.03)
    return torus_any(p, V(0, neck_y, 0.03), V(0, 1, 0.12), 0.087, 0.009) - 0.0025 * np.sin(ang * 26)

def surface_z(x, y):
    z = HEADC[2] + 0.25
    for _ in range(60):
        dd = head_sack(np.array([[x, y, z]]))[0]
        z -= dd * 0.9
    return z

def stitches():
    """thick black cross-stitches pulling the grin half shut"""
    vs, fs = [], []
    def add(v, f):
        fs.append(f + sum(len(q) for q in vs)); vs.append(v)
    for x in np.linspace(-MOUTH_W * 0.86, MOUTH_W * 0.86, 11):
        y = mouth_y(x)
        for sgn in (-1, 1):
            a = V(x - 0.008 * sgn, y + 0.017, 0); b = V(x + 0.008 * sgn, y - 0.017, 0)
            pts = []
            for t in np.linspace(0, 1, 5):
                q = a + (b - a) * t
                pts.append(V(q[0], q[1], surface_z(q[0], q[1]) + 0.002 + 0.002 * math.sin(t * math.pi)))
            add(*tube(np.array(pts), 0.0021, sides=5))
    v = np.vstack(vs); f = np.vstack(fs)
    col = mixc(np.tile(hexc("#17110c"), (len(v), 1)), hexc("#3d0605"), 0.4)
    return Prim(v, f, vertex_normals(v, f), col, {"name": "twine", "rough": 0.9}, "stitch")

def teeth_behind():
    """his own teeth, yellow and broken, just visible through the slit"""
    rng = np.random.default_rng(31)
    vs, fs = [], []
    for x in np.linspace(-0.05, 0.05, 12):
        if rng.random() < 0.25: continue
        y = mouth_y(x); z = surface_z(x, y) - 0.012
        for top in (1, -1):
            L = rng.uniform(0.008, 0.014)
            a = V(x, y + top * 0.012, z); b = V(x + rng.normal(0, 0.0015), y + top * (0.012 - L), z + 0.002)
            v, f = tube(np.array([a, b]), [0.0042, 0.0032], sides=6)
            fs.append(f + sum(len(q) for q in vs)); vs.append(v)
    v = np.vstack(vs); f = np.vstack(fs)
    col = mixc(np.tile(hexc("#b9a46a"), (len(v), 1)), hexc("#4a2a12"), smoothstep(0.0, 0.01, np.abs(v[:, 1] - mouth_y(v[:, 0]))) * 0.6)
    return Prim(v, f, vertex_normals(v, f), col, {"name": "teeth", "rough": 0.35}, "teeth")

def head_prims():
    hf = lambda p: np.minimum(head_sack(p), rope(p))
    hv, hfa, hn = mesh_sdf(hf, HEADC - V(0.2, 0.3, 0.2), HEADC + V(0.22, 0.26, 0.2), 0.0082, smooth=1)
    hao = sdf_ao(hf, hv, hn, 0.008, 5, 1.6)
    hl = (rope(hv) < head_sack(hv)).astype(int)
    weave = 0.93 + 0.07 * np.sin(hv[:, 0] * 420) * np.sin(hv[:, 1] * 420)
    hc = mixc(hexc("#7a6847"), hexc("#3e3220"), smoothstep(0.2, 0.9, 0.5 + 0.5 * fbm(hv * 9, 3, seed=90))) * weave[:, None]
    hc = mixc(hc, hexc("#2a2016"), smoothstep(0.5, 0.9, 0.5 + 0.5 * fbm(hv * 4, 2, seed=93)) * 0.6)      # grime and sweat
    hc = mixc(hc, hexc("#3d0605"), splatter(hv, 91, 0.35, 9))
    # the slit is black, the cloth round it soaked dark red, and it has run down the chin
    x = hv[:, 0]; dm = np.abs(hv[:, 1] - mouth_y(x))
    front = (hv[:, 2] > HEADC[2] + 0.02) & (np.abs(x) < MOUTH_W + 0.02)
    run = smoothstep(0.5, 0.75, 0.5 + 0.5 * vnoise(np.column_stack([x * 70, x * 0, x * 0]), 94)) * (hv[:, 1] < mouth_y(x)) * smoothstep(0.09, 0.0, mouth_y(x) - hv[:, 1])
    hc = mixc(hc, hexc("#2e0403"), (smoothstep(0.022, 0.004, dm) * 0.85 + run * 0.9).clip(0, 1) * front)
    for ex, ey, rx, ry in EYES:
        q = hv - (HEADC + V(ex, ey, 0.12))
        dd = np.sqrt((q[:, 0] / rx) ** 2 + (q[:, 1] / ry) ** 2)
        zf = HEADC[2] + 0.138 * math.sqrt(max(0.0, 1 - (ex / 0.132) ** 2 - (ey / 0.15) ** 2))
        hc = mixc(hc, hexc("#020101"), smoothstep(-0.0, -0.012, hv[:, 2] - zf) * smoothstep(1.5, 1.0, dd))   # black inside
        hc = mixc(hc, hexc("#1f1810"), smoothstep(1.8, 1.2, dd) * (hv[:, 2] > HEADC[2]) * 0.6)            # dirty, fraying rim
    hc = hc * (0.3 + 0.7 * hao)[:, None]
    hc[hl == 1] = (mixc(hexc("#5b4b30"), hexc("#2c2215"), 0.5 + 0.5 * fbm(hv[hl == 1] * 40, 2, seed=92)) * (0.4 + 0.6 * hao[hl == 1])[:, None])
    prims = split_by_label(hv, hfa, hn, hc, hl, {0: {"name": "burlap", "rough": 1.0}, 1: {"name": "twine", "rough": 1.0}}, "hood")
    prims.append(stitches()); prims.append(teeth_behind())
    # eyes deep in the holes, looking at you: yellowed, red-rimmed, tiny black pupils
    for ex, ey, rx, ry in EYES:
        zs = surface_z(ex, HEADC[1] + ey)
        cc = V(ex, HEADC[1] + ey, zs + 0.005)
        r = 0.0135
        ev, ef, en = mesh_sdf(lambda p, cc=cc: sphere(p, cc, r), cc - 0.02, cc + 0.02, 0.0036, smooth=1)
        rel = (ev - cc) / r
        ec = np.tile(hexc("#9a8d6a"), (len(ev), 1))
        ec = mixc(ec, hexc("#7a140c"), smoothstep(0.9, 0.4, rel[:, 2]) * 0.9)
        ec = mixc(ec, hexc("#4a3a1c"), smoothstep(0.6, 0.5, np.linalg.norm(rel[:, :2], axis=1)) * (rel[:, 2] > 0))   # muddy iris
        ec = mixc(ec, hexc("#000000"), smoothstep(0.26, 0.2, np.linalg.norm(rel[:, :2], axis=1)) * (rel[:, 2] > 0))   # pin-prick pupil
        prims.append(Prim(ev, ef, en, ec, {"name": "eye", "rough": 0.05, "emit": [0.03, 0.025, 0.02]}, "eye"))
    return prims

def hands():
    out = []
    for W, el, tgt in ((WRR, ELR, SAW_AT), (WRL, ELL, SAW_AT + SAW_R @ FRONT_GRIP)):
        fwd = tgt - W + (W - el) * 0.4; fwd /= np.linalg.norm(fwd)
        up = V(0, 1, 0) if W is WRR else V(0.2, 1, -0.6)
        kw = dict(size=2.05, curl=0.95, spread=0.12, finger_len=1.0, claw=0.0, knob=1.12)
        def f(p, W=W, fwd=fwd, up=up, el=el):
            d, nl = hand_sdf(p, W, fwd, up, **kw)
            return np.minimum(smin(d, rcone(p, el + (W - el) * 0.7, W, 0.046, 0.04), 0.02), nl)
        v, fa, n = mesh_sdf(f, W - 0.15, W + 0.15, 0.0095, smooth=1)
        d, nl = hand_sdf(v, W, fwd, up, **kw)
        lab = ((nl < d) & (nl < 0.004)).astype(int)
        ao = sdf_ao(f, v, n, 0.008, 4, 1.2)
        col = skin_colour(v, n, ao, base="#b08b74", dark="#6e4e3e", vein="#5a4a5a", seed=81)
        col = mixc(col, hexc("#3c0705"), smoothstep(0.45, 0.75, 0.5 + 0.5 * fbm(v * 14, 3, seed=82)) * 0.85)    # bloody hands
        col[lab == 1] = hexc("#3a2a20") * (0.5 + 0.5 * ao[lab == 1])[:, None]
        out += split_by_label(v, fa, n, col, lab, {0: {"name": "skin", "rough": 0.5}, 1: {"name": "nails", "rough": 0.4}}, "hand")
    return out

def splatter(v, seed, amt=0.5, scale=6):
    """blood: patches + fine spatter, as a 0..1 mask"""
    m = smoothstep(0.6 - amt * 0.4, 0.85 - amt * 0.35, 0.5 + 0.5 * fbm(v * scale, 4, seed=seed))
    sp = smoothstep(0.82, 0.9, 0.5 + 0.5 * vnoise(v * scale * 9, seed + 1)) * amt
    return np.clip(m + sp, 0, 1)

def body_prims():
    parts = [("vest", torso), ("skin", arms), ("sleeve", sleeves), ("apron", apron)]
    f, v, fa, n, lab = mesh_parts(parts, (V(-0.62, 0.68, -0.32), V(0.62, 1.8, 0.62)), 0.019)
    ao = sdf_ao(f, v, n, 0.02, 4, 1.4)
    col = np.zeros((len(v), 3))
    m = lab == 0
    col[m] = cloth_colour(v[m], ao[m], base="#6b6658", stain="#463a26", blood="#3e0605", blood_amt=0.5, seed=83, dirt_y=1.1)
    m = lab == 1
    col[m] = skin_colour(v[m], n[m], ao[m], base="#b08b74", dark="#6e4e3e", vein="#5a4a5a", seed=84)
    col[m] = mixc(col[m], hexc("#4a0806"), splatter(v[m], 85, 0.5, 8) * 0.9)
    m = lab == 2
    col[m] = cloth_colour(v[m], ao[m], base="#615c50", stain="#3e3324", blood="#3e0605", blood_amt=0.45, seed=86)
    m = lab == 3
    leather = mixc(hexc("#3a2c22"), hexc("#21170f"), 0.5 + 0.5 * fbm(v[m] * 5, 3, seed=87)) * (0.35 + 0.65 * ao[m])[:, None]
    bl = splatter(v[m], 88, 0.95, 5)
    bl = np.maximum(bl, smoothstep(1.25, 0.8, v[m][:, 1]) * smoothstep(0.4, 0.7, 0.5 + 0.5 * fbm(v[m] * np.array([18, 2, 18]), 2, seed=89)))  # runs down to the hem
    col[m] = mixc(leather, mixc(hexc("#5a0806"), hexc("#260202"), smoothstep(0.3, 0.9, bl)) * (0.4 + 0.6 * ao[m])[:, None], bl)
    mats = {0: {"name": "vest", "rough": 0.9}, 1: {"name": "skin", "rough": 0.5}, 2: {"name": "sleeve", "rough": 0.9},
            3: {"name": "apron", "rough": 0.3}}
    prims = split_by_label(v, fa, n, col, lab, mats, "body")
    prims += head_prims()
    prims += hands()
    return prims

# ---------------------------------------------------------------- legs
KN = V(0.015, -0.5, 0.07); AN = V(0.0, -0.97, 0.0)
def leg_parts():
    def pants(p):
        d = rcone(p, V(0, 0.03, 0), KN, 0.14, 0.095)
        d = smin(d, rcone(p, KN, AN + V(0, 0.16, 0), 0.095, 0.075), 0.04)
        d = smin(d, ellipsoid(p, KN + V(0, -0.15, -0.035), (0.085, 0.13, 0.085)), 0.05)   # calf
        d = smin(d, ellipsoid(p, V(0.0, -0.2, 0.02), (0.125, 0.2, 0.125)), 0.06)
        return d + 0.005 * fbm(p * np.array([10, 3, 10]), 2, seed=93)
    def boot(p):
        d = rcone(p, AN + V(0, 0.19, 0), AN + V(0, -0.02, 0), 0.075, 0.07)
        d = smin(d, ellipsoid(p, AN + V(0, -0.04, 0.09), (0.07, 0.055, 0.15)), 0.04)          # big round toe cap
        sole = rbox(p, AN + V(0, -0.085, 0.05), V(0.068, 0.016, 0.145), 0.014)
        d = smin(d, sole, 0.006)
        d = smax(d, -(p[:, 1] - (AN[1] - 0.1)), 0.004)
        return d + 0.0015 * fbm(p * 30, 2, seed=94)
    return [("pants", pants), ("boot", boot)]

def leg_prims():
    parts = leg_parts()
    f, v, fa, n, lab = mesh_parts(parts, (V(-0.18, -1.1, -0.18), V(0.18, 0.18, 0.32)), 0.016)
    ao = sdf_ao(f, v, n, 0.015, 4, 1.4)
    col = np.zeros((len(v), 3))
    m = lab == 0
    col[m] = cloth_colour(v[m], ao[m], base="#3d3a33", stain="#1d1b17", blood="#3a0605", blood_amt=0.45, seed=95, dirt_y=-0.6)
    m = lab == 1
    col[m] = mixc(hexc("#2b1f16"), hexc("#15100b"), 0.5 + 0.5 * fbm(v[m] * 12, 2, seed=96)) * (0.4 + 0.6 * ao[m])[:, None]
    col[m] = mixc(col[m], hexc("#3a0605"), splatter(v[m], 97, 0.4, 10) * 0.8)
    sole = (v[:, 1] < AN[1] - 0.07) & m
    col[sole] = hexc("#0d0c0b")
    return split_by_label(v, fa, n, col, lab, {0: {"name": "pants", "rough": 0.95}, 1: {"name": "boot", "rough": 0.45}}, "leg")

# ---------------------------------------------------------------- chainsaw
def box_mesh(c, half):
    s = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    v = c + s * half
    f = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]])
    return v, convex_outward(v, f, c)

BAR_Z0, BAR_Z1, BAR_H0, BAR_H1, BAR_Y = 0.3, 0.92, 0.05, 0.034, -0.035
def saw_parts():
    def housing(p):
        d = rbox(p, V(0, 0.0, 0.17), V(0.07, 0.075, 0.14), 0.03)
        d = smin(d, ellipsoid(p, V(0.0, 0.05, 0.1), (0.075, 0.06, 0.1)), 0.03)           # engine cover hump
        d = smin(d, rbox(p, V(0.0, -0.075, 0.12), V(0.06, 0.02, 0.17), 0.012), 0.01)      # base
        for i in range(7):                                                               # cooling fins on the side
            d = smin(d, rbox(p, V(0.075, 0.0, 0.1 + i * 0.022), V(0.01, 0.05, 0.004), 0.002), 0.003)
        d = smin(d, capsule(p, V(0.04, 0.07, 0.29), V(0.04, 0.1, 0.33), 0.012), 0.01)      # exhaust
        d = smax(d, -capsule(p, V(0.04, 0.1, 0.33), V(0.04, 0.12, 0.35), 0.007), 0.003)
        return d - 0.0012 * fbm(p * 60, 2, seed=101)
    def grips(p):
        rear = [V(0, 0.07, 0.07), V(0, 0.04, -0.1), V(0, -0.06, -0.11), V(0, -0.075, 0.05)]
        d = None
        for a, b in zip(rear[:-1], rear[1:]):
            e = capsule(p, a, b, 0.018); d = e if d is None else smin(d, e, 0.01)
        # the hoop of the front handle
        ang = np.linspace(-0.2, math.pi + 0.2, 10)
        pts = [V(0.11 * math.cos(a), 0.06 + 0.11 * math.sin(a), 0.2) for a in ang]
        for a, b in zip(pts[:-1], pts[1:]):
            d = smin(d, capsule(p, a, b, 0.014), 0.006)
        return d
    def bar(p):
        z = p[:, 2]; t = np.clip((z - BAR_Z0) / (BAR_Z1 - BAR_Z0), 0, 1)
        hh = BAR_H0 + (BAR_H1 - BAR_H0) * t
        d2 = np.abs(p[:, 1] - BAR_Y) - hh
        dz = z - BAR_Z1
        nose = np.sqrt(np.maximum(dz, 0) ** 2 + np.maximum(np.abs(p[:, 1] - BAR_Y) - 0.0, 0) ** 2) - BAR_H1
        d = np.where(dz > 0, nose, np.maximum(d2, BAR_Z0 - 0.04 - z))
        return np.maximum(d, np.abs(p[:, 0]) - 0.0045)
    return housing, grips, bar

def chain_teeth():
    """links round the edge of the bar, every other one a hooked cutter"""
    vs, fs, cut = [], [], []
    L = BAR_Z1 - BAR_Z0
    pts = []
    for z in np.arange(BAR_Z0 - 0.02, BAR_Z1, 0.0125):
        t = np.clip((z - BAR_Z0) / L, 0, 1); hh = BAR_H0 + (BAR_H1 - BAR_H0) * t
        pts.append((V(0, BAR_Y + hh + 0.004, z), V(0, 1, 0)))
        pts.append((V(0, BAR_Y - hh - 0.004, z), V(0, -1, 0)))
    for a in np.linspace(-math.pi / 2, math.pi / 2, 9):
        nrm = V(0, math.sin(a), math.cos(a))
        pts.append((V(0, BAR_Y, BAR_Z1) + nrm * (BAR_H1 + 0.004), nrm))
    off = 0
    for i, (c, nrm) in enumerate(pts):
        big = i % 4 in (0, 1)
        h = V(0.006, 0.0045 if not big else 0.007, 0.0055)
        cc = c + nrm * (0.002 if big else 0)
        v, f = box_mesh(cc, h)
        if abs(nrm[2]) > 0.3:      # turn the nose links to follow the curve
            R = rot_x(-math.atan2(nrm[2], nrm[1]))
            v = (v - cc) @ R.T + cc
        vs.append(v); fs.append(f + off); off += len(v)
    v = np.vstack(vs); f = np.vstack(fs)
    return v, f

def bar_mesh():
    """the guide bar as a thin slab: outline extruded 9 mm"""
    outl = []
    zs = np.linspace(BAR_Z0 - 0.04, BAR_Z1, 24)
    for z in zs:
        t = np.clip((z - BAR_Z0) / (BAR_Z1 - BAR_Z0), 0, 1); outl.append((z, BAR_Y + BAR_H0 + (BAR_H1 - BAR_H0) * t))
    for a in np.linspace(math.pi / 2, -math.pi / 2, 13)[1:-1]:
        outl.append((BAR_Z1 + BAR_H1 * math.cos(a), BAR_Y + BAR_H1 * math.sin(a)))
    for z in zs[::-1]:
        t = np.clip((z - BAR_Z0) / (BAR_Z1 - BAR_Z0), 0, 1); outl.append((z, BAR_Y - BAR_H0 - (BAR_H1 - BAR_H0) * t))
    o = np.array(outl); K = len(o)
    v = np.vstack([np.column_stack([np.full(K, x), o[:, 1], o[:, 0]]) for x in (-0.0045, 0.0045)])
    f = []
    for i in range(1, K - 1):        # fan each face (outline is convex)
        f.append([0, i, i + 1]); f.append([K, K + i + 1, K + i])
    for i in range(K):
        j = (i + 1) % K
        f += [[i, j, K + i], [j, K + j, K + i]]
    f = convex_outward(v, np.array(f), V(0, BAR_Y, (BAR_Z0 + BAR_Z1) / 2))
    # split shared verts so the flat faces shade flat
    v2 = v[f].reshape(-1, 3); f2 = np.arange(len(v2)).reshape(-1, 3)
    return v2, f2, vertex_normals(v2, f2)

def saw_prims():
    housing, grips, bar = saw_parts()
    hv, hf, hn = mesh_sdf(housing, V(-0.12, -0.12, -0.02), V(0.12, 0.14, 0.38), 0.0078, smooth=1)
    hao = sdf_ao(housing, hv, hn, 0.006, 4, 1.3)
    hc = mixc(hexc("#8c2216"), hexc("#4a120b"), smoothstep(0.3, 0.9, 0.5 + 0.5 * fbm(hv * 14, 3, seed=102)))
    hc = mixc(hc, hexc("#1c1712"), smoothstep(0.5, 0.85, 0.5 + 0.5 * fbm(hv * 7, 3, seed=103)) * 0.75)    # grease and grime
    fins = (hv[:, 0] > 0.068) & (hv[:, 2] > 0.085) & (hv[:, 2] < 0.25)
    hc[fins] = hexc("#2d2b29")
    hc = mixc(hc, hexc("#3a0403"), splatter(hv, 104, 0.35, 16) * 0.9)
    hc = hc * (0.35 + 0.65 * hao)[:, None]
    prims = [Prim(hv, hf, hn, hc, {"name": "saw_body", "rough": 0.45}, "housing")]
    gv, gf, gn = mesh_sdf(grips, V(-0.14, -0.11, -0.14), V(0.14, 0.2, 0.24), 0.0075, smooth=1)
    gc = mixc(hexc("#151515"), hexc("#262422"), 0.5 + 0.5 * fbm(gv * 30, 2, seed=105))
    gc = mixc(gc, hexc("#3a0403"), splatter(gv, 106, 0.4, 14))
    prims.append(Prim(gv, gf, gn, gc, {"name": "rubber", "rough": 0.7}, "grips"))
    bv, bf, bn = bar_mesh()
    bc = mixc(hexc("#8b8d90"), hexc("#55575a"), 0.5 + 0.5 * fbm(bv * 25, 2, seed=107))
    bl = splatter(bv, 108, 0.9, 12) * smoothstep(0.4, 0.75, bv[:, 2])
    bc = mixc(bc, mixc(hexc("#6a0906"), hexc("#2a0202"), bl), bl)
    prims.append(Prim(bv, bf, bn, bc, {"name": "steel", "rough": 0.3, "metal": 0.85}, "bar"))
    cv_, cf = chain_teeth()
    cc = mixc(hexc("#5c5d5f"), hexc("#4a0705"), smoothstep(0.35, 0.8, cv_[:, 2]) * 0.8)
    prims.append(Prim(cv_, cf, vertex_normals(cv_, cf), cc, {"name": "chain", "rough": 0.35, "metal": 0.9}, "chain"))
    # shift so the rear grip (where the hand closes) is the origin
    g0 = V(0, 0.055, -0.02)
    return [Prim(P.v - g0, P.f, P.n, P.c, P.mat, P.name) for P in prims]

def place(prims, R=None, t=V(0, 0, 0), sx=1):
    out = []
    for P in prims:
        v = P.v * V(sx, 1, 1); n = P.n * V(sx, 1, 1); f = P.f if sx == 1 else P.f[:, ::-1]
        if R is not None: v = v @ R.T; n = n @ R.T
        out.append(Prim(v + t, f, n, P.c, P.mat, P.name))
    return out

def build(which):
    S = saw_prims(); print("chainsaw", *write_glb(OUT + "chainsaw.glb", S, "chainsaw"))
    Lg = leg_prims(); print("butcher_leg", *write_glb(OUT + "butcher_leg.glb", Lg, "butcher_leg"))
    if "body" not in which: return
    B = [Prim(P.v + V(0, LIFT, 0), P.f, P.n, P.c, P.mat, P.name) for P in body_prims()]; print("butcher_body", *write_glb(OUT + "butcher_body.glb", B, "butcher_body"))
    H2 = HIP + V(0, LIFT, 0)
    full = B + place(Lg, None, H2) + place(Lg, None, H2 * V(-1, 1, 1), sx=-1) + place(S, SAW_R, SAW_AT + V(0, LIFT, 0))
    print('HIP', H2, 'SAW_AT', SAW_AT + V(0, LIFT, 0), 'SAW_R euler deg: rx -10 then ry -8')
    print("butcher", *write_glb(OUT + "butcher.glb", full, "butcher"))

if __name__ == "__main__":
    build(sys.argv[1:] or ["body"])
