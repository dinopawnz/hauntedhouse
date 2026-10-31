"""THE CHAINED WRAITH - three metres of rotten shroud, hunched, hood up. Inside the hood a
skull-tight face with a jaw hanging far too long and two cold points of light for eyes.
Skeletal arms out of wide sleeves, iron manacles, chains that drag on the floor behind it.
Original design. Faces +z; origin on the floor under it (it floats a hand's width off it)."""
import sys, math
import numpy as np
sys.path.insert(0, "/home/claude/hh13/tools")
from sculpt import *
from body import *

H = 2.95
SHL, SHR = V(-0.27, 2.36, -0.02), V(0.27, 2.36, -0.02)
ELL, ELR = V(-0.36, 1.86, 0.20), V(0.34, 1.9, 0.26)
WRL, WRR = V(-0.30, 1.50, 0.52), V(0.27, 1.58, 0.58)
HEAD = V(0, 2.62, 0.16)        # face centre, forward of the body: it's hunched

def robe(p):
    q = p - V(0, 0, 0)
    ang = np.arctan2(q[:, 0], q[:, 2] + 0.1)
    # main shroud: hunched cone from the shoulders to a wide, dragging hem
    top = 2.42; hem = 0.22
    t = np.clip((top - p[:, 1]) / (top - hem), 0, 1)
    rad = 0.24 + 0.34 * t ** 1.15 + (0.03 + 0.03 * t) * np.sin(ang * 7 + 0.4) + 0.018 * np.sin(ang * 13 + 2.0) * t
    cz = 0.06 * (1 - t) ** 2                                                    # leans forward at the top
    r_xz = np.sqrt(q[:, 0] ** 2 + ((q[:, 2] - cz) * 1.2) ** 2)
    d = np.maximum(r_xz - rad, p[:, 1] - top)
    d = smin(d, ellipsoid(p, V(0, 2.36, -0.04), (0.34, 0.16, 0.24)), 0.12)      # hunched shoulders
    d = smin(d, ellipsoid(p, V(0, 2.5, -0.14), (0.24, 0.18, 0.16)), 0.1)        # the hump of its back
    # hood: a deep cowl with the opening facing forward
    hood = ellipsoid(p, HEAD + V(0, 0.04, 0.0), (0.2, 0.26, 0.25))
    hood = smax(hood, -ellipsoid(p, HEAD + V(0, -0.02, 0.17), (0.13, 0.2, 0.2)), 0.02)
    hood = smin(hood, rcone(p, HEAD + V(0, 0.2, -0.12), HEAD + V(0, 0.32, -0.32), 0.07, 0.02), 0.08)   # the cowl's point
    d = smin(d, hood, 0.1)
    # wide sleeves, open at the forearm
    for sh, el, wr in ((SHL, ELL, WRL), (SHR, ELR, WRR)):
        end = el + (wr - el) * 0.3
        sl = rcone(p, sh, end, 0.11, 0.17)
        sl = smax(sl, -rcone(p, el, end + (end - el) * 0.4, 0.07, 0.14), 0.02)      # hollow mouth of the sleeve
        d = smin(d, sl, 0.06)
    # torn hem
    hemh = tatter(p[:, 1], p[:, 0], p[:, 2], 0.26, 0.09, 3.1, seed=40)
    d = smax(d, hemh - p[:, 1], 0.02)
    # heavy cloth folds
    d = d + 0.008 * fbm(p * np.array([7, 1.6, 7]), 3, seed=41) + 0.003 * fbm(p * 25, 2, seed=42)
    return d

def face(p):
    """skull-tight face, deep black sockets, a jaw dropped open far too long"""
    c = HEAD
    d = ellipsoid(p, c + V(0, 0.03, -0.03), (0.085, 0.1, 0.1))
    for s in (-1, 1):
        d = smin(d, ellipsoid(p, c + V(s * 0.05, -0.005, 0.045), (0.02, 0.015, 0.02)), 0.02)       # cheekbones
        d = smax(d, -ellipsoid(p, c + V(s * 0.06, -0.04, 0.035), (0.022, 0.035, 0.03)), 0.01)      # sunken cheeks
    d = smin(d, capsule(p, c + V(-0.045, 0.035, 0.07), c + V(0.045, 0.035, 0.07), 0.012), 0.02)   # brow
    for s in (-1, 1):
        d = smax(d, -ellipsoid(p, c + V(s * 0.032, 0.008, 0.08), (0.024, 0.024, 0.03)), 0.006)     # empty sockets
    d = smax(d, -ellipsoid(p, c + V(0, -0.025, 0.1), (0.012, 0.016, 0.02)), 0.004)               # nose hole
    # the long, dropped jaw and gaping mouth
    jaw_t = c + V(0, -0.07, 0.04); jaw_b = c + V(0, -0.27, 0.06)
    d = smin(d, rcone(p, jaw_t, jaw_b, 0.055, 0.03), 0.04)
    d = smin(d, rcone(p, c + V(0, 0.0, -0.02), c + V(0, -0.2, 0.0), 0.05, 0.04), 0.05)           # stretched neck/jaw skin
    mouth = rcone(p, c + V(0, -0.06, 0.09), c + V(0, -0.22, 0.09), 0.022, 0.016)
    d = smax(d, -mouth, 0.01)
    d = d - 0.0012 * (1 - ridged(p * 60, 2, seed=48)) + 0.0008 * fbm(p * 90, 2, seed=49)
    return d

def hands():
    out = []
    for W, el in ((WRL, ELL), (WRR, ELR)):
        fwd = (W - el); fwd = fwd / np.linalg.norm(fwd) + V(0, -0.5, 0.2)
        up = V(0, 1, 0.3)
        def f(p, W=W, fwd=fwd, up=up, el=el):
            d, nl = hand_sdf(p, W, fwd, up, size=1.35, curl=0.42, spread=0.4, finger_len=1.7, claw=0.8)
            arm = bony_limb(p, el + (W - el) * 0.1, W, 0.03, 0.02, 1.3)       # bone-thin forearm out of the sleeve
            return np.minimum(smin(d, arm, 0.015), nl)
        lo = np.minimum(W, el) - 0.2; hi = np.maximum(W, el) + 0.25
        v, fa, n = mesh_sdf(f, lo, hi, 0.0068, smooth=1)
        d, nl = hand_sdf(v, W, fwd, up, size=1.35, curl=0.42, spread=0.4, finger_len=1.7, claw=0.8)
        lab = ((nl < d) & (nl < 0.004)).astype(int)
        ao = sdf_ao(f, v, n, 0.008, 4, 1.2)
        col = skin_colour(v, n, ao, base="#b9b8ae", dark="#7a776c", vein="#59637a", seed=8)
        col[lab == 1] = hexc("#1a140f") * (0.4 + 0.6 * ao[lab == 1])[:, None]
        out += split_by_label(v, fa, n, col, lab, {0: {"name": "bone", "rough": 0.5}, 1: {"name": "claws", "rough": 0.25}}, "hand")
    return out

def torus_link(c, axis_a, axis_b, R=0.03, r=0.0075, stretch=1.45, nu=8, nv=5):
    """one oval chain link centred at c, lying in the plane of axis_a (long) and axis_b"""
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)[:, None]
    w = np.linspace(0, 2 * np.pi, nv, endpoint=False)[None, :]
    a = np.asarray(axis_a, float); a /= np.linalg.norm(a)
    b = np.asarray(axis_b, float); b = b - a * (a @ b); b /= np.linalg.norm(b)
    nrm = np.cross(a, b)
    ring_dir = np.cos(u)[..., None] * a * stretch + np.sin(u)[..., None] * b
    ring_dir_n = ring_dir / np.linalg.norm(ring_dir, axis=-1, keepdims=True)
    centre = c + ring_dir * R
    pts = centre + (np.cos(w)[..., None] * ring_dir_n + np.sin(w)[..., None] * nrm) * r
    pts = pts.reshape(-1, 3)
    faces = []
    for i in range(nu):
        for j in range(nv):
            p0 = i * nv + j; p1 = ((i + 1) % nu) * nv + j; p2 = i * nv + (j + 1) % nv; p3 = ((i + 1) % nu) * nv + (j + 1) % nv
            faces += [[p0, p1, p2], [p1, p3, p2]]
    faces = np.array(faces)
    # wind outward from each ring centre
    cen = np.broadcast_to(centre, (nu, nv, 3)).reshape(-1, 3)
    fn = np.cross(pts[faces[:, 1]] - pts[faces[:, 0]], pts[faces[:, 2]] - pts[faces[:, 0]])
    bad = (fn * (pts[faces].mean(1) - cen[faces[:, 0]])).sum(1) < 0
    faces[bad] = faces[bad][:, ::-1]
    return pts, faces

def chain_along(path, pitch=0.058, R=0.026, r=0.0072):
    """links along a polyline, alternating orientation like a real chain"""
    path = np.asarray(path, float)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1); cum = np.concatenate([[0], np.cumsum(seg)])
    L = cum[-1]; n = int(L / pitch)
    parts = []
    for i in range(n):
        s = (i + 0.5) * pitch
        k = np.searchsorted(cum, s) - 1; k = min(max(k, 0), len(seg) - 1)
        t = (s - cum[k]) / max(seg[k], 1e-9)
        c = path[k] + (path[k + 1] - path[k]) * t
        a = path[k + 1] - path[k]; a /= np.linalg.norm(a)
        side = np.cross(a, [0, 1, 0]) if abs(a[1]) < 0.95 else np.cross(a, [1, 0, 0])
        side /= np.linalg.norm(side)
        if i % 2: side = np.cross(a, side)
        import os
        lo = os.environ.get("CHAIN_LO") is not None   # light build: 6x3 links (36 tris) instead of 8x5 (80)
        parts.append(torus_link(c, a, side, R, r, nu=6 if lo else 8, nv=3 if lo else 5))
    return merge(parts)

def catenary(a, b, sag, n=40):
    t = np.linspace(0, 1, n)[:, None]
    p = a + (b - a) * t
    p[:, 1] -= sag * 4 * t[:, 0] * (1 - t[:, 0])
    return p

def chains():
    parts = []
    # from each manacle down to the floor and dragging behind
    for W, back in ((WRL, -1.2), (WRR, -1.05)):
        hang = catenary(W + V(0, -0.03, 0), V(W[0] * 1.6, 0.03, W[2] - 0.15), 0.08, 30)
        drag = np.array([V(W[0] * 1.6, 0.03, W[2] - 0.15), V(W[0] * 1.9, 0.025, -0.4), V(W[0] * 1.3, 0.025, back)])
        parts.append(chain_along(np.vstack([hang, drag[1:]])))
        # the manacle itself
        m = torus_link(W + V(0, 0.0, 0), V(1, 0, 0), V(0, 0, 1), R=0.045, r=0.016, stretch=1.0, nu=18, nv=8)
        parts.append(m)
    # a chain wrapped round its body, crossing the chest
    ring = []
    for k in range(48):
        a = k / 47 * 2 * np.pi
        ring.append(V(0.36 * math.sin(a), 1.95 + 0.25 * math.cos(a + 0.6), 0.27 * math.cos(a) + 0.02))
    parts.append(chain_along(np.array(ring)))
    v, f = merge(parts)
    n = vertex_normals(v, f)
    rust = smoothstep(0.2, 0.8, 0.5 + 0.5 * fbm(v * 30, 3, seed=55))
    col = mixc(hexc("#3a3634"), hexc("#5e321c"), rust * 0.8)
    return [Prim(v, f, n, col, {"name": "iron", "rough": 0.6, "metal": 0.75}, "chains")]

def build():
    v, fa, n = mesh_sdf(robe, V(-0.75, 0.1, -0.75), V(0.75, 3.05, 0.75), 0.024, smooth=2)
    ao = sdf_ao(robe, v, n, 0.03, 5, 1.4)
    col = cloth_colour(v, ao, base="#3a3b42", stain="#16161a", seed=43, dirt_y=0.9)
    col = mixc(col, hexc("#5a5c62") * (0.3 + 0.7 * ao)[:, None], smoothstep(0.6, 0.95, 0.5 + 0.5 * fbm(v * 3, 3, seed=44)) * 0.4)
    # the hood's mouth is black inside
    inner = smoothstep(0.17, 0.08, np.linalg.norm((v - (HEAD + V(0, -0.01, 0.06))) / np.array([1, 1.2, 1.4]), axis=1))
    col = col * (1 - 0.9 * inner)[:, None]
    prims = [Prim(v, fa, n, col, {"name": "shroud", "rough": 0.95}, "robe")]
    fv, ff, fn = mesh_sdf(face, HEAD - V(0.12, 0.33, 0.14), HEAD + V(0.12, 0.15, 0.13), 0.0055, smooth=2)
    fao = sdf_ao(face, fv, fn, 0.008, 5, 1.6)
    fc = skin_colour(fv, fn, fao, base="#a7a497", dark="#6b675d", vein="#4c5368", seed=9)
    fc = fc * (0.55 + 0.45 * smoothstep(-0.05, 0.1, fv[:, 2] - HEAD[2]))[:, None]   # in the shadow of the hood
    mdist = np.linalg.norm((fv - (HEAD + V(0, -0.14, 0.09))) / np.array([0.03, 0.12, 0.04]), axis=1)
    fc = mixc(fc, hexc("#0a0303"), smoothstep(1.4, 0.9, mdist))                       # black throat
    for s in (-1, 1):
        sd = np.linalg.norm(fv - (HEAD + V(s * 0.032, 0.008, 0.075)), axis=1)
        fc = mixc(fc, hexc("#050505"), smoothstep(0.03, 0.018, sd))                      # empty sockets
    prims.append(Prim(fv, ff, fn, fc, {"name": "bone", "rough": 0.5}, "face"))
    # cold points of light deep in the sockets
    for s in (-1, 1):
        c = HEAD + V(s * 0.032, 0.006, 0.062)
        ev, ef, en = mesh_sdf(lambda p, c=c: sphere(p, c, 0.0065), c - 0.01, c + 0.01, 0.0015, smooth=1)
        prims.append(Prim(ev, ef, en, np.tile(hexc("#d8ecff"), (len(ev), 1)), {"name": "glow", "rough": 0.1, "emit": [0.75, 0.9, 1.0]}, "eye"))
    # a few broken teeth in the gaping mouth
    tv, tf = [], []
    rng = np.random.default_rng(3)
    for i in range(7):
        x = -0.018 + i * 0.006
        for top in (True, False):
            if rng.random() < 0.3: continue
            y = HEAD[1] - (0.075 if top else 0.205); L = rng.uniform(0.008, 0.016)
            pth = np.array([V(x, y, HEAD[2] + 0.088), V(x + rng.normal(0, 0.002), y + (-L if top else L), HEAD[2] + 0.09)])
            vv, ff2 = tube(pth, [0.0028, 0.0012], sides=5)
            tf.append(ff2 + sum(len(a) for a in tv)); tv.append(vv)
    tv = np.vstack(tv); tf = np.vstack(tf)
    prims.append(Prim(tv, tf, vertex_normals(tv, tf), np.tile(hexc("#b8a77c"), (len(tv), 1)), {"name": "teeth", "rough": 0.3}, "teeth"))
    prims += hands() + chains() + tatters()
    tris, size = write_glb("/home/claude/hh13/out/wraith.glb", prims, "wraith")
    for P in prims: print("  ", P.name, len(P.f))
    print("wraith", tris, "tris", size // 1024, "KB")

def tatters():
    """long ragged strips hanging from the hem, almost to the floor"""
    rng = np.random.default_rng(9)
    vs, fs, off = [], [], 0
    for k in range(46):
        a = rng.uniform(0, 2 * np.pi)
        r0 = 0.56 + rng.normal(0, 0.03)
        top = V(r0 * math.sin(a), 0.42 + rng.uniform(-0.04, 0.06), r0 * math.cos(a) / 1.2 + 0.0)
        L = rng.uniform(0.2, 0.38); w = rng.uniform(0.03, 0.07)
        tang = V(math.cos(a), 0, -math.sin(a))
        rows = 8
        for i in range(rows + 1):
            t = i / rows
            c = top + V(0, -L * t, 0) + V(math.sin(a), 0, math.cos(a)) * (0.03 * t) + tang * 0.02 * math.sin(t * 5 + k)
            ww = w * (1 - 0.7 * t)
            vs.append(c - tang * ww / 2); vs.append(c + tang * ww / 2)
        for i in range(rows):
            p0 = off + i * 2
            fs += [[p0, p0 + 1, p0 + 2], [p0 + 1, p0 + 3, p0 + 2]]
        off += (rows + 1) * 2
    v = np.array(vs); f = np.array(fs)
    col = np.tile(hexc("#2e2f35"), (len(v), 1)) * (0.55 + 0.45 * (v[:, 1] > 0.25))[:, None]
    return [Prim(v, f, vertex_normals(v, f), col, {"name": "shroud_rags", "rough": 0.95, "double": True}, "rags")]

if __name__ == "__main__":
    build()
