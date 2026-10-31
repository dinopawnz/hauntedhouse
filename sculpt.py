"""
Tiny sculpting toolkit for the Hollow House monsters.

Shapes are signed distance functions (SDFs) blended together, meshed with
marching cubes, smoothed, coloured per vertex and written out as .glb files.
Everything is numpy; no Blender needed.
"""
import json, struct, math, os
H_SCALE = float(os.environ.get('HSCALE', '1'))      # >1 = coarser meshes, for light 'lo' versions
SUFFIX = os.environ.get('SUFFIX', '')
import numpy as np
from skimage.measure import marching_cubes
import scipy.sparse as sp

# ----------------------------------------------------------------------------
#  noise
# ----------------------------------------------------------------------------
def _hash(ix, iy, iz, seed):
    n = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 2654435761)
    n = (n ^ (n >> 13)) * 1274126177
    n = n ^ (n >> 16)
    return ((n & 0xFFFFFF).astype(np.float64) / 0xFFFFFF) * 2.0 - 1.0

def vnoise(p, seed=0):
    """smooth value noise, p (N,3) -> (N,) in about [-1,1]"""
    pi = np.floor(p).astype(np.int64)
    f = p - pi
    u = f * f * f * (f * (f * 6 - 15) + 10)
    x0, y0, z0 = pi[:, 0], pi[:, 1], pi[:, 2]
    out = 0.0
    for dx in (0, 1):
        wx = u[:, 0] if dx else 1 - u[:, 0]
        for dy in (0, 1):
            wy = u[:, 1] if dy else 1 - u[:, 1]
            for dz in (0, 1):
                wz = u[:, 2] if dz else 1 - u[:, 2]
                out = out + wx * wy * wz * _hash(x0 + dx, y0 + dy, z0 + dz, seed)
    return out

def fbm(p, octaves=4, seed=0, lac=2.0, gain=0.5):
    a, f, s = 1.0, 1.0, 0.0
    tot = 0.0
    for o in range(octaves):
        s = s + a * vnoise(p * f, seed + o * 17)
        tot += a
        a *= gain; f *= lac
    return s / tot

def ridged(p, octaves=3, seed=0):
    """sharp creases (cracks, veins, wrinkles): 0 on a crease, ~1 away from it"""
    s, a, tot = 0.0, 1.0, 0.0
    f = 1.0
    for o in range(octaves):
        s = s + a * np.abs(vnoise(p * f, seed + o * 31))
        tot += a; a *= 0.5; f *= 2.0
    return s / tot

# ----------------------------------------------------------------------------
#  SDF primitives (p is (N,3))
# ----------------------------------------------------------------------------
def V(*a): return np.array(a, dtype=np.float64)

def sphere(p, c, r):
    return np.linalg.norm(p - c, axis=1) - r

def ellipsoid(p, c, r):
    r = np.asarray(r, dtype=np.float64)
    q = (p - c) / r
    k0 = np.linalg.norm(q, axis=1)
    k1 = np.linalg.norm(q / r, axis=1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)

def capsule(p, a, b, r):
    pa = p - a; ba = b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0, 1)
    return np.linalg.norm(pa - h[:, None] * ba, axis=1) - r

def rcone(p, a, b, ra, rb):
    """capsule that tapers from radius ra at a to rb at b"""
    pa = p - a; ba = b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0, 1)
    return np.linalg.norm(pa - h[:, None] * ba, axis=1) - (ra + (rb - ra) * h)

def rbox(p, c, half, r=0.0):
    q = np.abs(p - c) - (np.asarray(half) - r)
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(np.max(q, axis=1), 0) - r

def torus_y(p, c, R, r):
    q = p - c
    d = np.sqrt(q[:, 0] ** 2 + q[:, 2] ** 2) - R
    return np.sqrt(d ** 2 + q[:, 1] ** 2) - r

def smin(a, b, k):
    if k <= 0: return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b + (a - b) * h - k * h * (1 - h)

def smax(a, b, k):
    return -smin(-a, -b, k)

def chain_capsules(p, pts, radii, k=0.0):
    """union of tapered capsules through a list of points"""
    d = None
    for i in range(len(pts) - 1):
        e = rcone(p, pts[i], pts[i + 1], radii[i], radii[i + 1])
        d = e if d is None else smin(d, e, k)
    return d

# rotations
def rot_x(a):
    c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def rot_y(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def rot_z(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

# ----------------------------------------------------------------------------
#  meshing
# ----------------------------------------------------------------------------
def mesh_sdf(f, lo, hi, h, chunk=400000, smooth=2, grad_normals=True):
    """f: callable (N,3)->(N,) signed distance. Returns verts (V,3), faces (F,3), normals (V,3)."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    h = h * H_SCALE
    n = np.ceil((hi - lo) / h).astype(int) + 1
    xs = lo[0] + np.arange(n[0]) * h
    ys = lo[1] + np.arange(n[1]) * h
    zs = lo[2] + np.arange(n[2]) * h
    vol = np.empty((n[0], n[1], n[2]), dtype=np.float32)
    # evaluate slab by slab along x to keep memory down
    YY, ZZ = np.meshgrid(ys, zs, indexing="ij")
    yz = np.stack([YY.ravel(), ZZ.ravel()], axis=1)
    per = max(1, chunk // len(yz))
    for i0 in range(0, n[0], per):
        i1 = min(n[0], i0 + per)
        xx = np.repeat(xs[i0:i1], len(yz))
        pts = np.column_stack([xx, np.tile(yz, (i1 - i0, 1))])
        vol[i0:i1] = f(pts).reshape(i1 - i0, n[1], n[2])
    # pad so the surface is closed
    vol = np.pad(vol, 1, constant_values=float(h * 4))
    v, fa, nn, _ = marching_cubes(vol, level=0.0, spacing=(h, h, h))
    v = v - h + lo
    v, fa = weld(v, fa)
    if smooth:
        v = taubin(v, fa, smooth)
    if grad_normals:
        nrm = sdf_normals(f, v, h * 0.5)
    else:
        nrm = vertex_normals(v, fa)
    fa = orient(v, fa, nrm)
    return v, fa, nrm

def orient(v, fa, nrm):
    """make every triangle wind counter-clockwise seen from outside (agrees with the outward normals)"""
    fn = np.cross(v[fa[:, 1]] - v[fa[:, 0]], v[fa[:, 2]] - v[fa[:, 0]])
    agree = (fn * nrm[fa].mean(1)).sum(1)
    fa = fa.copy()
    bad = agree < 0
    fa[bad] = fa[bad][:, ::-1]
    return fa

def weld(v, f, tol=1e-7):
    key = np.round(v / tol).astype(np.int64)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    v2 = v[idx]
    f2 = inv[f]
    ok = (f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2]) & (f2[:, 0] != f2[:, 2])
    return v2, f2[ok]

def adjacency(nv, f):
    i = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    j = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    A = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(nv, nv)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1
    return sp.diags(1.0 / deg) @ A

def taubin(v, f, iters=2, lam=0.5, mu=-0.53):
    W = adjacency(len(v), f)
    for _ in range(iters):
        v = v + lam * (W @ v - v)
        v = v + mu * (W @ v - v)
    return v

def vertex_normals(v, f):
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

def sdf_normals(f, v, e):
    g = np.zeros_like(v)
    for k in range(3):
        d = np.zeros(3); d[k] = e
        g[:, k] = f(v + d) - f(v - d)
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)

def sdf_ao(f, v, n, step=0.02, samples=5, k=1.0):
    """ambient occlusion from the distance field: dark in creases and under things"""
    occ = np.zeros(len(v))
    w = 1.0
    for i in range(1, samples + 1):
        d = step * i
        occ += w * np.maximum(0.0, d - f(v + n * d)) / d
        w *= 0.6
    return np.clip(1.0 - k * occ / 2.0, 0.0, 1.0)

# ----------------------------------------------------------------------------
#  parametric pieces (hair strands, chains, teeth...)
# ----------------------------------------------------------------------------
def tube(path, radii, sides=6, cap=True):
    """a tube along a polyline. path (K,3), radii (K,) -> verts, faces"""
    path = np.asarray(path, float); radii = np.broadcast_to(np.asarray(radii, float), (len(path),))
    K = len(path)
    t = np.gradient(path, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    # parallel transport frame
    up = np.array([0, 0, 1.0]) if abs(t[0, 1]) > 0.9 else np.array([0, 1.0, 0])
    n0 = np.cross(t[0], up); n0 /= np.linalg.norm(n0)
    N = [n0]
    for i in range(1, K):
        n = N[-1] - t[i] * (N[-1] @ t[i])
        nl = np.linalg.norm(n)
        n = n / nl if nl > 1e-9 else N[-1]
        N.append(n)
    N = np.array(N); B = np.cross(t, N)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = np.cos(ang)[None, :, None] * N[:, None, :] + np.sin(ang)[None, :, None] * B[:, None, :]
    verts = (path[:, None, :] + ring * radii[:, None, None]).reshape(-1, 3)
    faces = []
    for i in range(K - 1):
        for j in range(sides):
            a = i * sides + j; b = i * sides + (j + 1) % sides
            c = (i + 1) * sides + j; d = (i + 1) * sides + (j + 1) % sides
            faces += [[a, b, c], [b, d, c]]          # counter-clockwise seen from outside
    faces = np.array(faces)
    if cap:
        verts = np.vstack([verts, path[0], path[-1]])
        s0, s1 = len(verts) - 2, len(verts) - 1
        cf = [[s0, (j + 1) % sides, j] for j in range(sides)]
        cf += [[s1, (K - 1) * sides + j, (K - 1) * sides + (j + 1) % sides] for j in range(sides)]
        faces = np.vstack([faces, cf])
    return verts, faces

def merge(parts):
    """parts: list of (verts, faces) -> one (verts, faces)"""
    vs, fs, off = [], [], 0
    for v, f in parts:
        vs.append(v); fs.append(f + off); off += len(v)
    return np.vstack(vs), np.vstack(fs)

def transform(v, R=None, t=None, s=None):
    v = np.asarray(v, float)
    if s is not None: v = v * s
    if R is not None: v = v @ np.asarray(R).T
    if t is not None: v = v + t
    return v

# ----------------------------------------------------------------------------
#  colour helpers
# ----------------------------------------------------------------------------
def hexc(h):
    h = h.lstrip("#"); return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])

def mixc(a, b, t):
    t = np.clip(t, 0, 1)[..., None] if np.ndim(t) else np.clip(t, 0, 1)
    return a * (1 - t) + b * t

def srgb_to_linear(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)

# ----------------------------------------------------------------------------
#  a model = list of primitives, each with its own material
# ----------------------------------------------------------------------------
class Prim:
    def __init__(self, verts, faces, normals=None, colors=None, mat=None, name="part"):
        self.v = np.asarray(verts, np.float32)
        self.f = np.asarray(faces, np.int64)
        self.n = np.asarray(normals if normals is not None else vertex_normals(self.v.astype(float), self.f), np.float32)
        self.c = np.asarray(colors if colors is not None else np.ones((len(self.v), 3)), np.float32)
        self.mat = mat or {}
        self.name = name

def split_by_label(v, f, n, c, labels, mats, name="part"):
    """split one mesh into primitives by a per-vertex material label (face takes the majority)"""
    lf = labels[f]
    maj = np.where(lf[:, 1] == lf[:, 2], lf[:, 1], lf[:, 0])
    out = []
    for lab in np.unique(maj):
        fs = f[maj == lab]
        used = np.unique(fs)
        remap = -np.ones(len(v), np.int64); remap[used] = np.arange(len(used))
        out.append(Prim(v[used], remap[fs], n[used], c[used], mats[int(lab)], f"{name}_{int(lab)}"))
    return out

def write_glb(path, prims, name="model"):
    """write primitives (all in one mesh, one node) as a binary glTF"""
    if SUFFIX and path.endswith(".glb"): path = path[:-4] + SUFFIX + ".glb"
    bin_ = bytearray()
    views, accs, gprims, materials = [], [], [], []
    def add_view(data, target=None):
        while len(bin_) % 4: bin_.append(0)
        off = len(bin_); bin_.extend(data)
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target: v["target"] = target
        views.append(v); return len(views) - 1
    def add_acc(arr, ctype, typ, target, minmax=False, norm=False):
        bv = add_view(arr.tobytes(), target)
        a = {"bufferView": bv, "componentType": ctype, "count": int(arr.shape[0]), "type": typ}
        if norm: a["normalized"] = True
        if minmax:
            a["min"] = [float(x) for x in arr.min(axis=0)]; a["max"] = [float(x) for x in arr.max(axis=0)]
        accs.append(a); return len(accs) - 1
    matkey = {}
    for P in prims:
        if len(P.f) == 0: continue
        m = P.mat
        key = json.dumps(m, sort_keys=True)
        if key not in matkey:
            mm = {"name": m.get("name", "mat%d" % len(materials)),
                  "pbrMetallicRoughness": {"baseColorFactor": list(m.get("base", [1, 1, 1, 1])),
                                           "metallicFactor": float(m.get("metal", 0.0)),
                                           "roughnessFactor": float(m.get("rough", 0.6))}}
            if m.get("emit"): mm["emissiveFactor"] = [float(x) for x in m["emit"]]
            if m.get("double"): mm["doubleSided"] = True
            if m.get("alpha"):
                mm["alphaMode"] = "BLEND"
                mm["pbrMetallicRoughness"]["baseColorFactor"][3] = float(m["alpha"])
            materials.append(mm); matkey[key] = len(materials) - 1
        pos = P.v.astype(np.float32)
        nrm = P.n.astype(np.float32)
        col = srgb_to_linear(P.c.astype(np.float64)).astype(np.float32)
        ia = add_acc(pos, 5126, "VEC3", 34962, minmax=True)
        ib = add_acc(nrm, 5126, "VEC3", 34962)
        # 16-bit colours (RGBA, so every element stays 4-byte aligned): smaller, and no banding in the darks
        col16 = np.clip(np.round(col * 65535), 0, 65535).astype(np.uint16)
        col16 = np.ascontiguousarray(np.column_stack([col16, np.full(len(col16), 65535, np.uint16)]))
        ic = add_acc(col16, 5123, "VEC4", 34962, norm=True)
        if len(pos) < 65535:
            idx = P.f.astype(np.uint16).ravel(); ctype = 5123
        else:
            idx = P.f.astype(np.uint32).ravel(); ctype = 5125
        ii = add_acc(idx.reshape(-1, 1), ctype, "SCALAR", 34963)
        gprims.append({"attributes": {"POSITION": ia, "NORMAL": ib, "COLOR_0": ic}, "indices": ii,
                       "material": matkey[key], "mode": 4})
    while len(bin_) % 4: bin_.append(0)
    gltf = {"asset": {"version": "2.0", "generator": "hollow-house sculpt.py"},
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": name}],
            "meshes": [{"name": name, "primitives": gprims}], "materials": materials,
            "accessors": accs, "bufferViews": views, "buffers": [{"byteLength": len(bin_)}]}
    js = json.dumps(gltf, separators=(",", ":")).encode()
    while len(js) % 4: js += b" "
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(bin_))
    out += struct.pack("<II", len(js), 0x4E4F534A) + js
    out += struct.pack("<II", len(bin_), 0x004E4942) + bytes(bin_)
    with open(path, "wb") as fh: fh.write(out)
    tris = sum(len(P.f) for P in prims)
    return tris, len(out)

def read_glb(path):
    """read back our own glb (for checking + preview)"""
    b = open(path, "rb").read()
    magic, ver, L = struct.unpack("<III", b[:12])
    assert magic == 0x46546C67 and ver == 2 and L == len(b), "bad header"
    jl, jt = struct.unpack("<II", b[12:20]); assert jt == 0x4E4F534A
    g = json.loads(b[20:20 + jl])
    bl, bt = struct.unpack("<II", b[20 + jl:28 + jl]); assert bt == 0x004E4942
    binb = b[28 + jl:28 + jl + bl]
    assert g["buffers"][0]["byteLength"] == bl
    def acc(i):
        a = g["accessors"][i]; v = g["bufferViews"][a["bufferView"]]
        dt = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[a["componentType"]]
        nc = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[a["type"]]
        assert v["byteOffset"] % 4 == 0 and v["byteOffset"] + v["byteLength"] <= bl
        arr = np.frombuffer(binb, dt, a["count"] * nc, v["byteOffset"]).reshape(a["count"], nc)
        return arr
    out = []
    for pr in g["meshes"][0]["primitives"]:
        v = acc(pr["attributes"]["POSITION"]).astype(float)
        n = acc(pr["attributes"]["NORMAL"]).astype(float)
        ca = g["accessors"][pr["attributes"]["COLOR_0"]]
        c = acc(pr["attributes"]["COLOR_0"]).astype(float)
        if ca["componentType"] == 5123: c = c[:, :3] / 65535.0
        f = acc(pr["indices"]).reshape(-1, 3).astype(np.int64)
        assert f.max() < len(v)
        m = g["materials"][pr["material"]]
        out.append((v, f, n, c, m))
    return out, g

def convex_outward(v, f, center=None):
    """for a closed, roughly convex piece: wind every face away from its centre"""
    c = v.mean(0) if center is None else center
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    bad = (fn * (v[f].mean(1) - c)).sum(1) < 0
    f = f.copy(); f[bad] = f[bad][:, ::-1]
    return f
