"""Software preview renderer: reads a .glb we wrote and draws it lit, from a few angles."""
import sys, math
import numpy as np, cv2
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from sculpt import read_glb

def lin2srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)

def render(prims, yaw=25, pitch=8, size=700, ss=2, fov=28, dist=None, look=None, bg=(14, 12, 16), dark=False):
    allv = np.vstack([p[0] for p in prims])
    lo, hi = allv.min(0), allv.max(0)
    c = (lo + hi) / 2 if look is None else np.asarray(look, float)
    rad = np.linalg.norm(hi - lo) / 2
    if dist is None: dist = rad / math.tan(math.radians(fov / 2)) * 1.08
    yw, pt = math.radians(yaw), math.radians(pitch)
    eye = c + dist * np.array([math.sin(yw) * math.cos(pt), math.sin(pt), math.cos(yw) * math.cos(pt)])
    fwd = c - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    W = size * ss
    f = (W / 2) / math.tan(math.radians(fov / 2))
    img = np.zeros((W, W, 3), np.float32)
    img[:] = np.array(bg, np.float32) / 255.0
    # vertical background gradient
    g = np.linspace(1.15, 0.6, W)[:, None, None]
    img *= g
    key = np.array([-0.5, 0.75, 0.55]); key /= np.linalg.norm(key)
    rim = np.array([0.6, 0.3, -0.75]); rim /= np.linalg.norm(rim)
    tris_all = []
    for (v, fc, n, col, m) in prims:
        base = np.array(m["pbrMetallicRoughness"]["baseColorFactor"][:3])
        emit = np.array(m.get("emissiveFactor", [0, 0, 0]))
        rough = m["pbrMetallicRoughness"]["roughnessFactor"]
        rel = v - eye
        z = rel @ fwd
        x = rel @ right; y = rel @ up
        sx = W / 2 + f * x / np.maximum(z, 1e-3); sy = W / 2 - f * y / np.maximum(z, 1e-3)
        fn = n[fc].mean(1); fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9)
        two = m.get("doubleSided", False)
        view = -(v[fc].mean(1) - eye); view /= np.linalg.norm(view, axis=1, keepdims=True)
        facing = (fn * view).sum(1)
        if two:
            fn = np.where(facing[:, None] < 0, -fn, fn); facing = np.abs(facing)
        keep = facing > -0.05
        cc = col[fc].mean(1) * base
        dif = np.clip(fn @ key, 0, 1)
        h = (key + view); h /= np.linalg.norm(h, axis=1, keepdims=True)
        spec = np.clip((fn * h).sum(1), 0, 1) ** (2 + 60 * (1 - rough) ** 2) * (1 - rough) * 0.9
        rimv = np.clip(fn @ rim, 0, 1) ** 2 * 0.35
        amb = 0.10 + 0.08 * np.clip(fn[:, 1], 0, 1)
        lit = cc * (amb + 1.25 * dif)[:, None] + (spec + rimv)[:, None] * np.array([0.8, 0.85, 1.0]) + emit
        if dark: lit = cc * (0.05 + 0.4 * dif)[:, None] + emit + (rimv * 0.6)[:, None]
        zz = z[fc].mean(1)
        for i in np.nonzero(keep & (z[fc].min(1) > 0.01))[0]:
            tris_all.append((zz[i], np.stack([sx[fc[i]], sy[fc[i]]], 1), lit[i]))
    tris_all.sort(key=lambda t: -t[0])
    for _, pts, colr in tris_all:
        cv2.fillConvexPoly(img, np.round(pts * 4).astype(np.int32), [float(x) for x in colr], lineType=cv2.LINE_AA, shift=2)
    out = lin2srgb(img)
    out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
    return (out[:, :, ::-1] * 255).astype(np.uint8)

def sheet(path, out, views=((25, 8), (0, 2), (90, 5), (200, 10)), size=520, **kw):
    prims, g = read_glb(path)
    ims = [render(prims, yaw=a, pitch=b, size=size, **kw) for a, b in views]
    cv2.imwrite(out, np.hstack(ims))
    tris = sum(len(p[1]) for p in prims)
    return tris

if __name__ == "__main__":
    print(sheet(sys.argv[1], sys.argv[2]))
