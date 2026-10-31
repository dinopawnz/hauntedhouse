"""z-buffer rasteriser with smooth (per-vertex) shading for honest previews"""
import sys, math
import numpy as np, cv2
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from sculpt import read_glb

def lin2srgb(c):
    c = np.clip(c, 0, 1); return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)

def render(prims, yaw=25, pitch=8, size=640, fov=28, dist=None, look=None, bg=(14, 12, 16), lights=None, ss=2):
    allv = np.vstack([p[0] for p in prims])
    lo, hi = allv.min(0), allv.max(0)
    c = (lo + hi) / 2 if look is None else np.asarray(look, float)
    rad = np.linalg.norm(hi - lo) / 2
    if dist is None: dist = rad / math.tan(math.radians(fov / 2)) * 1.05
    yw, pt = math.radians(yaw), math.radians(pitch)
    eye = c + dist * np.array([math.sin(yw) * math.cos(pt), math.sin(pt), math.cos(yw) * math.cos(pt)])
    fwd = c - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
    W = size * ss; fpx = (W / 2) / math.tan(math.radians(fov / 2))
    img = np.zeros((W, W, 3), np.float32); img[:] = np.array(bg, np.float32) / 255 * np.linspace(1.2, 0.5, W)[:, None, None]
    zb = np.full((W, W), np.inf, np.float32)
    if lights is None:
        lights = [((-0.5, 0.75, 0.55), (1.25, 1.18, 1.05)), ((0.7, 0.25, -0.6), (0.35, 0.4, 0.55))]
    for (v, fc, n, col, m) in prims:
        base = np.array(m["pbrMetallicRoughness"]["baseColorFactor"][:3]); emit = np.array(m.get("emissiveFactor", [0, 0, 0]))
        rough = m["pbrMetallicRoughness"]["roughnessFactor"]; two = m.get("doubleSided", False)
        rel = v - eye; z = rel @ fwd; x = rel @ right; y = rel @ up
        sx = W / 2 + fpx * x / np.maximum(z, 1e-4); sy = W / 2 - fpx * y / np.maximum(z, 1e-4)
        view = -rel / np.linalg.norm(rel, axis=1, keepdims=True)
        nn = n.copy()
        if two:
            nn = np.where(((nn * view).sum(1) < 0)[:, None], -nn, nn)
        cc = col * base
        lit = cc * (0.07 + 0.06 * np.clip(nn[:, 1], 0, 1))[:, None]
        for L, Lc in lights:
            L = np.array(L, float); L /= np.linalg.norm(L)
            dif = np.clip(nn @ L, 0, 1)
            h = L + view; h /= np.linalg.norm(h, axis=1, keepdims=True)
            sp = np.clip((nn * h).sum(1), 0, 1) ** (2 + 80 * (1 - rough) ** 2) * (1 - rough) ** 1.5 * 1.2
            lit += cc * dif[:, None] * np.array(Lc) + sp[:, None] * np.array(Lc) * 0.8
        lit += emit
        # rasterise
        P = np.stack([sx, sy], 1)
        a, b, cidx = fc[:, 0], fc[:, 1], fc[:, 2]
        area = (P[b, 0] - P[a, 0]) * (P[cidx, 1] - P[a, 1]) - (P[cidx, 0] - P[a, 0]) * (P[b, 1] - P[a, 1])
        keep = (z[a] > 0.01) & (z[b] > 0.01) & (z[cidx] > 0.01) & (np.abs(area) > 1e-6)
        if not two: keep &= area < 0          # back-face cull (y is flipped)
        for t in np.nonzero(keep)[0]:
            ia, ib, ic = a[t], b[t], cidx[t]
            xs = P[[ia, ib, ic], 0]; ys = P[[ia, ib, ic], 1]
            x0, x1 = int(max(0, math.floor(xs.min()))), int(min(W - 1, math.ceil(xs.max())))
            y0, y1 = int(max(0, math.floor(ys.min()))), int(min(W - 1, math.ceil(ys.max())))
            if x1 < x0 or y1 < y0: continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            ar = area[t]
            w0 = ((xs[1] - gx) * (ys[2] - gy) - (xs[2] - gx) * (ys[1] - gy)) / ar
            w1 = ((xs[2] - gx) * (ys[0] - gy) - (xs[0] - gx) * (ys[2] - gy)) / ar
            w2 = 1 - w0 - w1
            ins = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
            if not ins.any(): continue
            zz = w0 * z[ia] + w1 * z[ib] + w2 * z[ic]
            sub = zb[y0:y1 + 1, x0:x1 + 1]
            upd = ins & (zz < sub)
            if not upd.any(): continue
            sub[upd] = zz[upd]
            colp = w0[..., None] * lit[ia] + w1[..., None] * lit[ib] + w2[..., None] * lit[ic]
            im = img[y0:y1 + 1, x0:x1 + 1]; im[upd] = colp[upd]
    out = lin2srgb(img); out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
    return (out[:, :, ::-1] * 255).astype(np.uint8)

def sheet(path, out, views, size=560, **kw):
    prims, g = read_glb(path)
    ims = []
    for vw in views:
        kk = dict(kw); kk.update(vw)
        ims.append(render(prims, size=size, **kk))
    cv2.imwrite(out, np.hstack(ims))
