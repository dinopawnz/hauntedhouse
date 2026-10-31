"""Shared body parts for the monsters: gaunt limbs, ribcages, long-fingered hands, cloth."""
import sys, math
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from sculpt import *

def limb(p, pts, radii, k=0.02):
    return chain_capsules(p, [np.asarray(q, float) for q in pts], radii, k)

def bony_limb(p, a, b, ra, rb, knob=1.25, k=0.012):
    """a limb segment with knobbly joints - reads as starved"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = rcone(p, a, b, ra, rb)
    d = smin(d, sphere(p, a, ra * knob), k)
    d = smin(d, sphere(p, b, rb * knob), k)
    return d

def ribcage(p, c, r, ribs=0.0025, freq=95.0, front_open=True):
    """ellipsoid chest with visible ribs (grooves) on the sides and front"""
    c = np.asarray(c, float)
    d = ellipsoid(p, c, r)
    q = p - c
    side = smoothstep(0.2, 0.9, np.abs(q[:, 0]) / r[0] + 0.4 * (q[:, 2] / r[2]))
    band = smoothstep(-r[1] * 0.85, -r[1] * 0.1, q[:, 1]) * smoothstep(r[1] * 0.75, r[1] * 0.2, q[:, 1])
    # rib grooves slope down towards the front
    phase = (q[:, 1] + 0.35 * np.abs(q[:, 0]) - 0.25 * q[:, 2]) * freq
    d = d + ribs * np.maximum(0, np.sin(phase)) * band * side
    if front_open:   # sternum ridge
        d = smin(d, capsule(p, c + V(0, r[1] * 0.5, r[2] * 0.85), c + V(0, -r[1] * 0.45, r[2] * 0.92), 0.008), 0.02)
    return d

def hand_sdf(p, wrist, fwd, up, size=1.0, curl=0.35, spread=0.18, finger_len=1.0, claw=0.0, knob=1.18):
    """a long-fingered hand. fwd = from wrist towards the fingertips, up = back of the hand.
    returns (sdf, nail_sdf)"""
    fwd = np.asarray(fwd, float); fwd /= np.linalg.norm(fwd)
    up = np.asarray(up, float); up = up - fwd * (up @ fwd); up /= np.linalg.norm(up)
    side = np.cross(up, fwd)
    W = np.asarray(wrist, float)
    s = size
    palm_c = W + fwd * 0.045 * s
    d = ellipsoid(p, palm_c, None) if False else None
    # palm as a flattened ellipsoid in the hand frame
    q = p - palm_c
    lq = np.stack([q @ side, q @ up, q @ fwd], 1)
    d = ellipsoid(lq, V(0, 0, 0), (0.038 * s, 0.013 * s, 0.05 * s))
    d = smin(d, capsule(p, W - fwd * 0.02 * s, W + fwd * 0.01 * s, 0.022 * s), 0.02 * s)     # wrist
    nails = None
    for i, (off, ln) in enumerate([(-0.024, 0.88), (-0.008, 1.0), (0.008, 0.96), (0.023, 0.8)]):
        base = palm_c + fwd * 0.045 * s + side * off * s * (1 + spread)
        dirn = fwd + side * off * spread * 6
        dirn /= np.linalg.norm(dirn)
        L = 0.034 * s * ln * finger_len
        pts = [base]
        cur = base.copy(); dd = dirn.copy()
        for seg, lf in enumerate((1.0, 0.75, 0.6)):
            dd = dd * math.cos(curl) - up * math.sin(curl)   # curl towards the palm
            dd /= np.linalg.norm(dd)
            cur = cur + dd * L * lf
            pts.append(cur.copy())
        rads = [0.0085 * s, 0.0075 * s, 0.0065 * s, 0.0052 * s]
        fd = None
        for a, b, ra, rb in zip(pts[:-1], pts[1:], rads[:-1], rads[1:]):
            e = bony_limb(p, a, b, ra, rb, knob=knob, k=0.004 * s)
            fd = e if fd is None else smin(fd, e, 0.004 * s)
        d = smin(d, fd, 0.008 * s)
        tip = pts[-1]
        nail_c = tip + dd * (0.004 + claw * 0.012) * s + up * 0.003 * s
        nl = capsule(p, tip - dd * 0.006 * s + up * 0.003 * s, nail_c, 0.0045 * s * (1 - 0.5 * claw))
        nails = nl if nails is None else np.minimum(nails, nl)
    # thumb
    tb = W + fwd * 0.02 * s - side * 0.03 * s
    td = (fwd * 0.7 - side * 0.6 - up * 0.2); td /= np.linalg.norm(td)
    t1 = tb + td * 0.035 * s; t2 = t1 + (td * 0.8 - up * 0.4) / np.linalg.norm(td * 0.8 - up * 0.4) * 0.03 * s
    d = smin(d, bony_limb(p, tb, t1, 0.011 * s, 0.009 * s, 1.15, 0.005 * s), 0.01 * s)
    d = smin(d, bony_limb(p, t1, t2, 0.009 * s, 0.0065 * s, 1.15, 0.004 * s), 0.004 * s)
    nl = capsule(p, t2 - td * 0.004 * s + up * 0.002 * s, t2 + td * (0.004 + claw * 0.012) * s, 0.0048 * s)
    nails = np.minimum(nails, nl)
    # knuckles and tendons
    d = d - 0.0012 * s * (1 - ridged(p * 160 / s, 2, seed=44))
    return d, nails

def tatter(y, x, z, base, amp, freq, seed):
    """a ragged hem height at each (x,z)"""
    ang = np.arctan2(x, z)
    pts = np.stack([np.cos(ang) * freq, np.sin(ang) * freq, np.zeros_like(ang)], 1)
    torn = fbm(pts, 3, seed=seed)
    spikes = np.abs(vnoise(pts * 2.7, seed + 7))
    return base + amp * torn + amp * 1.4 * spikes ** 3

def skin_colour(v, n, ao, base="#9c9a8e", dark="#5d5c58", vein="#4f5577", seed=1):
    col = mixc(hexc(base), hexc(dark), 0.5 + 0.5 * fbm(v * 6, 3, seed=seed))
    col = mixc(col, hexc(vein), smoothstep(0.72, 0.95, 1 - ridged(v * 22, 3, seed=seed + 5)) * 0.5)
    return col * (0.3 + 0.7 * ao)[:, None]

def cloth_colour(v, ao, base="#bdb5a3", stain="#5b4a32", blood="#4a0b08", blood_amt=0.0, seed=3, dirt_y=0.4):
    col = mixc(hexc(base), hexc(stain), smoothstep(0.1, 0.8, fbm(v * 4, 4, seed=seed)) * 0.6)
    col = mixc(col, hexc(stain), smoothstep(dirt_y, 0.0, v[:, 1]) * 0.7)            # filthy hem
    if blood_amt:
        b = smoothstep(0.55 - blood_amt * 0.5, 0.9 - blood_amt * 0.4, fbm(v * 7, 4, seed=seed + 9))
        col = mixc(col, hexc(blood), b)
    # folds read darker in the creases
    col = col * (0.85 + 0.15 * (0.5 + 0.5 * vnoise(v * np.array([18, 2, 18]), seed + 3)))[:, None]
    return col * (0.25 + 0.75 * ao)[:, None]

def mesh_parts(parts, bounds, h, labels_fn=None, smooth=2):
    """parts: list of (name, sdf_fn). meshes the union, then labels every vertex by the nearest part"""
    def f(p):
        d = None
        for _, g in parts:
            e = g(p); d = e if d is None else np.minimum(d, e)
        return d
    v, fa, n = mesh_sdf(f, bounds[0], bounds[1], h, smooth=smooth)
    dists = np.stack([g(v) for _, g in parts], 1)
    lab = np.argmin(dists, 1)
    return f, v, fa, n, lab
