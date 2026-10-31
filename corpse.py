"""THE DEAD - starved corpses frozen mid-crawl, for the room where they swarm over the walls.
Two poses. Modelled lying on a surface: up (+y) is away from the wall, the head points +z.
Origin = on the surface under the middle of the body. Light, because dozens appear at once."""
import sys, math
import numpy as np
sys.path.insert(0, "/home/claude/hh13/tools")
from sculpt import *
from body import *

def corpse_sdf(pose):
    def f(p):
        # torso low to the surface, chest raised a little, head up and reaching
        d = ribcage(p, V(0, 0.16, 0.18), (0.15, 0.1, 0.17), ribs=0.006, freq=70)
        d = smin(d, ellipsoid(p, V(0, 0.12, -0.25), (0.13, 0.085, 0.12)), 0.08)
        d = smin(d, rcone(p, V(0, 0.12, -0.22), V(0, 0.15, 0.12), 0.07, 0.1), 0.06)
        hx, hy, hz = pose["head"]
        d = smin(d, rcone(p, V(0, 0.2, 0.32), V(0, hy - 0.05, hz - 0.08), 0.05, 0.04), 0.04)      # neck stretched forward
        head = ellipsoid(p, V(hx, hy, hz), (0.075, 0.085, 0.095))
        head = smin(head, ellipsoid(p, V(hx, hy - 0.07, hz + 0.03), (0.05, 0.05, 0.06)), 0.03)   # hanging jaw
        for s in (-1, 1):
            head = smax(head, -sphere(p, V(hx + s * 0.03, hy + 0.01, hz + 0.08), 0.019), 0.006)   # sockets
        head = smax(head, -ellipsoid(p, V(hx, hy - 0.06, hz + 0.08), (0.025, 0.04, 0.03)), 0.008)  # mouth open, screaming
        d = smin(d, head, 0.03)
        # limbs splayed like a lizard's, gripping the surface
        for side, s in (("L", -1), ("R", 1)):
            sh = V(s * 0.17, 0.17, 0.25); el = pose["el" + side] * np.array([s, 1, 1]); wr = pose["wr" + side] * np.array([s, 1, 1])
            d = smin(d, bony_limb(p, sh, el, 0.042, 0.033, 1.25), 0.03)
            d = smin(d, bony_limb(p, el, wr, 0.033, 0.024, 1.25), 0.012)
            fing = wr + (wr - el) / np.linalg.norm(wr - el) * 0.09; fing[1] = 0.01
            d = smin(d, rcone(p, wr, fing, 0.022, 0.012), 0.02)                                      # flat clawing hand
            hp = V(s * 0.1, 0.12, -0.3); kn = pose["kn" + side] * np.array([s, 1, 1]); an = pose["an" + side] * np.array([s, 1, 1])
            d = smin(d, bony_limb(p, hp, kn, 0.062, 0.043, 1.2), 0.04)
            d = smin(d, bony_limb(p, kn, an, 0.043, 0.027, 1.25), 0.012)
            d = smin(d, ellipsoid(p, an + V(0, -0.02, -0.05), (0.03, 0.02, 0.07)), 0.02)
        d = smax(d, -p[:, 1] + 0.002, 0.01)          # flat where it touches the surface
        return d - 0.003 * fbm(p * 18, 2, seed=pose["seed"])
    return f

POSES = {
    "corpse_a": dict(seed=1, head=(0.0, 0.33, 0.55),
                     elL=V(0.38, 0.22, 0.42), wrL=V(0.42, 0.04, 0.72), elR=V(0.36, 0.25, 0.2), wrR=V(0.5, 0.04, 0.32),
                     knL=V(0.36, 0.2, -0.22), anL=V(0.34, 0.05, -0.62), knR=V(0.3, 0.18, -0.5), anR=V(0.2, 0.05, -0.9)),
    "corpse_b": dict(seed=2, head=(0.04, 0.27, 0.6),
                     elL=V(0.4, 0.2, 0.15), wrL=V(0.55, 0.04, 0.3), elR=V(0.36, 0.24, 0.45), wrR=V(0.36, 0.04, 0.8),
                     knL=V(0.32, 0.2, -0.45), anL=V(0.22, 0.05, -0.85), knR=V(0.38, 0.2, -0.2), anR=V(0.4, 0.05, -0.6)),
}

def build(name):
    f = corpse_sdf(POSES[name])
    v, fa, n = mesh_sdf(f, V(-0.68, -0.02, -1.05), V(0.68, 0.48, 0.9), 0.022, smooth=2)
    ao = sdf_ao(f, v, n, 0.025, 4, 1.4)
    col = skin_colour(v, n, ao, base="#8a9089", dark="#4e5551", vein="#3c4560", seed=POSES[name]["seed"] * 7)
    col = mixc(col, hexc("#3a1a14"), smoothstep(0.55, 0.85, 0.5 + 0.5 * fbm(v * 4, 3, seed=9)) * 0.55)     # rot and bruising
    hx, hy, hz = POSES[name]["head"]
    dm = np.linalg.norm((v - V(hx, hy - 0.06, hz + 0.075)) / np.array([0.03, 0.045, 0.03]), axis=1)
    col = mixc(col, hexc("#080303"), smoothstep(1.3, 0.8, dm))
    for s in (-1, 1):
        col = mixc(col, hexc("#030303"), smoothstep(0.026, 0.016, np.linalg.norm(v - V(hx + s * 0.03, hy + 0.01, hz + 0.075), axis=1)))
    prims = [Prim(v, fa, n, col, {"name": "deadskin", "rough": 0.7}, "corpse")]
    tris, size = write_glb(f"/home/claude/hh13/out/{name}.glb", prims, name)
    print(name, tris, "tris", size // 1024, "KB")

if __name__ == "__main__":
    for nm in POSES: build(nm)
