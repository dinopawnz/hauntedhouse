"""THE CRAWLER - an emaciated, wet, near-black figure on all fours with spider-long limbs,
head hung low under matted hair, a grin full of teeth and two pin-point white eyes.
Original design. Faces +z. Built in pieces so the house can animate the crawl:
  crawler_body  (origin: on the floor under the middle of the body)
  crawler_arm   (origin: the shoulder joint; a right arm - the left is the same model mirrored with sx=-1)
  crawler_leg   (origin: the hip joint; right leg, mirror for the left)
  crawler       (all of it in one piece, for when it doesn't need to move its limbs)"""
import sys, math
import numpy as np
sys.path.insert(0, "/home/claude/hh13/tools"); sys.path.insert(0, "/home/claude/hh13/models")
from sculpt import *
from body import *
import grinner as G

SH = V(0.19, 0.86, 0.30)      # right shoulder, in body space
HP = V(0.13, 0.72, -0.30)     # right hip
HEAD_AT = V(0, 0.66, 0.50)    # where the neck meets the head
# arm and leg, relative to their own joint
EL = V(0.16, -0.24, 0.13); WR = V(0.19, -0.79, 0.33)
KN = V(0.17, -0.27, 0.18); AN = V(0.13, -0.63, -0.22)

def torso(p):
    c = V(0, 0.82, 0.0)
    q = p - c
    # a long, starved, arched back: chest forward, hips behind
    d = ribcage(p, V(0, 0.82, 0.18), (0.15, 0.11, 0.17), ribs=0.0035, freq=80)
    d = smin(d, ellipsoid(p, V(0, 0.72, -0.27), (0.13, 0.09, 0.11)), 0.08)
    d = smin(d, rcone(p, V(0, 0.75, -0.25), V(0, 0.83, 0.15), 0.075, 0.11), 0.07)
    # the spine: a ridge of knuckles down the back
    for i in range(15):
        z = 0.24 - 0.53 * i / 14
        sp = V(0, 0.80 + 0.278 * (z + 0.27) - 0.012, z)
        d = smin(d, sphere(p, sp, 0.016), 0.014)
    for s in (-1, 1):
        d = smin(d, ellipsoid(p, V(s * 0.1, 0.9, 0.22), (0.07, 0.025, 0.08)), 0.03)   # shoulder blades jut up
        d = smin(d, sphere(p, V(s * SH[0], SH[1], SH[2]), 0.05), 0.05)
        d = smin(d, sphere(p, V(s * HP[0], HP[1], HP[2]), 0.06), 0.06)
    d = smin(d, rcone(p, V(0, 0.84, 0.3), HEAD_AT + V(0, 0.02, -0.02), 0.06, 0.042), 0.05)   # neck hanging forward
    return d - 0.0018 * fbm(p * 35, 2, seed=61)

def arm(p):
    o = V(0, 0, 0)
    d = bony_limb(p, o, EL, 0.034, 0.026, 1.3)
    d = smin(d, ellipsoid(p, EL * 0.3, (0.04, 0.06, 0.04)), 0.03)
    d = smin(d, bony_limb(p, EL, WR, 0.026, 0.017, 1.3), 0.012)
    d = smin(d, ellipsoid(p, EL + (WR - EL) * 0.22, (0.03, 0.07, 0.03)), 0.02)
    return d - 0.0015 * fbm(p * 35, 2, seed=62)

def leg(p):
    d = bony_limb(p, V(0, 0, 0), KN, 0.055, 0.034, 1.25)
    d = smin(d, bony_limb(p, KN, AN, 0.034, 0.02, 1.3), 0.015)
    d = smin(d, ellipsoid(p, KN + (AN - KN) * 0.25, (0.032, 0.07, 0.03)), 0.02)
    # long foot flat on the floor, toes splayed
    f = AN + V(0, -0.045, 0.06)
    d = smin(d, ellipsoid(p, f, (0.035, 0.02, 0.09)), 0.03)
    for i in range(5):
        b = f + V(-0.022 + i * 0.011, 0, 0.07)
        d = smin(d, bony_limb(p, b, b + V((i - 2) * 0.008, -0.012, 0.045), 0.0075, 0.005, 1.2), 0.006)
    return d - 0.0015 * fbm(p * 35, 2, seed=63)

def colour_skin(v, n, ao, seed=1):
    col = mixc(hexc("#26272b"), hexc("#0c0c0e"), 0.5 + 0.5 * fbm(v * 7, 3, seed=seed))
    col = mixc(col, hexc("#3c3f4a"), smoothstep(0.55, 0.9, 1 - ridged(v * 20, 3, seed=seed + 3)) * 0.4)   # wet sheen streaks
    col = mixc(col, hexc("#3a1210"), smoothstep(0.6, 0.85, fbm(v * 5, 2, seed=seed + 9)) * 0.35)          # raw patches
    return col * (0.3 + 0.7 * ao)[:, None]

SKIN = {"name": "wetskin", "rough": 0.28}

def mesh_one(f, lo, hi, h, seed):
    v, fa, n = mesh_sdf(f, lo, hi, h, smooth=2)
    ao = sdf_ao(f, v, n, 0.012, 5, 1.3)
    return Prim(v, fa, n, colour_skin(v, n, ao, seed), SKIN, "skin")

def hand(side_sign=1):
    W = WR; fwd = V(0.15, -0.1, 1.0); up = V(0, 1, 0)
    def f(p):
        d, nl = hand_sdf(p, W, fwd, up, size=1.3, curl=0.05, spread=0.55, finger_len=1.9, claw=1.0)
        return np.minimum(d, nl)
    v, fa, n = mesh_sdf(f, W + V(-0.13, -0.08, -0.06), W + V(0.15, 0.08, 0.32), 0.0056, smooth=1)
    d, nl = hand_sdf(v, W, fwd, up, size=1.3, curl=0.05, spread=0.55, finger_len=1.9, claw=1.0)
    ao = sdf_ao(f, v, n, 0.006, 4, 1.2)
    lab = (nl < d).astype(int)
    col = colour_skin(v, n, ao, 7)
    col[lab == 1] = (hexc("#191510") * (0.5 + 0.5 * ao[lab == 1])[:, None])
    return split_by_label(v, fa, n, col, lab, {0: SKIN, 1: {"name": "claws", "rough": 0.2}}, "hand")

def head_prims():
    """the Grinner's grin on a smaller, darker, wetter head, mostly hidden under matted hair"""
    v, fa, n = mesh_sdf(G.head_sdf, V(-0.11, -0.03, -0.13), V(0.11, 0.33, 0.135), 0.0056, smooth=2)
    ao = sdf_ao(G.head_sdf, v, n, 0.006, 5, 1.4)
    col = colour_skin(v, n, ao, 21)
    dx = np.interp(v[:, 0], G.MX, G.MY); dh = np.interp(v[:, 0], G.MX, G.MH); dz = np.interp(v[:, 0], G.MX, G.MZ)
    dm = np.abs(v[:, 1] - dx) - dh
    inside = smoothstep(-0.002, -0.012, v[:, 2] - dz) * smoothstep(0.004, -0.002, dm)
    col = mixc(col, hexc("#1a0303"), inside)
    col = mixc(col, hexc("#3b0d0a"), smoothstep(0.006, 0.0, dm) * smoothstep(-0.03, -0.004, v[:, 2] - dz))
    prims = [Prim(v, fa, n, col, SKIN, "head")]
    # pin-point eyes that glow white in the dark
    for s in (-1, 1):
        c = V(s * 0.033, 0.180, 0.078)
        ev, ef, en = mesh_sdf(lambda p, c=c: sphere(p, c, 0.0115), c - 0.016, c + 0.016, 0.002, smooth=1)
        ec = np.tile(hexc("#ffffff"), (len(ev), 1))
        prims.append(Prim(ev, ef, en, ec, {"name": "glow_eye", "rough": 0.1, "emit": [0.95, 0.95, 0.9]}, "eye"))
    prims += G.teeth_mesh()
    hair = G.hair_mesh(scale_len=1.25, seed=17, strands=80, seg=10)
    prims += hair
    return prims

def place(prims, R, t, s=1.0):
    return [Prim(transform(P.v, R, t, s), P.f, transform(P.n, R), P.c, P.mat, P.name) for P in prims]

def build():
    body = [mesh_one(torso, V(-0.3, 0.55, -0.48), V(0.3, 1.0, 0.5), 0.012, 1)]
    # head hangs low and forward, face looking ahead and up at you
    head = place(head_prims(), rot_x(math.radians(22)), HEAD_AT + V(0, -0.05, 0.02), 1.05)
    body += head
    armP = [mesh_one(arm, V(-0.08, -0.84, -0.08), V(0.26, 0.08, 0.42), 0.0085, 2)] + hand()
    legP = [mesh_one(leg, V(-0.09, -0.72, -0.32), V(0.26, 0.09, 0.32), 0.0095, 3)]
    write_glb("/home/claude/hh13/out/crawler_body.glb", body, "crawler_body")
    write_glb("/home/claude/hh13/out/crawler_arm.glb", armP, "crawler_arm")
    write_glb("/home/claude/hh13/out/crawler_leg.glb", legP, "crawler_leg")
    full = list(body)
    for s in (1, -1):
        M = np.diag([s, 1, 1.0])
        for parts, J in ((armP, SH), (legP, HP)):
            for P in parts:
                f = P.f if s == 1 else P.f[:, ::-1]
                full.append(Prim(P.v @ M + V(s * J[0], J[1], J[2]), f, P.n @ M, P.c, P.mat, P.name))
    tris, size = write_glb("/home/claude/hh13/out/crawler.glb", full, "crawler")
    for nm in ("crawler_body", "crawler_arm", "crawler_leg"):
        pr, g = read_glb(f"/home/claude/hh13/out/{nm}.glb"); print(nm, sum(len(x[1]) for x in pr), "tris")
    print("crawler", tris, "tris", size // 1024, "KB")

if __name__ == "__main__":
    build()
