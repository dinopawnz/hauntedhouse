import numpy as np, subprocess, wave, os
SR = 44100
rng = np.random.default_rng(7)
SRC = "/home/claude/dinopawnz/hauntedhouse/"
def save(name, x, br="96k"):
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.97
    w = "/tmp/claude-0/" + name + ".wav"
    with wave.open(w, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR)
        f.writeframes((x * 32767).astype(np.int16).tobytes())
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", w, "-codec:a", "libmp3lame", "-b:a", br, name + ".mp3"], check=True)
    print(name, round(len(x) / SR, 3), "s")
def load(name, rate=1.0):
    out = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", SRC + name, "-af", f"asetrate={int(SR*rate)},aresample={SR}", "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).astype(np.float64) / 32768
def lp(x, a):            # one-pole low-pass, a in (0,1): bigger = darker
    y = np.empty_like(x); s = 0.0
    for i in range(len(x)): s = s + (1 - a) * (x[i] - s); y[i] = s
    return y
def lpf(x, cut):          # fft brickwall-ish low-pass
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR); X *= 1 / (1 + (f / cut) ** 4); return np.fft.irfft(X, len(x))
def hpf(x, cut):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR); X *= (f / cut) ** 4 / (1 + (f / cut) ** 4); return np.fft.irfft(X, len(x))
def clink(t0, out, amp=1.0):
    n = int(0.09 * SR); t = np.arange(n) / SR
    parts = rng.uniform(1800, 2600) * np.array([1, 1.62, 2.31, 3.17, 4.4])
    s = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) * np.exp(-t * rng.uniform(40, 90)) * a for f, a in zip(parts, [1, .6, .45, .3, .2]))
    s += rng.normal(0, 1, n) * np.exp(-t * 400) * 0.6
    i = int(t0 * SR); m = min(n, len(out) - i)
    if m > 0: out[i:i + m] += s[:m] * amp

# 1. chains dragging: clusters of clinks + a scraping bed
def chains(dur=6.0, step=0.55):
    out = np.zeros(int(dur * SR))
    t = 0.05
    while t < dur - 0.2:
        for k in range(rng.integers(3, 9)):
            clink(t + rng.uniform(0, 0.22), out, rng.uniform(0.2, 1.0))
        t += step * rng.uniform(0.8, 1.25)
    scrape = hpf(lpf(rng.normal(0, 1, len(out)), 3500), 400) * 0.08
    env = 0.5 + 0.5 * np.sin(np.arange(len(out)) / SR * 2 * np.pi / (step * 2)) ** 2
    out += scrape * env
    return out
save("chain_rattle", chains())

# 2. the dead walking behind you: slowed heavy footsteps with chains
fs = load("footsteps.mp3", 0.82)
ch = chains(len(fs) / SR, 0.62)
save("ghost_walk", fs * 1.0 + ch * 0.55)

# 3. splat: a thump and a wet spray
n = int(1.3 * SR); t = np.arange(n) / SR
thump = np.sin(2 * np.pi * (70 * np.exp(-t * 6)) * t) * np.exp(-t * 14)
wet = lpf(rng.normal(0, 1, n), 1800) * np.exp(-t * 7) * (1 + 0.8 * np.sin(2 * np.pi * 23 * t) ** 8)
drips = np.zeros(n)
for k in range(9):
    i = int(rng.uniform(0.15, 1.1) * SR); m = int(0.04 * SR); tt = np.arange(m) / SR
    drips[i:i + m] += np.sin(2 * np.pi * rng.uniform(500, 900) * tt * (1 + 3 * tt)) * np.exp(-tt * 90) * rng.uniform(0.1, 0.3)
save("splat", thump * 1.2 + wet * 0.9 + drips)

# 4. bone crack: a sharp snap and a grinding crunch
n = int(0.9 * SR); t = np.arange(n) / SR
snap = hpf(rng.normal(0, 1, n), 1500) * np.exp(-t * 60)
crunch = np.zeros(n)
for k in range(30):
    i = int(rng.uniform(0.02, 0.45) * SR); m = int(0.006 * SR)
    crunch[i:i + m] += rng.normal(0, 1, m) * rng.uniform(0.2, 0.7)
crunch = lpf(crunch, 5000)
save("bone_crack", snap * 1.3 + crunch * 0.8)

# 5. demon roar: the growl dropped an octave and a half, a rumble under it, a scream on top
g = load("growl_x.mp3", 0.55)
th = load("thunder_x.mp3", 0.7)[: len(g)]
sc = load("screech_x.mp3", 0.75)[: len(g)]
n = max(len(g), len(th))
roar = np.zeros(n); roar[: len(g)] += g * 1.0; roar[: len(th)] += lpf(th, 300) * 0.9; roar[: len(sc)] += sc * 0.35
sub = np.sin(2 * np.pi * 38 * np.arange(n) / SR) * np.minimum(1, np.arange(n) / (0.4 * SR)) * np.exp(-np.arange(n) / SR * 0.5) * 0.35
save("demon_roar", np.tanh((roar + sub) * 1.6))

# 6. skitter: something on all fours coming fast - nails on boards
n = int(2.4 * SR); out = np.zeros(n); t = 0.0
while t < 2.3:
    m = int(0.012 * SR); i = int(t * SR)
    click = hpf(rng.normal(0, 1, m), 2500) * np.exp(-np.arange(m) / SR * 500)
    out[i:i + m] += click * rng.uniform(0.4, 1.0)
    t += rng.uniform(0.035, 0.09)
thud = np.zeros(n); t = 0.0
while t < 2.3:
    m = int(0.05 * SR); i = int(t * SR); tt = np.arange(m) / SR
    thud[i:i + m] += np.sin(2 * np.pi * 110 * tt) * np.exp(-tt * 60) * rng.uniform(0.3, 0.7)
    t += rng.uniform(0.14, 0.22)
save("skitter", out + thud * 0.8 + lpf(rng.normal(0, 1, n), 900) * 0.04)

# 7. the summoning: a low rising drone with a choir-ish beating
n = int(7 * SR); tt = np.arange(n) / SR
f0 = 42 + 18 * (tt / 7) ** 2
ph = 2 * np.pi * np.cumsum(f0) / SR
drone = sum(np.sin(ph * k * (1 + 0.003 * k)) / k for k in range(1, 9))
beat = 0.6 + 0.4 * np.sin(2 * np.pi * (2 + 4 * tt / 7) * tt)
whis = lpf(hpf(rng.normal(0, 1, n), 2000), 6000) * 0.15 * (tt / 7)
save("summon", (drone * beat * np.minimum(1, tt / 1.5) + whis) * np.minimum(1, (7 - tt) / 0.3))
