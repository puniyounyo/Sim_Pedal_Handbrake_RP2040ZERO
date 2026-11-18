# -*- coding: utf-8 -*-
"""
Printed M6-class bolt + nut for the desk clamp v3 base fixing.
NOT a copy of metal M6: coarse trapezoidal thread, pitch 2 mm,
sized to the existing holes (d6.5) and nut pockets (AF10.5 x 5.5)
of the v3 clamp body, so the body does not change.

Bolt (print frame, head at bottom):
  head   z 0..7     lobed knob r = 8.0 + 0.9*cos(6*theta)  (hand tighten)
  shank  z 7..18    smooth r 3.15 (d6.3, through base + plate hole)
  thread z 18..25.6 r 2.60 + tooth, height 0.65 (tip d6.5)
  tip    z 25.6..27.2 chamfer to a point cap
Nut (print frame):
  outer  z 0..5.5   hex AF10.2 (apothem 5.1) - fits AF10.5 pocket
  bore   through    root r 3.45, tooth height 0.55 (tip r 2.90),
                    phase shifted half pitch against the bolt
Estimated strength (PETG, vertical print): bolt root d5.2 layer
tensile ~450-650 N breaking. Working load target <= ~150 N/bolt.
"""
import numpy as np
import struct, os

OUT = os.path.dirname(os.path.abspath(__file__))
PITCH = 2.0

def tooth(zeta, height):
    z = np.mod(zeta, PITCH)
    h = np.zeros_like(z, dtype=float)
    rise = (z >= 0) & (z < 0.65)
    tip = (z >= 0.65) & (z < 1.0)
    fall = (z >= 1.0) & (z < 1.65)
    h[rise] = height * (z[rise] / 0.65)
    h[tip] = height
    h[fall] = height * (1 - (z[fall] - 1.0) / 0.65)
    return h

def make_consistent(verts, faces):
    from collections import defaultdict, deque
    edge_map = defaultdict(list)
    for fi, fc in enumerate(faces):
        for e in range(3):
            a, b = fc[e], fc[(e + 1) % 3]
            edge_map[(min(a, b), max(a, b))].append((fi, a, b))
    bad = sum(1 for v in edge_map.values() if len(v) != 2)
    faces = [list(fc) for fc in faces]
    seen = [False] * len(faces)
    dq = deque([0]); seen[0] = True
    while dq:
        fi = dq.popleft()
        fc = faces[fi]
        for e in range(3):
            a, b = fc[e], fc[(e + 1) % 3]
            for fj, x, y in edge_map[(min(a, b), max(a, b))]:
                if fj == fi or seen[fj]: continue
                if (x, y) == (a, b):
                    faces[fj] = [faces[fj][0], faces[fj][2], faces[fj][1]]
                seen[fj] = True
                dq.append(fj)
    faces = np.array(faces)
    tris = verts[faces]
    vol = np.einsum('ij,ij->i', tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6
    if vol < 0: faces = faces[:, [0, 2, 1]]
    print(f"  watertight check: non-manifold edges={bad}, volume={abs(vol)/1000:.2f} cm3")
    return faces

def save_stl(path, verts, faces):
    tris = verts[faces]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True); ln[ln == 0] = 1; n = n / ln
    with open(path, 'wb') as f:
        f.write(b'\0' * 80)
        f.write(struct.pack('<I', len(tris)))
        for i in range(len(tris)):
            f.write(struct.pack('<3f', *n[i]))
            for v in tris[i]: f.write(struct.pack('<3f', *v))
            f.write(struct.pack('<H', 0))
    print(f"{os.path.basename(path)}: {len(tris)} tris")

def revolve(radius_fn, z_rows, n_theta=120):
    verts, rings = [], []
    for z in z_rows:
        th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        r = radius_fn(th, z)
        if np.max(r) < 1e-9:
            rings.append(('point', len(verts))); verts.append([0.0, 0.0, z])
        else:
            ring = []
            for i in range(n_theta):
                ring.append(len(verts))
                verts.append([r[i] * np.cos(th[i]), r[i] * np.sin(th[i]), z])
            rings.append(('ring', ring))
    faces = []
    for k in range(len(z_rows) - 1):
        ta, a = rings[k]; tb, b = rings[k + 1]
        if ta == 'ring' and tb == 'ring':
            for i in range(n_theta):
                j = (i + 1) % n_theta
                faces.append([a[i], b[i], b[j]]); faces.append([a[i], b[j], a[j]])
        elif ta == 'point' and tb == 'ring':
            for i in range(n_theta):
                j = (i + 1) % n_theta
                faces.append([a, b[j], b[i]])
        elif ta == 'ring' and tb == 'point':
            for i in range(n_theta):
                j = (i + 1) % n_theta
                faces.append([a[i], a[j], b])
    return np.array(verts, float), np.array(faces, int)

# ---------------- bolt ----------------
CORE = 2.60
TOOTH_H = 0.65
def bolt_radius(th, z):
    phase = PITCH * th / (2 * np.pi)
    lobe = 8.0 + 0.9 * np.cos(6 * th)
    if z <= 0: return np.zeros_like(th)
    if z < 1.2: return lobe * (z / 1.2) ** 0.5 + 0.001
    if z < 5.5: return lobe
    if z < 7.0:
        t = (z - 5.5) / 1.5
        return lobe * (1 - t) + 3.15 * t
    if z < 18.0: return np.full_like(th, 3.15)
    if z < 25.6: return CORE + tooth(z - phase, TOOTH_H)
    if z < 26.6:
        t = (z - 25.6) / 1.0
        return np.full_like(th, (CORE + TOOTH_H) * (1 - t) + 2.0 * t)
    return np.zeros_like(th)

def build_bolt(thread_end, fname):
    def radius(th, z):
        phase = PITCH * th / (2 * np.pi)
        lobe = 8.0 + 0.9 * np.cos(6 * th)
        if z <= 0: return np.zeros_like(th)
        if z < 1.2: return lobe * (z / 1.2) ** 0.5 + 0.001
        if z < 5.5: return lobe
        if z < 7.0:
            t = (z - 5.5) / 1.5
            return lobe * (1 - t) + 3.15 * t
        if z < 18.0: return np.full_like(th, 3.15)
        if z < thread_end: return CORE + tooth(z - phase, TOOTH_H)
        if z < thread_end + 0.8:
            t = (z - thread_end) / 0.8
            return np.full_like(th, (CORE + TOOTH_H) * (1 - t) + 2.0 * t)
        return np.zeros_like(th)
    z_rows = [0, 0.7, 1.2, 3.0, 5.5, 6.2, 7.0, 12.0, 18.0]
    z_rows += list(np.arange(18.25, thread_end + 0.01, 0.25))
    z_rows += [thread_end + 0.8, thread_end + 1.4]
    v, f = revolve(radius, z_rows)
    f = make_consistent(v, f)
    save_stl(os.path.join(OUT, fname), v, f)

build_bolt(25.6, 'desk_clamp_bolt_m6.stl')        # all 3 holes (v5: front nut
# also sits in a plate trap, reached via the underside slide tunnel)

# ---------------- nut ----------------
BORE_ROOT = 3.45
BORE_TOOTH = 0.55
def hex_r(th, apothem=5.45):
    a = np.mod(np.degrees(th), 60.0) - 30.0
    return apothem / np.cos(np.radians(a))

def nut_mesh(n_theta=120):
    z_out = [0, 5.5]
    z_in = [0, 0.5] + list(np.arange(0.75, 5.01, 0.25)) + [5.5]
    verts, faces = [], []
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phase = PITCH * th / (2 * np.pi)
    def ring(z, radii):
        idx = []
        for i in range(n_theta):
            idx.append(len(verts))
            verts.append([radii[i] * np.cos(th[i]), radii[i] * np.sin(th[i]), z])
        return idx
    out_rings = [ring(z, hex_r(th)) for z in z_out]
    in_rings = []
    for z in z_in:
        if z < 0.5 or z > 5.0:
            r = np.full(n_theta, 3.05)
        else:
            r = BORE_ROOT - tooth(z - phase + PITCH / 2, BORE_TOOTH)
        in_rings.append(ring(z, r))
    def quad(a, b, flip=False):
        for i in range(n_theta):
            j = (i + 1) % n_theta
            if flip:
                faces.append([a[i], b[j], b[i]]); faces.append([a[i], a[j], b[j]])
            else:
                faces.append([a[i], b[i], b[j]]); faces.append([a[i], b[j], a[j]])
    quad(out_rings[0], out_rings[1])
    for k in range(len(z_in) - 1):
        quad(in_rings[k], in_rings[k + 1], flip=True)
    ob, ib = out_rings[0], in_rings[0]
    ot, it = out_rings[-1], in_rings[-1]
    for i in range(n_theta):
        j = (i + 1) % n_theta
        faces.append([ob[i], ib[i], ib[j]]); faces.append([ob[i], ib[j], ob[j]])
        faces.append([ot[i], it[j], it[i]]); faces.append([ot[i], ot[j], it[j]])
    return np.array(verts, float), np.array(faces, int)

v2, f2 = nut_mesh()
f2 = make_consistent(v2, f2)
save_stl(os.path.join(OUT, 'desk_clamp_nut_m6.stl'), v2, f2)

# ---------------- clearance check ----------------
ths = np.linspace(0, 2 * np.pi, 720, endpoint=False)
phase = PITCH * ths / (2 * np.pi)
worst = 1e9
for z in np.arange(19, 25, 0.5):
    bolt_r = CORE + tooth(z - phase, TOOTH_H)
    bore_r = BORE_ROOT - tooth(z - phase + PITCH / 2, BORE_TOOTH)
    worst = min(worst, (bore_r - bolt_r).min())
print(f"thread radial clearance (min): {worst:.2f} mm (must be > 0)")
