# -*- coding: utf-8 -*-
"""
Generate the one-piece printed knob+screw and the nut insert
for the desk clamp v2. Coarse trapezoidal thread, pitch 4 mm,
profile symmetric (so the insert mates even when flipped).

Meshes are built as surfaces of revolution r(theta, z) on a grid,
so each part is a single closed solid. Units: mm.
Adjust CLEARANCE if the thread prints too tight / loose.
"""
import numpy as np
import struct, os

OUT = os.path.dirname(os.path.abspath(__file__))
PITCH = 4.0
CLEARANCE = 0.0   # extra radial clearance added to the bore (mm)

# ---------- thread profile (symmetric trapezoid, period = PITCH) ----------
def tooth(zeta, height):
    """tooth height at phase zeta (periodic, symmetric about PITCH/2)"""
    z = np.mod(zeta, PITCH)
    h = np.zeros_like(z, dtype=float)
    rise = (z >= 0) & (z < 1.3)
    tip = (z >= 1.3) & (z < 2.0)
    fall = (z >= 2.0) & (z < 3.3)
    h[rise] = height * (z[rise] / 1.3)
    h[tip] = height
    h[fall] = height * (1 - (z[fall] - 2.0) / 1.3)
    return h


# ---------- winding repair: make faces consistent, then outward ----------
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
                if fj == fi or seen[fj]:
                    continue
                # neighbor must traverse shared edge opposite direction
                if (x, y) == (a, b):
                    faces[fj] = [faces[fj][0], faces[fj][2], faces[fj][1]]
                seen[fj] = True
                dq.append(fj)
    faces = np.array(faces)
    tris = verts[faces]
    vol = np.einsum('ij,ij->i', tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6
    if vol < 0:
        faces = faces[:, [0, 2, 1]]
    print(f"  winding repair: non-manifold edges={bad}, volume now {abs(vol)/1000:.1f} cm3")
    return faces

# ---------- generic revolved-surface mesh ----------
def revolve(radius_fn, z_rows, n_theta=120):
    """
    radius_fn(theta_array, z) -> radius array.
    z_rows: list of z heights (ascending). Returns (verts, faces).
    Caps: first/last rows are pinched to r=0 by radius_fn returning 0.
    """
    verts = []
    rings = []   # each entry: ('ring', [indices]) or ('point', index)
    for z in z_rows:
        th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        r = radius_fn(th, z)
        if r.max() < 1e-9:
            rings.append(('point', len(verts)))
            verts.append([0.0, 0.0, z])
        else:
            ring = []
            for i in range(n_theta):
                ring.append(len(verts))
                verts.append([r[i] * np.cos(th[i]), r[i] * np.sin(th[i]), z])
            rings.append(('ring', ring))
    faces = []
    for k in range(len(z_rows) - 1):
        ta, a = rings[k]
        tb, b = rings[k + 1]
        if ta == 'ring' and tb == 'ring':
            for i in range(n_theta):
                j = (i + 1) % n_theta
                faces.append([a[i], b[i], b[j]])
                faces.append([a[i], b[j], a[j]])
        elif ta == 'point' and tb == 'ring':
            for i in range(n_theta):
                j = (i + 1) % n_theta
                faces.append([a, b[j], b[i]])
        elif ta == 'ring' and tb == 'point':
            for i in range(n_theta):
                j = (i + 1) % n_theta
                faces.append([a[i], a[j], b])
    return np.array(verts, float), np.array(faces, int)

def save_stl(path, verts, faces):
    tris = verts[faces]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1
    n = n / ln
    with open(path, 'wb') as f:
        f.write(b'\0' * 80)
        f.write(struct.pack('<I', len(tris)))
        for i in range(len(tris)):
            f.write(struct.pack('<3f', *n[i]))
            for v in tris[i]:
                f.write(struct.pack('<3f', *v))
            f.write(struct.pack('<H', 0))
    # signed volume check
    vol = np.einsum('ij,ij->i', tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6
    print(f"{os.path.basename(path)}: {len(tris)} tris, volume {vol/1000:.1f} cm3")
    return vol

# =====================================================================
# SCREW + KNOB (one piece). Local z: 0 = knob bottom, up = tip.
#   knob   z 0..16    lobed grip r = 27 + 2.2*cos(6*theta)
#   neck   z 16..21   r 17 -> 10.4
#   plain  z 21..29   r 10.4
#   thread z 29..75   r 10.4 + tooth(z - phase), tooth height 2.2 (tip r 12.6)
#   tip    z 75..84   r 10.4 with snap groove r 8.6 at z 78..81
# =====================================================================
CORE = 10.4
TOOTH_H = 2.2

def screw_radius(th, z):
    phase = PITCH * th / (2 * np.pi)
    if z <= 0:
        return np.zeros_like(th)
    if z < 1.5:
        return (24.5 + 2.0 * np.cos(6 * th)) * (z / 1.5) ** 0.5 + 0.001
    if z < 14:
        return 27.0 + 2.2 * np.cos(6 * th)
    if z < 16:
        t = (z - 14) / 2
        return (27.0 + 2.2 * np.cos(6 * th)) * (1 - t) + 17.0 * t
    if z < 20.2:
        return np.full_like(th, 17.0)
    if z < 21.8:
        t = (z - 20.2) / 1.6
        return np.full_like(th, 17.0 * (1 - t) + CORE * t)
    if z < 29:
        return np.full_like(th, CORE)
    if z < 75:
        return CORE + tooth(z - phase, TOOTH_H)
    if z < 77.8:
        return np.full_like(th, CORE)
    if z < 78.4:
        t = (z - 77.8) / 0.6
        return np.full_like(th, CORE * (1 - t) + 8.6 * t)
    if z < 80.6:
        return np.full_like(th, 8.6)
    if z < 81.2:
        t = (z - 80.6) / 0.6
        return np.full_like(th, 8.6 * (1 - t) + CORE * t)
    if z < 83:
        return np.full_like(th, CORE)
    if z < 84:
        t = (z - 83) / 1.0
        return np.full_like(th, CORE * (1 - t) + 5.0 * t)
    return np.zeros_like(th)

z_rows = [0, 0.8, 1.5, 6, 10, 14, 15, 16, 18, 20.2, 21, 21.8, 25, 29]
z_rows += list(np.arange(29.5, 75.01, 0.5))
z_rows += [75.5, 77.8, 78.4, 79.5, 80.6, 81.2, 82, 83, 84, 84.6]
v, f = revolve(screw_radius, z_rows)
f = make_consistent(v, f)
if save_stl(os.path.join(OUT, 'desk_clamp_screw.stl'), v, f) < 0:
    print('  (negative volume: winding flipped, slicer will still handle)')

# =====================================================================
# NUT INSERT. Local z: 0 = flange bottom (print side), up = body end.
#   flange z 0..4     r 20 (d40)
#   body   z 4..23    hex AF29.2: r = 14.6 / cos(((theta mod 60) - 30) deg)
#   bore   through    r 13.3 - tooth(z - phase + PITCH/2), tooth h 2.0
# Built as outer grid + inner grid joined by top/bottom rings.
# Hex was AF30 originally; the v5 body's AF31.2 channel printed ~0.5/side
# tighter than CAD (mouth on the supported face worst), so the insert
# would not enter from the seat side. Reduced to AF29.2 (+0.4/side
# clearance) after the 2026-11-17 fit check. Flange/bore unchanged.
# =====================================================================
BORE_ROOT = 13.3 + CLEARANCE
BORE_TOOTH = 2.0
HEX_APOTHEM = 14.6   # AF29.2

def hex_r(th):
    a = np.mod(np.degrees(th), 60.0) - 30.0
    return HEX_APOTHEM / np.cos(np.radians(a))

def insert_mesh(n_theta=120):
    z_out = [0, 4, 4.8, 22.2, 23]
    z_in = [0, 1.0] + list(np.arange(1.5, 22.01, 0.5)) + [22.6, 23]
    verts, faces = [], []
    def ring(z, radii):
        idx = []
        th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        for i in range(n_theta):
            idx.append(len(verts))
            verts.append([radii[i] * np.cos(th[i]), radii[i] * np.sin(th[i]), z])
        return idx
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phase = PITCH * th / (2 * np.pi)
    out_rings = []
    for z in z_out:
        r = np.full(n_theta, 20.0) if z <= 4 else hex_r(th)
        if z == 4.8:
            r = hex_r(th)
        out_rings.append(ring(z, r))
    in_rings = []
    for z in z_in:
        if z < 1.0:
            r = np.full(n_theta, 12.3 + CLEARANCE)
        elif z > 22.0:
            r = np.full(n_theta, 11.9 + CLEARANCE)
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
    for k in range(len(z_out) - 1):
        quad(out_rings[k], out_rings[k + 1])
    for k in range(len(z_in) - 1):
        quad(in_rings[k], in_rings[k + 1], flip=True)   # bore faces inward
    # bottom ring (z=0): outer -> inner ; top ring (z=23): inner -> outer
    ob, ib = out_rings[0], in_rings[0]
    ot, it = out_rings[-1], in_rings[-1]
    for i in range(n_theta):
        j = (i + 1) % n_theta
        faces.append([ob[i], ib[i], ib[j]]); faces.append([ob[i], ib[j], ob[j]])
        faces.append([ot[i], it[j], it[i]]); faces.append([ot[i], ot[j], it[j]])
    return np.array(verts, float), np.array(faces, int)

v2, f2 = insert_mesh()
f2 = make_consistent(v2, f2)
save_stl(os.path.join(OUT, 'desk_clamp_nut_insert.stl'), v2, f2)

# ---------- numeric thread clearance check ----------
ths = np.linspace(0, 2 * np.pi, 720, endpoint=False)
phase = PITCH * ths / (2 * np.pi)
worst = 1e9
for z in np.arange(35, 70, 1.0):
    screw_r = CORE + tooth(z - phase, TOOTH_H)
    bore_r = BORE_ROOT - tooth(z - phase + PITCH / 2, BORE_TOOTH)
    worst = min(worst, (bore_r - screw_r).min())
print(f"thread radial clearance (min over engagement): {worst:.2f} mm (must be > 0)")
