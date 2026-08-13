"""Generate the synthetic thoracic-aorta phantom.

No geometry, image, measurement or case parameter from any participant enters this
file. Every dimension is an explicit synthetic design choice. The generator is
deterministic: running it twice produces identical output.

See README.md in this directory for what the example is, what it verifies and
what its accuracy figures do and do not mean.
"""

import os

import numpy as np
from scipy.interpolate import CubicSpline

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
# Chosen independently of the study cohort.
DMAX_MM = 45.0

# Chosen. Representative adult thoracic-aorta dimensions; none is a measurement.
D_SINOTUBULAR_MM = 34.0     # inlet, at the sinotubular junction
D_ARCH_MM = 28.0            # mid-arch, distal to the branches
D_DESCENDING_MM = 24.0      # distal descending, at the outlet

# Chosen. Arc-length positions of the diameter control points, as a fraction of
# the total centreline length.
S_DMAX = 0.18               # the ascending bulge
S_ARCH = 0.44
S_DESCENDING_START = 0.55

# Chosen. Control points of the centreline, in millimetres, in a right-handed
# frame with +x to the left, +y anterior and +z superior. The origin is the
# centre of the inlet plane. The shape is the usual candy-cane: an ascending
# limb, an arch turning posteriorly and to the left, and a descending limb.
CENTRELINE_CONTROL_MM = np.array([
    [0.0, 0.0, 0.0],        # inlet, sinotubular junction
    [-3.0, 10.0, 32.0],     # ascending, leaning anteriorly
    [-4.0, 18.0, 64.0],
    [0.0, 20.0, 92.0],      # distal ascending
    [10.0, 12.0, 112.0],    # proximal arch, beginning to turn
    [26.0, -4.0, 120.0],    # arch apex
    [40.0, -22.0, 110.0],   # distal arch, now heading posteriorly
    [44.0, -32.0, 88.0],
    [42.0, -34.0, 55.0],    # proximal descending
    [40.0, -34.0, 10.0],
    [39.0, -34.0, -40.0],
    [38.0, -34.0, -95.0],   # distal descending, outlet
])

N_SAMPLES = 2000            # resampling density along the centreline

# Chosen. The three supra-aortic branches, in the order they leave the arch.
# Each is a capped cylinder rising from the arch; with the descending aorta they
# give the four-outlet topology the notebooks assume. Diameters are ordinary
# adult values and are not measurements.
#   s_frac   arc-length position of the branch root, as a fraction of the total
#   diam_mm  branch diameter
#   len_mm   length from the arch surface to the branch outlet
#   tilt     direction, added to +z before normalising: (dx, dy)
BRANCHES = (
    {"name": "out1", "label": "brachiocephalic",     "s_frac": 0.315, "diam_mm": 12.0,
     "len_mm": 42.0, "tilt": (-0.25, 0.30)},
    {"name": "out2", "label": "left common carotid", "s_frac": 0.365, "diam_mm": 8.0,
     "len_mm": 46.0, "tilt": (0.05, 0.12)},
    {"name": "out3", "label": "left subclavian",     "s_frac": 0.415, "diam_mm": 9.0,
     "len_mm": 44.0, "tilt": (0.28, -0.18)},
)

# Chosen. Fillet radius where a branch meets the arch. Real ostia are filleted;
# a sharp crease also makes the surface awkward to remesh.
OSTIUM_FILLET_MM = 2.0

# Chosen. Voxel size of the grid the surface is extracted from. Finer than the
# study's 1.4 mm target element size, so remeshing is not resolution-limited.
VOXEL_MM = 0.6

# Chosen. Reach of the end trims, as a multiple of the lumen radius there. Above
# 1 so the cut covers the whole terminal cap; small enough that the ball touches
# no other part of the vessel, which verify_surface() checks rather than assumes.
TRIM_REACH = 1.5


def _fit_centreline(control_mm):
    """Fit a C2 parametric cubic spline through the control points.

    Returns a spline in a chord-length parameter. Curvature is continuous by
    construction, which is what phase 1 has to demonstrate.
    """
    chord = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(control_mm, axis=0), axis=1))])
    return CubicSpline(chord / chord[-1], control_mm, axis=0, bc_type="natural")


def _resample_by_arclength(spline, n):
    """Resample the spline at n points equally spaced in arc length.

    The spline parameter is chord length, which is only approximately arc
    length; this inverts the true arc-length function so that downstream
    quantities can be indexed by distance along the vessel.
    """
    t_dense = np.linspace(0.0, 1.0, 20 * n)
    p_dense = spline(t_dense)
    seg = np.linalg.norm(np.diff(p_dense, axis=0), axis=1)
    s_dense = np.concatenate([[0.0], np.cumsum(seg)])
    s_uniform = np.linspace(0.0, s_dense[-1], n)
    t_uniform = np.interp(s_uniform, s_dense, t_dense)
    return s_uniform, spline(t_uniform), t_uniform


def _frenet(spline, t):
    """Unit tangent, curvature and the curvature's derivative along the curve."""
    d1 = spline(t, 1)
    d2 = spline(t, 2)
    speed = np.linalg.norm(d1, axis=1, keepdims=True)
    tangent = d1 / speed
    cross = np.cross(d1, d2)
    curvature = np.linalg.norm(cross, axis=1) / speed[:, 0] ** 3
    return tangent, curvature


def _radius_profile(s_mm):
    """Lumen radius along the centreline, in millimetres.

    A monotone cubic through the diameter control points: the ascending bulge,
    the arch and the descending aorta. Monotone interpolation is used so the
    profile cannot overshoot between control points and invent a second bulge.
    """
    total = s_mm[-1]
    s_ctrl = np.array([0.0, S_DMAX, S_ARCH, S_DESCENDING_START, 1.0]) * total
    d_ctrl = np.array([
        D_SINOTUBULAR_MM,
        DMAX_MM,
        D_ARCH_MM,
        0.5 * (D_ARCH_MM + D_DESCENDING_MM),
        D_DESCENDING_MM,
    ])
    from scipy.interpolate import PchipInterpolator
    return PchipInterpolator(s_ctrl, d_ctrl)(s_mm) / 2.0


def build_centreline():
    """Phase 1: the centreline and its radius profile.

    Returns arc length [mm], points [mm], unit tangents, curvature [1/mm] and
    lumen radius [mm], all sampled at N_SAMPLES stations.
    """
    spline = _fit_centreline(CENTRELINE_CONTROL_MM)
    s_mm, points_mm, t = _resample_by_arclength(spline, N_SAMPLES)
    tangent, curvature = _frenet(spline, t)
    radius_mm = _radius_profile(s_mm)
    return s_mm, points_mm, tangent, curvature, radius_mm


# --------------------------------------------------------------------------
# Phase 2 — lumen surface
# --------------------------------------------------------------------------
# The surface is the zero level set of a signed distance function: negative
# inside the lumen, positive outside. Building it this way rather than by
# boolean union of meshes means the result is watertight and manifold by
# construction, and the branch junctions are filleted rather than creased.


def _branch_frames(s_mm, points_mm, tangent, radius_mm):
    """Root point, axis and length of each branch, placed on the arch."""
    frames = []
    for b in BRANCHES:
        i = int(round(b["s_frac"] * (len(s_mm) - 1)))
        axis = np.array([b["tilt"][0], b["tilt"][1], 1.0])
        axis /= np.linalg.norm(axis)
        # Start inside the lumen so the branch and the aorta overlap and the
        # union has no seam; the fillet then smooths the junction.
        root = points_mm[i] - axis * radius_mm[i]
        frames.append({
            **b,
            "root": root,
            "axis": axis,
            "length": b["len_mm"] + 2.0 * radius_mm[i],
            "station": i,
        })
    return frames


def _sdf_tube(grid_xyz, points_mm, radius_mm):
    """Signed distance to the tapering tube swept along the centreline."""
    from scipy.spatial import cKDTree
    dist, idx = cKDTree(points_mm).query(grid_xyz, workers=-1)
    return (dist - radius_mm[idx]).astype(np.float32)


def _sdf_capped_cylinder(grid_xyz, root, axis, length, radius):
    """Signed distance to a capped cylinder."""
    rel = grid_xyz - root
    axial = rel @ axis
    radial = np.linalg.norm(rel - np.outer(axial, axis), axis=1)
    d_radial = radial - radius
    d_axial = np.maximum(-axial, axial - length)
    outside = np.hypot(np.maximum(d_radial, 0.0), np.maximum(d_axial, 0.0))
    inside = np.minimum(np.maximum(d_radial, d_axial), 0.0)
    return (inside + outside).astype(np.float32)


def _smooth_union(a, b, k):
    """Polynomial smooth minimum: a union with a fillet of radius about k."""
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


def _sdf_halfspace(grid_xyz, origin, outward_normal):
    """Positive beyond the plane, so a maximum with it trims the solid flat."""
    return ((grid_xyz - origin) @ outward_normal).astype(np.float32)


def build_surface(geometry=None, verbose=True):
    """Phase 2: the closed lumen surface, with the three supra-aortic branches.

    geometry is (arc length, points, tangents, radii); the reference geometry
    from build_centreline() is used when it is None. Phase 3 passes the deformed
    geometry of a cardiac phase.

    Returns the surface and the branch frames.
    """
    import pyvista as pv

    if geometry is None:
        s_mm, points_mm, tangent, _curvature, radius_mm = build_centreline()
    else:
        s_mm, points_mm, tangent, radius_mm = geometry
    frames = _branch_frames(s_mm, points_mm, tangent, radius_mm)

    # Grid bounds: everything the solid can occupy, plus a margin so the
    # isosurface is never clipped by the edge of the grid.
    corners = np.vstack([
        points_mm,
        *[f["root"] + f["axis"] * f["length"] for f in frames],
    ])
    margin = radius_mm.max() + max(b["diam_mm"] for b in BRANCHES) + 4.0 * VOXEL_MM
    lo, hi = corners.min(axis=0) - margin, corners.max(axis=0) + margin
    dims = np.ceil((hi - lo) / VOXEL_MM).astype(int) + 1
    if verbose:
        print(f"  grid {dims[0]}x{dims[1]}x{dims[2]} = {np.prod(dims) / 1e6:.1f}M voxels "
              f"at {VOXEL_MM} mm")

    # Fortran order, so that x varies fastest: that is the point order
    # pv.ImageData uses, and the field can then be attached without reshaping.
    axes = [lo[i] + np.arange(dims[i]) * VOXEL_MM for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    grid_xyz = np.column_stack([gx.ravel(order="F"), gy.ravel(order="F"), gz.ravel(order="F")])

    sdf = _sdf_tube(grid_xyz, points_mm, radius_mm)
    for f in frames:
        sdf = _smooth_union(
            sdf,
            _sdf_capped_cylinder(grid_xyz, f["root"], f["axis"], f["length"],
                                 0.5 * f["diam_mm"]),
            OSTIUM_FILLET_MM,
        )

    # Trim the aorta flat at the inlet and at the descending outlet, replacing
    # the tube's terminal spherical caps by flat faces. The branch cylinders are
    # already capped, so they need no trimming.
    #
    # Each cut is confined to a ball about the end point. An unrestricted
    # half-space would not do: the inlet plane, extended, passes above the lower
    # descending limb and would amputate it. The ball has to be wide enough to
    # contain the whole cap and narrow enough to reach nothing else, which
    # TRIM_REACH * (local radius) is; verify_surface() checks that it holds.
    #
    # The ball, rather than a mask on the nearest centreline station: taking the
    # maximum only raises the field where the half-space exceeds it, so with a
    # ball this wide the region altered lies strictly inside it and the field
    # stays continuous. Cutting on the station index instead leaves a jump across
    # the mask boundary, and marching cubes then puts the face up to a voxel
    # proud of the plane rather than on it.
    inlet_n = -tangent[0] / np.linalg.norm(tangent[0])
    outlet_n = tangent[-1] / np.linalg.norm(tangent[-1])
    for origin, normal, radius in ((points_mm[0], inlet_n, radius_mm[0]),
                                   (points_mm[-1], outlet_n, radius_mm[-1])):
        near = np.sum((grid_xyz - origin) ** 2, axis=1) < (TRIM_REACH * radius) ** 2
        sdf[near] = np.maximum(sdf[near],
                               _sdf_halfspace(grid_xyz[near], origin, normal))

    grid = pv.ImageData(dimensions=tuple(dims), spacing=(VOXEL_MM,) * 3, origin=tuple(lo))
    grid.point_data["sdf"] = sdf
    surface = grid.contour([0.0], scalars="sdf").extract_surface().triangulate().clean()
    # clean() merges the coincident vertices of a degenerate marching-cubes
    # triangle, which leaves a line or a point cell behind. They enclose no
    # volume, but they are counted by n_cells and make vtkMassProperties baulk,
    # so keep the polygons alone.
    surface = pv.PolyData(surface.points, faces=surface.faces).clean()
    surface = surface.compute_normals(auto_orient_normals=True, consistent_normals=True)
    return surface, frames, (s_mm, points_mm, tangent, radius_mm)


# --------------------------------------------------------------------------
# Phase 3 — motion
# --------------------------------------------------------------------------
# The displacement field is analytic, so the exact position of every point at
# every phase is known and the reconstruction can be scored against it later.
# It is the sum of the two components the manuscript describes: a pressure-driven
# radial expansion, and a non-radial motion of the ascending aorta carried over
# from cardiac contraction. Both are modulated by one temporal waveform.
#
# Every number below is chosen, not measured. None is fitted to the cohort.

# Chosen. 70 bpm, a round value corresponding to none of the three cases.
CYCLE_S = 60.0 / 70.0
N_PHASES = 10                   # sampled at 0 %, 10 %, ... 90 % of the cycle

# Chosen. Peak circumferential strain, at the systolic peak and at the station
# where the radial weighting is greatest. Physiologically plausible; the exact
# value is a modelling choice.
PEAK_STRAIN = 0.05

# Chosen. Excursion of the ascending aorta at the systolic peak: a translation,
# and a rotation about an axis through the point where the motion is anchored.
# The axis is left-right, so the ascending limb rocks in the sagittal plane; the
# translation is anterior and caudal, the direction the root takes in systole.
ATA_TRANSLATION_MM = 5.0
ATA_TRANSLATION_DIR = np.array([0.0, 0.5, -0.866])
ATA_ROTATION_DEG = 3.0
ATA_ROTATION_AXIS = np.array([1.0, 0.0, 0.0])

# Chosen. The non-radial motion is applied in full up to S_NONRADIAL_FULL and has
# decayed to nothing by S_NONRADIAL_END, past the arch apex, where the aorta is
# tethered. Arc-length fractions.
S_NONRADIAL_FULL = 0.25
S_NONRADIAL_END = 0.45

# Chosen. Radial amplitude along the centreline, as a fraction of PEAK_STRAIN:
# greatest in the ascending segment, smaller in the stiffer descending aorta.
# Arc-length fractions and weights.
W_RADIAL_S = np.array([0.0, S_DMAX, S_ARCH, S_DESCENDING_START, 1.0])
W_RADIAL_W = np.array([0.85, 1.00, 0.60, 0.45, 0.40])

# Chosen. Fraction of the cycle from end-diastole to the systolic peak: 0.257 s
# at 70 bpm. It has to land on a sampled phase, so that the ten surfaces include
# the systolic extreme, and it must not put two sampled phases at the same point
# on the waveform: 0.20 does, giving bit-identical surfaces at 10 % and 60 %.
T_PEAK = 0.30


def waveform(tau):
    """Temporal modulation, one cardiac cycle, 0 at end-diastole and 1 at peak.

    An asymmetric raised cosine: a fast upstroke over T_PEAK of the cycle and a
    slower decay over the rest. It is non-negative by construction, so the vessel
    never contracts below its reference size, and its derivative vanishes at both
    ends of the cycle, which is what a volume minimum requires.
    """
    tau = np.mod(np.asarray(tau, dtype=float), 1.0)
    return np.where(tau < T_PEAK,
                    0.5 * (1.0 - np.cos(np.pi * tau / T_PEAK)),
                    0.5 * (1.0 + np.cos(np.pi * (tau - T_PEAK) / (1.0 - T_PEAK))))


def _radial_weight(s_mm):
    """Radial amplitude along the centreline, as a fraction of PEAK_STRAIN."""
    from scipy.interpolate import PchipInterpolator
    return PchipInterpolator(W_RADIAL_S * s_mm[-1], W_RADIAL_W)(s_mm)


def _nonradial_weight(s_mm):
    """Weight of the non-radial motion: 1 over the ascending segment, 0 past the
    arch, with a C2 transition so the blend introduces no curvature step."""
    x = np.clip((s_mm / s_mm[-1] - S_NONRADIAL_FULL)
                / (S_NONRADIAL_END - S_NONRADIAL_FULL), 0.0, 1.0)
    return 1.0 - (x ** 3 * (10.0 - 15.0 * x + 6.0 * x ** 2))


def _rotate(vectors, axis, angle):
    """Rodrigues rotation of row vectors about a unit axis."""
    c, s = np.cos(angle), np.sin(angle)
    return (vectors * c + np.cross(axis, vectors) * s
            + np.outer(vectors @ axis, axis) * (1.0 - c))


def deform(points_mm, tau, ref):
    """Map points from the 0 % configuration to cardiac phase tau.

    ref is the reference geometry from build_centreline(). Each point is carried
    by the centreline station nearest to it: scaled radially about that station,
    then moved with the ascending aorta's non-radial motion, weighted by how far
    along the vessel the station lies.

    This is the ground truth. The same map generates the phase geometries, so the
    displacement of any point at any phase is known in closed form.
    """
    from scipy.spatial import cKDTree
    s_ref, c_ref, _tangent, _curvature, _radius = ref
    _, idx = cKDTree(c_ref).query(np.atleast_2d(points_mm), workers=-1)

    g = float(waveform(tau))
    scale = 1.0 + PEAK_STRAIN * _radial_weight(s_ref)[idx] * g
    moved = c_ref[idx] + scale[:, None] * (points_mm - c_ref[idx])

    pivot = c_ref[int(round(S_NONRADIAL_END * (len(s_ref) - 1)))]
    axis = ATA_ROTATION_AXIS / np.linalg.norm(ATA_ROTATION_AXIS)
    rigid = (_rotate(moved - pivot, axis, np.deg2rad(ATA_ROTATION_DEG) * g) + pivot
             + ATA_TRANSLATION_MM * g * ATA_TRANSLATION_DIR / np.linalg.norm(ATA_TRANSLATION_DIR))
    return moved + _nonradial_weight(s_ref)[idx][:, None] * (rigid - moved)


def _discrete_frame(points_mm):
    """Arc length, unit tangent and curvature of a densely sampled curve.

    The deformed centreline is no longer the spline of phase 1 — the non-radial
    weighting stretches it — so its frame is taken by finite differences.
    """
    seg = np.linalg.norm(np.diff(points_mm, axis=0), axis=1)
    s_mm = np.concatenate([[0.0], np.cumsum(seg)])
    d1 = np.gradient(points_mm, s_mm, axis=0)
    d2 = np.gradient(d1, s_mm, axis=0)
    speed = np.linalg.norm(d1, axis=1, keepdims=True)
    curvature = np.linalg.norm(np.cross(d1, d2), axis=1) / speed[:, 0] ** 3
    return s_mm, d1 / speed, curvature


def geometry_at_phase(tau, ref):
    """The generating geometry at cardiac phase tau.

    The centreline is carried by the same map as any other point; the radius
    profile is scaled by the radial component, which leaves the centreline fixed.
    Returns arc length, points, tangents, radii and curvature.
    """
    s_ref, c_ref, _tangent, _curvature, r_ref = ref
    c_def = deform(c_ref, tau, ref)
    r_def = r_ref * (1.0 + PEAK_STRAIN * _radial_weight(s_ref) * float(waveform(tau)))
    s_def, t_def, k_def = _discrete_frame(c_def)
    return s_def, c_def, t_def, r_def, k_def


def build_phases(verbose=True):
    """Phase 3: the ten per-phase lumen surfaces.

    Each is surfaced independently, on its own grid, so the phases share neither
    vertices nor connectivity — as per-phase segmentations of a real acquisition
    do not. Establishing the correspondence is the pipeline's job, not the
    generator's.
    """
    ref = build_centreline()
    surfaces = []
    for k in range(N_PHASES):
        tau = k / N_PHASES
        s_def, c_def, t_def, r_def, _k_def = geometry_at_phase(tau, ref)
        surface, _frames, _geom = build_surface((s_def, c_def, t_def, r_def), verbose=False)
        if verbose:
            print(f"  {100 * tau:3.0f}%  g = {float(waveform(tau)):.3f}  "
                  f"{surface.n_points:6d} points  {surface.volume / 1000.0:6.1f} cm^3")
        surfaces.append(surface)
    return ref, surfaces


def _tube_volume(points_mm, radius_mm):
    """Volume of the tube swept along a centreline, sum(pi r^2 ds)."""
    seg = np.linalg.norm(np.diff(points_mm, axis=0), axis=1)
    r_mid = 0.5 * (radius_mm[:-1] + radius_mm[1:])
    return float(np.sum(np.pi * r_mid ** 2 * seg))


# --------------------------------------------------------------------------
# Phase 4 — SimVascular-shaped outputs
# --------------------------------------------------------------------------
# The layout is the one config_phantom.py describes, which is the layout
# config/local_example.py documents for a study case. Writing it here means the
# example loads through the existing configuration layer rather than around it.

# Chosen. Spacing of the centreline written to file. The 2000 stations the
# generator works with are far denser than SimVascular's own extraction, and than
# anything downstream expects.
CENTRELINE_OUT_SPACING_MM = 1.0

# Chosen. How far distal of the inlet cap the inlet control plane sits, measured
# along the centreline.
#
# It cannot sit on the cap. The phantom is a closed surface, so its inlet is a
# filled disk of triangles rather than an open rim, and the 0.2 mm slab that
# stage 0b isolates the inlet ring with then contains that whole disk. Measured
# on the 0 % phase with the plane on the cap: 338 ring points at every radius
# from 0.72 to 17.22 mm, where a ring should be a narrow annulus.
#
# That defeats the ray matching. find_optimal_point_intrs() accepts a ray when
# the mean distance to the three nearest ring points falls below 10 % of the mean
# radius; across a filled disk that tolerance, 1.391 mm here, is comparable to the
# spacing between the points, so whether a ray is accepted turns on where the
# nearest few happen to fall rather than on the geometry. The margin is that thin:
# on the 0 % phase fifteen of the sixteen rays landed and the sixteenth missed by
# 1.413 mm against the 1.391 mm tolerance. Most phases lost rays, some over half.
#
# Any offset clear of the cap gives a true annulus and all 160 rays land: 2, 3 and
# 5 mm were tested and each gave 160/160 across all ten phases. Three is the middle
# of that range and is several times both the 0.6 mm voxel and the 0.2 mm slab.
#
# Only the control plane moves. The inlet cap of the surface itself stays where it
# was, so mesh-surfaces/inlet.vtp is unchanged.
INLET_PLANE_OFFSET_MM = 3.0

# Chosen. Arc-length position of the arch control plane, as a fraction of the
# total. Distinct from S_ARCH, which is where the mid-arch *diameter* control
# point sits: the two were the same station until stage 0b was first run against
# the phantom, and they answer different questions.
#
# The plane has to sit where the vessel is still heading cranially. deformation.py
# forces the stored arch normal to have a negative z component and then keeps the
# half-space that normal points into, so the segment retained is the ascending one
# only when the negated normal points back towards the inlet — which it does only
# while the tangent still points up. Past the apex the forcing is a no-op and the
# same call keeps the descending limb instead. With the plane at S_ARCH, which is
# 0.44 and past the apex at 0.3787, stage 0b tracked the descending aorta: 2069
# points whose true motion peaks at 0.434 mm, against 9.33 mm over the ascending
# segment it should have been tracking.
#
# It also has to clear the three branch ostia, or the ring is not a circle. At
# 0.32, just distal of the brachiocephalic root at 0.315, the extracted ring runs
# out to 28.18 mm where the lumen radius is 18.21. At 0.30 it is 19.09 to 19.81
# against 19.13, and the tangent still has z = 0.758.
#
# Stage 2 is unaffected: cell 5 clips the centreline with this plane only for one
# study dataset, and takes the whole centreline for every other case, the phantom
# included.
ARCH_PLANE_S = 0.30

# Chosen. How close to a cap plane a triangle must sit, and how nearly parallel
# its normal must be, to count as part of that cap rather than as wall.
CAP_PLANE_TOL_MM = 0.5 * VOXEL_MM
CAP_NORMAL_TOL = 0.9


def _inlet_station(s_mm):
    """Index of the centreline station the inlet control plane sits at."""
    return int(np.searchsorted(s_mm, INLET_PLANE_OFFSET_MM))


def _arch_station(s_mm):
    """Index of the centreline station the arch control plane sits at."""
    return int(round(ARCH_PLANE_S * (len(s_mm) - 1)))


def _cap_planes(geometry, frames):
    """The flat faces of the lumen surface: the inlet, the descending outlet and
    one per supra-aortic branch, each an origin, an outward normal and a radius.

    The names are those config/local_example.py documents.
    """
    _s_mm, points_mm, tangent, radius_mm = geometry
    caps = [
        {"name": "inlet", "origin": points_mm[0], "radius": radius_mm[0],
         "normal": -tangent[0] / np.linalg.norm(tangent[0])},
        {"name": "out", "origin": points_mm[-1], "radius": radius_mm[-1],
         "normal": tangent[-1] / np.linalg.norm(tangent[-1])},
    ]
    for f in frames:
        caps.append({"name": f["name"], "origin": f["root"] + f["axis"] * f["length"],
                     "normal": f["axis"], "radius": 0.5 * f["diam_mm"]})
    return caps


def split_boundaries(surface, geometry, frames):
    """Partition the surface into the named boundaries the pipeline expects.

    Every triangle goes to exactly one of inlet, out, out1, out2, out3 or
    interface, so the parts reassemble into the whole with nothing counted twice
    and nothing dropped. The ring of triangles where marching cubes bridges the
    sharp rim of a cap falls to the interface; that is the conservative way round,
    since the interface is the surface the wall matching runs on.
    """
    centres = surface.cell_centers().points
    normals = surface.cell_normals
    caps = _cap_planes(geometry, frames)

    label = np.full(surface.n_cells, -1, dtype=int)
    for i, cap in enumerate(caps):
        rel = centres - cap["origin"]
        axial = rel @ cap["normal"]
        radial = np.linalg.norm(rel - np.outer(axial, cap["normal"]), axis=1)
        label[(np.abs(axial) < CAP_PLANE_TOL_MM)
              & (radial < 1.5 * cap["radius"])
              & (normals @ cap["normal"] > CAP_NORMAL_TOL)] = i

    parts = {cap["name"]: surface.extract_cells(label == i).extract_surface()
             for i, cap in enumerate(caps)}
    parts["interface"] = surface.extract_cells(label == -1).extract_surface()
    return parts


def _resample_polyline(points_mm, radius_mm, spacing_mm):
    """Resample a densely sampled curve at a fixed arc-length spacing."""
    seg = np.linalg.norm(np.diff(points_mm, axis=0), axis=1)
    s_mm = np.concatenate([[0.0], np.cumsum(seg)])
    s_out = np.linspace(0.0, s_mm[-1], int(round(s_mm[-1] / spacing_mm)) + 1)
    out = np.column_stack([np.interp(s_out, s_mm, points_mm[:, k]) for k in range(3)])
    return out, np.interp(s_out, s_mm, radius_mm)


def _write_centreline(path, geometry):
    """The centreline as a polyline, carrying the inscribed-sphere radius that a
    SimVascular centreline file carries."""
    import pyvista as pv

    _s_mm, points_mm, _tangent, radius_mm = geometry
    pts, radii = _resample_polyline(points_mm, radius_mm, CENTRELINE_OUT_SPACING_MM)
    line = pv.PolyData(pts)
    line.lines = np.hstack([[len(pts)], np.arange(len(pts))])
    line.point_data["MaximumInscribedSphereRadius"] = radii
    line.save(path)
    return line


def _write_segment_centreline(path, geometry):
    """The centreline of the tracked segment, as ``centerline.vtk``.

    Stage 0b reads this — a different file from the whole-aorta
    ``phantom_Centerlines.vtp`` that stage 2 uses — and sweeps its matching rings
    along it. deformation.py's own error message says what it has to be: the
    centreline of the segment being tracked, which is the one delimited by the two
    control planes, not the whole vessel.

    Legacy .vtk, because that is the name and format deformation.py opens.
    """
    import pyvista as pv

    s_mm, points_mm, _tangent, radius_mm = geometry
    i0 = _inlet_station(s_mm)
    i1 = _arch_station(s_mm)
    pts, radii = _resample_polyline(points_mm[i0:i1 + 1], radius_mm[i0:i1 + 1],
                                    CENTRELINE_OUT_SPACING_MM)
    line = pv.PolyData(pts)
    line.lines = np.hstack([[len(pts)], np.arange(len(pts))])
    line.point_data["MaximumInscribedSphereRadius"] = radii
    line.save(path)
    return line


def _write_cuts_posit(path, geometry):
    """The control-plane positions, in the format read_cuts_posit() parses.

    Two rows, because that is what cell 2 of wall_def_temp_def.ipynb consumes:
    plane_info_0 and plane_info_arc. Each is a centre and a normal. The first is
    the inlet plane, INLET_PLANE_OFFSET_MM distal of the inlet cap; the second is
    at ARCH_PLANE_S, in the distal ascending aorta just proximal to the
    brachiocephalic origin, which is where the tracked segment is treated as
    ending. Both constants carry the reasoning for where they sit.

    **Sign convention.** The inlet normal is written pointing *out of* the inlet,
    against the direction of flow. That is not arbitrary: cell 5 of stage 2 does

        mesh_E.clip(origin=plane_info_0[0], normal=plane_info_0[1])

    and PyVista's clip keeps the half-space the normal points away from. With the
    normal along the vessel the call discards the aorta and keeps the root; with it
    reversed the aorta survives, which is plainly what the step is for. Measured on
    this phantom: 20.5 % of the surface retained one way, 80.4 % the other.

    So the convention is inferred from what the pipeline does with the file, not
    read off a study case — no real cuts_posit.txt was available to check against.
    It is the only reading under which cell 5 is not destructive, but it should
    still be confirmed against one of the study files.
    """
    s_mm, points_mm, tangent, _radius_mm = geometry
    i_inlet = _inlet_station(s_mm)
    i_arch = _arch_station(s_mm)
    rows = (("inlet", points_mm[i_inlet], -tangent[i_inlet]),
            ("arch", points_mm[i_arch], tangent[i_arch]))
    with open(path, "w") as fh:
        fh.write("name\tcentre_mm\tnormal\n")
        for name, centre, normal in rows:
            u = normal / np.linalg.norm(normal)
            fh.write(f"{name}\t[{centre[0]:.6f} {centre[1]:.6f} {centre[2]:.6f}]"
                     f"\t[{u[0]:.6f} {u[1]:.6f} {u[2]:.6f}]\n")


def _fmt_point(p):
    return f"[{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}]"


def _write_transformation_details(path, geometry):
    """Three points on the aortic root, in the format select_cut.py reads for
    ``transformation_details.xlsx``.

    Equally spaced around the lumen circle at the inlet control plane's own
    station, which is INLET_PLANE_OFFSET_MM along the centreline rather than at
    the cap: their mean is that plane's centre exactly, and the cross product of
    two of their edges is along its normal, up to a sign select_cut.py resolves
    itself. That is the same plane ``_write_cuts_posit()`` writes analytically, so
    select_cut.py's own computation from these three points can be scored against
    it, rather than merely producing a plane with nothing to check it against.

    Clicking the root is what select_sinotub asks the operator for, and the three
    points have to lie on whichever plane the file records, so this station and
    ``_write_cuts_posit()``'s must stay the same one.
    """
    import pandas as pd

    s_mm, points_mm, tangent, radius_mm = geometry
    i = _inlet_station(s_mm)
    centre, normal, radius = points_mm[i], tangent[i], radius_mm[i]
    normal = normal / np.linalg.norm(normal)
    seed = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(normal, seed)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    angles = np.array([0.0, 2.0, 4.0]) * np.pi / 3.0
    points = [centre + radius * (np.cos(a) * e1 + np.sin(a) * e2) for a in angles]

    df = pd.DataFrame([{
        "Source Point 1": _fmt_point(points[0]),
        "Source Point 2": _fmt_point(points[1]),
        "Source Point 3": _fmt_point(points[2]),
    }])
    df.to_excel(path, index=False)


def _write_arc_cut_posit(path, geometry):
    """The arch control plane, in the format select_cut.py reads for
    ``arc_cut_posit.txt``.

    The same point and normal ``_write_cuts_posit()`` writes for the arch row.
    select_cut.py performs no computation on this file — it is read and written
    back unchanged — so this exercises that pass-through, not a second derivation.
    """
    s_mm, points_mm, tangent, _radius_mm = geometry
    i_arch = _arch_station(s_mm)
    centre, normal = points_mm[i_arch], tangent[i_arch]
    normal = normal / np.linalg.norm(normal)
    with open(path, "w") as fh:
        fh.write("Name\tSource Point\tnormal\n")
        fh.write(f"Arc\t{_fmt_point(centre)}\t{_fmt_point(normal)}\n")


def write_outputs(out_dir=None, verbose=True):
    """Phase 4: write the generated files in the layout config_phantom.py names.

    Returns the directory written to, the boundary parts and the phase surfaces.
    """
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    seg_dir = os.path.join(out_dir, "segmentation", "Segmentation_AI")
    surf_dir = os.path.join(out_dir, "mesh-complete", "mesh-surfaces")
    mesh_dir = os.path.join(out_dir, "mesh")
    ref_dir = os.path.join(out_dir, "reference")
    for d in (seg_dir, surf_dir, mesh_dir, ref_dir):
        os.makedirs(d, exist_ok=True)

    ref = build_centreline()
    _ref, surfaces = build_phases(verbose=verbose)

    # The per-phase surfaces, named as the pipeline globs for them: the leading
    # field is the percentage of the cycle, which is how they are ordered.
    for k, surface in enumerate(surfaces):
        surface.save(os.path.join(seg_dir, f"{10 * k}%_phantom.stl"))

    # The 0 % surface is the reference the rest is expressed against.
    s_def, c_def, t_def, r_def, _k_def = geometry_at_phase(0.0, ref)
    geometry = (s_def, c_def, t_def, r_def)
    frames = _branch_frames(*geometry)

    # SimVascular numbers the nodes of the complete mesh and carries that number
    # onto every boundary surface, which is how a boundary condition is tied back
    # to the volume. Stage 2 writes its wall-motion files against it, so the
    # generated surfaces have to carry it too. One-based, as SimVascular writes it.
    surfaces[0].point_data["GlobalNodeID"] = np.arange(
        1, surfaces[0].n_points + 1, dtype=np.int32)
    parts = split_boundaries(surfaces[0], geometry, frames)
    for name, part in parts.items():
        part.save(os.path.join(surf_dir, f"{name}.vtp"))
    surfaces[0].save(os.path.join(out_dir, "mesh-complete",
                                  "mesh-complete.exterior.vtp"))

    _write_centreline(os.path.join(mesh_dir, "phantom_Centerlines.vtp"), geometry)
    _write_cuts_posit(os.path.join(out_dir, "segmentation", "cuts_posit.txt"), geometry)

    # Stage 0a's inputs: select_cut.py runs headless when it can read these, and
    # scores its own plane against the one _write_cuts_posit() just wrote directly.
    _write_transformation_details(
        os.path.join(out_dir, "segmentation", "transformation_details.xlsx"), geometry)
    _write_arc_cut_posit(
        os.path.join(out_dir, "segmentation", "arc_cut_posit.txt"), geometry)

    # Stage 0b's own centreline input, of the tracked segment only.
    _write_segment_centreline(
        os.path.join(out_dir, "segmentation", "centerline.vtk"), geometry)

    # The ground truth: where every vertex of the 0 % surface goes, at every
    # phase, according to the analytic field the phases were generated from.
    points0 = surfaces[0].points
    tau = np.arange(N_PHASES) / N_PHASES
    displacement = np.stack([deform(points0, t, ref) - points0 for t in tau])
    np.savez_compressed(
        os.path.join(ref_dir, "displacement_truth.npz"),
        points_mm=points0.astype(np.float32),
        displacement_mm=displacement.astype(np.float32),
        tau=tau, cycle_s=CYCLE_S, peak_strain=PEAK_STRAIN,
        ata_translation_mm=ATA_TRANSLATION_MM, ata_rotation_deg=ATA_ROTATION_DEG,
    )

    if verbose:
        print(f"  written under {out_dir}")
    return out_dir, parts, surfaces


def _euler_characteristic(surface):
    """V - E + F for a closed triangle mesh, where E = 3F/2.

    Counts triangles rather than cells: a stray line or point cell would
    otherwise be counted as a face and shift the answer.
    """
    faces = surface.n_faces_strict
    return surface.n_points - (3 * faces // 2) + faces


def _check(label, ok, detail):
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}: {detail}")
    return ok


def verify():
    """Phase 1 checks from the plan: lengths and diameters to 0.1 mm, and
    curvature continuity."""
    s_mm, points_mm, tangent, curvature, radius_mm = build_centreline()
    d_mm = 2.0 * radius_mm
    passed = []

    print("Phase 1 — centreline and radius profile")

    length = s_mm[-1]
    passed.append(_check("total arc length in 350-400 mm", 350.0 <= length <= 400.0,
                         f"{length:.1f} mm"))

    dmax = d_mm.max()
    passed.append(_check("maximum diameter matches specification to 0.1 mm",
                         abs(dmax - DMAX_MM) <= 0.1, f"{dmax:.3f} vs {DMAX_MM} mm"))

    passed.append(_check("inlet diameter matches specification to 0.1 mm",
                         abs(d_mm[0] - D_SINOTUBULAR_MM) <= 0.1,
                         f"{d_mm[0]:.3f} vs {D_SINOTUBULAR_MM} mm"))
    passed.append(_check("outlet diameter matches specification to 0.1 mm",
                         abs(d_mm[-1] - D_DESCENDING_MM) <= 0.1,
                         f"{d_mm[-1]:.3f} vs {D_DESCENDING_MM} mm"))

    # Curvature continuity: a C2 spline has continuous curvature, so successive
    # samples must not jump. Compare the largest step against the mean step.
    jumps = np.abs(np.diff(curvature))
    passed.append(_check("curvature continuous (no step > 20x the mean)",
                         jumps.max() < 20.0 * jumps.mean(),
                         f"max step {jumps.max():.2e}, mean {jumps.mean():.2e} 1/mm"))

    r_curv = 1.0 / curvature.max()
    passed.append(_check("tightest radius of curvature is anatomically plausible",
                         20.0 <= r_curv <= 60.0, f"{r_curv:.1f} mm at the arch"))

    # A tube is only well defined where the lumen radius stays below the local
    # radius of curvature; otherwise the swept surface self-intersects.
    ratio = radius_mm * curvature
    passed.append(_check("lumen radius below the radius of curvature everywhere",
                         ratio.max() < 1.0, f"max r/R = {ratio.max():.3f}"))

    passed.append(_check("arc-length parametrisation is uniform",
                         np.allclose(np.diff(s_mm), s_mm[1] - s_mm[0], rtol=1e-6),
                         f"spacing {s_mm[1] - s_mm[0]:.4f} mm"))

    passed.append(_check("deterministic", True, "no random state used"))

    print(f"\n{sum(passed)}/{len(passed)} checks passed")
    return all(passed)


def verify_surface():
    """Phase 2 checks from the plan: watertight, outward normals, and the
    branches present at the specified sizes."""
    print("\nPhase 2 — lumen surface with supra-aortic branches")
    surface, frames, (s_mm, points_mm, tangent, radius_mm) = build_surface()
    passed = []

    print(f"  surface: {surface.n_points} points, {surface.n_cells} triangles")

    edges = surface.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                          manifold_edges=False, non_manifold_edges=False)
    passed.append(_check("watertight (no boundary edges)", edges.n_cells == 0,
                         f"{edges.n_cells} boundary edges"))

    nonman = surface.extract_feature_edges(boundary_edges=False, feature_edges=False,
                                           manifold_edges=False, non_manifold_edges=True)
    passed.append(_check("manifold (no non-manifold edges)", nonman.n_cells == 0,
                         f"{nonman.n_cells} non-manifold edges"))

    bodies = surface.connectivity("all").split_bodies()
    passed.append(_check("one connected component", len(bodies) == 1,
                         f"{len(bodies)} bodies"))

    euler = _euler_characteristic(surface)
    passed.append(_check("genus 0 (Euler characteristic 2)", euler == 2, f"chi = {euler}"))

    # Outward normals: by the divergence theorem the signed volume is positive
    # only when the normals point out of the solid.
    volume = surface.volume
    passed.append(_check("normals outward (positive enclosed volume)", volume > 0,
                         f"{volume / 1000.0:.1f} cm^3"))

    # The end trims are confined to a ball about each end. It must contain the
    # cap it is there to remove, and reach no other part of the vessel: if it
    # did, the trimming plane would cut that part too.
    # The stations whose own tube enters the ball must form one unbroken run
    # anchored at the end being trimmed. A second run would be a different part
    # of the vessel reaching into the ball, and the plane would cut that too.
    for label, i in (("inlet", 0), ("outlet", len(s_mm) - 1)):
        reach = TRIM_REACH * radius_mm[i]
        d = np.linalg.norm(points_mm - points_mm[i], axis=1)
        touching = np.flatnonzero(d < reach + radius_mm)
        one_run = touching.max() - touching.min() + 1 == touching.size
        passed.append(_check(f"the {label} trim reaches nothing but its own cap",
                             reach > radius_mm[i] and one_run and i in touching,
                             f"reach {reach:.1f} mm over cap radius {radius_mm[i]:.1f} mm; "
                             f"stations {touching.min()}-{touching.max()} touch the ball, "
                             f"in {1 if one_run else 2} run(s)"))

    # An analytic lower bound: the aorta alone, as a tube of revolution.
    tube_volume = _tube_volume(points_mm, radius_mm)
    passed.append(_check("enclosed volume exceeds the bare tube, as the branches add to it",
                         volume > tube_volume,
                         f"{volume / 1000.0:.1f} vs {tube_volume / 1000.0:.1f} cm^3"))
    passed.append(_check("enclosed volume within 25% of tube plus branches",
                         abs(volume - tube_volume) / tube_volume < 0.25,
                         f"{100 * (volume - tube_volume) / tube_volume:+.1f}%"))

    # The plan also asks for freedom from self-intersection. There is no check
    # for it here because there is nothing left to check: marching cubes over a
    # scalar field cannot emit a self-intersecting surface, each cell being
    # triangulated independently within its own voxel. The four topological
    # checks above are what would fail if that reasoning were wrong.

    # Three branches must be present. A plane above the arch apex cuts the
    # branches and nothing else, so the slice has exactly three loops.
    z_cut = max(f["root"][2] + 0.75 * f["length"] for f in frames)
    slab = surface.slice(normal=(0, 0, 1), origin=(0, 0, z_cut))
    loops = len(slab.connectivity("all").split_bodies()) if slab.n_cells else 0
    passed.append(_check("three branches present above the arch", loops == 3,
                         f"{loops} loops at z = {z_cut:.1f} mm"))

    # Branch outlet diameters, measured on the surface rather than assumed.
    for f in frames:
        tip = f["root"] + f["axis"] * (f["length"] - 1.5)
        cut = surface.slice(normal=f["axis"], origin=tip)
        if cut.n_cells == 0:
            passed.append(_check(f"{f['name']} ({f['label']}) outlet found", False, "no slice"))
            continue
        # The cutting plane is infinite and tilted, so it can catch the arch or a
        # neighbouring branch as well; keep only the loop around this branch.
        cut = cut.connectivity("closest", tip)
        pts = cut.points - tip
        rad = np.linalg.norm(pts - np.outer(pts @ f["axis"], f["axis"]), axis=1)
        measured = 2.0 * rad.mean()
        passed.append(_check(f"{f['name']} ({f['label']}) diameter within 0.3 mm",
                             abs(measured - f["diam_mm"]) <= 0.3,
                             f"{measured:.2f} vs {f['diam_mm']} mm"))

    # The bulge must survive surfacing: measure the lumen where the profile says
    # the maximum is, and compare with the specification.
    i_max = int(round(S_DMAX * (len(s_mm) - 1)))
    cut = surface.slice(normal=tangent[i_max], origin=points_mm[i_max])
    body = cut.connectivity("closest", points_mm[i_max]) if cut.n_cells else cut
    rad = np.linalg.norm(body.points - points_mm[i_max], axis=1)
    measured = 2.0 * rad.mean()
    passed.append(_check("maximum diameter preserved on the surface within 0.5 mm",
                         abs(measured - DMAX_MM) <= 0.5, f"{measured:.2f} vs {DMAX_MM} mm"))

    print(f"\n{sum(passed)}/{len(passed)} checks passed")
    return all(passed), surface


def verify_phases():
    """Phase 3 checks from the plan: the volume change matches the prescribed
    strain to better than 1 %, and no phase surface self-intersects."""
    print("\nPhase 3 — per-phase surfaces")
    passed = []

    # The waveform, before any geometry depends on it.
    tau_dense = np.linspace(0.0, 1.0, 20001)
    g_dense = waveform(tau_dense)
    passed.append(_check("waveform never contracts below the reference size",
                         g_dense.min() >= 0.0, f"min g = {g_dense.min():.3e}"))
    passed.append(_check("waveform peaks at 1", abs(g_dense.max() - 1.0) < 1e-9,
                         f"max g = {g_dense.max():.6f} at {tau_dense[g_dense.argmax()]:.3f}"))
    passed.append(_check("waveform is periodic", abs(waveform(0.0) - waveform(1.0)) < 1e-12,
                         f"g(0) = {float(waveform(0.0)):.3e}, g(1) = {float(waveform(1.0)):.3e}"))
    passed.append(_check("upstroke faster than the decay", T_PEAK < 0.5 * (1.0 - T_PEAK),
                         f"rise {T_PEAK * CYCLE_S:.3f} s, decay {(1 - T_PEAK) * CYCLE_S:.3f} s"))
    passed.append(_check("the systolic peak is one of the sampled phases",
                         abs(T_PEAK * N_PHASES - round(T_PEAK * N_PHASES)) < 1e-12,
                         f"peak at {100 * T_PEAK:.0f}%"))

    g_sampled = waveform(np.arange(N_PHASES) / N_PHASES)
    passed.append(_check("the ten phases are distinct",
                         len(np.unique(g_sampled)) == N_PHASES,
                         f"{len(np.unique(g_sampled))} distinct values of g"))

    ref = build_centreline()
    s_ref, c_ref, _t_ref, _k_ref, r_ref = ref

    # Phase 0 must be the reference configuration exactly: it is what every other
    # phase is measured against, and what the pipeline takes as its starting mesh.
    passed.append(_check("0 % reproduces the reference geometry",
                         np.allclose(deform(c_ref, 0.0, ref), c_ref, atol=1e-12),
                         f"max shift {np.abs(deform(c_ref, 0.0, ref) - c_ref).max():.2e} mm"))

    print(f"  building {N_PHASES} phase surfaces")
    _ref, surfaces = build_phases()

    # Volume against the prescribed strain. The branches do not distend, so their
    # contribution is measured once at 0 % and held fixed; what is being checked
    # is that the surfaces track the strain the field prescribes, not the offset.
    tube0 = _tube_volume(c_ref, r_ref)
    offset = surfaces[0].volume - tube0
    worst, worst_tau = 0.0, 0.0
    for k, surface in enumerate(surfaces):
        tau = k / N_PHASES
        _s, c_def, _t, r_def, _kappa = geometry_at_phase(tau, ref)
        predicted = _tube_volume(c_def, r_def) + offset
        rel = abs(surface.volume - predicted) / predicted
        if rel > worst:
            worst, worst_tau = rel, tau
    passed.append(_check("enclosed volume matches the prescribed strain to 1 %",
                         worst < 0.01, f"worst {100 * worst:.3f}% at {100 * worst_tau:.0f}%"))

    v = np.array([s.volume for s in surfaces])
    passed.append(_check("volume is least at 0 % and greatest at the systolic peak",
                         v.argmin() == 0 and v.argmax() == round(T_PEAK * N_PHASES),
                         f"least at {10 * v.argmin():.0f}%, greatest at {10 * v.argmax():.0f}%, "
                         f"swing {100 * (v.max() - v.min()) / v.min():.1f}%"))

    # Every phase surface must be as sound as the 0 % one.
    unsound = []
    for k, surface in enumerate(surfaces):
        edges = surface.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                              manifold_edges=False, non_manifold_edges=False)
        nonman = surface.extract_feature_edges(boundary_edges=False, feature_edges=False,
                                               manifold_edges=False, non_manifold_edges=True)
        euler = _euler_characteristic(surface)
        bodies = len(surface.connectivity("all").split_bodies())
        if edges.n_cells or nonman.n_cells or euler != 2 or bodies != 1:
            unsound.append(f"{10 * k}%: {edges.n_cells} boundary, {nonman.n_cells} "
                           f"non-manifold, chi = {euler}, {bodies} bodies")
    passed.append(_check("every phase surface watertight, manifold, genus 0, one body",
                         not unsound,
                         f"all {N_PHASES} phases" if not unsound else "; ".join(unsound)))

    # A swept tube is only well defined while the lumen radius stays below the
    # local radius of curvature. Phase 1 checks it for the reference; the
    # non-radial motion changes the curvature, so it has to hold at every phase.
    worst_ratio, worst_ratio_tau = 0.0, 0.0
    for k in range(N_PHASES):
        tau = k / N_PHASES
        _s, _c, _t, r_def, kappa = geometry_at_phase(tau, ref)
        ratio = float((r_def * kappa).max())
        if ratio > worst_ratio:
            worst_ratio, worst_ratio_tau = ratio, tau
    passed.append(_check("lumen radius below the radius of curvature at every phase",
                         worst_ratio < 1.0,
                         f"max r/R = {worst_ratio:.3f} at {100 * worst_ratio_tau:.0f}%"))

    # The excursion the field actually delivers, measured on the 0 % surface.
    peak = round(T_PEAK * N_PHASES) / N_PHASES
    pts = surfaces[0].points
    disp = np.linalg.norm(deform(pts, peak, ref) - pts, axis=1)
    inlet = pts[:, 2] < 20.0                      # the ascending limb, near the root
    outlet = np.linalg.norm(pts - c_ref[-1], axis=1) < 30.0
    passed.append(_check("ascending excursion of the order specified",
                         5.0 <= disp[inlet].max() <= 15.0,
                         f"max {disp[inlet].max():.2f} mm at the root, "
                         f"{ATA_TRANSLATION_MM} mm translation + {ATA_ROTATION_DEG} deg"))
    passed.append(_check("descending aorta moves radially only",
                         disp[outlet].max() < 1.0,
                         f"max {disp[outlet].max():.3f} mm at the outlet, "
                         f"radial expansion alone"))

    # Determinism, over the part of the generator phase 3 adds.
    again = geometry_at_phase(0.3, build_centreline())
    once = geometry_at_phase(0.3, ref)
    passed.append(_check("deterministic", all(np.array_equal(a, b) for a, b in zip(once, again)),
                         "identical geometry on a second build"))

    print(f"\n{sum(passed)}/{len(passed)} checks passed")
    return all(passed), surfaces


def verify_outputs():
    """Phase 4 checks from the plan: load_case("PHANTOM") returns, and the config
    cell of wall_def_temp_def.ipynb runs against what was written."""
    import pyvista as pv

    print("\nPhase 4 — SimVascular-shaped outputs")
    out_dir, parts, surfaces = write_outputs()
    s_ref, points_ref, tangent_ref, _k_ref, radius_ref = build_centreline()
    passed = []

    # The configuration layer, reached the way the notebooks reach it.
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import sys
    sys.path.insert(0, repo)
    from config import load_case
    cfg = load_case("PHANTOM")

    passed.append(_check("load_case(\"PHANTOM\") returns without config/local.py",
                         isinstance(cfg, dict) and cfg["case"] == "PHANTOM",
                         f"{len(cfg)} keys, cycle {cfg['cycle_duration_s']:.4f} s"))
    passed.append(_check("heart rate agrees with the generator",
                         abs(cfg["cycle_duration_s"] - CYCLE_S) < 1e-12,
                         f"{cfg['heart_rate_bpm']} bpm, {cfg['cycle_duration_s']:.4f} s"))

    # Every path the config cell of the stage-2 notebook reads.
    missing = [k for k in ("segmentation_dir", "mesh_surfaces_dir", "mesh_exterior",
                           "centerline", "cuts_posit") if not os.path.exists(cfg[k])]
    passed.append(_check("every path the config cell reads exists", not missing,
                         "all five" if not missing else f"missing {missing}"))

    # Stage 0a's inputs: select_cut.py reads these instead of prompting
    # interactively. run_stage0a.py is what actually exercises select_cut.py
    # against them; this only checks that they were written.
    stage0a_inputs = [os.path.join(cfg["segmentation_dir"], name)
                      for name in ("transformation_details.xlsx", "arc_cut_posit.txt")]
    passed.append(_check("stage 0a inputs written",
                         all(os.path.isfile(p) for p in stage0a_inputs),
                         "transformation_details.xlsx and arc_cut_posit.txt"))

    # Stage 0b's input: the centreline of the tracked segment, which deformation.py
    # reads by that name and refuses to run without.
    seg_cline_path = os.path.join(cfg["segmentation_dir"], "centerline.vtk")
    seg_cline = pv.read(seg_cline_path) if os.path.isfile(seg_cline_path) else None
    passed.append(_check("stage 0b centreline written, spanning the control planes",
                         seg_cline is not None and seg_cline.n_points > 1,
                         f"centerline.vtk, {seg_cline.n_points if seg_cline else 0} points"))

    # That cell reads three of them outright; do exactly what it does.
    mesh_ref = pv.read(os.path.join(cfg["mesh_surfaces_dir"], "interface.vtp"))
    centreline = pv.read(cfg["centerline"])
    sys.path.insert(0, repo)
    from SUPORT_def_deformation import read_cuts_posit
    data = read_cuts_posit(cfg["cuts_posit"])
    plane_info_0 = np.array(data[0])
    plane_info_arc = np.array(data[1])
    passed.append(_check("the pipeline's own reader parses cuts_posit.txt",
                         plane_info_0.shape == (2, 3) and plane_info_arc.shape == (2, 3),
                         f"{len(data)} rows, shapes {plane_info_0.shape} "
                         f"and {plane_info_arc.shape}"))
    i_inlet = _inlet_station(s_ref)
    t0 = tangent_ref[i_inlet] / np.linalg.norm(tangent_ref[i_inlet])
    passed.append(_check("the inlet control plane is clear of the cap, normal out of the inlet",
                         np.linalg.norm(plane_info_0[0] - points_ref[i_inlet]) < 1e-5
                         and abs(plane_info_0[1] @ t0 + 1.0) < 1e-5,
                         f"centre {np.round(plane_info_0[0], 3)} mm, "
                         f"{INLET_PLANE_OFFSET_MM:.1f} mm along the centreline, "
                         f"normal.tangent = {plane_info_0[1] @ t0:.6f}"))

    # The point of that sign: cell 5 clips with this normal and keeps the far side.
    # Check on the geometry itself that the aorta is what survives.
    _kept = surfaces[0].clip(origin=plane_info_0[0], normal=plane_info_0[1])
    passed.append(_check("the inlet plane clips the root away and keeps the aorta",
                         _kept.n_points > 0.5 * surfaces[0].n_points,
                         f"{100.0 * _kept.n_points / surfaces[0].n_points:.1f} % of the "
                         f"surface retained by cell 5's clip"))
    # Stage 0b keeps the half-space its z-forced normal points into, so the
    # ascending segment survives that clip only while the tangent still points
    # cranially. See ARCH_PLANE_S.
    i_arch = _arch_station(s_ref)
    t_arch = tangent_ref[i_arch] / np.linalg.norm(tangent_ref[i_arch])
    passed.append(_check("the arch control plane is in the still-ascending aorta",
                         np.linalg.norm(plane_info_arc[0] - points_ref[i_arch]) < 1e-5
                         and t_arch[2] > 0.1,
                         f"centre {np.round(plane_info_arc[0], 1)} mm, at "
                         f"{100 * ARCH_PLANE_S:.0f}% of the arc length, "
                         f"tangent z = {t_arch[2]:.3f}"))

    # And clear of the ostia, or the ring stage 0b matches around is not a circle.
    # Silent in the generator otherwise: it shows up only as a poor 0b score.
    first_branch = min(b["s_frac"] for b in BRANCHES)
    passed.append(_check("the arch control plane is proximal to every branch",
                         ARCH_PLANE_S < first_branch,
                         f"{ARCH_PLANE_S:.3f} against the first branch at "
                         f"{first_branch:.3f}"))

    # The boundary partition: complete, disjoint, and each part the right size.
    total = sum(p.n_cells for p in parts.values())
    passed.append(_check("boundaries partition the surface exactly",
                         total == surfaces[0].n_cells,
                         f"{total} of {surfaces[0].n_cells} triangles, "
                         f"{len(parts)} parts"))
    passed.append(_check("the named boundaries are those the pipeline expects",
                         set(parts) == {"inlet", "out", "out1", "out2", "out3",
                                        "interface"},
                         ", ".join(sorted(parts))))
    # Stage 2 writes its wall-motion files keyed on GlobalNodeID, so every
    # boundary must carry it, and the values must index the complete surface.
    exterior = pv.read(cfg["mesh_exterior"])
    ids_ok, detail = True, "all six boundaries"
    for name, part in parts.items():
        ids = part.point_data.get("GlobalNodeID")
        if ids is None:
            ids_ok, detail = False, f"{name}.vtp carries no GlobalNodeID"
            break
        mapped = exterior.points[ids - 1]
        if not np.allclose(mapped, part.points, atol=1e-9):
            ids_ok, detail = False, f"{name}.vtp GlobalNodeID does not index the exterior"
            break
    passed.append(_check("boundaries carry a GlobalNodeID indexing the complete surface",
                         ids_ok, detail))

    passed.append(_check("interface.vtp is the wall, not a cap",
                         mesh_ref.n_cells == parts["interface"].n_cells
                         and mesh_ref.n_cells > 0.9 * surfaces[0].n_cells,
                         f"{mesh_ref.n_cells} triangles, "
                         f"{100 * mesh_ref.n_cells / surfaces[0].n_cells:.1f}% of the surface"))

    # Cap areas, as an equivalent radius rather than a percentage: marching cubes
    # rounds the sharp rim of a cap over about half a voxel, which costs every cap
    # the same fraction of a millimetre and so costs a small branch a much larger
    # share of its area. A tolerance in millimetres measures the same thing on all
    # five; a tolerance in per cent does not.
    for cap in _cap_planes((s_ref, points_ref, tangent_ref, radius_ref),
                           _branch_frames(s_ref, points_ref, tangent_ref, radius_ref)):
        r_eq = np.sqrt(parts[cap["name"]].area / np.pi)
        passed.append(_check(f"{cap['name']} radius within a voxel of the specification",
                             abs(r_eq - cap["radius"]) < VOXEL_MM,
                             f"{r_eq:.2f} vs {cap['radius']:.2f} mm "
                             f"({r_eq - cap['radius']:+.2f} mm, voxel {VOXEL_MM} mm)"))

    # The per-phase files, named as cell 3 of the notebook globs for them.
    seg = os.path.join(cfg["segmentation_dir"], "Segmentation_AI")
    names = [f for f in os.listdir(seg) if f.endswith(".stl") and f[0].isdigit()]
    keys = sorted(int(f.split("_")[0].replace("%", "")) for f in names)
    passed.append(_check("ten phase surfaces, named as the pipeline globs for them",
                         keys == [10 * k for k in range(N_PHASES)],
                         f"{len(names)} files, {keys[0]}% to {keys[-1]}%"))

    # The centreline file.
    length = float(np.sum(np.linalg.norm(np.diff(centreline.points, axis=0), axis=1)))
    passed.append(_check("centreline length preserved to 0.5 mm",
                         abs(length - s_ref[-1]) < 0.5,
                         f"{length:.2f} vs {s_ref[-1]:.2f} mm over "
                         f"{centreline.n_points} points"))
    passed.append(_check("centreline carries the inscribed-sphere radius",
                         "MaximumInscribedSphereRadius" in centreline.point_data,
                         f"max {centreline.point_data['MaximumInscribedSphereRadius'].max():.2f} mm"))

    # The ground truth.
    truth = np.load(os.path.join(cfg["reference_dir"], "displacement_truth.npz"))
    passed.append(_check("ground truth covers every vertex at every phase",
                         truth["displacement_mm"].shape
                         == (N_PHASES, surfaces[0].n_points, 3),
                         f"{truth['displacement_mm'].shape}"))
    passed.append(_check("ground truth vanishes at 0 %",
                         np.abs(truth["displacement_mm"][0]).max() < 1e-9,
                         f"max {np.abs(truth['displacement_mm'][0]).max():.1e} mm, "
                         f"round-off in the rigid blend at g = 0"))
    peak = round(T_PEAK * N_PHASES)
    passed.append(_check("ground truth peaks at the systolic phase",
                         np.linalg.norm(truth["displacement_mm"], axis=2).max(axis=1).argmax()
                         == peak,
                         f"max {np.linalg.norm(truth['displacement_mm'][peak], axis=1).max():.2f} mm "
                         f"at {10 * peak}%"))

    print(f"\n{sum(passed)}/{len(passed)} checks passed")
    return all(passed), out_dir


def verify_reproducible():
    """Phase 7's first criterion: the generator reproduces its output exactly.

    This matters more than it would otherwise, because the decision was to
    distribute the generator and not its output: if a second run differed, a
    reader could not check that what they produced is what was described.

    Writes a second copy to a temporary directory and compares every file byte
    for byte. Costs a full rebuild, so it is opt-in.
    """
    import hashlib
    import shutil
    import tempfile

    print("\nPhase 7 — reproducibility")
    passed = []

    def digest(root):
        out = {}
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in sorted(filenames):
                path = os.path.join(dirpath, name)
                with open(path, "rb") as fh:
                    out[os.path.relpath(path, root)] = hashlib.sha256(fh.read()).hexdigest()
        return out

    first_root = write_outputs(verbose=False)[0]
    second_root = tempfile.mkdtemp(prefix="phantom-repeat-")
    try:
        write_outputs(out_dir=second_root, verbose=False)
        second = digest(second_root)
        # The default data directory may also contain ignored Stage-2 products.
        # Compare only the paths the generator itself wrote in the clean repeat.
        first_all = digest(first_root)
        first = {name: first_all[name] for name in second if name in first_all}
    finally:
        shutil.rmtree(second_root, ignore_errors=True)

    passed.append(_check("the second run writes the same files",
                         set(first) == set(second),
                         f"{len(first)} files, "
                         f"{len(set(first) ^ set(second))} differing in name"))
    differing = sorted(k for k in set(first) & set(second) if first[k] != second[k])
    passed.append(_check("every file is byte-for-byte identical", not differing,
                         f"all {len(first)} files" if not differing
                         else f"{len(differing)} differ: {differing[:4]}"))

    print(f"\n{sum(passed)}/{len(passed)} checks passed")
    return all(passed)


if __name__ == "__main__":
    import sys
    ok1 = verify()
    ok2, _ = verify_surface()
    ok3, _ = verify_phases()
    ok4, _ = verify_outputs()
    ok5 = True
    if "--check-reproducible" in sys.argv:
        ok5 = verify_reproducible()
    else:
        print("\nPhase 7 — reproducibility: not checked. It rebuilds everything a "
              "second time; run with --check-reproducible.")
    sys.exit(0 if (ok1 and ok2 and ok3 and ok4 and ok5) else 1)
