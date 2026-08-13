"""Stage 0a — pick the two control planes and write ``cuts_posit.txt``.

Supplied by the author in August 2026, together with ``deformation.py``, as the step
that produces ``dispm/``. Run it once per case, before ``deformation.py`` and before
stage 2; both read the file it writes.

    conda activate d_view
    AORTA_CASE=A python select_cut.py

It writes ``<segmentation_dir>/cuts_posit.txt``: two rows, each a centre and a normal.
The first bounds the ascending aorta at the aortic root, the second at the arch. It
does nothing if the file already exists — the planes are picked by hand and are not
meant to be re-picked.

**The sign convention lives here.** The inlet normal is negated whenever its z
component is positive, so the file always stores it pointing caudally. Its two
consumers are consistent with that and with each other: ``deformation.py`` re-applies
the same rule on reading and clips with PyVista's default ``invert=True``, and cell 5
of ``wall_def_temp_def.ipynb`` clips the same way, keeping everything cranial to the
plane. ``examples/phantom`` writes its own file to the same convention. This was an
open question in the deposit until this script arrived; it is not open any more.

The inlet plane comes from one of two routes:

- ``transformation_details.xlsx``, three points on the aortic root in the columns
  ``Source Point 1..3``, if the case has one. These are per-case files from the
  author's archive, in the patient's own frame, so they are not distributed.
- otherwise the root is picked interactively, three clicks, through
  ``select_sinotub``.

The file ``arc_cut_posit.txt`` plays the same role for the arch plane. Without it the
arch plane is picked interactively on ``centerline.vtk``, a centreline generated
upstream in SimVascular. The repository does not generate that centreline. This
explicit input replaces the former fallback to an unavailable VMTK class.

Reading the ``.xlsx`` needs ``openpyxl``, which ``environment.yml`` pins for this.
"""

import os

import numpy as np
import pandas as pd
import pyacvd
import pyvista as pv

import select_sinotub as snt
from config import load_case
from SUPORT_def_deformation import string_to_matrix

COLUMNS = ["Name", "Source Point", "normal"]


def select_cut_ataa(mesh, centerline, point1, point2):
    """Pick the arch plane on the centreline, in a PyVista window.

    Click a point: the nearest centreline point becomes the plane centre and the
    step to it from its predecessor becomes the normal. Close the window to return.
    """
    centerline = np.array(centerline)

    # Convert the centerline to a PyVista PolyData object for visualization
    centerline_mesh = pv.PolyData(centerline)
    centerline_mesh["index"] = np.arange(len(centerline))  # Add index for selection

    # Create a plotter
    plotter = pv.Plotter(notebook=False)

    center = centerline.mean(axis=0)
    vector1 = center - centerline[0]
    vector2 = centerline[len(centerline)-2] - centerline[0]
    normalc = np.cross(vector1, vector2)
    normalc_norm = np.linalg.norm(normalc)
    if normalc_norm == 0:
        raise ValueError("centreline does not define a stable display plane")
    normalc = normalc / normalc_norm

    plotter.add_mesh(mesh.clip(origin=center,normal=normalc), opacity=0.5, color="lightgrey")
    plotter.add_points(np.array([point1]), color="green", point_size=10)
    plotter.add_points(np.array([point2]), color="blue", point_size=10)
    plotter.add_mesh(centerline_mesh, color="red", point_size=5)

    picked_center=None
    normal=None

    def point_picker(point):
        nonlocal picked_center, normal
        picked_center=(point)
        print(point)
            # Find the closest point in the centerline
        distances = np.linalg.norm(centerline - picked_center, axis=1)
        selected_idx = np.argmin(distances)

        # Define the center of the cut plane
        center = centerline[selected_idx]

        # Calculate the normal vector using the previous point in the centerline
        if selected_idx == 0:
            print("Selected the first point in the centerline; normal cannot be defined.")
            return None

        previous_point = centerline[selected_idx - 1]
        normal = center - previous_point
        normal_norm = np.linalg.norm(normal)
        if normal_norm == 0:
            print("Selected centreline segment has zero length; choose another point.")
            normal = None
            return
        normal = normal / normal_norm
                        # Create the plane
        plane = pv.Plane(center=picked_center, direction=normal, i_size=50, j_size=50)
        plotter.add_mesh(plane, color='g', opacity=0.5)

    # Enable point picking
    plotter.enable_point_picking(callback=point_picker,pickable_window=True)
    plotter.show(title="Select a Point on the Centerline")

    if picked_center is None or normal is None:
        raise RuntimeError("no valid arch point was selected")
    return picked_center, normal


def main(case):
    cfg = load_case(case)
    seg_dir = cfg["segmentation_dir"]
    file_path = cfg["cuts_posit"]

    if os.path.exists(file_path):
        print(f"{file_path} exists; nothing to do.")
        return

    direct = os.path.join(seg_dir, "Segmentation_AI")
    infolder_total = os.listdir(direct)

    mesh_name = None
    for file_name in infolder_total:
        if file_name.endswith(".stl") and file_name[0].isdigit():
            if file_name[0]=='0':
                mesh_name=file_name
                break
    if mesh_name is None:
        raise FileNotFoundError(f"no 0 % surface in {direct}")
    print("mesh name:",mesh_name)

    mesh_r=pv.read(os.path.join(direct, mesh_name))
    mesh_r=mesh_r.clean().fill_holes(50)
    clus = pyacvd.Clustering(mesh_r)
    clus.cluster(20000)
    mesh_rem = clus.create_mesh()
    mesh_rem.fill_holes(1,inplace=True)

# check inlet

    details = os.path.join(seg_dir, "transformation_details.xlsx")
    if os.path.exists(details):

        df_read = pd.read_excel(details)
        points=[string_to_matrix(df_read["Source Point 1"][0]),string_to_matrix(df_read["Source Point 2"][0]),string_to_matrix(df_read["Source Point 3"][0])]

        vector1 = points[1] - points[0]
        vector2 = points[2] - points[0]
        normali = np.cross(vector1, vector2)
        normali = normali / np.linalg.norm(normali)

        # The convention: the stored inlet normal always points caudally. See the
        # module docstring for the two consumers that depend on it.
        if normali[2]>0:
            normali=-normali
            print("normal changed")

        plane_info_0=[np.mean(points, axis=0),normali]

    else:
        # it will create the points
        plane_def=snt.sinotub(mesh_rem)
        plane_info_0=plane_def.plane_data

# check art
    plane_info_arc=[[0,0,0],[0,1,0]]

    mesh_resp=mesh_rem.copy()

    mesh_resp.clip(normal=plane_info_0[1],origin=plane_info_0[0],inplace=True)
    mesh_resp.clip(normal="y",origin=[0,mesh_resp.bounds[3]-30, 0], invert=True, inplace=True)
    mesh_resp=mesh_resp.connectivity('largest')

    mesh_temp = mesh_resp.clip("y",origin=[0,mesh_resp.bounds[3]-3,0], invert=False)
    mesh_temp=mesh_temp.connectivity('largest')
    center_arc=mesh_temp.center_of_mass()

    arc_cut = os.path.join(seg_dir, "arc_cut_posit.txt")
    if os.path.exists(arc_cut):
        # Read the text file
        df_read = pd.read_csv(arc_cut, sep="\t")
        # Extract points
        points = [
            string_to_matrix(df_read["Source Point"][0]),
            string_to_matrix(df_read["normal"][0])
        ]

        plane_info_arc[0], plane_info_arc[1] = points[0], points[1]
    else:
        centerline_path = os.path.join(seg_dir, "centerline.vtk")
        if not os.path.isfile(centerline_path):
            raise FileNotFoundError(
                "Stage 0a needs either arc_cut_posit.txt or a saved SimVascular "
                f"centreline at {centerline_path}"
            )
        Cline = pv.read(centerline_path)
        if Cline.n_points < 2:
            raise ValueError(f"centreline has fewer than two points: {centerline_path}")
        Cline_temp = Cline.clip(normal=plane_info_0[1], origin=plane_info_0[0], invert=True)
        if Cline_temp.n_points < 2:
            raise ValueError("no usable centreline remains above the inlet plane")
        plane_info_arc[0], plane_info_arc[1] = select_cut_ataa(mesh_resp, Cline_temp.points,point1=plane_info_0[0] + plane_info_0[1],point2=center_arc)

    data = [
        ["Inlet", str(plane_info_0[0]), str(plane_info_0[1])],
        ["Arc", str(plane_info_arc[0]), str(plane_info_arc[1])]
    ]

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Write to a text file
    with open(file_path, "w") as f:
        # Write header
        f.write("\t".join(COLUMNS) + "\n")
        # Write data
        for row in data:
            f.write("\t".join(row) + "\n")
    print("wrote", file_path)


if __name__ == "__main__":
    selected_case = os.environ.get("AORTA_CASE")
    if not selected_case:
        raise SystemExit(
            "Set AORTA_CASE to a label defined in the private config/local.py "
            "before running select_cut.py."
        )
    main(selected_case)
