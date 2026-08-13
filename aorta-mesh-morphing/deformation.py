"""Stage 0b — track the ascending-aorta wall motion and write ``dispm/``.

Supplied by the author in August 2026. This is the step whose absence the deposit
recorded until then: cell 4 of ``wall_def_temp_def.ipynb`` reads ``dispm/mw_<n>.vtp``
and nothing here produced them. It is the implementation of the deformation-tracking
framework of the earlier paper (IEEE TBME 2026, doi:10.1109/TBME.2026.3681119), whose
authors agreed in August 2026 to its being published here.

    conda activate d_view
    AORTA_CASE=A python deformation.py

Run ``select_cut.py`` first — it writes the ``cuts_posit.txt`` this reads — and stage 2
afterwards. Inputs are the per-phase surfaces in ``<segmentation_dir>/Segmentation_AI``
and a saved ``<segmentation_dir>/centerline.vtk``; the output is
``<segmentation_dir>/dispm/mw_<n>.vtp``, one per phase, each carrying a
``Displacement`` array on the 0 % surface.

The method, in the order the code runs it:

1. Align each phase to the 0 % phase over an 80 mm cube about the inlet plane, so that
   the aortic root is registered rather than the whole aorta.
2. Trim to the ascending segment between the two control planes, and match 16 points
   around the inlet ring and 16 around the arch ring.
3. A first RBF interpolation on those 32 points, which removes the large non-radial
   motion driven by cardiac contraction.
4. Rings swept along the centreline, matched between the original and the once-morphed
   surface, and a second RBF interpolation for the radial expansion.
5. The two displacement fields are summed, smoothed, and written.

**The registration in step 1 is ICP, not RANSAC.** ``point_to_points_reg`` is called
here and only here, and its body is ``registration_icp`` with a point-to-plane
estimator, initialised from the identity — there is no RANSAC stage in it. The
manuscript describes this step as *ICP registration of the aortic valve using a
RANSAC-based point-to-plane metric*; RANSAC-then-ICP is what the phantom verification
notebook of the earlier paper does, not what produced the published displacement
fields. The wording needs correcting before submission.

**Two constants differed between the author's two copies of this step.** The batch
driver this file comes from used an 80 mm alignment cube and rings over the first 80 %
of the centreline; his interactive notebook of the same step used 100 mm and 30 %. Asked
in August 2026 which produced the published ``dispm/``, he answered that he is almost
certain it was the notebook's, and that the step is not sensitive to either — any cube
that contains the aortic valve will do, and the larger one was probably for a patient
with a larger valve. The notebook's values are the ones set below; the batch driver's
were 80 and 0.8, recorded here because his answer is a recollection rather than a
record.

Changes made in depositing it: paths come from ``config`` instead of an absolute
Windows archive root, and the loop over every dataset in that archive is replaced by
one case; ``os.path.exists`` was called with two arguments, which raises TypeError, so
the centreline was never read; ``meshes`` was a module-level global; the output
directory is now passed to ``save_deformation_vtp`` rather than appended inside it; and
an unused list of colour names is gone. Nothing else is altered.
"""

import os

import numpy as np
import pyacvd
import pyvista as pv

from config import load_case
from SUPORT_def_deformation import *

# Rings swept along the centreline, as a fraction of its points. The author's
# notebook's value; the batch driver used 0.8. See the module docstring.
RING_FRACTION = 0.3

# Side of the cube about the inlet plane over which the phases are registered. It has
# to contain the aortic valve and is otherwise not critical, per the author. His
# notebook's value; the batch driver used 80.
ALIGNMENT_CUBE_MM = 100

_NO_CENTERLINE = (
    "{path} not found. The tracking step needs a centreline of the ascending "
    "segment, saved as centerline.vtk next to the segmentations. centerline_simp's "
    "VMTK path cannot supply it — see that module's docstring — so generate it in "
    "SimVascular, as the study did, and save it there."
)


def main(case, save_figures=False):
    cfg = load_case(case)
    number = cfg["dataset_id"]
    seg_dir = cfg["segmentation_dir"]

    direct = os.path.join(seg_dir, "Segmentation_AI")
    infolder_total = os.listdir(direct)
    organized_files={}
    for file_name in infolder_total:
            if file_name.endswith(".stl") and file_name[0].isdigit():
                first_digit = int(file_name.split('_')[0].replace('%', ''))  # Extract the first digit
                organized_files.setdefault(first_digit, []).append(file_name)

    load_name_mesh,load_name_mesh_load=[],[]

    for digit in sorted(organized_files.keys(), key=lambda x: f"{x:03}"):
        load_name_mesh.append(organized_files[digit][0])
        load_name_mesh_load.append(os.path.join(direct, organized_files[digit][0]))

    meshes = []
    for file in load_name_mesh_load:
            mesh_r=pv.read(file)
            mesh_r=mesh_r.clean().fill_holes(50)
            clus = pyacvd.Clustering(mesh_r)
            clus.cluster(10000)
            mesh_rem = clus.create_mesh()
            mesh_rem.fill_holes(1,inplace=True)
            meshes.append(mesh_rem)

    orginal_0=meshes[0].copy()

    data = read_cuts_posit(cfg["cuts_posit"])
    plane_info_0=np.array(data[0])
    plane_info_arc=np.array(data[1])
    # Both normals are stored pointing caudally; select_cut.py enforces that for the
    # inlet plane when it writes the file, and this restates it for both.
    if plane_info_0[1][2]>0: plane_info_0[1]=-plane_info_0[1]
    if plane_info_arc[1][2]>0: plane_info_arc[1]=-plane_info_arc[1]
#####start with inlet

    meshes_to_align=[]
    cut_cube=pv.Cube(plane_info_0[0],ALIGNMENT_CUBE_MM,ALIGNMENT_CUBE_MM,ALIGNMENT_CUBE_MM)
    for num_mesh,mesh in enumerate(meshes):
        meshes_to_align.append(mesh.clip_box(cut_cube,invert=False))

    transformation_matrix_list=[]

    for num_mesh,mesh in enumerate(meshes_to_align):

        if num_mesh==0:
            transformation_matrix = np.identity(4)
            source_pcd1 = o3d.geometry.PointCloud()
            source_pcd1.points = o3d.utility.Vector3dVector(np.array(mesh.points))
            source_pcd1_DW = source_pcd1.voxel_down_sample(voxel_size=0.5)
        else:
            target_pcd1 = o3d.geometry.PointCloud()
            target_pcd1.points = o3d.utility.Vector3dVector(np.array(mesh.points))
            target_pcd1_DW= target_pcd1.voxel_down_sample(voxel_size=0.5)

            # Point-to-plane ICP from the identity. Not RANSAC, whatever the name of
            # the variable and of the manuscript's sentence — see the docstring.
            ransac_result = point_to_points_reg(target_pcd1_DW,source_pcd1_DW)
            transformation_matrix= np.array(ransac_result.transformation)

            meshes[num_mesh]=meshes[num_mesh].transform(transformation_matrix)
        transformation_matrix_list.append(transformation_matrix)
    print("plot")

    plotter = pv.Plotter()
    for num_mesh, mesh in enumerate(meshes_to_align):
        plotter.add_mesh(mesh.transform(transformation_matrix_list[num_mesh]),opacity=0.5)
    plotter.show()

    for num_mesh,mesh in enumerate(meshes):
        mesh_T=mesh.clip(origin=plane_info_0[0],normal=plane_info_0[1])
        meshes[num_mesh]=mesh_T.connectivity('largest')
#####select point on inlet

    ring_inlet_list_orig=[None]*len(meshes)

    for num_mesh,mesh in enumerate(meshes):
            ring_T =mesh.clip(normal=plane_info_0[1],origin=plane_info_0[0]-plane_info_0[1]*0.2, invert=False)
            ring_inlet_list_orig[num_mesh]=select_closest(ring_T,plane_info_0[0])
            if num_mesh==0: plane_info_0[0]=ring_inlet_list_orig[num_mesh].center_of_mass()
    vectors_inlet,point_matching_inlet_T=calculate_points_ref(ring_inlet_list_orig,plane_info_0[1])

    for i,mesh in enumerate(meshes):
        if len(point_matching_inlet_T[i])!=16:
            print("erro inlet points, more than 16")

    plane_info=[None]*len(meshes)
    point_matching_inlet=[None]*len(point_matching_inlet_T)

    for num_mesh,mesh in enumerate(meshes):
        inverse_matrix = np.linalg.inv(transformation_matrix_list[num_mesh])
        meshes[num_mesh]=mesh.transform(inverse_matrix, inplace=True)
        plane_info[num_mesh]=trasform_plane(plane_info_0,inverse_matrix)
        point_matching_inlet[num_mesh]=trasform_points(point_matching_inlet_T[num_mesh],inverse_matrix)

#####start arch cut
    mesh_T=meshes[0].clip(normal=plane_info_arc[1], origin=plane_info_arc[0], inplace=False,invert=False)
    mesh_T=mesh_T.connectivity('largest')

    for num_mesh,mesh in enumerate(meshes):

        mesh_T=mesh.clip(normal=plane_info_arc[1], origin=plane_info_arc[0],invert=False)
        meshes[num_mesh]=select_closest(mesh_T, plane_info_arc[0])
#####select point on arch

    ring_arc_list_orig=[]
    point_matching_arc_temp=[]
    vectors_arc=None
    ring_Tval=[]
    center_arc_ring=[]

    for num_mesh,mesh in enumerate(meshes):
        ring =mesh.clip(normal=plane_info_arc[1],origin=plane_info_arc[0]+plane_info_arc[1]*2)
        ring=ring.connectivity('largest')
        center_arc_ring.append(plane_info_arc[0])
        ring_arc_list_orig.append(ring)
        vectors_arc,point_matching_arc_temp2,ring_T=calculate_points_dif (ring,plane_info_arc,vectors_arc,number_vectors=16)
        ring_Tval.append(ring_T)
        point_matching_arc_temp.append(point_matching_arc_temp2)

    point_matching_arc = correct_t(point_matching_arc_temp,ring_Tval, center_arc_ring, vectors_arc)

#### first RBF
    for i in range(len(point_matching_arc)):
        if len(point_matching_inlet[i]) !=len(point_matching_arc[i]):
            print(i,"point_matching_inlet",len(point_matching_inlet[i]))
            print(i,"point_matching_arc",len(point_matching_arc[i]))
    original_points=np.concatenate((point_matching_inlet,point_matching_arc),axis=1)

    meshes_new,deformation_init=rbf_calculation(meshes,original_points,original_points[0])
#### second RBF
    centerline_path = os.path.join(seg_dir, "centerline.vtk")
    if not os.path.exists(centerline_path):
        raise FileNotFoundError(_NO_CENTERLINE.format(path=centerline_path))
    Cline = pv.read(centerline_path)

    number_rings =int(len(Cline.points)*RING_FRACTION)

    ring_centers=[None]*(len(meshes))

    for num,mesh in enumerate(meshes):
        Cline_len=centerline_length(Cline.points,plane_info[num][0])
        ring_centers_per_mesh=[]

        for i in range(0,number_rings-1):
            interval_Cline=[[0,0,0],[0,0,0]]
            if i==0 : Cline_posit=Cline_len
            else: Cline_posit=(Cline_len / number_rings) * (i)

            interval_Cline[0],interval_Cline[1]=calc_cut_posit(Cline.points,Cline_posit)
            if interval_Cline[0] is not None: ring_centers_per_mesh.append(interval_Cline)

        ring_centers[num]=(ring_centers_per_mesh)
    deformation_secod=[[]]*len(meshes)

    target_points=[]
    origin_points=[]
    for num in range(len(meshes)):
        point_matching_list_orig,point_matching_list_orig_new = [original_points[num]],[original_points[num]]

        for ring_num in range(len(ring_centers[num])):
            ring=cut_ring(meshes[num], ring_centers[num][ring_num][0], ring_centers[num][ring_num][1])
            ring_new=cut_ring(meshes_new[num], ring_centers[num][ring_num][0], ring_centers[num][ring_num][1])
            if ring is not None and ring_new is not None:

                vectors_center,point_matching_li,point_matching_li_new =calculate_points_dif_2ring(ring,ring_new, ring_centers[num][ring_num][1])

                if len(point_matching_li_new)==len(point_matching_li):
                    point_matching_li, point_matching_li_new = remove_close_points(point_matching_list_orig, point_matching_list_orig_new, point_matching_li, point_matching_li_new)
                    if len(point_matching_li_new)!=0:
                        point_matching_list_orig.append(point_matching_li)
                        point_matching_list_orig_new.append(point_matching_li_new)

        Origin= [item for sublist in point_matching_list_orig_new for item in sublist]
        Target= [item for sublist in point_matching_list_orig for item in sublist]
        target_points.append(Target)
        origin_points.append(Origin)
        deformation_secod[num]=rbf_calculation_def(meshes_new[num],Target,Origin)
###save
    deformation_init_temp = np.array(deformation_init)
    deformation_secod_temp = np.array(deformation_secod)
    deformation =  deformation_secod_temp+deformation_init_temp

    meshes_new_new=[None]*len(meshes)
    smoothed_vectors=[]

    for num,mesh in enumerate(meshes):
        mesh_points=meshes[0].copy()
        deformed_mesh_points = np.stack([mesh_points.points[:,0]+deformation[num][:,0],mesh_points.points[:,1]+deformation[num][:,1],mesh_points.points[:,2]+deformation[num][:,2]], axis=-1)
        mesh_points.points = deformed_mesh_points
        meshes_new_new[num]=mesh_points

    for timesD in deformation:
            smoothed_vectors.append(smooth_displacement_vectors(meshes[0].points, timesD))
    smoothed_vectors=np.array(smoothed_vectors)

    new_percentage = range(0, len(smoothed_vectors))
    # dispm/ is where stage 2 reads them from. Passed in rather than appended by
    # save_deformation_vtp, which stage 2 also calls, with a different destination.
    out_dir = os.path.join(seg_dir, "dispm")
    save_deformation_vtp(out_dir,smoothed_vectors,meshes[0],new_percentage)
    print(f"wrote {len(smoothed_vectors)} phases to {out_dir}")

    if save_figures:
        # Needs imageio for the GIF, which environment.yml does not pin.
        save_gif(meshes[0],os.path.join(seg_dir,f"{number}.gif"),smoothed_vectors,number)
        save_screenshot(orginal_0,meshes_new_new[0],Cline,os.path.join(seg_dir,f"{number}.png"),number)


if __name__ == "__main__":
    selected_case = os.environ.get("AORTA_CASE")
    if not selected_case:
        raise SystemExit(
            "Set AORTA_CASE to a label defined in the private config/local.py "
            "before running deformation.py."
        )
    main(selected_case)
