
from scipy.optimize import minimize
import open3d as o3d
from scipy.interpolate import Rbf
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
import numpy as np
import pyvista as pv
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
#funtions for deformation


def plot_mesh(mesh1=None,mesh2=None, c_points1=np.array([]),c_points2=np.array([]),point_last=np.array([]),save=None,innotebook=True,number=None):    
    plotter = pv.Plotter(notebook=innotebook)
    if mesh1!=None: plotter.add_mesh(mesh1, color="b",opacity=0.5)
    if mesh2!=None: plotter.add_mesh(mesh2, color="red",opacity=0.5)
    if c_points1.size!=0: plotter.add_points(c_points1, color="green", point_size=2)
    if c_points2.size!=0: plotter.add_points(c_points2, color="red", point_size=2)
    if point_last.size!=0: plotter.add_points(point_last, color="red", point_size=10)
    plotter.add_axes()
    plotter.camera_position = 'yz'
    if number is not None: plotter.add_text(f"NUMBER {number}", position="lower_right", font_size=15)
    plotter.show(jupyter_backend='static')

    if save is not None: plotter.screenshot(save)
    
cores=['r','g','b','y','c','m']

def plot_4_meshes(meshes,meshes2,points,number, innotebook=True):
    plotter = pv.Plotter(shape=(4, 4),notebook=innotebook)
    plotter.add_text(f"NUMBER {number}", position="lower_right", font_size=15)
    plotter.add_axes()
    # plotter.camera_position = 'yz'
    # plotter.camera_azimuth = 20

    for val,posit in zip([2,3,4,5,  7,8,9,10,   11,13,14,15,    16,17,18,19],[[0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3],[3,0],[3,1],[3,2],[3,3]]):
        plotter.subplot(posit[0], posit[1]) 
    
        plotter.add_text(f"Mesh {val*5}", position="upper_edge", font_size=10)
        plotter.add_mesh(meshes[val], color="r",opacity=0.5)
        plotter.add_mesh(meshes2[val], color="b",opacity=0.5)
        plotter.add_points(np.array(points[val]), color="g", point_size=5)

    plotter.show(jupyter_backend='static')
    return

def save_screenshot(mesh_0,mesh1,centerline, output_path,number):

    # Set up plotter
    pl = pv.Plotter(off_screen=True,shape=(1, 2))
    pl.subplot(0, 0)
    # Add mesh_0 to the first subplot
    pl.add_mesh(mesh_0, color="b",opacity=0.5)
    pl.add_mesh(mesh1, color="r",opacity=0.5)
    pl.add_mesh(centerline, color="g")
    pl.camera_position = 'yz'
    pl.camera.elevation = 5
    pl.camera.azimuth = 300
    pl.add_axes()
    pl.add_text(f"NUMBER {number}", position="lower_right", font_size=15)
    # Add mesh_0 to the second subplot
    pl.subplot(0, 1)
    pl.add_mesh(mesh_0, color="b",opacity=0.5)
    pl.add_mesh(mesh1, color="r",opacity=0.5)
    pl.add_mesh(centerline, color="g")
    pl.camera_position = 'yz'
    pl.camera.elevation = 5
    pl.camera.azimuth = 10
    pl.add_axes()
    # pl.show(jupyter_backend='static')
    pl.screenshot(output_path)



def save_gif(mesh_0, output_gif_path, deformation,number):
    """Animate the displacement over the cardiac cycle, two views side by side.

    Not called anywhere in this repository; kept as a visualisation helper.
    Writing the GIF needs `imageio`, which environment.yml does not pin for that
    reason — `pip install imageio` before using this.

    Mutates mesh_0's points frame by frame and restores them before returning.
    """
    sargs = dict(title="Displacement (mm)", font_family="arial", shadow=True, n_labels=5, color='black')
    sargs2 = dict(position_x=10)

    # The mapper has to be bound to a named array for the per-frame update below
    # to reach it; without this it would render whatever happened to be active.
    scalars_name = "Displacement (mm)"
    mesh_0[scalars_name] = np.zeros(mesh_0.n_points)

    # Set up plotter
    pl = pv.Plotter(off_screen=True, shape=(1, 2))
    pl.subplot(0, 0)
    # Add mesh_0 to the first subplot
    pl.add_mesh(mesh_0, scalars=scalars_name, lighting=True, cmap="jet", scalar_bar_args=sargs2, clim=[0, 15])
    pl.camera_position = 'yz'
    pl.camera.elevation = 5
    pl.camera.azimuth = 300
    pl.add_axes()
    pl.add_text(f"NUMBER {number}", position="lower_right", font_size=15)
    # Add mesh_0 to the second subplot
    pl.subplot(0, 1)
    pl.add_mesh(mesh_0, scalars=scalars_name, lighting=True, cmap="jet", scalar_bar_args=sargs, clim=[0, 15])
    pl.camera_position = 'yz'
    pl.camera.elevation = 5
    pl.camera.azimuth = 200
    pl.add_axes()

    pl.open_gif(output_gif_path)
    reference_points = mesh_0.points.copy()

    # Plotter.update_coordinates() and update_scalars() were removed in PyVista
    # 0.44; both subplots hold the same mesh object, so mutating it in place is
    # what replaces them.
    for deform_t in deformation:
        mesh_0.points = reference_points + deform_t
        mesh_0[scalars_name] = np.linalg.norm(deform_t, axis=1)
        pl.write_frame()

    mesh_0.points = reference_points
    pl.close()


def string_to_matrix(s):
    # Remove '[' and ']' from the string and split it by whitespace
    values = s.replace('[', '').replace(']', '').replace(',', '').split()
    # Convert each value to float
    return np.array([float(value) for value in values])

def point_to_points_reg (source, target):
    """Point-to-plane ICP between two Open3D point clouds. Never called here.

    The name and the original comment said RANSAC-based point-to-point; the body
    is plain `registration_icp` with a point-to-plane estimator, and no RANSAC
    stage. Corrected rather than renamed, since a reader comparing the deposit
    with the manuscript needs to find it.

    The registration the manuscript describes — RANSAC followed by ICP, on the
    aortic valve — belongs to the upstream tracking step that produces `dispm/`,
    which is not part of this repository. Nothing in this file performs it, and
    nothing here calls this function.
    """
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=30, max_nn=500))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=30, max_nn=500))
    threshold =  1
    ransac_result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000))
    
    return ransac_result
def distance_min(t,mesh, cpoint,vector):
    t_scalar = float(np.asarray(t).reshape(-1)[0])
    moving_point = cpoint + t_scalar * vector
    distances_t=np.linalg.norm(mesh.points - moving_point, axis=1)
    closest_indices = np.argsort(distances_t)[:3]
    closest_points = mesh.points[closest_indices]
    distances=np.linalg.norm(closest_points - moving_point, axis=1)
    return  np.mean(distances)



def find_optimal_point_intrs(mesh, cpoint,vector):
    """Return the closest accepted point along a ray, or ``(None, None)``.

    A miss used to return one-element zero arrays, which callers mistook for a
    valid three-dimensional point. The explicit ``None`` result prevents a failed
    intersection from silently entering an RBF control-point array.
    """
    distances=np.linalg.norm(mesh.points - cpoint, axis=1)
    init_t = np.mean(distances)
    
    max_Dist=init_t*0.1

    result = minimize(distance_min, init_t, args=(mesh, cpoint,vector), method='BFGS')
    t_value = float(np.asarray(result.x).reshape(-1)[0])
    dist_final=distance_min(t_value,mesh, cpoint,vector)
    # plot_mesh(mesh1=mesh,c_points1=closest_points,c_points2=np.array( cpoint + result.x * vector))
    if dist_final< max_Dist:
        return np.asarray(cpoint) + t_value * np.asarray(vector), t_value
    return None, None



def rodrigues_rotation(v, k, theta):
    k = k / np.linalg.norm(k) 
    v_rot = (v * np.cos(theta) +np.cross(k, v) * np.sin(theta) +k * (np.dot(k, v)) * (1 - np.cos(theta)))
    return v_rot

def calculate_vectors(vector_main,number_vectors=16,first_vector=None):
    rotation_angle = 2 * np.pi / number_vectors
    new_vectors=[]
    if first_vector is not None: 
        projection_main = np.dot(first_vector, vector_main) * vector_main
        vectors = first_vector - projection_main
    else:
            perp_vector = np.cross(vector_main, [0, 0, 1])# z-axis as arbitrary vector
            vectors =np.array(  perp_vector / np.linalg.norm(perp_vector)) # Normalize
    
    for i in range(0, number_vectors):
        rotated_vector = rodrigues_rotation(vectors, vector_main, rotation_angle * i)
        rotated_vector = rotated_vector / np.linalg.norm(rotated_vector)
        new_vectors.append(rotated_vector)
    return new_vectors
    # return vectors

def calculate_points_ref (rings,vector_normal):
    
    vectors=calculate_vectors(vector_normal)
    point_matching=[]

    for ring_index, ring in enumerate(rings):
            ring_sec=[]
            for ray_index, vector in enumerate(vectors):
                point_find,_=find_optimal_point_intrs(ring, ring.center_of_mass(),vector)
                if point_find is None:
                    raise RuntimeError(
                        f"ray intersection failed for ring {ring_index}, ray {ray_index}"
                    )
                ring_sec.append(point_find)

            point_matching.append(np.array(ring_sec))
    return vectors,point_matching

def trasform_points(list_points,transformation_matrix):
    point_lit_res=[]
    for i in list_points:
        center_cut=np.dot(transformation_matrix,[i[0],i[1],i[2],1])
        point_lit_res.append(center_cut[:3])
    return point_lit_res

def trasform_plane(plane_info_arc,transformation_matrix):
    vector_cut=np.dot(transformation_matrix,[plane_info_arc[1][0],plane_info_arc[1][1],plane_info_arc[1][2],0])
    center_cut=np.dot(transformation_matrix,[plane_info_arc[0][0],plane_info_arc[0][1],plane_info_arc[0][2],1])
    return center_cut[:3],vector_cut[:3]
def select_closest(mesh, point):
    tree = cKDTree(mesh.points)
    _, index = tree.query(point)
    components = mesh.connectivity("point_seed",point_ids=index)

    return components
def cut_ring(mesh, cpoint, normal1):
    C_dist_1 = mesh.clip(normal=normal1, origin=cpoint + normal1 * 0.5, invert=True)
    C_dist = C_dist_1.clip(normal=normal1, origin=cpoint - normal1 * 0.5, invert=False)
    if len(C_dist.points) == 0:
        return None
    # body = check_close_body(C_dist, cpoint)
    body=C_dist.connectivity('largest')
    return body

def circularity_deviation(vector, cpoint, mesh1):
    vector_normal = (vector) / np.linalg.norm(vector)
    body = cut_ring(mesh1, cpoint, vector_normal)
    if body is None:
        return None, None 
    distances = np.linalg.norm(body.points - body.center_of_mass(), axis=1)
    return np.mean(distances), np.std(distances)
#calculate cut
#create the meshes to aligh
def find_ATAA(mesh,centerline):
        
        std_desv2=[]
        optimal_vect_T=[]
        center=[]
        mean_calc=0
        fix_enum=0
        i=0
        for enum2 in range(len(centerline)-1):  
            if enum2>25:
                mean_dev,std_desv=circularity_deviation(centerline[enum2] - centerline[enum2 + 1], centerline[enum2], mesh)
                if mean_dev is None:
                    std_desv2.append(0)
                else:std_desv2.append(std_desv)
            elif enum2>5:
                _,stdd=circularity_deviation(centerline[enum2] - centerline[enum2 + 1], centerline[enum2], mesh)
                std_desv2.append(0)
                mean_calc+=stdd
                i+=1
            else:
                std_desv2.append(0)
        max_temp=mean_calc/i
        for num2,std in enumerate(std_desv2):
            
            if std>max_temp+2:
                num3=num2-4
                optimal_vect=centerline[num3] - centerline[num3+1]
                optimal_vect_T = -optimal_vect / np.linalg.norm(optimal_vect)
                center=centerline[num3]
                fix_enum=num3
                return std_desv2,optimal_vect_T,center,fix_enum

def calculate_points_dif (ring,ring_posit,vectors=None,number_vectors=16):
    vector_normal=ring_posit[1]
    center_temp= ring_posit[0]
    if vectors is None:vectors=calculate_vectors(vector_normal,number_vectors,first_vector=np.array([0,0,1]))
    
    ring_sec=[]
    result_t=[]
    for ray_index, vector in enumerate(vectors):
        points_finded,t=find_optimal_point_intrs(ring,center_temp,vector)
        if points_finded is None:
            raise RuntimeError(f"ray intersection failed for ray {ray_index}")
        result_t.append(t)
        ring_sec.append(points_finded)
    return vectors,ring_sec,result_t

def read_cuts_posit(file_path):
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    
    data = []
    with open(file_path, "r") as f:
        # Read the header
        header = f.readline().strip().split("\t")
        # Read the data rows
        for line in f:
            row = line.strip().split("	")
            first=row[1][1:-1].split(' ')
            second=row[2][1:-1].split(' ')
            
            def remotion(array):
                arrayR=[]
                for i in array:
                    if not i=='':
                        arrayR.append(float(i))
                return arrayR
            
            first=remotion(first)
            second=remotion(second) 

            data.append([first,second])

    return data

def correct_t(point_matching_arc_temp,corrected_t, center, vectors, threshold=2):
    ring_sec_final = point_matching_arc_temp

    for i, vector in enumerate(vectors):
        vector_t = []
        for val_mes, ring_center in enumerate(center):
            vector_t.append(corrected_t[val_mes][i])

        vector_t = np.array(vector_t)  # Convert to NumPy array for advanced indexing
        mediat = np.mean(vector_t)
        dpt = np.std(vector_t)

        outliers = (vector_t < mediat - threshold * dpt) | (vector_t > mediat + threshold * dpt)
        

        
        if np.any(outliers):
            # print(i)
            # plt.figure()
            # plt.ylim([0, 30])
            # plt.hlines(mediat - threshold * dpt, 0, 16)
            # plt.hlines(mediat + threshold * dpt, 0, 16)
            # plt.plot(vector_t)
            # plt.show()
            vector_t[outliers] = np.nan  # Now you can index with boolean array
            indices_validos = np.where(~np.isnan(vector_t))[0]  # Indices of valid values
            valores_validos = vector_t[indices_validos]
            spline = CubicSpline(indices_validos, valores_validos)
            indices_outliers = np.where(outliers)[0]
            vector_t[indices_outliers] = spline(indices_outliers)
            # plt.figure()
            # plt.ylim([0, 30])
            # plt.hlines(mediat - threshold * dpt, 0, 16)
            # plt.hlines(mediat + threshold * dpt, 0, 16)
            # plt.plot(vector_t)
            # plt.plot(indices_outliers, vector_t[indices_outliers], 'ro')
            # plt.show()
            for val_mesin in indices_outliers:      
                ring_sec_final[val_mesin][i] = np.array(np.array(center[val_mesin]) + vector_t[val_mesin] * np.array(vector))
    return ring_sec_final
def rbf_calculation(meshes,point_matching_list_origT,point_matching_list_orig_newT):
    deformation_secod=[[]]*len(meshes)
    meshes_new=[[]]*len(meshes)
    x_mesh_ref = meshes[0].points[:,0]
    y_mesh_ref = meshes[0].points[:,1]
    z_mesh_ref = meshes[0].points[:,2]
    original_points=point_matching_list_orig_newT

    for num_mesh,mesh in enumerate(meshes):
        
        target_points=np.array(point_matching_list_origT[num_mesh])
        displacements = target_points - original_points
        
        x_orig, y_orig, z_orig = original_points.T
        dx, dy, dz = displacements.T
        rbf_x = Rbf(x_orig, y_orig, z_orig, dx, function='gaussian')
        rbf_y = Rbf(x_orig, y_orig, z_orig, dy, function='gaussian')
        rbf_z = Rbf(x_orig, y_orig, z_orig, dz, function='gaussian')

        x_deformed = rbf_x(x_mesh_ref, y_mesh_ref, z_mesh_ref)
        y_deformed = rbf_y(x_mesh_ref, y_mesh_ref, z_mesh_ref)
        z_deformed = rbf_z(x_mesh_ref, y_mesh_ref, z_mesh_ref)
        
        deformation_secod[num_mesh]=np.stack([x_deformed,y_deformed,z_deformed], axis=-1)
        
        deformed_mesh_points = np.stack([x_mesh_ref+x_deformed,y_mesh_ref+ y_deformed,z_mesh_ref+ z_deformed], axis=-1)
        
        mesh_temp= meshes[0].copy()
        mesh_temp.points = deformed_mesh_points
        meshes_new[num_mesh]=mesh_temp
        
    return meshes_new,deformation_secod
def centerline_length(points,inlet_center):
    length = 0


    for i in range(len(points)-1):
        # Calculate the Euclidean distance between consecutive points
        distance = np.linalg.norm(np.array(points[i + 1]) - np.array(points[i]))
        length += distance
    d= np.array(points[-1]) - np.array(inlet_center)
    v= (np.array(points[-1]) - np.array(points[-2]))
    length += np.dot(d, v) / np.linalg.norm(v)
    
    return length


def calc_cut_posit(points, leng):
    """Return the point and unit tangent at arclength ``leng`` along a polyline."""
    if leng < 0:
        raise ValueError("leng must be non-negative")

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("points must have shape (n, 3) with n >= 2")

    cumulative_distance = 0.0
    for i in range(len(points) - 1):
        segment = points[i + 1] - points[i]
        segment_length = np.linalg.norm(segment)
        if segment_length == 0:
            continue

        direction = segment / segment_length
        if cumulative_distance + segment_length >= leng:
            remaining_distance = leng - cumulative_distance
            new_point = points[i] + remaining_distance * direction
            return new_point, direction
        cumulative_distance += segment_length

    return None, None

def  calculate_points_dif_2ring(ring,ring_new, rings_vector):
    
    point_matching=[]
    point_matching_new=[]
    
    vectors=calculate_vectors(rings_vector,first_vector=np.array([0,0,1]))
    center=ring.center_of_mass()
    for ray_index, vector in enumerate(vectors):
        points_finded,_=find_optimal_point_intrs(ring, center,vector)
        points_finded_new,_=find_optimal_point_intrs(ring_new, center,vector)
        if points_finded is None or points_finded_new is None:
            # The two arrays must retain one-to-one correspondence. Omitting the
            # ray from both is safe for this optional intermediate-ring control
            # set; required inlet/arch rings use the hard-error paths above.
            continue
        point_matching.append(points_finded)
        point_matching_new.append(points_finded_new)

    if len(point_matching) < 3:
        raise RuntimeError("fewer than three paired-ring intersections remain")

    return vectors,point_matching,point_matching_new

def rbf_calculation_def(meshes,target_points,original_points):

    x_mesh_ref = meshes.points[:,0]
    y_mesh_ref = meshes.points[:,1]
    z_mesh_ref = meshes.points[:,2]
    
    displacements = np.array(target_points) - np.array(original_points)
    
    x_orig, y_orig, z_orig =  np.array(original_points).T
    dx, dy, dz = displacements.T

    rbf_x = Rbf(x_orig, y_orig, z_orig, dx, function='linear')
    rbf_y = Rbf(x_orig, y_orig, z_orig, dy, function='linear')
    rbf_z = Rbf(x_orig, y_orig, z_orig, dz, function='linear')

    x_deformed = rbf_x(x_mesh_ref, y_mesh_ref, z_mesh_ref)
    y_deformed = rbf_y(x_mesh_ref, y_mesh_ref, z_mesh_ref)
    z_deformed = rbf_z(x_mesh_ref, y_mesh_ref, z_mesh_ref)
    
    deformation_secod=np.stack([x_deformed,y_deformed,z_deformed], axis=-1)

        
    return deformation_secod
def is_point_close(p1, p2, threshold=5):
    """
    Check if two points are close based on a distance threshold.
    """
    distance = np.linalg.norm(np.array(p1) - np.array(p2))
    return distance < threshold

def remove_close_points(point_matching_list_orig, point_matching_list_orig_new, 
                        point_matching_li, point_matching_li_new):
    """
    Remove points that are too close to existing points in the main lists.
    If a point is removed from one list, remove its corresponding point from the other list.
    """
    indices_to_remove = []
    
    # Iterate through each point in point_matching_li and check if it's close to any point in point_matching_list_orig
    for i, (point_li, point_li_new) in enumerate(zip(point_matching_li, point_matching_li_new)):
        # Compare with all points in the existing lists
        for orig_points, orig_points_new in zip(point_matching_list_orig, point_matching_list_orig_new):
            for p_orig, p_orig_new in zip(orig_points, orig_points_new):
                # If point is too close to any existing point in point_matching_list_orig
                if is_point_close(point_li, p_orig) or is_point_close(point_li_new, p_orig_new):
                    indices_to_remove.append(i)  # Mark the index for removal
                    break  # No need to check further for this point
    
    # Remove points based on the collected indices
    point_matching_li_filtered = [p for i, p in enumerate(point_matching_li) if i not in indices_to_remove]
    point_matching_li_new_filtered = [p for i, p in enumerate(point_matching_li_new) if i not in indices_to_remove]
    return point_matching_li_filtered, point_matching_li_new_filtered
def gaussian_weights(distances, sigma):
    return np.exp(-0.5 * (distances / sigma)**2)

def smooth_displacement_vectors(nodes, displacement_vectors, radius=2, smoothing_factor=0.8, sigma=1.0):
    tree = cKDTree(nodes)
    smoothed_vectors = np.zeros_like(displacement_vectors)
    for i, node in enumerate(nodes):
        # Find the indices of nearby nodes within the specified radius
        indices = tree.query_ball_point(node, radius)
        if i in indices:    indices.remove(i)  # Remove the index of the current node if it's included
        if len(indices)>2:
            # Calculate distances to nearby nodes
            distances = np.linalg.norm(nodes[indices] - node, axis=1)
            # Calculate Gaussian weights
            weights = gaussian_weights(distances, sigma)
            weights /= np.sum(weights)  # Normalize weights
            # Apply weighted average to the displacement vectors of nearby nodes
            sum_deformation_close = np.sum(displacement_vectors[indices] * weights[:, np.newaxis], axis=0)
            # Apply smoothing to the displacement vector of the current node
            smoothed_vectors[i] = (1 - smoothing_factor) * displacement_vectors[i] + smoothing_factor * sum_deformation_close
        else:
            smoothed_vectors[i] = displacement_vectors[i]
    return smoothed_vectors

def add_middle_points(deformation_vectors, num_middle_points=5):

    num_vector, num_points, _ = deformation_vectors.shape
    extended_vectors = np.zeros((num_middle_points, num_points, 3))
    old_s=np.linspace(0, 1, num_vector)
    new_s=np.linspace(old_s.min(), old_s.max(), num_middle_points)
    
    for i in range(num_points):
        for j in range(3):
            extended_vectors[:, i, j]  = interp1d(old_s, deformation_vectors[:, i, j], kind="linear")(new_s)
    return extended_vectors
def save_deformation_vtp(dir,deformation_vectors,mesh_0,new_percentage):

    os.makedirs(dir, exist_ok=True)

    if len(deformation_vectors) == 0:
        return

    first_item = deformation_vectors[0]
    if hasattr(first_item, "save") and hasattr(first_item, "points"):
        for i, mesh in enumerate(deformation_vectors):
            name = "mw_{}.vtp".format(new_percentage[i])
            filename = os.path.join(dir, name)
            mesh.save(filename)
        return

    polydata = pv.PolyData(mesh_0.points,mesh_0.faces)

    # Add deformation vectors as point data
    for i in range(len(deformation_vectors)):
        displacement = np.asarray(deformation_vectors[i])
        if displacement.shape[0] != polydata.n_points:
            raise ValueError(
                f"Number of scalars ({displacement.shape[0]}) must match the number of mesh points ({polydata.n_points})."
            )

        polydata["Displacement"] = displacement
        # polydata["Jacobian"] = jacobian_matrices[i]
        name= "mw_{}.vtp".format((new_percentage[i]))
        filename = os.path.join(dir,name)
        for attempt in range(5):
            try:
                polydata.save(filename)
                break
            except OSError as error:
                if attempt == 4 or "Resource temporarily unavailable" not in str(error):
                    raise
                time.sleep(0.2)
