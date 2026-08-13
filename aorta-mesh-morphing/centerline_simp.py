"""Centreline resampling for the wall-motion tracking step, as supplied by the author.

Imported by `deformation.py` for its resampling helpers.

The optional `centerline` class uses VMTK. VMTK is not part of the pinned environment;
the class raises an actionable import error when it is unavailable instead of a late
`NameError`. The supported pipeline reads centrelines generated in SimVascular.

That is consistent with the rest of the project rather than at odds with it: the
centrelines were generated in SimVascular, as the top-level README says, and both
callers read a saved centreline when one exists. Neither ever built one here. The
class is kept because it documents what the alternative path was meant to be; the
free functions above it — `resample_points` in particular — are ordinary code.

Note that the tracking step's centreline is not stage 2's. This one is built on the
ascending segment between the two control planes and saved next to the segmentations
as `centerline.vtk`; stage 2 reads the SimVascular centreline of the whole thoracic
aorta, `<sim>/mesh/<id>_Centerlines.vtp`. They are different curves in different
files, and both are needed.
"""

import pyvista as pv
import numpy as np
import math
try:
    from vmtk import vmtkcenterlines
except ImportError:  # optional upstream tool, not part of environment.yml
    vmtkcenterlines = None
from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splev
# self.surface .ctline e .points .plane
def calculate_distances(points):
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return distances

# Function to resample points along a curve with given minimum distance
def resample_points(points_pbv, dist_min=1):
    points=np.array(points_pbv)
    # Perform spline interpolation to fit a curve through the points
    tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]])

    # Calculate total curve length
    distances = calculate_distances(points)
    total_length = np.sum(distances)

    # Number of points required to have distances of dist_min
    num_new_points = int(total_length // dist_min) + 1

    # Generate new evenly spaced parameters
    u_new = np.linspace(0, 1, num_new_points)

    # Evaluate the spline at the new parameter values to get new points
    new_points = splev(u_new, tck)
    new_points = np.vstack(new_points).T

    return new_points

def find_closest_point( Ctlines_list, point):
        distances = np.linalg.norm(Ctlines_list - point, axis=1)
        return np.argmin(distances)

def calculate_normal_vector( Ctlines, idx):
    """Calculate the normal vector at the closest centerline point."""
    # Extract the position of the closest centerline point
    closest_point = Ctlines.points[idx]

    # Assuming the centerline has a tangent, you can get the next point to find the tangent vector
    if idx < len(Ctlines.points) - 1:
        tangent_vector = Ctlines.points[idx + 1] - closest_point
    else:
        tangent_vector = closest_point - Ctlines.points[idx - 1]

    # Normalize the tangent vector
    tangent_vector /= np.linalg.norm(tangent_vector)
    return tangent_vector

# def clip_centerline(Ctlines, point1, normal_vector1,point):
#     """Clip the centerline at the specified point using the normal vector."""
#     # Define the plane for clipping
#     plane_origin = point1
#     plane_normal = normal_vector1
#     # Clip the centerline using the plane
#     clipped_c1 = Ctlines.clip(normal=plane_normal, origin=plane_origin, invert=False)
#     clipped_c2 = Ctlines.clip(normal=plane_normal, origin=plane_origin, invert=True)
    
#     if clipped_c1.n_points > clipped_c2.n_points:
#         clipped_centerline = clipped_c1
#     else:
#         clipped_centerline = clipped_c2
#     # Here you could store or visualize the clipped_centerline as needed
#     # plotter=pv.Plotter(notebook=False)
#     # plotter.add_points(point, color='red', point_size=20)
#     # plotter.add_mesh(clipped_centerline, color='red', line_width=10)
#     # plotter.add_mesh(Ctlines, color='blue', line_width=10)
#     # planess = pv.Plane(center=plane_origin, direction=plane_normal, i_size=10, j_size=10)
#     # plotter.add_mesh(planess, color='green', point_size=20)
#     # plotter.show()
#     return clipped_centerline



class centerline:


    def __init__(self, surf_read,Point1,Point2,dist_min=1):
        if vmtkcenterlines is None:
            raise ImportError(
                "VMTK is required to construct a centreline here; use the supported "
                "SimVascular centreline input or install VMTK separately"
            )
        surf_read=surf_read.clean()
        surf_read= surf_read.smooth()
        surf_read= surf_read.fill_holes(80)
        
        vmtk= vmtkcenterlines.vmtkCenterlines()
        vmtk.Surface= surf_read
        vmtk.SeedSelectorName='pointlist'
        if Point1[2]<Point2[2]:
            vmtk.SourcePoints= Point1.tolist()
            vmtk.TargetPoints= Point2.tolist()
        else:
            vmtk.SourcePoints= Point2.tolist()
            vmtk.TargetPoints= Point1.tolist()
        #correct centerline
        vmtk.Resampling = True
        vmtk.ResamplingStepLength =0.1
        vmtk.ResamplingStepLengthType = 'relative'
        vmtk.Execute()

        Ctlines=pv.PolyData(vmtk.Centerlines)
        # p = pv.Plotter()
        # p.add_mesh(surf_read, color="lightblue", opacity=0.5, show_edges=True)
        # p.add_points(Point1,color="blue", point_size=10)

        # p.add_points(Point2,color="green", point_size=10)
        # p.add_mesh(Ctlines, color="red", line_width=10)
        # p.show()

        Ctlines_list=Ctlines.points
        
        # plotter=pv.Plotter(notebook=False)
        # plotter.add_mesh(surf_read, color='blue', opacity=0.5)
        # plotter.add_mesh(Ctlines, color='blue', line_width=10)
        # plotter.add_points(Point1, color='red', point_size=20)
        # plotter.add_points(Point2, color='red', point_size=20)
        # plotter.show()
        # for point in [Point1, Point2]:
        #             # Find the closest point on the centerline
        #             closest_point_idx = find_closest_point(Ctlines_list, point)
        #             closest_point = Ctlines_list[closest_point_idx]
        #             # Calculate the normal vector at the closest centerline point
        #             normal_vector = calculate_normal_vector(Ctlines, closest_point_idx)

        #             # Clip the centerline
        #             # Ctlines=clip_centerline(Ctlines, point, normal_vector,point)
        #             # Ctlines_list=Ctlines.points
        if Point2[2]>Point1[2]:
            Ctlines_list=np.append(np.array([Point2]),Ctlines_list,axis=0)
            Ctlines_list=np.append(Ctlines_list,[Point1],axis=0)
        else:
            Ctlines_list=np.append(np.array([Point1]),Ctlines_list,axis=0)
            Ctlines_list=np.append(Ctlines_list,[Point2],axis=0)

        Ctlines_pointsnp = [Ctlines_list[0]]  
        
        for num1 in range(1, len(Ctlines_list)):
            point = Ctlines_list[num1]
            prev_point = Ctlines_pointsnp[-1]

            if not math.dist(point, prev_point) < 0.01:
                Ctlines_pointsnp.append(point)
                
        
        Ctlines_pointsnp = resample_points(Ctlines_pointsnp, dist_min)
        
        self.centerl=pv.PolyData(Ctlines_pointsnp)
        
        
        # plotter=pv.Plotter(notebook=False)
        # plotter.add_mesh(surf_read, color='blue', opacity=0.5)
        # plotter.add_mesh(self.centerl, color='blue', line_width=10)
        # plotter.add_points(Point1, color='red', point_size=20)
        # plotter.add_points(Point2, color='red', point_size=20)
        # plotter.show()
