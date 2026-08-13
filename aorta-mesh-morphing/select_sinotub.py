"""Pick the sinotubular junction by hand, for stage 0a. Supplied by the author.

`select_cut.py` uses this when a case has no `transformation_details.xlsx`. Three
points are clicked on the aortic root; the normal of their plane becomes the inlet
control plane. Press `u` to undo the last point, close the window to accept.

It restates the sign convention independently: the normal is negated whenever its z
component is positive, so the plane written to `cuts_posit.txt` always points
caudally, whichever of the two routes built it.

`read_cuts_posit` below duplicates the function of the same name in
`SUPORT_def_deformation.py` and is called by nothing here. It is left as delivered,
with the `os` import it needs added — without it the function raises NameError on
its first line.
"""

import os

import numpy as np
import pyvista as pv


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
def select_points(mesh, title='Select points'):
    picked_points = []
    point_actor = pv.Plotter(notebook=False)
    point_actor_ref = None  # To keep track of the actor for points
    plane_actor_ref = None  # To keep track of the plane actor

    def point_picker(point):
        nonlocal point_actor_ref, plane_actor_ref
        picked_points.append(point)
        
        # Update points visualization
        if point_actor_ref:
            point_actor.remove_actor(point_actor_ref)  # Remove the previous points actor
        points_polydata = pv.PolyData(np.array(picked_points))
        point_actor_ref = point_actor.add_mesh(points_polydata, color='r', point_size=10, render_points_as_spheres=True)
        
        # Handle plane
        if len(picked_points) == 3:
            # Calculate the normal vector
            vector1 = picked_points[1] - picked_points[0]
            vector2 = picked_points[2] - picked_points[0]
            normal = np.cross(vector1, vector2)
            normal = normal / np.linalg.norm(normal)  # Normalize the vector

            # Create and add the plane
            plane = pv.Plane(center=np.mean(picked_points, axis=0), direction=normal, i_size=50, j_size=50)
            if plane_actor_ref:
                point_actor.remove_actor(plane_actor_ref)  # Remove any existing plane
            plane_actor_ref = point_actor.add_mesh(plane, color='g', opacity=0.5)
        elif plane_actor_ref:
            # Remove the plane if it exists and the number of points is not 3
            point_actor.remove_actor(plane_actor_ref)
            plane_actor_ref = None

    def remove_last_point():
        nonlocal point_actor_ref, plane_actor_ref
        if picked_points:
            picked_points.pop()  # Remove the last point
            
            # Update points visualization
            if point_actor_ref:
                point_actor.remove_actor(point_actor_ref)  # Remove the previous points actor
            if picked_points:
                points_polydata = pv.PolyData(np.array(picked_points))
                point_actor_ref = point_actor.add_mesh(points_polydata, color='r', point_size=10, render_points_as_spheres=True)
            else:
                point_actor_ref = None

            # Remove the plane if the number of points is no longer 3
            if plane_actor_ref:
                point_actor.remove_actor(plane_actor_ref)
                plane_actor_ref = None

    # Add meshes and text
    point_actor.add_mesh(mesh)
    point_actor.enable_point_picking(callback=point_picker, pickable_window=True)
    point_actor.add_text(title, position='lower_edge', font_size=20, color='k')

    # Add instruction text
    point_actor.add_text("Press 'u' to remove the last point", position=(10, 10), font_size=14, color='blue')

    # Key binding for removing the last point
    point_actor.add_key_event("u", remove_last_point)  # Press 'r' to remove the last point

    point_actor.show()
    
    return np.array(picked_points)



class sinotub:

    def __init__(self,surf):
        points = select_points(surf, title=f'Select 3 points')
        self.points3=points
        vector1 = points[1] - points[0]
        vector2 = points[2] - points[0]
        normal = np.cross(vector1, vector2)
        normal = normal / np.linalg.norm(normal)  # Normalize the vector
        if normal[2]>0:
            normal=-normal
        self.plane_data=[np.mean(points, axis=0),normal]
        