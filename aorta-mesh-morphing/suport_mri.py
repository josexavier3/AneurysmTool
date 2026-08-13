"""4D flow DICOM reading and the geometric helpers the imaging stage uses.

`inlet_velocity_MRI_segmentation.ipynb` imports this module as `ut`. This is the
author's own file, as used for the study, supplied by him in August 2026 — not a
reconstruction. It replaces an earlier vendored subset of flow4D that stood in
for it while it was missing, and which was not what produced the published
results.

Provenance, because it is not all one thing. The DICOM reader was written by
working from flow4D and is in substantial part that code: `get_dz` is identical
to flow4D's, and `read_acquisition` differs from it in some twenty lines out of
a hundred and twenty — reading with `stop_before_pixels`, carrying the file path
rather than the decoded pixel array, and leaving the VENC detection out.
`seriesData_to_arrayData` has been reworked more substantially. flow4D is MIT
licensed, which permits this and asks only that the notice travel with the code:

    flow4D — code to process 4D flow MRI data and create inlet velocity profiles
    for numerical simulations.
    https://github.com/saitta-s/flow4D
    doi:10.5281/zenodo.7236015

    MIT License
    Copyright (c) 2022 saitta-s

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

The rest of the module — rigid registration, point selection, the `cuts_posit.txt`
reader, the mesh and interpolation helpers — is the author's own and has no
counterpart in flow4D. flow4D should be cited in the manuscript's *Software
implementation* section, which does not currently list it.

`select_points` and `select_mri_point_with_fixed_ct` open a Matplotlib window and
wait for clicks. `environment.yml` pins `matplotlib-base`, which carries no GUI
backend, so those two need an interactive backend installed to be used; nothing
else here does.
"""

import sys
import os
from collections import Counter
from tqdm import tqdm
import pydicom
from os.path import join
from itertools import groupby
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import zoom
import trimesh
import pyvista as pv
import trimesh
# import gmsh

from scipy.ndimage import map_coordinates



def get_dz(ds):
    try:
        dz = float(ds.SpacingBetweenSlices)
    except:
        dz = float(ds.SliceThickness)
    return dz


def read_acquisition(dataDir):
    
    series0 = []
    series1 = []
    series2 = []
    series3 = []
    series = []
    sNum = []
    for root, dirs, files in os.walk(dataDir):
        for file in tqdm(files, desc='Reading images', disable=len(files) == 0):
            full_path = join(root, file)
            ds = pydicom.dcmread(full_path, stop_before_pixels=True)
            # ds.decompress('gdcm')
            sNum.append(ds.SeriesNumber)
            dataTemp = dict()
            dataTemp['FileName'] = file
            dataTemp['FilePath'] = full_path
            # dataTemp['pixel_array'] = ds.pixel_array.astype(np.int16).astype(float)
            dataTemp['info'] = ds 
            series.append(dataTemp)
    counter = Counter(sNum)
    sNum = np.unique(sNum)

    if len(counter) == 4:
        for i in range(len(series)):
            if int(series[i]['info'].SeriesNumber) == sNum[0]:
                series0.append(series[i])
            elif int(series[i]['info'].SeriesNumber) == sNum[1]:
                series1.append(series[i])
            elif int(series[i]['info'].SeriesNumber) == sNum[2]:
                series2.append(series[i])
            elif int(series[i]['info'].SeriesNumber) == sNum[3]:
                series3.append(series[i])
            else:
                print('Series number not found.')
                print(series[i]['info'].SeriesNumber)
                sys.exit(0)

    elif len(counter) == 2:
        num_imgs = list(counter.values())

        if num_imgs[0] > num_imgs[1]:
            assert num_imgs[0] == 3 * num_imgs[1]
            series_count = 0
            for i in range(len(series)):
                if int(series[i]['info'].SeriesNumber) == sNum[0]:
                    if series_count < num_imgs[1]:
                        series0.append(series[i])
                        series_count += 1
                    elif series_count < 2 * num_imgs[1]:
                        series1.append(series[i])
                        series_count += 1
                    elif series_count < 3 * num_imgs[1]:
                        series2.append(series[i])
                        series_count += 1
                else:
                    series3.append(series[i])

        if num_imgs[0] < num_imgs[1]:
            assert num_imgs[1] == 3 * num_imgs[0]
            series_count = 0
            for i in range(len(series)):
                if int(series[i]['info'].SeriesNumber) == sNum[1]:
                    if series_count < num_imgs[1]:
                        series0.append(series[i])
                        series_count += 1
                    elif series_count < 2 * num_imgs[1]:
                        series1.append(series[i])
                        series_count += 1
                    elif series_count < 3 * num_imgs[1]:
                        series2.append(series[i])
                        series_count += 1
                else:
                    series3.append(series[i])

    K = []
    for k, v in groupby(series0, key=lambda x: x['info'].SliceLocation):
        K.append(k)

    vendor = ds.Manufacturer
    slices = len(set(K))
    if slices ==0: slices =1
    frames = len(series0) // slices
    rows = ds.Rows
    columns = ds.Columns
    # origin      = ds.ImagePositionPatient
    origin = [0.0, 0.0, 0.0]
    orientation = ds.ImageOrientationPatient
    position = ds.PatientPosition
    # period      = float(ds.NominalInterval) / 1000
    spacing = [float(ds.PixelSpacing[1]), float(ds.PixelSpacing[0]), get_dz(ds)]
    spacing = [s / 1000 for s in spacing]

    series0 = sorted(series0, key=lambda k: k['FileName'])
    series1 = sorted(series1, key=lambda k: k['FileName'])
    series2 = sorted(series2, key=lambda k: k['FileName'])
    series3 = sorted(series3, key=lambda k: k['FileName'])

    meta = {'vendor': vendor,
            'num_slices': slices,
            'num_frames': frames,
            'num_rows': rows,
            'num_cols': columns,
            'origin': origin,
            'orientation': orientation,
            'position': position,
            'spacing': spacing,
            'HighBit': ds.HighBit
    }

    series_data = {'series0': series0,
            'series1': series1,
            'series2': series2,
            'series3': series3
    }

    # # venc detection
    # venc = get_venc(series_data)
    # if np.mean(venc) > 80:
    #     venc = [vv * 0.01 for vv in venc]
    # meta['venc'] = venc

    return series_data, meta

def plot_velocity_magnitude_slices(magTemp, velTemp, num_slices=12, frame=0):
    """
    Plot Vx, Vy, Vz and |V| for selected number of slices from the given frame.
    
    Parameters:
        magTemp (ndarray): 4D array (slices, rows, cols, frames) with magnitude data.
        velTemp (ndarray): 5D array (slices, rows, cols, frames, components) with velocity data.
        num_slices (int): Number of slices to display.
        frame (int): Which frame to visualize.
    """
    total_slices = min(num_slices, velTemp.shape[0])
    
    fig, axes = plt.subplots(total_slices, 4, figsize=(16, 2.5 * total_slices))
    if total_slices == 1:
        axes = axes.reshape(1, 4)

    for i in range(total_slices):
        im0 = axes[i, 0].imshow(velTemp[i, :, :, frame, 0], cmap='bwr', vmin=-np.max(np.abs(velTemp)), vmax=np.max(np.abs(velTemp)))
        axes[i, 0].set_title(f"Slice {i+1} - Vx")
        plt.colorbar(im0, ax=axes[i, 0], shrink=0.6)

        im1 = axes[i, 1].imshow(velTemp[i, :, :, frame, 1], cmap='bwr', vmin=-np.max(np.abs(velTemp)), vmax=np.max(np.abs(velTemp)))
        axes[i, 1].set_title(f"Slice {i+1} - Vy")
        plt.colorbar(im1, ax=axes[i, 1], shrink=0.6)

        im2 = axes[i, 2].imshow(velTemp[i, :, :, frame, 2], cmap='bwr', vmin=-np.max(np.abs(velTemp)), vmax=np.max(np.abs(velTemp)))
        axes[i, 2].set_title(f"Slice {i+1} - Vz")
        plt.colorbar(im2, ax=axes[i, 2], shrink=0.6)

        im3 = axes[i, 3].imshow(magTemp[i, :, :, frame], cmap='gray')
        axes[i, 3].set_title(f"Slice {i+1} - |V|")
        plt.colorbar(im3, ax=axes[i, 3], shrink=0.6)

    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def seriesData_to_arrayData(seriesData, meta):
    arrayData = []
    for s in seriesData.keys():
        series = seriesData[s]
        newArr = np.zeros((meta['num_rows'], meta['num_cols'], meta['num_slices'], meta['num_frames']))
        try:
            #IPP = []
            # Optimization: Group by frame first
            series_by_frame = {}
            for item in series:
                fid = int(item['info'].TemporalPositionIdentifier)
                if fid not in series_by_frame: series_by_frame[fid] = []
                series_by_frame[fid].append(item)

            for j in range(1, meta['num_frames'] + 1):
                if j in series_by_frame:
                    frameBlock = series_by_frame[j]
                    frameBlock = sorted(frameBlock, key=lambda k: k['info'].SliceLocation)
                    for i in range(min(len(frameBlock), meta['num_slices'])):
                        ds_pixels = pydicom.dcmread(frameBlock[i]['FilePath'])
                        newArr[:, :, i, j - 1] = ds_pixels.pixel_array.astype(np.int16).astype(float)
                    #IPP.append(frameBlock[i]['IPP'])
            arrayData.append(newArr)
        except:
            series = sorted(series, key=lambda k: k['info'].SliceLocation)
            #series = sorted(series, key=lambda k: k['FileName'])
            ids = np.arange(0, meta['num_slices'] * meta['num_frames'] - meta['num_frames'], meta['num_frames'])
            for i in range(len(ids)):
                for j in range(meta['num_frames']):
                    idx = ids[i] + j
                    if idx < len(series):
                        ds_pixels = pydicom.dcmread(series[idx]['FilePath'])
                        newArr[:, :, i, j] = ds_pixels.pixel_array.astype(np.int16).astype(float)
            arrayData.append(newArr)

    return arrayData


def plot_magnitude_slices(magTemp, num_slices=5, frame=0):
    """
    Plot selected number of magnitude slices from a given frame.

    Parameters:
        magTemp (ndarray): 4D array (slices, rows, cols, frames).
        num_slices (int): Number of slices to display.
        frame (int): Frame index to display (default=0).
    """
    total_slices = min(num_slices, magTemp.shape[0])
    fig, axes = plt.subplots(1, total_slices, figsize=(3 * total_slices, 3))

    if total_slices == 1:
        axes = [axes]  # ensure iterable

    for i in range(total_slices):
        ax = axes[i]
        im = ax.imshow(magTemp[i, :, :, frame], cmap='gray')
        ax.set_title(f"Slice {i+1}")
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.7)

    plt.tight_layout()
    plt.show()
    
    #####################################################################
    
def read_folder(folder_name):
    if not os.path.exists(folder_name):
        raise ValueError(f"Folder '{folder_name}' does not exist.")

    # Initialize lists to store 2D DICOM images and metadata for each slice
    slice_images = []
    dicom_metadata = []

    # Iterate through the DICOM files in the specified folder and sort them by SliceLocation
    dicom_files = sorted([f for f in os.listdir(folder_name) if f.endswith(".dcm")],
                        key=lambda x: pydicom.dcmread(os.path.join(folder_name, x)).SliceLocation)

    for fname in dicom_files:
        dicom_data = pydicom.dcmread(os.path.join(folder_name, fname))

    # Initialize lists to store 2D DICOM images and metadata for each slice
    slice_images = []
    dicom_metadata = []

    # Iterate through the DICOM files in the specified folder and sort them by SliceLocation
    dicom_files = sorted([f for f in os.listdir(folder_name) if f.endswith(".dcm")],
                        key=lambda x: pydicom.dcmread(os.path.join(folder_name, x)).SliceLocation)

    for fname in dicom_files:
        dicom_data = pydicom.dcmread(os.path.join(folder_name, fname))
        dicom_metadata.append(dicom_data)
        dicom_hu = dicom_data.pixel_array * dicom_data.RescaleSlope + dicom_data.RescaleIntercept
        slice_images.append(dicom_hu)

    # Create a 3D NumPy array from the list of DICOM images
    img3d = np.array(slice_images)
    return img3d, dicom_metadata[0]





################################################

def rigid_transform(P, Q):
    # P: Nx3 MRI points
    # Q: Nx3 CT points
    assert P.shape == Q.shape

    # 1. Subtract centroids
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)

    # 2. Compute covariance matrix
    H = Pc.T @ Qc

    # 3. SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 4. Fix improper rotation (reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 5. Compute translation
    t = Q.mean(axis=0) - R @ P.mean(axis=0)

    return R, t

def build_transform_matrix(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def select_points(volume, label, existing_points=None, n_points=1):
    """
    Interactively select one or more points on a 3D volume (Z, Y, X) using a slice slider.

    Args:
        volume: 3D numpy array indexed as [z, y, x].
        label: Title label for the figure.
        existing_points: Optional iterable of (x, y, z) points to overlay for reference.
        n_points: Number of points to select. If 1, returns a single (x, y, z) tuple.
                  If >1, returns a list of (x, y, z) tuples.
    """
    selected_points = []

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(left=0.25, bottom=0.25)
    slice_idx = int(volume.shape[0] // 2)
    img = ax.imshow(volume[slice_idx, :, :], cmap='gray')
    ax.set_title(f"{label}: Slice {slice_idx} | Select 1 of {n_points}")

    # Add a slider for slices
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, volume.shape[0] - 1, valinit=slice_idx, valstep=1)

    # Store artists so we can clear/redraw cleanly
    existing_scatters = []
    existing_texts = []
    selected_scatters = []
    selected_texts = []

    def clear_artists(artists):
        while artists:
            artist = artists.pop()
            try:
                artist.remove()
            except Exception:
                pass

    def plot_existing_points(slice_z):
        # Clear previous existing point markers and labels
        clear_artists(existing_scatters)
        clear_artists(existing_texts)

        if existing_points is not None:
            for i, point in enumerate(existing_points):
                if point is None:
                    continue
                x, y, z = point
                if abs(z - slice_z) <= 2:
                    alpha = 1.0 if z == slice_z else 0.5
                    size = 100 if z == slice_z else 50
                    scatter = ax.scatter(x, y, c='red', s=size, alpha=alpha,
                                         marker='x', linewidth=3)
                    existing_scatters.append(scatter)
                    txt = ax.annotate(f'P{i+1}', (x, y), xytext=(5, 5),
                                      textcoords='offset points', color='red',
                                      fontweight='bold', fontsize=12)
                    existing_texts.append(txt)

    def plot_selected_points():
        # Clear and redraw selected points (always on current view)
        clear_artists(selected_scatters)
        clear_artists(selected_texts)
        for i, (x, y, z) in enumerate(selected_points):
            if z == int(slider.val):
                sc = ax.scatter(x, y, c='lime', s=150, marker='o', linewidth=2,
                                edgecolor='black', alpha=0.8)
                selected_scatters.append(sc)
                txt = ax.annotate(f'S{i+1}', (x, y), xytext=(5, 5),
                                  textcoords='offset points', color='lime',
                                  fontweight='bold', fontsize=12)
                selected_texts.append(txt)

    def update(val):
        idx = int(slider.val)
        img.set_data(volume[idx, :, :])
        ax.set_title(f"{label}: Slice {idx} | Select {min(len(selected_points)+1, n_points)} of {n_points}")
        plot_existing_points(idx)
        plot_selected_points()
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Initial draw
    plot_existing_points(slice_idx)
    plot_selected_points()

    def onclick(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        if len(selected_points) >= n_points:
            return
        x, y = int(event.xdata), int(event.ydata)
        z = int(slider.val)
        selected_points.append((x, y, z))
        plot_selected_points()
        ax.set_title(f"{label}: Slice {z} | Select {min(len(selected_points)+1, n_points)} of {n_points}")
        fig.canvas.draw_idle()

        if len(selected_points) >= n_points:
            plt.close(fig)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    # Wait until points are selected or window closed
    while len(selected_points) < n_points and plt.get_fignums():
        plt.pause(0.1)

    if len(selected_points) == 0:
        print("No point selected.")
        return None if n_points == 1 else []

    if n_points == 1:
        return selected_points[0]
    return selected_points
    
    
def select_mri_point_with_fixed_ct(ct_image, mri_volume, ct_point, n_points=1):

    mri_points = []


    fig, (ax_ct, ax_mri) = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9, wspace=0.3)

    # Fixed CT image (assumed 2D slice)
    ax_ct.imshow(ct_image[:, :], cmap='gray')
    for i in range(n_points):
        print(ct_point)
        x_ct, y_ct, z_ct = ct_point[i]
        ax_ct.set_title(f"CT: Slice {z_ct}")
        ax_ct.scatter(x_ct, y_ct, c='lime', s=150, marker='o', linewidth=2, edgecolor='black', alpha=0.9)
        ax_ct.annotate('CT Point', (x_ct, y_ct), xytext=(5, 5),
                       textcoords='offset points', color='lime',
                       fontweight='bold', fontsize=10)

    # MRI image (navigable)
    mri_slice_idx = int(mri_volume.shape[0] // 2)
    mri_img = ax_mri.imshow(mri_volume[mri_slice_idx, :, :], cmap='gray')
    ax_mri.set_title(f"MRI: Slice {mri_slice_idx} | Select 1 of {n_points}")

    # MRI slice slider
    ax_mri_slider = plt.axes([0.3, 0.1, 0.4, 0.03])
    mri_slider = Slider(ax_mri_slider, 'MRI Slice', 0, mri_volume.shape[0] - 1,
                        valinit=mri_slice_idx, valstep=1)

    selected_scatters = []
    selected_texts = []

    def clear_artists(artists):
        while artists:
            artist = artists.pop()
            try:
                artist.remove()
            except Exception:
                pass

    def plot_selected_points():
        clear_artists(selected_scatters)
        clear_artists(selected_texts)
        current_z = int(mri_slider.val)
        for i, (x, y, z) in enumerate(mri_points):
            if z == current_z:
                sc = ax_mri.scatter(x, y, c='red', s=150, marker='o', linewidth=2,
                                    edgecolor='black', alpha=0.9)
                selected_scatters.append(sc)
                txt = ax_mri.annotate(f'S{i+1}', (x, y), xytext=(5, 5),
                                      textcoords='offset points', color='red',
                                      fontweight='bold', fontsize=12)
                selected_texts.append(txt)

    def update_mri(val):
        idx = int(mri_slider.val)
        mri_img.set_data(mri_volume[idx, :, :])
        ax_mri.set_title(f"MRI: Slice {idx} | Select {min(len(mri_points)+1, n_points)} of {n_points}")
        plot_selected_points()
        fig.canvas.draw_idle()

    mri_slider.on_changed(update_mri)

    def onclick(event):
        if event.inaxes != ax_mri or event.xdata is None or event.ydata is None:
            return
        if len(mri_points) >= n_points:
            return
        x, y = int(event.xdata), int(event.ydata)
        z = int(mri_slider.val)
        mri_points.append((x, y, z))
        print(f"MRI point selected: ({x}, {y}, {z}) [{len(mri_points)}/{n_points}]")
        plot_selected_points()
        ax_mri.set_title(f"MRI: Slice {z} | Select {min(len(mri_points)+1, n_points)} of {n_points}")
        fig.canvas.draw_idle()

        if len(mri_points) >= n_points:
            plt.close(fig)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    # Wait until enough points are selected or window is closed
    while len(mri_points) < n_points and plt.get_fignums():
        plt.pause(0.1)

    if len(mri_points) == 0:
        print("MRI point selection cancelled.")
        return None if n_points == 1 else []

    if n_points == 1:
        return mri_points[0]
    return mri_points


def compute_rigid_transform_with_vectors(A, B):
    """
    Compute rigid transform mapping B->A using anchor + two vectors.
    
    Inputs:
      - A: (N, 3) or (N_s, N, 3), each triplet [p1, p2, p3] for CT (target)
      - B: (N, 3) or (N_s, N, 3), each triplet [p1, p2, p3] for MRI (source)
    
    Returns:
      - T: 4x4 homogeneous transform
    """

    A = np.asarray(A)
    B = np.asarray(B)
    if A.shape != B.shape:
        raise ValueError(f"A and B must have same shape, got {A.shape} vs {B.shape}")
    if A.shape[-2:] != (3,):  # need triplets
        raise ValueError("Each input must contain triplets of 3D points")

    # Flatten slices if necessary
    if A.ndim == 3:  # (N_s, 3, 3)
        A = A.reshape(-1, 3, 3)
        B = B.reshape(-1, 3, 3)
    elif A.ndim == 2:  # single triplet
        A = A[None, ...]
        B = B[None, ...]

    def frame_from_triplet(P):
        p1, p2, p3 = P
        v1 = p2 - p1
        v2 = p3 - p1
        # Orthonormal basis via Gram-Schmidt
        e1 = v1 / np.linalg.norm(v1)
        v2_proj = v2 - np.dot(v2, e1) * e1
        e2 = v2_proj / np.linalg.norm(v2_proj)
        e3 = np.cross(e1, e2)
        R = np.stack([e1, e2, e3], axis=1)  # 3x3 rotation matrix
        return p1, R

    # If multiple triplets, average over them
    Rs, ts = [], []
    for a_triplet, b_triplet in zip(A, B):
        p1_A, R_A = frame_from_triplet(a_triplet)
        p1_B, R_B = frame_from_triplet(b_triplet)
        R = R_A @ R_B.T
        t = p1_A - R @ p1_B
        Rs.append(R)
        ts.append(t)

    R = np.mean(Rs, axis=0)  # crude averaging
    # Project averaged R back to SO(3) with SVD
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    t = np.mean(ts, axis=0)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


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

def pv_to_trimesh(pv_mesh: pv.PolyData)-> trimesh.Trimesh:
    faces = pv_mesh.faces.reshape((-1, 4))[:, 1:]  # drop face length
    return trimesh.Trimesh(vertices=pv_mesh.points, faces=faces, process=True)



# def generate_volume_mesh(stl_file, element_size=1.0):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("volume_mesh")

    gmsh.merge(stl_file)

    # Classify surfaces
    angle = 40
    force_parametrizable_patches = True
    include_boundary = True
    gmsh.model.mesh.classifySurfaces(
        angle * (3.14159 / 180.0), include_boundary, force_parametrizable_patches
    )
    gmsh.model.mesh.createGeometry()
    gmsh.model.geo.synchronize()

    surfaces = gmsh.model.getEntities(dim=2)
    surface_loop = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfaces])
    gmsh.model.geo.addVolume([surface_loop])
    gmsh.model.geo.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", element_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", element_size)

    gmsh.model.mesh.generate(3)

    # Save to temp file
    temp_vtk_file = "__temp_mesh.vtk"
    gmsh.write(temp_vtk_file)
    gmsh.finalize()

    # Load into PyVista and return it
    volume = pv.read(temp_vtk_file)
    os.remove(temp_vtk_file)  # optional: clean up

    return volume

def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    n_points = points.shape[0]
    points_h = np.hstack([points, np.ones((n_points, 1))])  # To homogeneous coords
    transformed_h = points_h @ transform.T
    transformed = transformed_h[:, :3] / transformed_h[:, 3, np.newaxis]
    return transformed


def interpolate_scalar_field_at_points(points, field, origin_mm, spacing_mm):
    pts_voxel = (points - origin_mm) / spacing_mm
    pts_voxel = pts_voxel[:, [2, 1, 0]].T  # reorder to Z, Y, X for indexing
    values = map_coordinates(field, pts_voxel, order=1, mode='nearest')
    return values

def interpolate_velocity_at_points(points, vel_field, origin_mm, spacing_mm):
    pts_voxel = (points - origin_mm) / spacing_mm
    pts_voxel = pts_voxel[:, [2, 1, 0]].T  # reorder to Z, Y, X

    u = map_coordinates(vel_field[..., 0], pts_voxel, order=1, mode='nearest')
    v = map_coordinates(vel_field[..., 1], pts_voxel, order=1, mode='nearest')
    w = map_coordinates(vel_field[..., 2], pts_voxel, order=1, mode='nearest')
    return np.vstack((u, v, w)).T