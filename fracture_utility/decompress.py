"""Data decompression code.

We assume the data are organized as follows:
```
$DATA_ROOT/
├──── $CATEGORY/
│     |──── $MESH/
|     |     |──── compressed_data.npz
|     |     |──── compressed_mesh.obj
|     │     |──── $FRACTURE/
|     |     |     |──── compressed_fracture.npy
•     •     •
•     •     •
```

You can use `decompress_mesh()` to decompress all fractures of a single mesh,
    or use `decompress_category()` to decompress all fractures of all meshes in
    a category.

The provided code under `__main__` can decompress the entire Breaking Bad
    dataset which consists of three subsets `everyday`, `artiface`, `other`.
"""

import os
import time
import argparse

import numpy as np
from tqdm import tqdm
from scipy.sparse import load_npz
import igl

ALL_CATEGORY = [
    'BeerBottle', 'Bowl', 'Cup', 'DrinkingUtensil', 'Mug', 'Plate', 'Spoon',
    'Teacup', 'ToyFigure', 'WineBottle', 'Bottle', 'Cookie', 'DrinkBottle',
    'Mirror', 'PillBottle', 'Ring', 'Statue', 'Teapot', 'Vase', 'WineGlass'
]
ALL_SUBSET = ['everyday', 'artifact', 'other']


def decompress_mesh(mesh_dir_full_path, save_dir):
    """Decompress all the fractures of a mesh."""
    # Skip failed meshes
    if not os.path.isdir(mesh_dir_full_path) or \
            len(os.listdir(mesh_dir_full_path)) == 0:
        return 0
    # Read main mesh and data
    num_fracs = 0
    compressed_mesh_path = os.path.join(mesh_dir_full_path,
                                        "compressed_mesh.obj")
    compressed_data_path = os.path.join(mesh_dir_full_path,
                                        "compressed_data.npz")
    fine_vertices, fine_triangles = igl.read_triangle_mesh(
        compressed_mesh_path)
    piece_to_fine_vertices_matrix = load_npz(compressed_data_path)
    # Now, go over all fractures
    for frac_dir in os.listdir(mesh_dir_full_path):
        frac_dir_full_path = os.path.join(mesh_dir_full_path, frac_dir)
        if not os.path.isdir(frac_dir_full_path):
            continue
        # Make new directory for decompressed fracture
        frac_save_path = os.path.join(save_dir, frac_dir)
        os.makedirs(frac_save_path, exist_ok=True)
        # Load fracture data
        frac_data_path = os.path.join(frac_dir_full_path,
                                      "compressed_fracture.npy")
        piece_labels_after_impact = np.load(frac_data_path)
        # Now actually construct the meshes to write
        fine_vertex_labels_after_impact = \
            piece_to_fine_vertices_matrix @ piece_labels_after_impact
        n_pieces_after_impact = int(np.max(piece_labels_after_impact) + 1)
        for i in range(n_pieces_after_impact):
            tri_labels = \
                fine_vertex_labels_after_impact[fine_triangles[:, 0]]
            if np.any(tri_labels == i):
                vi, fi = igl.remove_unreferenced(
                    fine_vertices, fine_triangles[tri_labels == i, :])[:2]
            else:
                continue
            ui, I, J, _ = igl.remove_duplicate_vertices(vi, fi, 1e-10)
            gi = J[fi]
            # Now we write the mesh ui, gi
            write_file_name = os.path.join(frac_save_path,
                                           "piece_" + str(i) + ".obj")
            igl.write_triangle_mesh(write_file_name, ui, gi)
            num_fracs = num_fracs + 1

    return num_fracs


def decompress_category(category_dir, save_dir):
    """Decompress all shapes belonging to a category."""
    if not os.path.isdir(category_dir):
        return
    print("Processing", category_dir)
    num_fracs = 0
    t0 = time.time()
    for mesh_dir in tqdm(os.listdir(category_dir)):
        mesh_dir_full_path = os.path.join(category_dir, mesh_dir)
        mesh_save_dir = os.path.join(save_dir, mesh_dir)
        num_fracs += decompress_mesh(mesh_dir_full_path, mesh_save_dir)

    total_time = time.time() - t0
    print("Decompressed a total of", str(num_fracs), "fracture pieces in",
          round(total_time, 3), "seconds.")


def process_everyday(data_root, category):
    if not os.path.isdir(os.path.join(data_root, 'everyday_compressed')):
        print('compressed everyday subset does not exist, skipping...')
        return
    if category.lower() == 'all':
        category = ALL_CATEGORY.copy()
    else:
        category = [category]
    for cat in category:
        cat_dir = os.path.join(data_root, 'everyday_compressed', cat)
        save_dir = os.path.join(data_root, 'everyday', cat)
        decompress_category(cat_dir, save_dir)


def process_artifact(data_root):
    cat_dir = os.path.join(data_root, 'artifact_compressed')
    if not os.path.isdir(os.path.join(data_root, 'artifact_compressed')):
        print('compressed artifact subset does not exist, skipping...')
        return
    save_dir = os.path.join(data_root, 'artifact')
    decompress_category(cat_dir, save_dir)


def process_other(data_root):
    cat_dir = os.path.join(data_root, 'other_compressed')
    if not os.path.isdir(os.path.join(data_root, 'other_compressed')):
        print('compressed other subset does not exist, skipping...')
        return
    save_dir = os.path.join(data_root, 'other')
    decompress_category(cat_dir, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data decompression')
    parser.add_argument('--data_root', required=True, type=str)
    parser.add_argument(
        '--subset',
        type=str,
        required=True,
        choices=ALL_SUBSET + [
            'all',
        ],
        help='data subset')
    parser.add_argument(
        '--category',
        type=str,
        default='all',
        choices=ALL_CATEGORY + [
            'all',
        ],
        help='category in everyday subset')
    args = parser.parse_args()

    if args.subset == 'all':
        subsets = ALL_SUBSET
    else:
        subsets = [args.subset]
    for subset in subsets:
        if subset == 'everyday':
            process_everyday(args.data_root, args.category)
        elif subset == 'artifact':
            process_artifact(args.data_root)
        elif subset == 'other':
            process_other(args.data_root)
        else:
            raise NotImplementedError('Unknown subset:', subset)