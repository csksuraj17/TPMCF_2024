import numpy as np
import os

def create_sparse_matrix(input_path, output_dir, densities):
    data = np.loadtxt(input_path)
    np.random.seed(42)

    for d in densities:
        density = d / 100.0
        mask = np.random.rand(*data.shape) < density
        sparse_data = data * mask

        output_path = os.path.join(output_dir, f"sparse_{d}.npy")
        np.save(output_path, sparse_data)
        print(f"Saved sparse matrix with {d}% density to {output_path}")

def generate_sparse_matrices():
    input_files = {
        "RT": "data/rtdata.txt",
        "TP": "data/tpdata.txt"
    }
    densities = [5, 10, 15, 20]
    output_dir = "outputs/sparse_matrices"
    os.makedirs(output_dir, exist_ok=True)

    for name, path in input_files.items():
        create_sparse_matrix(path, output_dir, densities)
