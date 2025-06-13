import numpy as np
import pandas as pd
import copy
import os


def load_full_tensor(file_path, value_col, shape=(142, 4500, 64)):
    df = pd.read_csv(file_path, header=None, sep=' ', names=['user_id', 'serv_id', 'time_stamp', value_col])
    tensor = np.zeros(shape, dtype=np.float64)
    for i in range(len(df)):
        u = df['user_id'].iloc[i]
        s = df['serv_id'].iloc[i]
        t = df['time_stamp'].iloc[i]
        val = df[value_col].iloc[i]
        tensor[u, s, t] = val
    return tensor


def training_matrix_generator(per, data):
    """
    For each time slice, randomly zeros out (1 - percentage) of non-zero entries.
    """
    data = copy.deepcopy(data)
    for k in range(data.shape[2]):
        temp = data[:, :, k]
        idx = np.flatnonzero(temp)
        retain_count = int(round(per * temp.size))
        zero_count = np.count_nonzero(temp) - retain_count
        if zero_count > 0:
            zero_indices = np.random.choice(idx, size=zero_count, replace=False)
            np.put(temp, zero_indices, 0)
        data[:, :, k] = temp
    return data


def save_sparse_versions(name, full_tensor, output_dir="outputs/sparse", steps=range(5, 55, 5)):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{name}_full.npy"), full_tensor)
    print(f"Saved: {name}_full.npy")

    for percent in steps:
        keep_ratio = percent / 100
        sparse_tensor = training_matrix_generator(keep_ratio, full_tensor)
        file_path = os.path.join(output_dir, f"{name}_{percent}.npy")
        np.save(file_path, sparse_tensor)
        actual_density = np.count_nonzero(sparse_tensor) * 100 / (142 * 4500 * 64)
        print(f"{name} {percent}% Target → Actual: {actual_density:.2f}% | Saved: {file_path}")


def generate_sparse_matrices():
    input_paths = {
        "rt": "data/rtdata.txt",
        "tp": "data/tpdata.txt"
    }

    print("Generating RT full + sparse matrices...")
    rt_tensor = load_full_tensor(input_paths["rt"], value_col='rt')
    save_sparse_versions("rtdata", rt_tensor)

    print("\nGenerating TP full + sparse matrices...")
    tp_tensor = load_full_tensor(input_paths["tp"], value_col='tp')
    save_sparse_versions("tpdata", tp_tensor)


if __name__ == "__main__":
    generate_sparse_matrices()
