import numpy as np
import os
from sklearn.ensemble import IsolationForest
from tqdm import tqdm


def get_outlier(x_trn, x_ind, outlier_ratio, shape):
    outlier_indicator = np.ones(shape)

    clf = IsolationForest(max_samples=len(x_trn), random_state=0, contamination=outlier_ratio)
    clf.fit(x_trn)
    y_pred = clf.predict(x_trn)

    for idx, (i, j, k) in enumerate(x_ind):
        if y_pred[idx] == -1:
            outlier_indicator[i, j, k] = 0

    return outlier_indicator


def run_outlier_detection(dataset_name, data_train, output_dir):
    """
    Runs outlier detection for a dataset and saves multiple masks for 2-10% contamination.
    """
    os.makedirs(output_dir, exist_ok=True)
    x_train, x_index = [], []

    for i in range(data_train.shape[0]):
        for j in range(data_train.shape[1]):
            for k in range(data_train.shape[2]):
                if data_train[i, j, k] > 0:
                    x_train.append(data_train[i, j, k])
                    x_index.append((i, j, k))

    x_train = np.array(x_train).reshape(-1, 1)

    for ratio in tqdm(range(2, 12, 2), desc=f"{dataset_name} Outlier Ratios"):
        outlier_ratio = ratio / 100
        outlier_mask = get_outlier(x_train, x_index, outlier_ratio, shape=data_train.shape)
        out_path = os.path.join(output_dir, f"{dataset_name}_outlier_{ratio}.npy")
        np.save(out_path, outlier_mask)
        print(f"Saved: {out_path}")


def find_outliers():
    input_paths = {
        "RT": "data/rtdata.txt",
        "TP": "data/tpdata.txt"
    }

    shape_map = {
        "RT": (142, 4500, 64),  # (users, services, time)
        "TP": (142, 4500, 64)
    }

    for name, path in input_paths.items():
        print(f"Loading and reshaping {name} dataset...")
        flat = np.loadtxt(path)
        reshaped = flat.reshape(shape_map[name])
        run_outlier_detection(name, reshaped, output_dir="outputs/outliers")


if __name__ == "__main__":
    find_outliers()
