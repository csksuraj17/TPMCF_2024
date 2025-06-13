import numpy as np
import os
from sklearn.ensemble import IsolationForest

def detect_outliers(input_path, output_path):
    data = np.loadtxt(input_path)
    print(f"Loaded data from {input_path} with shape {data.shape}")

    model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    inlier_mask = model.fit_predict(data) == 1  # Keep only inliers

    cleaned_data = data[inlier_mask]
    np.save(output_path, cleaned_data)
    print(f"Saved cleaned data to {output_path} with shape {cleaned_data.shape}")

def find_outliers():
    input_paths = {
        "RT": "data/rtdata.txt",
        "TP": "data/tpdata.txt"
    }
    output_dir = "outputs/cleaned_full"
    os.makedirs(output_dir, exist_ok=True)

    for name, path in input_paths.items():
        output_path = os.path.join(output_dir, f"{name.lower()}data_cleaned.npy")
        detect_outliers(path, output_path)
