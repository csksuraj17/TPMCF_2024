import time
import logging
import numpy as np
from scripts import create_sparse_matrix, detect_outliers, prepare_initial_features_and_graph, csfe_module, pte_module

# =======================
# Global Configuration
# =======================
DENSITY_LIST = [0.05, 0.10, 0.15, 0.20]      # Sparse matrix densities
DATASETS = ['rt', 'tp']                      # Dataset types
TIMESTAMP_INDEX = 63                         # Time step index used for prediction
NUM_USERS = 142                              # Number of users
NUM_SERVICES = 4500                          # Number of services
NUM_TIME_STEPS = 64                          # Total time steps
PREV_TIME_STEPS = 8                          # Previous time steps used for features
FEATURE_DIR = "features/"                    # Path to feature directory
ADJ_DIR = "adj_matrices/"                    # Path to adjacency matrices
DATA_DIR = "data/"                           # Path to dataset directory
OUT_DIR = "outputs/"                         # Path to output directory

# =======================
# Logging Configuration
# =======================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =======================
# Pipeline Execution
# =======================
def run_pipeline():
    for dataset in DATASETS:
        for density in DENSITY_LIST:
            logging.info(f"========== Running pipeline for {dataset.upper()} with {int(density * 100)}% density ==========")
            percent = density

            # Step 1: Sparse Matrix Creation
            logging.info("Step 1: Creating Sparse Matrices...")
            create_sparse_matrix.generate_sparse_matrices(dataset, percent)

            # Step 2: Outlier Detection
            logging.info("Step 2: Outlier Detection...")
            detect_outliers.find_outliers(dataset, percent)

            # Step 3: Initial Feature Generation and Graph Preparation
            logging.info("Step 3: Initial Feature Generation and Graph Preparation...")
            prepare_initial_features_and_graph.run_feature_generation(dataset, percent)

            # Step 4: Collaborative Spatial Feature Extraction (GCN)
            logging.info("Step 4: Collaborative Spatial Feature Extraction...")
            csfe_module.t_gcn_training(
                NUM_USERS, NUM_SERVICES, NUM_TIME_STEPS, PREV_TIME_STEPS,
                FEATURE_DIR, ADJ_DIR, DATA_DIR, OUT_DIR, dataset, percent
            )

            # Step 5: Transformer Training & Prediction
            logging.info("Step 5: Transformer Training and Prediction...")
            try:
                data_path = f"{DATA_DIR}/data_train_{dataset}_{int(percent * 100)}.npy"
                gcn_path = f"{FEATURE_DIR}/gcn_features_{dataset}_{int(percent * 100)}.npy"

                data_train = np.load(data_path)
                gcn_features = np.load(gcn_path, allow_pickle=True)

                pte_module.run_pte(gcn_features, data_train, percent, k=TIMESTAMP_INDEX)
              
            except Exception as e:
                logging.error(f"Transformer step failed for {dataset.upper()} with {int(percent * 100)}% density: {e}")
                continue

            logging.info(f"Completed {dataset.upper()} with {int(density * 100)}% density.\n")

# =======================
# Main Execution
# =======================
if __name__ == "__main__":
    start_time = time.time()
    run_pipeline()
    logging.info("Total pipeline execution time: {:.2f} seconds".format(time.time() - start_time))
