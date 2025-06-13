# QoS Prediction Framework

This repository implements a step-by-step framework for temporal QoS prediction Using
multi-source collaborative features published in IEEE-TNSM 2024.

## 🔗 Paper Link

> Add your paper title and link here  
> **Title**: _TPMCF: Temporal QoS Prediction Using
Multi-Source Collaborative Features_  
> **Link**: [https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10518168]

## 📁 Project Structure

- `data/`: Contains input datasets (RT and TP).
- `scripts/`: Python scripts for each step in the pipeline.
- `utils/`: Utility functions for preprocessing and evaluation.
- `outputs/`: Folder where generated outputs will be saved.
- `main.py`: Main pipeline to execute all steps.
- `requirements.txt`: Required dependencies.

## 🧪 Steps in Pipeline

1. **Sparse Matrix Creation**  
   Vary sparsity to create input matrices from RT and TP datasets.

2. **Outlier Detection**  
   Use Isolation Forest to identify and remove outliers.

3. **Feature Generation**  
   Generate features from sparse matrices for learning.

4. **Experiment Execution**  
   Evaluate the framework for different densities (5%, 10%, 15%, 20%) with `lambda=10`.

## 🚀 How to Run

```bash
# Step 1: Clone the repo
git clone https://github.com/csksuraj17/QoS-Prediction.git
cd QoS-Prediction

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the pipeline
python main.py
