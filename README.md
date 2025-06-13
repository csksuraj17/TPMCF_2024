# TPMCF 2024

This repository implements a step-by-step framework for temporal QoS prediction Using
multi-source collaborative features paper published in IEEE TNSM 2024.

## Paper Link

> **Title**: _TPMCF: Temporal QoS Prediction Using
Multi-Source Collaborative Features_  
> **Link**: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10518168

## Project Structure

- `data/`: Put input datasets (RT and TP).
- `outputs/`: Folder where generated outputs will be saved.
- `scripts/`: Python scripts for each step in the pipeline.
- `main.py`: Main pipeline to execute all steps.
- Dataset Credits: https://github.com/wsdream/AMF/tree/master/data/dataset%232

## Steps in Pipeline

1. **Sparse Matrix Creation**  

   Generates sparse matrices from raw RT and TP datasets by varying sparsity levels (5%, 10%, 15%, 20%).

3. **Outlier Detection**  

   Applies Isolation Forest to identify and remove anomalous QoS values.

4. **Feature Generation**  

   Produces handcrafted features and graph-based structures from preprocessed data.

5. **Collaborative Spatial Feature Extraction Using GCMF**

   Trains a Graph Convolutional Matrix Factorization (GCMF) model to learn user-service spatial features.

7. **Temporal QoS Prediction**

   Uses a Transformer encoder to model temporal dynamics and predict future QoS values.


## How to Run

```bash
Step 1:
git clone https://github.com/csksuraj17/tpmcf_2024.git

Step 2: 
conda env create -f environment.yml
conda activate tpmcf_env

Step 3:
python main.py

```
## Citation
```bash
@ARTICLE{10518168,
  author={Kumar, Suraj and Chattopadhyay, Soumi and Adak, Chandranath},
  journal={IEEE Transactions on Network and Service Management}, 
  title={TPMCF: Temporal QoS Prediction Using Multi-Source Collaborative Features}, 
  year={2024},
  volume={21},
  number={4},
  pages={3945-3955},
  doi={10.1109/TNSM.2024.3395428}}
