import scanpy as sc
import pandas as pd
import numpy as np
from typing import Tuple, Optional

import os
#adata = sc.read_h5ad("../datasets/competition_support_set/k562_gwps.h5")


h5_path = "datasets/1M/1m_test.h5ad"
adata = sc.read_h5ad(h5_path)
pert_label = "cytokine"

perturbation_list = adata.obs[pert_label].value_counts()      

print(perturbation_list) 

# 1. Get the list of unique perturbation names (the index of the value_counts)
perturbations = perturbation_list.index.tolist()

# 2. Define the output path
output_file = "datasets/1M/test_cytokines.txt"

# 3. Save to a text file (one perturbation per line)
with open(output_file, "w") as f:
    for p in perturbations:
        f.write(f"{p}\n")

print(f"Successfully saved {len(perturbations)} perturbation names to {output_file}")