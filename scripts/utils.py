import os
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc 


def calc_sparsity(cellstate):
    positive = 0
    total = len(cellstate)

    for eachcount in cellstate:
        if eachcount > 0:
            positive = positive + 1
    
    sparsity = (total - positive)/total

    print(sparsity)
    #return (sparsity)


def prepare_dataset(h5ad_path, pert_label):
    adata = sc.read(h5ad_path)
    print(adata.obs[pert_label].unique())

    sc.pp.highly_variable_genes(adata,n_top_genes=2000, subset=False, flavor="seurat_v3")
    adata.obsm["X_hvg"] = adata[:, adata.var.highly_variable].X.copy()
    print("x_hvg done")

    output_path = h5ad_path.replace('.h5ad', '_processed.h5ad')
    adata.write(output_path)


    print("all done, ready for dataloader from cell-load")


def main():
    prepare_dataset("datasets/1M/1m_train.h5ad", "cytokine")




if __name__ == "__main__":
    main()