import pandas as pd
import torch
import sys
sys.path.append('/groups/cherkasvgrp/tsajed/ddlight')
from gpuvina import get_vina_scores_mul_gpu#, QuickVina2GPU
from glide_dock import get_glide_scores_mul_gpu
from easydict import EasyDict
import yaml
from typing import List, Tuple

def sort_by_pred_proba(
    mol_list: List[Tuple[str, str, float]]
) -> List[Tuple[str, str, float]]:
    return sorted(mol_list, key=lambda x: x[2], reverse=True)

with open('/home/tsajed/phd/ddlight/configs/params.yml', 'r') as f:
    config = EasyDict(yaml.safe_load(f))

def get_topK_mols(all_docked_mols, all_virtual_hits, config, topK=1000, dock_tolerance = 0.1):
    '''dock tolerance : what % more molecules to dock beyond topK to get real topK'''
    dock_thresh = all_docked_mols.train.cutoff
    result = []
    for key in all_docked_mols:
        mols_ids = all_docked_mols[key].mol_ids
        smiles = all_docked_mols[key].smiles
        dock_scores = all_docked_mols[key].dock_scores
        for m,s,d in (zip(mols_ids,smiles,dock_scores)):
            if d<dock_thresh:
                result.append((m,s,d))
    top_virt_hits = sort_by_pred_proba(all_virtual_hits)[0:int(1+dock_tolerance)*(max(topK-len(result),0))]
    print(len(top_virt_hits))
    
    # dock top virt hits 
    dock_mol_ids = [mol[0] for mol in top_virt_hits]
    dock_smiles_list = [mol[1] for mol in top_virt_hits]
    if config.global_params.dock_pgm =='vina':
        dock_scores, dock_mols = get_vina_scores_mul_gpu(dock_smiles_list, None, config, num_gpus=config.model_hps.num_gpus, 
                                            output_dir=f"{config.global_params.project_path}/{config.global_params.project_name}/final_docking/",
                                            dockscore_gt=None)
    elif config.global_params.dock_pgm == 'glide':
        batches = EasyDict({'train':EasyDict({'smiles':dock_smiles_list,
                                            'libID':dock_mol_ids}),
                                })
        dock_scores, _, _ = get_glide_scores_mul_gpu(batches, 0, config)
                
    if config.global_params.dock_pgm == 'vina':
        top_virt_result = [(m,s,d,mol) for m,s,d,mol in zip(dock_mol_ids,dock_smiles_list, dock_scores, dock_mols)]
    elif config.global_params.dock_pgm == 'glide':
        top_virt_result = [(m,s,d) for m,s,d in zip(dock_mol_ids,dock_smiles_list, dock_scores)]
    
    # Combine both and sort by docking score
    combined = result + top_virt_result
    combined_sorted = sorted(combined, key=lambda x: x[2])  # Lower score = better

    # Take top-K
    topK_mols = combined_sorted[:topK]

    # Convert to DataFrame
    if config.global_params.dock_pgm == 'vina':
        df_topK = pd.DataFrame(topK_mols, columns=["mol_id", "smiles", "dock_score", "mol"])
    elif config.global_params.dock_pgm == 'glide':
        df_topK = pd.DataFrame(topK_mols, columns=["mol_id", "smiles", "dock_score"])
    
    return df_topK
    
# topkdf = get_topK_mols(all_docked_mols, all_virutal_hits,config, topK=2000)
# topkdf.tail(100)

import os, pickle, concurrent.futures, itertools
from pathlib import Path

def _load_vhits_one(pkl_path):
    with open(pkl_path, "rb") as f:
        _, virtual_hits = pickle.load(f)  # [(mol_id, smiles, proba), ...]
    return virtual_hits

import os, pickle, concurrent.futures, itertools
from pathlib import Path

def _load_vhits_one(pkl_path):
    with open(pkl_path, "rb") as f:
        _all_molecules, virtual_hits = pickle.load(f)  # [(mol_id, smiles, proba), ...]
    return virtual_hits

def load_all_virtual_hits_parallel(iteration_dir, workers=None, chunksize=8, progress_every=1):
    """
    Read all allmols_virthits_*.pkl under `iteration_dir` in parallel processes
    and return a single list of (mol_id, smiles, proba). Prints running totals.
    """
    pkls = sorted(Path(iteration_dir).glob("allmols_virthits_*.pkl"))
    if not pkls:
        print("[vhits] no pickle shards found")
        return []

    if workers is None:
        workers = os.cpu_count() or 8  # processes, not threads

    print(f"number of workers: {workers}")

    total_rows = 0
    out = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        parts_iter = ex.map(_load_vhits_one, pkls, chunksize=chunksize)
        for i, shard in enumerate(parts_iter, 1):
            out.extend(shard)
            total_rows += len(shard)
            if (i % progress_every == 0) or (i == len(pkls)):
                print(f"[vhits] loaded {i}/{len(pkls)} shards, accumulated rows: {total_rows:,}")

    print(f"[vhits] done. total rows: {total_rows:,}")
    return out


def fetch_probs():
    import os

    iteration = 4
    iteration_dir = os.path.join(config.global_params.project_path, 
                                     config.global_params.project_name, f'iteration_{iteration}')
        
    topkdf = pd.read_csv(f"{iteration_dir}/final_dock_res.csv")
    vhits = load_all_virtual_hits_parallel(iteration_dir, chunksize=1)
    vh_df = pd.DataFrame(vhits, columns=["mol_id", "smiles", "proba"])
    if vh_df["mol_id"].duplicated().any():
        print("There are duplicate mol_ids in the virtual hits")


    # 3) Attach proba by key — use a map (fast) or merge (equivalent)
    proba_by_id = pd.Series(vh_df["proba"].values, index=vh_df["mol_id"])
    topkdf["proba"] = topkdf["mol_id"].map(proba_by_id)

    # 4) Quick sanity check
    matched = topkdf["proba"].notna().sum()
    print(f"[join] matched probabilities for {matched:,} / {len(topkdf):,} rows")

    # 5) Save cleanly (retain dock-score sort; no index column)
    topkdf.to_csv(f"{iteration_dir}/final_dock_proba_res.csv", index=False)


def main():
    import pickle
    import glob
    import os
    import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-it','--iteration',required=True)
    # args = parser.parse_args()

    from ALHelpers_molformer import write_final_sdf_from_df

    iteration = 4
    vina = None

    train_data_class = EasyDict(train=EasyDict(mol_ids=[], smiles=[], dock_scores=[], cutoff=-8.5))
    iteration_dir = os.path.join(config.global_params.project_path, 
                                     config.global_params.project_name, f'iteration_{iteration}')
        
    #Final DOCKING
    all_mols_virthits_files = glob.glob(f'{iteration_dir}/allmols_virthits_*.pkl')
    all_virtual_hits = []
    # for f in all_mols_virthits_files:
    #     all_virtual_hits.extend(pickle.load(open(f,'rb'))[1])
    #     print("all_virtual_hits ", len(all_virtual_hits))

    all_virtual_hits = load_all_virtual_hits_parallel(iteration_dir, chunksize=1)
    
    topkdf = get_topK_mols(all_docked_mols=train_data_class,
                           all_virtual_hits=all_virtual_hits, config = config, topK = config.global_params.topK, 
                           dock_tolerance=0.1)
    
    cols = [c for c in topkdf.columns if c != "mol"]
    topkdf.to_csv(f'{iteration_dir}/final_dock_res.csv', columns=cols)
    write_final_sdf_from_df(topkdf, f'{iteration_dir}/final_dock_mols.sdf')

    if vina is not None: vina._teardown()


if __name__ == "__main__":
    main()
