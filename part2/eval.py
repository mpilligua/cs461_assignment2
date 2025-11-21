from __future__ import annotations

from typing import Any, Dict

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle
from utils import locate, parse_args, set_seed, load_cfg, build_model, load_full_dataset
from tabulate import tabulate
import os
import sys
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score

def calculate_metrics(outputs, predictions, ground_truth):
    accuracy = accuracy_score(ground_truth, predictions)
    balanced_accuracy = balanced_accuracy_score(ground_truth, predictions)
    f1_macro = f1_score(ground_truth, predictions, average='macro')
    roc_auc_macro = roc_auc_score(ground_truth, outputs, average='macro', multi_class='ovo')
    return {
        'accuracy': accuracy,
        'f1_score': f1_macro,
        'balanced_accuracy': balanced_accuracy,
        'roc_auc': roc_auc_macro
    }


def _evaluate_loader(
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device | str,
):
    all_preds = []
    all_labels = []
    all_outputs = []
    for batch_idx, batch in enumerate(loader):
        # Unpack batch: first n-1 elements are inputs, last element is targets
        *inputs, targets = batch
        
        # Move inputs to device (handles single or multiple inputs)
        inputs = [inp.to(device) if torch.is_tensor(inp) else inp for inp in inputs]
        targets = targets.to(device)
        
        # Pass inputs to model (unpacks if multiple inputs)
        if len(inputs) == 1:
            outputs = model(inputs[0])
        else:
            outputs = model(*inputs)
        
        preds = outputs.argmax(dim=1)
        all_outputs.extend(outputs.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(targets.cpu().tolist())
    return calculate_metrics(np.array(all_outputs), np.array(all_preds), np.array(all_labels))

def evaluate_from_config(cfg_or_path: str, print_results=True):
    set_seed(42)
    cfg = load_cfg(cfg_or_path)
    dataset_cls, dataset_args = load_full_dataset(
        cfg.get("dataset"), additional_config={"split": "train"}
    )
    dataset = dataset_cls(**dataset_args)
    model_cls, model_args = build_model(cfg)
    model = model_cls.load_weights(cfg.model.best_weight_path)

    eval_cfg = cfg.get("evaluation", {})
    model = model.to(torch.device(
        eval_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    ))
    model.eval()
    if hasattr(cfg.dataset, "collate_fn"):
        collate_fn = locate(cfg.dataset.collate_fn)
    else:
        print("No collate_fn specified in config; using default collate.")
        collate_fn = None
    loader = DataLoader(
        dataset, batch_size=eval_cfg.get("batch_size", 128), shuffle=False,
        collate_fn=collate_fn
    )
    return _evaluate_loader(
        model, loader,
        torch.device(
            eval_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        ),
    )




def evaluate(all_cfgs: list | None ,
             cache_dir=".cache/",
             force_recompute=False):
    os.makedirs(cache_dir, exist_ok=True)
    baselines = ["linear_baseline"]
    for cfg in all_cfgs:
        if cfg not in baselines:
            baselines.append(cfg)
    all_results = {}

    for baseline in baselines:
        try:
            if not force_recompute:
                all_results[baseline] = pickle.load(
                    open(f"{cache_dir}/{baseline}_results.pkl", "rb")
                )
            else:
                raise Exception("Forcing recompute")
        except Exception:
            all_results[baseline] = evaluate_from_config(
                f"configs/{baseline}.yaml"
            )
            pickle.dump(
                all_results[baseline],
                open(f"{cache_dir}/{baseline}_results.pkl", "wb"),
            )
            print(f"Saved results for {baseline} to {cache_dir}/{baseline}_results.pkl")

    print("Evaluation Results:")
    pretty_print_comparison(all_results)

    return all_results


def pretty_print_comparison(all_results: Dict[str, Any]):
    headers = ["Method"] + list(next(iter(all_results.values())).keys())
    table = []
    for method, results in all_results.items():
        row = [method] + [f"{v:.4f}" for v in results.values()]
        table.append(row)
    print(tabulate(table, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    all_cfgs = sys.argv[1:] if len(sys.argv) > 1 else []
    res = evaluate(
        all_cfgs, cache_dir=".cache/",
        force_recompute=False
    )