from __future__ import annotations

from typing import Any, Dict

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

import pickle
from tta import TTAMethod
from utils import locate, parse_args, set_seed, load_cfg, build_model, load_full_dataset
from tabulate import tabulate
import os
import sys




def _evaluate_loader(
    model: TTAMethod,
    loader: DataLoader,
    device: torch.device | str,
    max_batches: int | None,
    corruption: str | None = None,
):
    total = 0
    correct = 0
    for batch_idx, batch in tqdm(enumerate(loader), leave=False, desc=f"Evaluating Corruption: {corruption}"):
        inputs, targets, corruption = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break
    accuracy = correct / total if total else 0.0
    return {"correct": correct, "total": total, "accuracy": accuracy}


def _safe_reset(tta: TTAMethod):
    reset = getattr(tta, "reset", None)
    if callable(reset):
        reset()


def _summarize(results: Dict[str, Dict[str, Any]]):
    totals = [metrics["total"] for metrics in results.values()]
    corrects = [metrics["correct"] for metrics in results.values()]
    if not totals:
        return {"mean_accuracy": 0.0}
    total = sum(totals)
    correct = sum(corrects)
    mean_acc = correct / total if total else 0.0
    return {"mean_accuracy": mean_acc, "num_scenarios": len(results)}


def pretty_print_results(results: Dict[str, Any]):
    # Prepare per-scenario table
    scenario_data = []
    for scenario, metrics in results.get("per_scenario", {}).items():
        accuracy = metrics.get("accuracy", 0.0)
        scenario_data.append([scenario, f"{accuracy:.4f}"])

    # Print per-scenario results
    if scenario_data:
        print("\nPer-Scenario Results:")
        print(
            tabulate(scenario_data, headers=["Scenario", "Accuracy"], tablefmt="grid")
        )

    # Print aggregate results
    aggregate = results.get("aggregate", {})
    mean_accuracy = aggregate.get("mean_accuracy", 0.0)
    num_scenarios = aggregate.get("num_scenarios", 0)

    print("\nAggregate Results:")
    aggregate_data = [
        ["Mean Accuracy", f"{mean_accuracy:.4f}"],
        ["Number of Scenarios", num_scenarios],
    ]
    print(tabulate(aggregate_data, tablefmt="grid"))


def pretty_print_comparison(all_results: Dict[str, Dict[str, Any]]):
    """
    Print comparison table with methods as columns.

    Args:
        all_results: Dict mapping method names to their results dictionaries
    """
    # Get all unique scenarios across all methods
    all_scenarios = set()
    for method_results in all_results.values():
        all_scenarios.update(method_results.get("per_scenario", {}).keys())
    all_scenarios = sorted(all_scenarios)

    # Prepare table data
    table_data = []
    headers = ["Scenario"] + list(all_results.keys())

    # Add per-scenario rows
    for scenario in all_scenarios:
        row = [scenario]
        for method in all_results.keys():
            per_scenario = all_results[method].get("per_scenario", {})
            accuracy = per_scenario.get(scenario, {}).get("accuracy", None)
            row.append(f"{accuracy:.4f}" if accuracy is not None else "N/A")
        table_data.append(row)

    # Add separator and aggregate row
    table_data.append(["---"] * len(headers))  # Separator
    aggregate_row = ["Mean Accuracy"]
    for method in all_results.keys():
        aggregate = all_results[method].get("aggregate", {})
        mean_acc = aggregate.get("mean_accuracy", None)
        aggregate_row.append(f"{mean_acc:.4f}" if mean_acc is not None else "N/A")
    table_data.append(aggregate_row)

    # Print table
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))


def evaluate_from_config(
    cfg_or_path, corruptions=None, print_results=True, baseline=None
) -> Dict[str, Any]:
    # load cfg
    cfg = load_cfg(cfg_or_path)

    # load full_dataset
    dataset_cls, dataset_args = load_full_dataset(
        cfg.dataset, {"kind": "public_test_bench"}
    )
    full_dataset = dataset_cls(**dataset_args)

    # load base model
    model = build_model(cfg)

    # load tta
    tta_cfg = cfg.get("tta")
    tta_args = parse_args(tta_cfg.get("args", {}))
    tta_method_cls = locate(cfg.tta.class_path)
    tta_method = tta_method_cls(model, **tta_args)

    eval_cfg = cfg.get("evaluation", {})
    device = torch.device(
        eval_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    tta_method.to(device)
    model.to(device)

    if corruptions is None:
        corruptions = full_dataset.get_available_corruptions()
    else:
        assert len(set(corruptions)) & len(
            set(full_dataset.get_available_corruptions())
        ) == len(set(corruptions)), (
            f"Invalid corruptions specified: {set(corruptions) - set(full_dataset.get_available_corruptions())}"
        )

    results = {}
    for corruption in tqdm(corruptions, desc=f"Evaluating Corruptions for {baseline or 'method'}", leave=False):
        scenario_dataset = full_dataset.filter_by_corruption(corruption)
        loader = torch.utils.data.DataLoader(
            scenario_dataset, batch_size=eval_cfg.get("batch_size", 128), shuffle=False
        )

        _safe_reset(tta_method)
        scenario_results = _evaluate_loader(
            tta_method, loader, device, eval_cfg.get("max_batches"), corruption=corruption
        )
        results[corruption] = scenario_results

    aggregate = _summarize(results)

    all_res = {"per_scenario": results, "aggregate": aggregate}
    if print_results:
        pretty_print_results(all_res)
    return all_res


def evaluate(all_cfgs: list | None, cache_dir=".cache/", force_recompute=True):
    os.makedirs(cache_dir, exist_ok=True)
    all_results = {}
    baselines = ["unadapted", "norm"] 
    for cfg in all_cfgs:
        if cfg not in baselines:
            baselines.append(cfg)
    for baseline in tqdm(baselines, desc="Evaluating Baselines"):
        try:
            if not force_recompute:
                all_results[baseline] = pickle.load(
                    open(f"{cache_dir}/{baseline}_results.pkl", "rb")
                )
            else:
                raise Exception("Forcing recompute")
        except Exception:
            all_results[baseline] = evaluate_from_config(
                f"configs/{baseline}.yaml", print_results=False,
                baseline=baseline
            )
            pickle.dump(
                all_results[baseline],
                open(f"{cache_dir}/{baseline}_results.pkl", "wb"),
            )
            print(f"Saved results for {baseline} to {cache_dir}/{baseline}_results.pkl")
        
    pretty_print_comparison(all_results)

    return all_results


if __name__ == "__main__":
    set_seed(42)
    all_cfgs = sys.argv[1:] if len(sys.argv) > 1 else []
    res = evaluate(
        all_cfgs, cache_dir=".cache/",
        force_recompute=True
    )
        
