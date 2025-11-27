#!/usr/bin/env python3
"""
Parallelized version of process_docs.py for processing multiple documents concurrently.
"""

import json
from zensols.calamr.cli import ApplicationFactory
import random
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from multiprocessing import cpu_count
from pyg_export import PyTorchGeometricExporter as PyGExport
import logging
import time

logging.getLogger("zensols").setLevel(logging.WARNING)

pyg_export = PyGExport()


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _convert_to_labels_map(final_decisions: list[dict]) -> dict[str, int]:
    return {decision["id"]: decision["final_decision"] for decision in final_decisions}


def _get_psiloqa_docs(subset: str = "test"):
    datasets_dir = f"datasets/psiloqa/calamr_input/{subset}"
    calamr_data = json.load(open(f"{datasets_dir}/calamr_data2.json"))
    final_decisions = json.load(open(f"{datasets_dir}/final_decisions.json"))
    return calamr_data, _convert_to_labels_map(final_decisions)


def _get_pubmedqa_docs(subset: str = "labeled"):
    datasets_dir = f"datasets/pubmedqa/calamr_input/{subset}"
    calamr_data = json.load(open(f"{datasets_dir}/calamr_data.json"))
    final_decisions = json.load(open(f"{datasets_dir}/final_decisions.json"))
    return calamr_data, _convert_to_labels_map(final_decisions)


def process_single_doc(doc: dict[str, str], label: int, output_path: str):
    """Process a single document. This function runs in a worker process."""
    # Get the document id
    doc_id = doc["id"]

    # Create resource per worker (important for thread safety)
    app_factory = ApplicationFactory()
    resource = app_factory.get_resource()

    # Process the document
    parsed_doc = resource.parse_documents(doc)
    doc_graph = resource.create_graph(parsed_doc)
    result = resource.align(doc_graph)

    # export to pyg
    pyg_data = pyg_export.export_alignment_graph(result)
    pyg_data.y = torch.tensor([label], dtype=torch.long)
    torch.save(pyg_data, f"{output_path}/pyg/{doc_id}.pt")

    # Save the csvresult
    result.df.to_csv(f"{output_path}/csv/{doc_id}.csv", index=False)

    return doc_id


def parallelize_docs(
    calamr_data: list[dict[str, str]],
    final_decisions: dict[str, int],
    output_path: str,
    num_workers: int = cpu_count(),
):
    start_time = time.time()
    print(f"Processing {len(calamr_data)} documents with {num_workers} workers...")

    # create csv output path
    csv_output_path = output_path / "csv"
    csv_output_path.mkdir(parents=True, exist_ok=True)

    # create pyg output path
    pyg_output_path = output_path / "pyg"
    pyg_output_path.mkdir(parents=True, exist_ok=True)

    # Process documents in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_doc = {
            executor.submit(
                process_single_doc,
                doc,
                final_decisions[doc["id"]],
                output_path=str(output_path),
            ): doc
            for doc in calamr_data
        }

        # Process completed tasks as they finish
        completed = 0
        for future in as_completed(future_to_doc):
            doc = future_to_doc[future]
            try:
                doc_id = future.result()
                completed += 1
                print(f"[{completed}/{len(calamr_data)}] Completed {doc_id}")
            except Exception as exc:
                print(
                    f"Document {doc.get('id', 'unknown')} generated an exception: {exc}"
                )

    total_time = time.time() - start_time
    print(f"Finished processing {completed}/{len(calamr_data)} documents")
    print(f"Total time: {total_time:.2f}s ({total_time / 60:.2f} minutes)")


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import config

    parser = argparse.ArgumentParser(
        description="Process documents in parallel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--subset", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", type=str, default="a1")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Number of parallel workers (default: {cpu_count()} CPU cores)",
    )
    args = parser.parse_args()

    _set_seed(args.seed)

    # create output path
    output_path = Path(f"{config.RESULTS_DIR}/pubmedqa/{args.version}/{args.subset}")
    output_path.mkdir(parents=True, exist_ok=True)

    calamr_data, final_decisions = _get_pubmedqa_docs(subset=args.subset)
    parallelize_docs(calamr_data, final_decisions, output_path, args.workers)
