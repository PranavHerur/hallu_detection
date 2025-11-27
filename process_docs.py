import json
from zensols.calamr.cli import ApplicationFactory
import random
import numpy as np
import torch
import logging

# disable logging for zensols
logging.getLogger("zensols").setLevel(logging.WARNING)

app_factory = ApplicationFactory()
resource = app_factory.get_resource()


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _get_psiloqa_docs(subset: str = "test"):
    datasets_dir = f"datasets/psiloqa/calamr_input/{subset}"
    calamr_data = json.load(open(f"{datasets_dir}/calamr_data.json"))
    final_decisions = json.load(open(f"{datasets_dir}/final_decisions.json"))
    return calamr_data, final_decisions


def _process_docs(doc: dict[str, str]):
    parsed_doc = resource.parse_documents(doc)

    doc_graph = resource.create_graph(parsed_doc)

    result = resource.align(doc_graph)

    return result


if __name__ == "__main__":
    import argparse

    from pathlib import Path
    import config

    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", type=str, default="a1")
    args = parser.parse_args()

    _set_seed(args.seed)

    output_path = Path(f"{config.RESULTS_DIR}/psiloqa/{args.version}/{args.subset}")
    output_path.mkdir(parents=True, exist_ok=True)

    calamr_data, final_decisions = _get_psiloqa_docs(subset=args.subset)
    for doc in calamr_data:
        doc_id = doc["id"]
        print(f"Processing {doc_id}")
        result = _process_docs(doc)
        result.df.to_csv(output_path / f"{doc_id}.csv", index=False)
