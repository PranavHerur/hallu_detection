import os
import logging

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s[%(levelname)s]:%(name)s: %(message)s"
)
logging.getLogger("zensols.calamr").setLevel(logging.DEBUG)
logging.getLogger("zensols.amr").setLevel(logging.DEBUG)
# logging.getLogger("zensols").setLevel(logging.DEBUG)


def _delete_cache():
    print("clearing cache")
    import shutil
    from pathlib import Path

    ddir = Path("~/.cache/calamr/data").expanduser()
    if ddir.is_dir():
        shutil.rmtree(ddir)
    print("cleared cache")


def main(align: bool = False, export_pyg: bool = False, delete_cache: bool = False):
    if delete_cache:
        _delete_cache()

    print("importing")
    from zensols.calamr.cli import ApplicationFactory

    print("starting")
    if 1:
        from zensols.calamr.cli import ApplicationFactory

        harn = ApplicationFactory.create_harness(relocate=False)
        config = harn.get_config_factory().config
        # config.write()
        with open("config.yaml", "w") as f:
            config.write(writer=f)

    print("getting app factory")
    app_factory = ApplicationFactory()
    print("getting resource")
    resource = app_factory.get_resource()

    doc = {
        "id": "liu-example",
        "comment": "original 2014 Liu et al example",
        "body": "I saw Joe's dog, which was running in the garden. The dog was chasing a cat.",
        "summary": "Joe's dog was chasing a cat in the garden.",
    }

    doc = {
        "id": "test-431-truth",
        "comment": "test-431-truth",
        "body": "Abso Lutely Productions is an American film and television production company owned by actors Tim Heidecker and Eric Wareheim and producer Dave Kneebone. It is known for producing TV shows such as Tom Goes to the Mayor;  Nathan for You; The Eric Andre Show;  Tim and Eric Awesome Show, Great Job!;  and Check It Out! with Dr. Steve Brule.\nTim Heidecker's father has been featured in the company's vanity logo since 2006. Sourced from a home video with a June 28, 1991 time stamp, he says, \"Abso-lutely,\" providing inspiration for the company name. This was in response to Tim (then 15 years old) asking him to sum up his vacation in two words.",
        "summary": "Tom Goes to the Mayor.",
    }

    print("parsing documents")
    parsed_doc = tuple(resource.parse_documents([doc]))[0]
    for sent in parsed_doc.amr.sents:
        print("parsed_doc.amr", sent.graph_string)

    if align:
        # Add token alignment
        print("adding token alignments")
        from zensols.amr.align import AmrAlignmentPopulator

        alignment_populator = AmrAlignmentPopulator(aligner="rule")
        for item in parsed_doc:
            if hasattr(item, "amr"):
                # alignment_populator.aligner is the _RuleAligner instance
                # which is callable via __call__
                alignment_populator.aligner(item.amr)

    print("creating graph")
    doc_graph = resource.create_graph(parsed_doc)

    result = resource.align(doc_graph)

    if export_pyg:
        print("exporting to pyg")
        from pyg_export import PyTorchGeometricExporter as PyGExport

        pyg_export = PyGExport()
        data = pyg_export.export_alignment_graph(result)
        print(data)
        print("exported to pyg")

    # Iterate through aligned tokens
    from zensols.calamr import (
        GraphNode,
        SentenceGraphAttribute,
        Flow,
        FlowDocumentGraph,
    )
    import json

    def tok_aligns(node: GraphNode) -> str:
        """Extract token alignments from a node."""
        spans = None
        if isinstance(node, SentenceGraphAttribute):
            spans = tuple(map(lambda t: (t.norm, t.lexspan.astuple), node.tokens))
        spans = None if spans is not None and len(spans) == 0 else spans
        return None if spans is None else json.dumps(spans)

    # Get the flow document graph from the result
    flow_doc_graph: FlowDocumentGraph = result.doc_graph.children["reversed_source"]

    # Iterate through components and flows
    for cname, graph in flow_doc_graph.components_by_name.items():
        flow: Flow
        for flow in graph.flows:
            src: str = tok_aligns(flow.source)
            trg: str = tok_aligns(flow.target)
            print(f"{flow}: {src} -> {trg}")

    output_dir = f"alignments/"
    os.makedirs(output_dir, exist_ok=True)

    filename = "alignments.csv" if align else "alignments_no_align.csv"
    filename = "alignments_no_align.csv"
    result.df.to_csv(f"{output_dir}/{filename}", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--align", action="store_true")
    parser.add_argument("--delete-cache", action="store_true")
    args = parser.parse_args()

    main(args.align, args.delete_cache)
