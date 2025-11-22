def main(align: bool = False):
    print("importing")
    from zensols.calamr.cli import ApplicationFactory

    print("starting")

    print("getting app factory")
    app_factory = ApplicationFactory()
    print("getting resource")
    resource = app_factory.get_resource()

    doc =     	{
            "id": "liu-example",
            "comment": "original 2014 Liu et al example",
            "body": "I saw Joe's dog, which was running in the garden. The dog was chasing a cat.",
            "summary": "Joe's dog was chasing a cat in the garden."
        }

    print("parsing documents")
    parsed_doc = resource.parse_documents(doc)

    if align:
        # Add token alignment
        print("adding token alignments")
        from zensols.amr.align import AmrAlignmentPopulator

        alignment_populator = AmrAlignmentPopulator(aligner='rule')
        for item in parsed_doc:
            if hasattr(item, 'amr'):
                # alignment_populator.aligner is the _RuleAligner instance
                # which is callable via __call__
                alignment_populator.aligner(item.amr)

    print("creating graph")
    doc_graph = resource.create_graph(parsed_doc)
    print(doc_graph)


    result = resource.align(doc_graph)

    # Iterate through aligned tokens
    from zensols.calamr import GraphNode, SentenceGraphAttribute, Flow, FlowDocumentGraph
    import json

    def tok_aligns(node: GraphNode) -> str:
        """Extract token alignments from a node."""
        spans = None
        if isinstance(node, SentenceGraphAttribute):
            spans = tuple(map(
                lambda t: (t.norm, t.lexspan.astuple), node.tokens))
        spans = None if spans is not None and len(spans) == 0 else spans
        return None if spans is None else json.dumps(spans)

    # Get the flow document graph from the result
    flow_doc_graph: FlowDocumentGraph = result.doc_graph.children['reversed_source']

    # Iterate through components and flows
    for cname, graph in flow_doc_graph.components_by_name.items():
        flow: Flow
        for flow in graph.flows:
            src: str = tok_aligns(flow.source)
            trg: str = tok_aligns(flow.target)
            print(f'{flow}: {src} -> {trg}')
            
    filename = 'alignments.csv' if align else 'alignments_no_align.csv'
    result.df.to_csv(filename, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--align', action='store_true')
    args = parser.parse_args()

    main(args.align)