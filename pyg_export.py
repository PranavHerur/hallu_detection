"""PyTorch Geometric export functionality for CALAMR alignment results."""

from __future__ import annotations

from pandas import DataFrame

__author__ = "Pranav Herur"

from typing import Dict, Any
from dataclasses import dataclass, field
import logging
from pathlib import Path
import torch
from torch import Tensor
import numpy as np
from zensols.calamr import (
    FlowGraphResult,
    DocumentGraphComponent,
    GraphNode,
    GraphEdge,
    ConceptGraphNode,
    AttributeGraphNode,
    SentenceGraphNode,
    DocumentGraphNode,
    RoleGraphEdge,
)

logger = logging.getLogger(__name__)

try:
    from torch_geometric.data import Data

    HAS_TORCH_GEOMETRIC = True
except ImportError:
    logger.warning(
        "PyTorch Geometric not available. Install with: pip install torch-geometric"
    )
    HAS_TORCH_GEOMETRIC = False

    # Create dummy Data class for type hints
    class Data:
        def __init__(self, **kwargs):
            pass


@dataclass
class PyTorchGeometricExporter:
    """Exports CALAMR alignment results to PyTorch Geometric format.

    This class converts CALAMR's document graphs and alignment results into
    PyTorch Geometric Data objects suitable for graph neural networks.
    """

    include_embeddings: bool = field(default=True)
    """Whether to include SBERT embeddings as node features."""

    include_metadata: bool = field(default=True)
    """Whether to include additional metadata as node/edge attributes."""

    def __post_init__(self):
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError(
                "PyTorch Geometric is required for this functionality. "
                "Install with: pip install torch-geometric"
            )

    def _get_node_features(self, node: GraphNode, component_type: str = None) -> Tensor:
        """Extract node features from a CALAMR graph node.

        :param node: the graph node to extract features from
        :param component_type: "source" or "summary" for bipartite graphs
        :return: tensor with node features
        """
        features = []

        # Node type encoding (categorical)
        if isinstance(node, ConceptGraphNode):
            node_type = 0
        elif isinstance(node, AttributeGraphNode):
            node_type = 1
        elif isinstance(node, SentenceGraphNode):
            node_type = 2
        elif isinstance(node, DocumentGraphNode):
            node_type = 3
        else:
            node_type = 4  # Unknown/other

        # Start with node type
        features.append(float(node_type))

        # Add component type (NEW!)
        if component_type == "source":
            features.append(0.0)  # Source component
        elif component_type == "summary":
            features.append(1.0)  # Summary component
        else:
            features.append(-1.0)  # Unknown/not applicable

        # Add SBERT embedding if available and requested
        if self.include_embeddings and hasattr(node, "embedding"):
            try:
                embedding = node.embedding
                if embedding is not None:
                    if isinstance(embedding, Tensor):
                        features.extend(embedding.tolist())
                    else:
                        # Convert numpy or other types to list
                        features.extend(np.array(embedding).flatten().tolist())
                else:
                    # Use zero embedding of standard size (768 for SBERT)
                    features.extend([0.0] * 768)
            except Exception as e:
                logger.warning(f"Could not extract embedding from node {node}: {e}")
                features.extend([0.0] * 768)

        # Add additional metadata if requested
        if self.include_metadata:
            # Add concept-specific features
            if isinstance(node, ConceptGraphNode):
                # Add concept confidence or other metrics if available
                features.append(1.0 if hasattr(node, "concept") else 0.0)
            else:
                features.append(0.0)

        return torch.tensor(features, dtype=torch.float)

    def _get_edge_attributes(self, edge: GraphEdge) -> Dict[str, Any]:
        """Extract edge attributes from a CALAMR graph edge.

        :param edge: the graph edge to extract attributes from
        :return: dictionary of edge attributes
        """
        attrs = {}

        # Edge type
        if isinstance(edge, RoleGraphEdge):
            attrs["edge_type"] = 0
            if hasattr(edge, "role") and edge.role:
                attrs["role"] = edge.role
        else:
            attrs["edge_type"] = 1

        # Add capacity and flow if available
        if hasattr(edge, "capacity"):
            attrs["capacity"] = (
                float(edge.capacity) if edge.capacity is not None else 0.0
            )

        if hasattr(edge, "flow"):
            attrs["flow"] = float(edge.flow) if edge.flow is not None else 0.0
            logger.info(f"Flow: {attrs['flow']}")

        return attrs

    def _component_to_pyg(
        self, component: DocumentGraphComponent, component_name: str = ""
    ) -> Data:
        """Convert a DocumentGraphComponent to PyTorch Geometric Data object.

        :param component: the document graph component to convert
        :param component_name: name/identifier for this component
        :return: PyTorch Geometric Data object
        """
        if component.graph is None or component.graph.vcount() == 0:
            # Return empty graph
            return Data(
                x=torch.empty((0, 1)),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                num_nodes=0,
                component_name=component_name,
            )

        # Extract nodes and their features
        nodes = []
        node_features = []
        node_id_map = {}  # Map original node IDs to new indices

        for i, node_id in enumerate(component.graph.vs.indices):
            # Get node from graph attribute 'ga'
            node = component.graph.vs[node_id]["ga"]
            nodes.append(node)
            node_features.append(self._get_node_features(node, component_name))
            node_id_map[node_id] = i

        # Stack node features
        if node_features:
            x = torch.stack(node_features)
        else:
            x = torch.empty((0, 1))

        # Extract edges
        edge_indices = []
        edge_attrs = []

        for edge in component.graph.es:
            source_id = edge.source
            target_id = edge.target

            # Map to new indices
            if source_id in node_id_map and target_id in node_id_map:
                edge_indices.append([node_id_map[source_id], node_id_map[target_id]])

                # Extract edge attributes using 'ga' attribute
                edge_obj = edge["ga"] if "ga" in edge.attributes() else None
                if edge_obj:
                    edge_attrs.append(self._get_edge_attributes(edge_obj))
                else:
                    edge_attrs.append({"edge_type": 1, "capacity": 0.0, "flow": 0.0})

        # Convert edges to PyG format
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Convert edge attributes to tensors
        edge_attr = None
        if edge_attrs and self.include_metadata:
            # Extract numerical edge features
            edge_features = []
            for attrs in edge_attrs:
                features = [
                    attrs.get("edge_type", 0),
                    attrs.get("capacity", 0.0),
                    attrs.get("flow", 0.0),
                ]
                edge_features.append(features)
            edge_attr = torch.tensor(edge_features, dtype=torch.float)

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(nodes),
            component_name=component_name,
        )

        return data

    def export_source_graph(self, result: FlowGraphResult) -> Data:
        """Export the source component as a PyTorch Geometric Data object.

        :param result: the flow graph result containing alignment data
        :return: PyG Data object for the source component
        """
        doc_graph = result.doc_graph
        components_by_name = doc_graph.components_by_name
        source_component = components_by_name.get("source")

        if source_component is None:
            raise ValueError("No source component found in document graph")

        data = self._component_to_pyg(source_component, "source")

        # Add graph-level metadata
        if self.include_metadata and hasattr(result, "stats"):
            stats = result.stats
            if hasattr(stats, "components") and "source" in stats.components:
                source_stats = stats.components["source"]
                data.root_flow = torch.tensor([source_stats.get("root_flow", 0.0)])
                data.reentrancies = torch.tensor([source_stats.get("reentrancies", 0)])

        return data

    def export_summary_graph(self, result: FlowGraphResult) -> Data:
        """Export the summary component as a PyTorch Geometric Data object.

        :param result: the flow graph result containing alignment data
        :return: PyG Data object for the summary component
        """
        doc_graph = result.doc_graph
        components_by_name = doc_graph.components_by_name
        summary_component = components_by_name.get("summary")

        if summary_component is None:
            raise ValueError("No summary component found in document graph")

        data = self._component_to_pyg(summary_component, "summary")

        # Add graph-level metadata
        if self.include_metadata and hasattr(result, "stats"):
            stats = result.stats
            if hasattr(stats, "components") and "summary" in stats.components:
                summary_stats = stats.components["summary"]
                data.root_flow = torch.tensor([summary_stats.get("root_flow", 0.0)])
                data.reentrancies = torch.tensor([summary_stats.get("reentrancies", 0)])

        return data

    def export_alignment_graph(self, result: FlowGraphResult) -> Data:
        graph, _ = self._export_alignment_graph(result)
        return graph

    def export_alignment_graph_with_df(
        self, result: FlowGraphResult
    ) -> tuple[Data, DataFrame]:
        graph, df = self._export_alignment_graph(result)
        return graph, df

    def _export_alignment_graph(
        self, result: FlowGraphResult
    ) -> tuple[Data, DataFrame]:
        """Export the full bipartite alignment graph as a PyTorch Geometric Data object.

        This creates a combined graph with both source and summary nodes,
        connected by alignment edges.

        :param result: the flow graph result containing alignment data
        :return: PyG Data object for the bipartite alignment graph
        """
        doc_graph = result.doc_graph

        # Get both components
        components_by_name = doc_graph.components_by_name
        source_component = components_by_name.get("source")
        summary_component = components_by_name.get("summary")

        if source_component is None or summary_component is None:
            raise ValueError("Both source and summary components required")

        # Collect all nodes
        all_nodes = []
        all_features = []
        node_id_map = {}
        current_idx = 0

        # CORRECT APPROACH: Use the dataframe to determine component membership
        # The component graphs contain duplicated nodes, but the dataframe has the correct separation

        df = result.df
        source_rows = df[(df["name"] == "source") & (df["edge_type"] == "role")]
        summary_rows = df[(df["name"] == "summary") & (df["edge_type"] == "role")]

        print(
            f"DEBUG: DataFrame shows {len(source_rows)} source rows, {len(summary_rows)} summary rows"
        )

        # Create mapping from node description to actual node objects
        # We'll use the main document graph which contains all nodes
        desc_to_node = {}
        if doc_graph.graph:
            for vertex_id in doc_graph.graph.vs.indices:
                node = doc_graph.graph.vs[vertex_id]["ga"]
                desc_to_node[str(node)] = (vertex_id, node)

        # Add source nodes - filter by 'role' edges to get unique nodes
        source_node_count = 0
        source_role_rows = source_rows[source_rows["edge_type"] == "role"]
        unique_source_descrs = set()
        unique_source_descrs.update(source_role_rows["s_descr"].unique())
        unique_source_descrs.update(source_role_rows["t_descr"].unique())

        for s_descr in unique_source_descrs:
            if s_descr in desc_to_node:
                vertex_id, node = desc_to_node[s_descr]
                all_nodes.append(("source", node))
                all_features.append(self._get_node_features(node, "source"))
                node_id_map[("source", vertex_id)] = current_idx
                current_idx += 1
                source_node_count += 1

        # Add summary nodes - filter by 'role' edges to get unique nodes
        summary_node_count = 0
        summary_role_rows = summary_rows[summary_rows["edge_type"] == "role"]
        unique_summary_descrs = set()
        unique_summary_descrs.update(summary_role_rows["s_descr"].unique())
        unique_summary_descrs.update(summary_role_rows["t_descr"].unique())

        for s_descr in unique_summary_descrs:
            if s_descr in desc_to_node:
                vertex_id, node = desc_to_node[s_descr]
                all_nodes.append(("summary", node))
                all_features.append(self._get_node_features(node, "summary"))
                node_id_map[("summary", vertex_id)] = current_idx
                current_idx += 1
                summary_node_count += 1

        # Stack features
        if all_features:
            x = torch.stack(all_features)
        else:
            x = torch.empty((0, 1))

        # Collect all edges
        edge_indices = []
        edge_attrs = []

        # Add internal edges based on dataframe
        # Process all edges from dataframe, filtering by component
        for _, row in df.iterrows():
            component_name = row["name"]
            edge_type = row["edge_type"]
            s_descr = row["s_descr"]
            t_descr = row["t_descr"]

            # Skip alignment edges (handle separately)
            if edge_type == "align":
                continue

            # Find source and target nodes in our node mapping
            s_node_info = desc_to_node.get(s_descr)
            t_node_info = desc_to_node.get(t_descr)

            if s_node_info and t_node_info:
                s_vertex_id, s_node = s_node_info
                t_vertex_id, t_node = t_node_info

                source_key = (component_name, s_vertex_id)
                target_key = (component_name, t_vertex_id)

                if source_key in node_id_map and target_key in node_id_map:
                    edge_indices.append(
                        [node_id_map[source_key], node_id_map[target_key]]
                    )
                    edge_attrs.append(
                        {
                            "edge_type": 1,  # Internal edge
                            "capacity": (
                                row.get("capacity", 0.0) if "capacity" in row else 0.0
                            ),
                            "flow": row.get("flow", 0.0) if "flow" in row else 0.0,
                            "is_alignment": 0,
                        }
                    )

        # Add alignment edges (cross-component edges)
        alignment_rows = df[df["edge_type"] == "align"]
        alignment_edges_added = 0

        for _, row in alignment_rows.iterrows():
            s_descr = row["s_descr"]
            t_descr = row["t_descr"]
            s_name = row["name"]  # Source component name

            # Find source and target nodes
            s_node_info = desc_to_node.get(s_descr)
            t_node_info = desc_to_node.get(t_descr)

            if s_node_info and t_node_info:
                s_vertex_id, s_node = s_node_info
                t_vertex_id, t_node = t_node_info

                # Alignment edges connect across components
                # Source nodes are from the source component
                # Target nodes should be from the summary component (or vice versa)
                source_key = (s_name, s_vertex_id)  # Use the component from the row
                target_key = (
                    "summary" if s_name == "source" else "source",
                    t_vertex_id,
                )

                if source_key in node_id_map and target_key in node_id_map:
                    edge_indices.append(
                        [node_id_map[source_key], node_id_map[target_key]]
                    )
                    edge_attrs.append(
                        {
                            "edge_type": 2,  # Alignment edge
                            "capacity": (
                                row.get("capacity", 0.0) if "capacity" in row else 0.0
                            ),
                            "flow": row.get("flow", 0.0) if "flow" in row else 0.0,
                            "is_alignment": 1,
                        }
                    )
                    alignment_edges_added += 1

        print(f"DEBUG: Added {alignment_edges_added} alignment edges")

        # # ACCESS THE REAL ALIGNMENT DATA from FlowGraphResult.df!
        # print("DEBUG: Accessing alignment data from FlowGraphResult.df:")
        if hasattr(result, "df"):
            df = result.df

            # Filter for alignment edges only
            alignment_df = df[df["edge_type"] == "align"]
            # print(f"DEBUG: Found {len(alignment_df)} alignment edges in DataFrame")

            if len(alignment_df) > 0:
                # print("DEBUG: First few alignment edges:")
                # for idx, row in alignment_df.head(10).iterrows():
                #     print(
                #         f"  {row['s_descr']} -> {row['t_descr']} (flow: {row['flow']}, bipartite: {row['is_bipartite']})"
                #     )

                # Store the alignment mappings for later use
                alignment_mappings = []
                for _, row in alignment_df.iterrows():
                    alignment_mappings.append(
                        {
                            "source_id": row["s_id"],
                            "target_id": row["t_id"],
                            "source_desc": row["s_descr"],
                            "target_desc": row["t_descr"],
                            "flow": row["flow"],
                            "is_bipartite": row["is_bipartite"],
                        }
                    )
                # print(
                #     f"DEBUG: Total alignment mappings found: {len(alignment_mappings)}"
                # )
            else:
                # print("DEBUG: No alignment edges found in DataFrame!")
                alignment_mappings = []
        else:
            # print("DEBUG: No DataFrame found in FlowGraphResult!")
            alignment_mappings = []

        # FIXED APPROACH: Use alignment mappings and create edges properly
        # The key insight: we need to find the node in the right component context
        if alignment_mappings:
            # print(f"DEBUG: Adding {len(alignment_mappings)} alignment edges from DataFrame")

            # Create separate mappings for source and summary nodes
            source_desc_to_idx = {}
            summary_desc_to_idx = {}

            for i, (comp_type, node) in enumerate(all_nodes):
                node_desc = str(node)
                if comp_type == "source":
                    source_desc_to_idx[node_desc] = i
                else:  # comp_type == "summary"
                    summary_desc_to_idx[node_desc] = i

            alignment_edges_added = 0
            for mapping in alignment_mappings:
                source_desc = mapping["source_desc"]
                target_desc = mapping["target_desc"]
                flow_value = mapping["flow"]

                # For each alignment, try all combinations to find cross-component matches
                # summary->source alignments
                if (
                    source_desc in summary_desc_to_idx
                    and target_desc in source_desc_to_idx
                ):
                    source_idx = summary_desc_to_idx[
                        source_desc
                    ]  # source node is in summary component
                    target_idx = source_desc_to_idx[
                        target_desc
                    ]  # target node is in source component

                    # print(f"DEBUG: Adding alignment edge: {source_desc} (summary) -> {target_desc} (source)")
                    edge_indices.append([source_idx, target_idx])
                    edge_attrs.append(
                        {
                            "edge_type": 2,  # Alignment edge type
                            "capacity": 1.0,
                            "flow": flow_value,
                            "is_alignment": 1,
                        }
                    )
                    alignment_edges_added += 1

                # source->summary alignments
                elif (
                    source_desc in source_desc_to_idx
                    and target_desc in summary_desc_to_idx
                ):
                    source_idx = source_desc_to_idx[
                        source_desc
                    ]  # source node is in source component
                    target_idx = summary_desc_to_idx[
                        target_desc
                    ]  # target node is in summary component

                    # print(f"DEBUG: Adding alignment edge: {source_desc} (source) -> {target_desc} (summary)")
                    edge_indices.append([source_idx, target_idx])
                    edge_attrs.append(
                        {
                            "edge_type": 2,  # Alignment edge type
                            "capacity": 1.0,
                            "flow": flow_value,
                            "is_alignment": 1,
                        }
                    )
                    alignment_edges_added += 1
                # else:
                #     print(f"DEBUG: Skipping same-component alignment: {source_desc} -> {target_desc}")

            # print(f"DEBUG: Successfully added {alignment_edges_added} alignment edges to PyG graph")

        # Clean up: The main alignment edges are now added from DataFrame

        # Convert edges to PyG format
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Convert edge attributes
        edge_attr = None
        if edge_attrs and self.include_metadata:
            edge_features = []
            for attrs in edge_attrs:
                features = [
                    attrs.get("edge_type", 0),
                    attrs.get("capacity", 0.0),
                    attrs.get("flow", 0.0),
                    attrs.get("is_alignment", 0),
                ]
                # print("features", features)
                edge_features.append(features)
            edge_attr = torch.tensor(edge_features, dtype=torch.float)

        # MODIFIED: Don't put bipartite labels in y - keep y for document-level predictions
        # Store component information in a separate field instead
        component_labels = []
        for comp_type, _ in all_nodes:
            component_labels.append(0 if comp_type == "source" else 1)

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(all_nodes),
            component_name="bipartite_alignment",
            component_labels=torch.tensor(
                component_labels, dtype=torch.long
            ),  # NEW: Store separately
        )

        # Add graph-level statistics
        if self.include_metadata and hasattr(result, "stats"):
            stats = result.stats
            if hasattr(stats, "agg"):
                agg_stats = stats.agg
                data.aligned_portion = torch.tensor(
                    [agg_stats.get("aligned_portion", 0.0)]
                )
                data.mean_flow = torch.tensor([agg_stats.get("mean_flow", 0.0)])
                data.tot_alignable = torch.tensor([agg_stats.get("tot_alignable", 0)])
                data.tot_aligned = torch.tensor([agg_stats.get("tot_aligned", 0)])

        return data, df

    def save_data(self, data: Data, output_path: Path):
        """Save a PyTorch Geometric Data object to disk.

        :param data: the PyG Data object to save
        :param output_path: path where to save the data
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, output_path)
        logger.info(f"Saved PyG data to: {output_path}")

    def export_all(
        self, result: FlowGraphResult, output_dir: Path, key: str = "data"
    ) -> Dict[str, Path]:
        """Export all graph types and save them to disk.

        :param result: the flow graph result to export
        :param output_dir: directory to save the exported data
        :param key: identifier for this data instance
        :return: dictionary mapping graph type to output path
        """
        output_paths = {}

        try:
            # Export source graph
            source_data = self.export_source_graph(result)
            source_path = output_dir / f"{key}_source_graph.pt"
            self.save_data(source_data, source_path)
            output_paths["source"] = source_path
        except Exception as e:
            logger.warning(f"Could not export source graph: {e}")

        try:
            # Export summary graph
            summary_data = self.export_summary_graph(result)
            summary_path = output_dir / f"{key}_summary_graph.pt"
            self.save_data(summary_data, summary_path)
            output_paths["summary"] = summary_path
        except Exception as e:
            logger.warning(f"Could not export summary graph: {e}")

        try:
            # Export alignment graph
            alignment_data = self.export_alignment_graph(result)
            alignment_path = output_dir / f"{key}_alignment_graph.pt"
            self.save_data(alignment_data, alignment_path)
            output_paths["alignment"] = alignment_path
        except Exception as e:
            logger.warning(f"Could not export alignment graph: {e}")

        return output_paths
