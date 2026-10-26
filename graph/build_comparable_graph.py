"""Build an explainable comparable-property graph from Malaysia_Property_Processed_Dataset.csv.

Run from the project root:
    python build_comparable_graph.py

Outputs:
    outputs/comparable_property_edges.csv  -- weighted edges + component scores
    outputs/comparable_property_graph.gexf -- graph for Gephi / NetworkX
    outputs/comparable_property_graph.png -- visual sample of the graph

The dataset's numeric attributes appear preprocessed/encoded. Similarities are therefore
computed in the dataset's current feature space. The script does not interpret normalized
price/area values as Ringgit or square feet.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

DATA_PATH = PROJECT_ROOT / "dataset" / "Malaysia_Property_Processed_Dataset.csv"
OUTPUT_DIR = ROOT / "outputs"

# Graph settings: tune these after checking the resulting graph.
TOP_K_NEIGHBOURS = 8
MIN_EDGE_WEIGHT = 0.55

# Relative importance of interpretable components. Weights are normalized below.
WEIGHTS = {
    "location": 0.25,
    "price": 0.20,
    "property_type": 0.15,
    "size": 0.15,
    "rooms": 0.10,
    "amenities": 0.10,
    "furnishing_parking": 0.05,
}

AMENITY_COLUMNS = [
    "train_station_count_3km",
    "shopping_mall_count_3km",
    "supermarket_count_3km",
    "cafe_count_3km",
    "school_count_3km",
    "hospital_count_3km",
    "pharmacy_count_3km",
    "park_count_1km",
    "office_count_3km",
    "university_count_3km",
]


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing)
            + "\nCheck that DATA_PATH points to Malaysia_Property_Processed_Dataset.csv."
        )


def numeric_similarity(a: float, b: float, scale: float) -> float:
    """Return a bounded [0, 1] similarity based on normalized absolute difference."""
    if pd.isna(a) or pd.isna(b):
        return 0.0
    if scale <= 0 or not np.isfinite(scale):
        return float(a == b)
    return float(max(0.0, 1.0 - abs(float(a) - float(b)) / scale))


def categorical_similarity(a, b) -> float:
    if pd.isna(a) or pd.isna(b):
        return 0.0
    # Unknown/missing sentinel values are not treated as a meaningful match.
    if str(a) in {"-1", "-1.0", "nan"} or str(b) in {"-1", "-1.0", "nan"}:
        return 0.0
    return float(a == b)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find dataset at {DATA_PATH}. Put the CSV in the project root "
            "or update DATA_PATH near the top of this script."
        )

    df = pd.read_csv(DATA_PATH)
    needed = [
        "Property_ID", "price", "property_type", "built_up_area", "bedroom",
        "bathroom", "area", "district", "state", "furnishing", "car_parks",
        *AMENITY_COLUMNS,
    ]
    require_columns(df, needed)

    # Stable, unique graph node identifiers.
    df = df.drop_duplicates(subset=["Property_ID"]).copy().reset_index(drop=True)
    df["Property_ID"] = df["Property_ID"].astype(str)
    n = len(df)
    if n < 2:
        raise ValueError("At least two distinct properties are required to build a graph.")

    # Scales are based on robust dataset spreads. Most columns in this processed dataset
    # are normalized; robust scales keep similarity meaningful if a column is not.
    price_scale = max(float(df["price"].quantile(0.90) - df["price"].quantile(0.10)), 1e-9)
    size_scale = max(float(df["built_up_area"].quantile(0.90) - df["built_up_area"].quantile(0.10)), 1e-9)
    bedroom_scale = max(float(df["bedroom"].quantile(0.90) - df["bedroom"].quantile(0.10)), 1e-9)
    bathroom_scale = max(float(df["bathroom"].quantile(0.90) - df["bathroom"].quantile(0.10)), 1e-9)
    parking_scale = max(float(df["car_parks"].quantile(0.90) - df["car_parks"].quantile(0.10)), 1e-9)

    # Amenity columns can be counts, so scale each by its observed 90th percentile.
    amenity_scales = {
        c: max(float(df[c].quantile(0.90)), 1.0) for c in AMENITY_COLUMNS
    }
    weights_total = sum(WEIGHTS.values())
    weights = {k: v / weights_total for k, v in WEIGHTS.items()}

    # Convert columns to arrays once so pairwise scoring is vectorized and fast enough
    # for the full 3,361-row dataset.
    ids = df["Property_ID"].astype(str).to_numpy()
    col = {name: pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
           for name in ["price", "property_type", "built_up_area", "bedroom", "bathroom",
                        "area", "district", "state", "furnishing", "car_parks", *AMENITY_COLUMNS]}
    amenity_matrix = np.column_stack([col[c] for c in AMENITY_COLUMNS])
    amenity_scale_array = np.array([amenity_scales[c] for c in AMENITY_COLUMNS], dtype=float)
    directed_candidates: dict[tuple[str, str], dict] = {}

    def cat_sim_array(values: np.ndarray, target: float) -> np.ndarray:
        valid = (values != -1) & np.isfinite(values) & np.isfinite(target) & (target != -1)
        return (values == target).astype(float) * valid

    def num_sim_array(values: np.ndarray, target: float, scale: float) -> np.ndarray:
        if not np.isfinite(target):
            return np.zeros(len(values), dtype=float)
        return np.clip(1.0 - np.abs(values - target) / scale, 0.0, 1.0) * np.isfinite(values)

    for i in range(n):
        area_sim = cat_sim_array(col["area"], col["area"][i])
        district_sim = cat_sim_array(col["district"], col["district"][i])
        state_sim = cat_sim_array(col["state"], col["state"][i])
        location = 0.60 * area_sim + 0.25 * district_sim + 0.15 * state_sim

        price = num_sim_array(col["price"], col["price"][i], price_scale)
        property_type = cat_sim_array(col["property_type"], col["property_type"][i])
        size = num_sim_array(col["built_up_area"], col["built_up_area"][i], size_scale)
        bedroom = num_sim_array(col["bedroom"], col["bedroom"][i], bedroom_scale)
        bathroom = num_sim_array(col["bathroom"], col["bathroom"][i], bathroom_scale)
        rooms = 0.5 * bedroom + 0.5 * bathroom

        amenity_similarity = np.clip(
            1.0 - np.abs(amenity_matrix - amenity_matrix[i]) / amenity_scale_array,
            0.0, 1.0
        )
        amenities = np.nan_to_num(amenity_similarity, nan=0.0).mean(axis=1)
        furnishing = cat_sim_array(col["furnishing"], col["furnishing"][i])
        parking = num_sim_array(col["car_parks"], col["car_parks"][i], parking_scale)
        furnishing_parking = 0.5 * furnishing + 0.5 * parking

        scores = (weights["location"] * location
                  + weights["price"] * price
                  + weights["property_type"] * property_type
                  + weights["size"] * size
                  + weights["rooms"] * rooms
                  + weights["amenities"] * amenities
                  + weights["furnishing_parking"] * furnishing_parking)
        scores[i] = -1.0  # exclude self
        # Partial selection avoids sorting every property for every source node.
        candidate_count = min(TOP_K_NEIGHBOURS, n - 1)
        candidate_idx = np.argpartition(scores, -candidate_count)[-candidate_count:]
        candidate_idx = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]
        kept = 0
        for j in candidate_idx:
            score = float(scores[j])
            if score < MIN_EDGE_WEIGHT:
                continue
            pair = tuple(sorted((ids[i], ids[j])))
            components = {
                "sim_location": float(location[j]),
                "sim_price": float(price[j]),
                "sim_property_type": float(property_type[j]),
                "sim_size": float(size[j]),
                "sim_rooms": float(rooms[j]),
                "sim_amenities": float(amenities[j]),
                "sim_furnishing_parking": float(furnishing_parking[j]),
            }
            item = {"weight": score, **components}
            if pair not in directed_candidates or score > directed_candidates[pair]["weight"]:
                directed_candidates[pair] = item
            kept += 1
            if kept >= TOP_K_NEIGHBOURS:
                break

    # Build graph, with property attributes kept on nodes for later inspection.
    graph = nx.Graph(name="Comparable Property Graph")
    indexed = df.set_index("Property_ID", drop=False)
    for _, row in df.iterrows():
        node_attrs = {
            "property_type": str(row["property_type"]),
            "price_processed": float(row["price"]),
            "built_up_area_processed": float(row["built_up_area"]),
            "bedroom_processed": float(row["bedroom"]),
            "bathroom_processed": float(row["bathroom"]),
            "area_code": str(row["area"]),
            "district_code": str(row["district"]),
            "state_code": str(row["state"]),
        }
        if "original_address" in df.columns:
            node_attrs["address"] = str(row["original_address"])[:250]
        if "original_link" in df.columns:
            node_attrs["listing_url"] = str(row["original_link"])
        graph.add_node(str(row["Property_ID"]), **node_attrs)

    for (u, v), attrs in directed_candidates.items():
        graph.add_edge(u, v, **attrs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    edges = []
    for u, v, data in graph.edges(data=True):
        edges.append({"source_Property_ID": u, "target_Property_ID": v, **data})
    edge_df = pd.DataFrame(edges)
    if not edge_df.empty:
        edge_df = edge_df.sort_values("weight", ascending=False)
    edge_df.to_csv(OUTPUT_DIR / "comparable_property_edges.csv", index=False)
    nx.write_gexf(graph, OUTPUT_DIR / "comparable_property_graph.gexf")

    # Draw a legible sample: highest-degree nodes and their neighbours.
    if graph.number_of_edges() > 0:
        degree_order = sorted(graph.degree, key=lambda x: x[1], reverse=True)
        selected = {node for node, _ in degree_order[:45]}
        for node in list(selected):
            selected.update(list(graph.neighbors(node))[:3])
        sample = graph.subgraph(selected).copy()
        plt.figure(figsize=(15, 11))
        pos = nx.spring_layout(sample, seed=42, weight="weight", k=0.65)
        widths = [0.5 + 3.0 * sample[u][v]["weight"] for u, v in sample.edges()]
        nx.draw_networkx_edges(sample, pos, alpha=0.25, width=widths, edge_color="#64748b")
        nx.draw_networkx_nodes(sample, pos, node_size=75, node_color="#0f766e", alpha=0.9)
        # Labels are omitted for readability; IDs are available in the CSV/GEXF.
        plt.title("Comparable-Property Graph (sample of dataset)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "comparable_property_graph.png", dpi=180, bbox_inches="tight")
        plt.close()
    else:
        warnings.warn("No edges met MIN_EDGE_WEIGHT. Lower MIN_EDGE_WEIGHT or increase TOP_K_NEIGHBOURS.")

    print("Comparable-property graph created successfully.")
    print(f"Dataset properties: {n:,}")
    print(f"Graph nodes: {graph.number_of_nodes():,}")
    print(f"Graph edges: {graph.number_of_edges():,}")
    print(f"Output folder: {OUTPUT_DIR}")
    print("Note: similarity weights are a transparent heuristic; validate/tune them for your study.")


if __name__ == "__main__":
    main()
