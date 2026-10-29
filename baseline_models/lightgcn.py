import os
import random
from pathlib import Path
import time
import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Official benchmark: one search profile per row (frozen_queries_set.csv).
BENCHMARK_PATH = str(PROJECT_ROOT / "frozen_queries_set.csv")

# Current deduplicated catalogue. LightGCN only needs the property ID set
# (property embeddings are learned), so both paths point to the same file;
# this keeps the ID set identical to the one used by CBR and CPG.
PROPERTY_PATH = str(PROJECT_ROOT / "dataset" / "Malaysia_Property_Cleaned_Raw.csv")
RAW_PROPERTY_PATH = PROPERTY_PATH

OUTPUT_DIR = str(SCRIPT_DIR / "lightgcn_outputs")

# Hard-constraint fields of a search profile used as query features.
QUERY_NUMERIC_COLUMNS = [
    "max_price",
    "min_bedroom",
    "min_bathroom",
    "parking_required",
    "furnished_only",
    "tenure_freehold",
    "tenure_leasehold",
]
FURNISHED_VALUES = {"partially furnished", "fully furnished"}

# Candidate-sufficiency rule (benchmark/apply_eligibility_rule.py).
# Only queries with benchmark_role == "ranking" (>= 5 eligible properties)
# enter ranking metrics; empty and sparse queries are excluded.
ELIGIBILITY_MANIFEST_PATH = str(PROJECT_ROOT / "benchmark_eligibility_manifest.csv")
RANKING_QUERY_IDS = set()


def load_ranking_query_ids():
    if not os.path.exists(ELIGIBILITY_MANIFEST_PATH):
        raise FileNotFoundError(
            "benchmark_eligibility_manifest.csv not found. Run:\n"
            "    python benchmark/apply_eligibility_rule.py"
        )
    manifest = pd.read_csv(ELIGIBILITY_MANIFEST_PATH)
    return set(
        manifest.loc[
            manifest["benchmark_role"] == "ranking", "query_id"
        ].astype(str)
    )

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "lightgcn_model.pt"
)

RESULT_PATH = os.path.join(
    OUTPUT_DIR,
    "lightgcn_results.csv"
)

RECOMMENDATION_PATH = os.path.join(
    OUTPUT_DIR,
    "Q001_recommendations.csv"
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

TRAIN_SIZE = 0.70
VAL_SIZE = 0.10
TEST_SIZE = 0.20

SEEDS = [42, 43, 44]

EMBEDDING_DIM = 64
N_LAYERS = 3

LEARNING_RATE = 0.001
REG_WEIGHT = 1e-4

EPOCHS = 100
PATIENCE = 15

NEGATIVES_PER_POSITIVE = 1

TOP_K_VALUES = [5, 10]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DEVICE
# ============================================================

print(f"Device: {DEVICE}")


# ============================================================
# BENCHMARK SCHEMA (frozen_queries_set.csv)
# ============================================================

def standardise_benchmark(
    queries,
    catalogue
):
    """Map frozen_queries_set.csv columns to the fields this model uses.

    preferred_area -> district (hard constraint), max_budget_rm -> max_price,
    min_bedrooms/min_bathrooms -> min_bedroom/min_bathroom, plus parking,
    furnishing and tenure requirements. The queries file has no state column,
    so state is derived from the catalogue (majority state of the district);
    it is only a query feature, never a relevance filter.
    """

    required = [
        "query_id", "preferred_area", "max_budget_rm", "property_type",
        "min_bedrooms", "min_bathrooms", "parking_required",
        "furnished_only", "tenure",
    ]
    missing = [c for c in required if c not in queries.columns]
    if missing:
        raise ValueError(
            "frozen_queries_set.csv is missing columns: " + ", ".join(missing)
        )

    district_state = (
        catalogue.assign(
            _d=catalogue["district"].astype(str).str.strip().str.lower()
        )
        .groupby("_d")["state"]
        .agg(lambda s: s.value_counts().index[0])
    )

    out = pd.DataFrame({
        "query_id": queries["query_id"].astype(str).str.strip(),
        "district": queries["preferred_area"].astype(str).str.strip().str.lower(),
        "property_type": queries["property_type"].astype(str).str.strip(),
        "max_price": pd.to_numeric(queries["max_budget_rm"], errors="coerce"),
        "min_bedroom": pd.to_numeric(queries["min_bedrooms"], errors="coerce"),
        "min_bathroom": pd.to_numeric(queries["min_bathrooms"], errors="coerce"),
        "parking_required": pd.to_numeric(
            queries["parking_required"], errors="coerce").fillna(0).astype(int),
        "furnished_only": pd.to_numeric(
            queries["furnished_only"], errors="coerce").fillna(0).astype(int),
        "tenure": queries["tenure"].astype(str).str.strip(),
    })
    out["state"] = out["district"].map(district_state).fillna("unknown")
    tenure_key = out["tenure"].str.lower()
    out["tenure_freehold"] = (tenure_key == "freehold").astype(int)
    out["tenure_leasehold"] = (tenure_key == "leasehold").astype(int)

    unknown = sorted(set(out.loc[out["state"] == "unknown", "district"]))
    if unknown:
        print(f"WARNING: districts not in catalogue: {unknown}")
    return out


# ============================================================
# DATA LOADING
# ============================================================

def load_data():

    print("\n===== LOADING DATA =====")

    raw_properties = pd.read_csv(
        RAW_PROPERTY_PATH
    )

    properties = raw_properties.copy()

    benchmark = standardise_benchmark(
        pd.read_csv(BENCHMARK_PATH),
        raw_properties
    )

    print(
        f"Benchmark queries : {len(benchmark)}"
    )

    print(
        f"Processed properties : {len(properties)}"
    )

    print(
        f"Raw properties : {len(raw_properties)}"
    )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "price",
        "built_up_area",
        "land_area",
        "bedroom",
        "bathroom",
        "car_parks",
        "latitude",
        "longitude",
        "population_density",
    ]

    for df in [
        properties,
        raw_properties,
    ]:

        for column in numeric_columns:

            if column in df.columns:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

    # --------------------------------------------------------
    # Normalize string columns
    # --------------------------------------------------------

    string_columns = [
        "property_type",
        "furnishing",
        "tenure",
        "land_title",
        "lot_type",
        "title_type",
        "unit_type",
        "facing_direction",
        "occupancy",
        "area",
        "district",
        "state",
    ]

    for df in [
        benchmark,
        properties,
        raw_properties,
    ]:

        for column in string_columns:

            if column in df.columns:

                df[column] = (
                    df[column]
                    .astype(str)
                    .str.strip()
                )

    # --------------------------------------------------------
    # Property ID
    # --------------------------------------------------------

    for df in [
        properties,
        raw_properties,
    ]:

        df["property_id"] = (
            pd.to_numeric(
                df["Property_ID"],
                errors="coerce"
            )
        )

    benchmark["query_id"] = (
        benchmark["query_id"]
        .astype(str)
        .str.strip()
    )

    print("\n===== DATASET VALUE CHECK =====")

    print(
        "Unique benchmark queries:",
        benchmark["query_id"].nunique()
    )

    print(
        "Unique processed properties:",
        properties["property_id"].nunique()
    )

    print(
        "Unique raw properties:",
        raw_properties["property_id"].nunique()
    )

    return (
        benchmark,
        properties,
        raw_properties
    )


# ============================================================
# HARD-CONSTRAINT RELEVANCE
# ============================================================

def get_relevant_properties(
    query,
    raw_properties
):

    candidates = raw_properties.copy()

    # Price
    candidates = candidates[
        candidates["price"] <= query["max_price"]
    ]

    # District (preferred_area is a hard constraint in the benchmark)
    candidates = candidates[
        candidates["district"].astype(str).str.strip().str.lower()
        == str(query["district"]).strip().lower()
    ]

    # Property type
    candidates = candidates[
        candidates["property_type"].astype(str).str.strip().str.lower()
        == str(query["property_type"]).strip().lower()
    ]

    # Bedrooms / bathrooms
    candidates = candidates[
        candidates["bedroom"] >= query["min_bedroom"]
    ]
    candidates = candidates[
        candidates["bathroom"] >= query["min_bathroom"]
    ]

    # Parking required: at least one parking space
    if int(query["parking_required"]) == 1:
        candidates = candidates[
            candidates["car_parks"] >= 1
        ]

    # Furnished only: partially or fully furnished
    if int(query["furnished_only"]) == 1:
        candidates = candidates[
            candidates["furnishing"].astype(str).str.strip().str.lower()
            .isin(FURNISHED_VALUES)
        ]

    # Tenure (skipped when "No preference")
    if str(query["tenure"]).strip().lower() in {"freehold", "leasehold"}:
        candidates = candidates[
            candidates["tenure"].astype(str).str.strip().str.lower()
            == str(query["tenure"]).strip().lower()
        ]

    return candidates


# ============================================================
# BUILD ALL QUERY-PROPERTY INTERACTIONS
# ============================================================

def build_interactions(
    benchmark,
    raw_properties
):

    print("\n===== BUILDING QUERY-PROPERTY INTERACTIONS =====")

    interactions = []

    query_positive_counts = []

    for _, query in benchmark.iterrows():

        relevant = get_relevant_properties(
            query,
            raw_properties
        )

        query_id = str(query["query_id"])

        query_positive_counts.append(
            len(relevant)
        )

        for property_id in relevant["property_id"]:

            interactions.append(
                (
                    query_id,
                    int(property_id)
                )
            )

    interactions = pd.DataFrame(
        interactions,
        columns=[
            "query_id",
            "property_id"
        ]
    )

    print(
        f"Total positive interactions: "
        f"{len(interactions)}"
    )

    print(
        f"Queries with >= 1 relevant property: "
        f"{sum(x > 0 for x in query_positive_counts)}"
    )

    print(
        f"Queries with 0 relevant properties: "
        f"{sum(x == 0 for x in query_positive_counts)}"
    )

    if len(query_positive_counts) > 0:

        print(
            f"Average relevant properties/query: "
            f"{np.mean(query_positive_counts):.2f}"
        )

        print(
            f"Minimum relevant properties: "
            f"{np.min(query_positive_counts)}"
        )

        print(
            f"Maximum relevant properties: "
            f"{np.max(query_positive_counts)}"
        )

    return interactions


# ============================================================
# SCENARIO-LEVEL SPLIT
# ============================================================

def split_scenarios(
    benchmark,
    seed
):

    query_ids = (
        benchmark["query_id"]
        .astype(str)
        .unique()
        .tolist()
    )

    rng = np.random.RandomState(seed)

    rng.shuffle(query_ids)

    n_queries = len(query_ids)

    n_train = int(
        n_queries * TRAIN_SIZE
    )

    n_val = int(
        n_queries * VAL_SIZE
    )

    train_ids = query_ids[
        :n_train
    ]

    val_ids = query_ids[
        n_train:n_train + n_val
    ]

    test_ids = query_ids[
        n_train + n_val:
    ]

    print("\n===== SCENARIO SPLIT =====")

    print(
        f"Train scenarios: {len(train_ids)}"
    )

    print(
        f"Validation scenarios: {len(val_ids)}"
    )

    print(
        f"Test scenarios: {len(test_ids)}"
    )

    return (
        train_ids,
        val_ids,
        test_ids
    )


# ============================================================
# SPLIT INTERACTIONS BY SCENARIO
# ============================================================

def split_interactions_by_scenario(
    interactions,
    train_ids,
    val_ids,
    test_ids
):

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    train_interactions = interactions[
        interactions["query_id"].isin(train_set)
    ].copy()

    val_interactions = interactions[
        interactions["query_id"].isin(val_set)
    ].copy()

    test_interactions = interactions[
        interactions["query_id"].isin(test_set)
    ].copy()

    print("\n===== INTERACTION SPLIT =====")

    print(
        f"Train interactions: "
        f"{len(train_interactions)}"
    )

    print(
        f"Validation interactions: "
        f"{len(val_interactions)}"
    )

    print(
        f"Test interactions: "
        f"{len(test_interactions)}"
    )

    return (
        train_interactions,
        val_interactions,
        test_interactions
    )


# ============================================================
# POSITIVE DICTIONARY
# ============================================================

def build_positive_dict(
    interactions
):

    positive_dict = {}

    for _, row in interactions.iterrows():

        query_id = str(
            row["query_id"]
        )

        property_id = int(
            row["property_id"]
        )

        if query_id not in positive_dict:

            positive_dict[query_id] = set()

        positive_dict[
            query_id
        ].add(property_id)

    return positive_dict


# ============================================================
# QUERY FEATURE PROCESSOR
# ============================================================

class QueryFeatureProcessor:

    def __init__(self):

        self.state_mapping = {}
        self.district_mapping = {}
        self.property_type_mapping = {}

        self.numeric_stats = {}

    # --------------------------------------------------------
    # Fit ONLY on training scenarios
    # --------------------------------------------------------

    def fit(
        self,
        train_benchmark
    ):

        self.state_mapping = (
            self._build_mapping(
                train_benchmark["state"]
            )
        )

        self.district_mapping = (
            self._build_mapping(
                train_benchmark["district"]
            )
        )

        self.property_type_mapping = (
            self._build_mapping(
                train_benchmark["property_type"]
            )
        )

        for column in QUERY_NUMERIC_COLUMNS:

            values = pd.to_numeric(
                train_benchmark[column],
                errors="coerce"
            ).fillna(0).values

            self.numeric_stats[column] = {
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    # --------------------------------------------------------
    # Category mapping
    # --------------------------------------------------------

    @staticmethod
    def _build_mapping(
        series
    ):

        values = sorted(
            series
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        # 0 = unknown
        return {
            value: index + 1
            for index, value in enumerate(values)
        }

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    def transform(
        self,
        query
    ):

        state = str(
            query["state"]
        ).strip()

        district = str(
            query["district"]
        ).strip()

        property_type = str(
            query["property_type"]
        ).strip()

        state_id = self.state_mapping.get(
            state,
            0
        )

        district_id = self.district_mapping.get(
            district,
            0
        )

        property_type_id = (
            self.property_type_mapping.get(
                property_type,
                0
            )
        )

        numeric_values = []

        for column in QUERY_NUMERIC_COLUMNS:

            value = float(
                pd.to_numeric(
                    query[column],
                    errors="coerce"
                )
            )

            stats = self.numeric_stats[column]

            minimum = stats["min"]
            maximum = stats["max"]

            if maximum > minimum:

                value = (
                    value - minimum
                ) / (
                    maximum - minimum
                )

            else:

                value = 0.0

            value = np.clip(
                value,
                0.0,
                1.0
            )

            numeric_values.append(
                value
            )

        return {
            "state": state_id,
            "district": district_id,
            "property_type": property_type_id,
            "numeric": numeric_values,
        }


# ============================================================
# QUERY ENCODER
# ============================================================

class QueryEncoder(nn.Module):

    def __init__(
        self,
        num_states,
        num_districts,
        num_property_types,
        embedding_dim
    ):

        super().__init__()

        categorical_dim = 16

        self.state_embedding = nn.Embedding(
            num_states,
            categorical_dim
        )

        self.district_embedding = nn.Embedding(
            num_districts,
            categorical_dim
        )

        self.property_type_embedding = nn.Embedding(
            num_property_types,
            categorical_dim
        )

        self.numeric_network = nn.Sequential(
            nn.Linear(len(QUERY_NUMERIC_COLUMNS), 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )

        self.output_network = nn.Sequential(
            nn.Linear(
                categorical_dim * 3 + 16,
                embedding_dim
            ),
            nn.ReLU(),
            nn.Linear(
                embedding_dim,
                embedding_dim
            )
        )

    def forward(
        self,
        state,
        district,
        property_type,
        numeric
    ):

        state_emb = self.state_embedding(
            state
        )

        district_emb = self.district_embedding(
            district
        )

        property_type_emb = (
            self.property_type_embedding(
                property_type
            )
        )

        numeric_emb = (
            self.numeric_network(
                numeric
            )
        )

        combined = torch.cat(
            [
                state_emb,
                district_emb,
                property_type_emb,
                numeric_emb,
            ],
            dim=1
        )

        return self.output_network(
            combined
        )


# ============================================================
# TRUE LIGHTGCN
# ============================================================

class LightGCN(nn.Module):

    def __init__(
        self,
        num_train_queries,
        num_properties,
        query_encoder,
        embedding_dim=64,
        n_layers=3
    ):

        super().__init__()

        self.num_train_queries = (
            num_train_queries
        )

        self.num_properties = (
            num_properties
        )

        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

        self.query_encoder = query_encoder

        # ----------------------------------------------------
        # Learnable property embeddings
        # ----------------------------------------------------

        self.property_embedding = nn.Embedding(
            num_properties,
            embedding_dim
        )

        nn.init.normal_(
            self.property_embedding.weight,
            std=0.1
        )

    # --------------------------------------------------------
    # TRUE LIGHTGCN PROPAGATION
    # --------------------------------------------------------

    def propagate(
        self,
        adjacency,
        query_embeddings
    ):

        property_embeddings = (
            self.property_embedding.weight
        )

        # Initial node embeddings
        embeddings = torch.cat(
            [
                query_embeddings,
                property_embeddings,
            ],
            dim=0
        )

        # Layer 0
        all_embeddings = [
            embeddings
        ]

        # ----------------------------------------------------
        # LightGCN message passing
        # ----------------------------------------------------

        for layer in range(
            self.n_layers
        ):

            embeddings = torch.sparse.mm(
                adjacency,
                embeddings
            )

            all_embeddings.append(
                embeddings
            )

        # ----------------------------------------------------
        # Layer combination
        # ----------------------------------------------------

        final_embeddings = torch.stack(
            all_embeddings,
            dim=1
        ).mean(
            dim=1
        )

        query_final = (
            final_embeddings[
                :self.num_train_queries
            ]
        )

        property_final = (
            final_embeddings[
                self.num_train_queries:
            ]
        )

        return (
            query_final,
            property_final
        )


# ============================================================
# CREATE PROPERTY MAPPING
# ============================================================

def create_property_mappings(
    properties
):

    property_ids = (
        properties["property_id"]
        .astype(int)
        .tolist()
    )

    property_to_idx = {
        property_id: idx
        for idx, property_id
        in enumerate(property_ids)
    }

    idx_to_property = {
        idx: property_id
        for property_id, idx
        in property_to_idx.items()
    }

    return (
        property_to_idx,
        idx_to_property
    )


# ============================================================
# QUERY FEATURES TO TENSORS
# ============================================================

def query_to_features(
    query,
    processor
):

    features = processor.transform(
        query
    )

    state = torch.tensor(
        [features["state"]],
        dtype=torch.long,
        device=DEVICE
    )

    district = torch.tensor(
        [features["district"]],
        dtype=torch.long,
        device=DEVICE
    )

    property_type = torch.tensor(
        [features["property_type"]],
        dtype=torch.long,
        device=DEVICE
    )

    numeric = torch.tensor(
        [features["numeric"]],
        dtype=torch.float32,
        device=DEVICE
    )

    return (
        state,
        district,
        property_type,
        numeric
    )


# ============================================================
# BATCH QUERY FEATURES
# ============================================================

def queries_to_batch_features(
    benchmark,
    query_ids,
    processor
):

    rows = []

    for query_id in query_ids:

        row = benchmark[
            benchmark["query_id"].astype(str)
            == str(query_id)
        ]

        if len(row) == 0:
            continue

        rows.append(
            row.iloc[0]
        )

    states = []
    districts = []
    property_types = []
    numeric = []

    for query in rows:

        features = processor.transform(
            query
        )

        states.append(
            features["state"]
        )

        districts.append(
            features["district"]
        )

        property_types.append(
            features["property_type"]
        )

        numeric.append(
            features["numeric"]
        )

    states = torch.tensor(
        states,
        dtype=torch.long,
        device=DEVICE
    )

    districts = torch.tensor(
        districts,
        dtype=torch.long,
        device=DEVICE
    )

    property_types = torch.tensor(
        property_types,
        dtype=torch.long,
        device=DEVICE
    )

    numeric = torch.tensor(
        numeric,
        dtype=torch.float32,
        device=DEVICE
    )

    return (
        states,
        districts,
        property_types,
        numeric
    )


# ============================================================
# BUILD TRAINING ADJACENCY
# ============================================================

def build_training_adjacency(
    train_query_ids,
    train_interactions,
    property_to_idx
):

    num_queries = len(
        train_query_ids
    )

    num_properties = len(
        property_to_idx
    )

    num_nodes = (
        num_queries +
        num_properties
    )

    query_to_idx = {
        query_id: idx
        for idx, query_id
        in enumerate(train_query_ids)
    }

    rows = []
    cols = []

    for _, interaction in (
        train_interactions.iterrows()
    ):

        query_id = str(
            interaction["query_id"]
        )

        property_id = int(
            interaction["property_id"]
        )

        if query_id not in query_to_idx:
            continue

        if property_id not in property_to_idx:
            continue

        q_idx = query_to_idx[
            query_id
        ]

        p_idx = (
            num_queries
            + property_to_idx[property_id]
        )

        # Query -> Property
        rows.append(q_idx)
        cols.append(p_idx)

        # Property -> Query
        rows.append(p_idx)
        cols.append(q_idx)

    # --------------------------------------------------------
    # Remove duplicate edges
    # --------------------------------------------------------

    edge_set = set(
        zip(rows, cols)
    )

    rows = [
        x[0]
        for x in edge_set
    ]

    cols = [
        x[1]
        for x in edge_set
    ]

    values = np.ones(
        len(rows),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Degree
    # --------------------------------------------------------

    degree = np.zeros(
        num_nodes,
        dtype=np.float32
    )

    for row in rows:
        degree[row] += 1.0

    # D^-1/2
    degree_inv_sqrt = np.zeros_like(
        degree
    )

    nonzero = degree > 0

    degree_inv_sqrt[
        nonzero
    ] = 1.0 / np.sqrt(
        degree[nonzero]
    )

    # --------------------------------------------------------
    # Normalized adjacency
    #
    # A_hat = D^-1/2 A D^-1/2
    # --------------------------------------------------------

    normalized_values = []

    for row, col in zip(
        rows,
        cols
    ):

        value = (
            degree_inv_sqrt[row]
            * values[0]
            * degree_inv_sqrt[col]
        )

        normalized_values.append(
            value
        )

    indices = torch.tensor(
        [rows, cols],
        dtype=torch.long,
        device=DEVICE
    )

    values_tensor = torch.tensor(
        normalized_values,
        dtype=torch.float32,
        device=DEVICE
    )

    adjacency = torch.sparse_coo_tensor(
        indices,
        values_tensor,
        size=(
            num_nodes,
            num_nodes
        ),
        device=DEVICE
    ).coalesce()

    print("\n===== TRAINING GRAPH =====")

    print(
        f"Graph nodes: {num_nodes}"
    )

    print(
        f"Graph edges: {len(rows)}"
    )

    print(
        f"Training query nodes: "
        f"{num_queries}"
    )

    print(
        f"Property nodes: "
        f"{num_properties}"
    )

    return (
        adjacency,
        query_to_idx
    )


# ============================================================
# BPR LOSS
# ============================================================

def bpr_loss(
    positive_scores,
    negative_scores
):

    difference = (
        positive_scores
        - negative_scores
    )

    return -torch.mean(
        F.logsigmoid(
            difference
        )
    )


# ============================================================
# NEGATIVE SAMPLING
# ============================================================

def sample_training_pairs(
    train_interactions,
    positive_dict,
    property_ids,
    property_to_idx
):

    query_ids = (
        train_interactions["query_id"]
        .astype(str)
        .values
    )

    positive_ids = (
        train_interactions["property_id"]
        .astype(int)
        .values
    )

    negative_ids = []

    all_property_ids = list(
        property_ids
    )

    for query_id in query_ids:

        positives = positive_dict.get(
            query_id,
            set()
        )

        available = [
            p
            for p in all_property_ids
            if p not in positives
        ]

        if not available:

            negative_ids.append(
                random.choice(
                    all_property_ids
                )
            )

        else:

            negative_ids.append(
                random.choice(
                    available
                )
            )

    return (
        list(query_ids),
        list(positive_ids),
        negative_ids
    )


# ============================================================
# GET TRAIN QUERY EMBEDDINGS
# ============================================================

def get_train_query_embeddings(
    model,
    benchmark,
    train_query_ids,
    processor
):

    (
        states,
        districts,
        property_types,
        numeric
    ) = queries_to_batch_features(
        benchmark,
        train_query_ids,
        processor
    )

    query_embeddings = (
        model.query_encoder(
            states,
            districts,
            property_types,
            numeric
        )
    )

    return query_embeddings


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(
    model,
    benchmark,
    train_query_ids,
    train_interactions,
    positive_dict,
    property_ids,
    property_to_idx,
    adjacency,
    processor,
    val_query_ids,
    val_interactions
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_state = None

    best_val_ndcg = -np.inf

    patience_counter = 0

    train_query_to_idx = {
        query_id: idx
        for idx, query_id
        in enumerate(train_query_ids)
    }

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Query encoder
        # ----------------------------------------------------

        train_query_embeddings = (
            get_train_query_embeddings(
                model,
                benchmark,
                train_query_ids,
                processor
            )
        )

        # ----------------------------------------------------
        # REAL LIGHTGCN PROPAGATION
        # ----------------------------------------------------

        propagated_queries, propagated_properties = (
            model.propagate(
                adjacency,
                train_query_embeddings
            )
        )

        # ----------------------------------------------------
        # BPR samples
        # ----------------------------------------------------

        (
            sampled_query_ids,
            positive_property_ids,
            negative_property_ids
        ) = sample_training_pairs(
            train_interactions,
            positive_dict,
            property_ids,
            property_to_idx
        )

        query_indices = torch.tensor(
            [
                train_query_to_idx[q]
                for q in sampled_query_ids
            ],
            dtype=torch.long,
            device=DEVICE
        )

        positive_indices = torch.tensor(
            [
                property_to_idx[p]
                for p in positive_property_ids
            ],
            dtype=torch.long,
            device=DEVICE
        )

        negative_indices = torch.tensor(
            [
                property_to_idx[p]
                for p in negative_property_ids
            ],
            dtype=torch.long,
            device=DEVICE
        )

        query_batch = (
            propagated_queries[
                query_indices
            ]
        )

        positive_batch = (
            propagated_properties[
                positive_indices
            ]
        )

        negative_batch = (
            propagated_properties[
                negative_indices
            ]
        )

        positive_scores = torch.sum(
            query_batch
            * positive_batch,
            dim=1
        )

        negative_scores = torch.sum(
            query_batch
            * negative_batch,
            dim=1
        )

        loss_bpr = bpr_loss(
            positive_scores,
            negative_scores
        )

        # ----------------------------------------------------
        # L2 regularization
        # ----------------------------------------------------

        reg_loss = (
            torch.mean(
                query_batch ** 2
            )
            +
            torch.mean(
                positive_batch ** 2
            )
            +
            torch.mean(
                negative_batch ** 2
            )
        )

        loss = (
            loss_bpr
            +
            REG_WEIGHT * reg_loss
        )

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_ndcg = evaluate_validation(
            model,
            benchmark,
            val_query_ids,
            val_interactions,
            property_ids,
            property_to_idx,
            adjacency,
            processor
        )

        if epoch == 1 or epoch % 10 == 0:

            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss.item():.6f} | "
                f"Val NDCG@10: {val_ndcg:.6f}"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if val_ndcg > best_val_ndcg:

            best_val_ndcg = val_ndcg

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                f"Early stopping at epoch "
                f"{epoch}"
            )

            break

    # --------------------------------------------------------
    # Restore best validation model
    # --------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return model


# ============================================================
# GET TRAINING-GRAPH PROPERTY EMBEDDINGS
# ============================================================

@torch.no_grad()
def get_property_embeddings(
    model,
    benchmark,
    train_query_ids,
    adjacency,
    processor
):

    model.eval()

    train_query_embeddings = (
        get_train_query_embeddings(
            model,
            benchmark,
            train_query_ids,
            processor
        )
    )

    _, property_embeddings = (
        model.propagate(
            adjacency,
            train_query_embeddings
        )
    )

    return property_embeddings


# ============================================================
# GET UNSEEN QUERY EMBEDDING
# ============================================================

@torch.no_grad()
def get_unseen_query_embedding(
    model,
    query,
    processor
):

    model.eval()

    (
        state,
        district,
        property_type,
        numeric
    ) = query_to_features(
        query,
        processor
    )

    query_embedding = (
        model.query_encoder(
            state,
            district,
            property_type,
            numeric
        )
    )

    return query_embedding


# ============================================================
# RANK PROPERTIES
# ============================================================

@torch.no_grad()
def rank_properties(
    model,
    query,
    property_embeddings,
    idx_to_property,
    processor
):

    query_embedding = (
        get_unseen_query_embedding(
            model,
            query,
            processor
        )
    )

    scores = torch.matmul(
        query_embedding,
        property_embeddings.T
    ).squeeze(0)

    sorted_indices = torch.argsort(
        scores,
        descending=True
    )

    results = []

    for idx in sorted_indices.tolist():

        property_id = (
            idx_to_property[idx]
        )

        score = float(
            scores[idx].item()
        )

        results.append(
            (
                property_id,
                score
            )
        )

    return results


# ============================================================
# GET QUERY BY ID
# ============================================================

def get_query(
    benchmark,
    query_id
):

    rows = benchmark[
        benchmark["query_id"].astype(str)
        == str(query_id)
    ]

    if len(rows) == 0:

        raise ValueError(
            f"Query {query_id} not found."
        )

    return rows.iloc[0]


# ============================================================
# GET TEST RELEVANT PROPERTY SET
# ============================================================

def get_relevant_ids(
    query_id,
    benchmark,
    raw_properties
):

    query = get_query(
        benchmark,
        query_id
    )

    relevant = get_relevant_properties(
        query,
        raw_properties
    )

    return set(
        relevant["property_id"]
        .astype(int)
        .tolist()
    )


# ============================================================
# PRECISION@K
# ============================================================

def precision_at_k(
    ranked_ids,
    relevant_ids,
    k
):

    top_k = ranked_ids[:k]

    if len(top_k) == 0:
        return 0.0

    hits = sum(
        1
        for p in top_k
        if p in relevant_ids
    )

    return hits / k


# ============================================================
# RECALL@K
# ============================================================

def recall_at_k(
    ranked_ids,
    relevant_ids,
    k
):

    if len(relevant_ids) == 0:
        return None

    top_k = ranked_ids[:k]

    hits = sum(
        1
        for p in top_k
        if p in relevant_ids
    )

    return hits / len(
        relevant_ids
    )


# ============================================================
# MRR
# ============================================================

def reciprocal_rank(
    ranked_ids,
    relevant_ids
):

    for rank, property_id in enumerate(
        ranked_ids,
        start=1
    ):

        if property_id in relevant_ids:

            return 1.0 / rank

    return 0.0


# ============================================================
# NDCG@K
# ============================================================

def ndcg_at_k(
    ranked_ids,
    relevant_ids,
    k
):

    if len(relevant_ids) == 0:
        return None

    top_k = ranked_ids[:k]

    dcg = 0.0

    for rank, property_id in enumerate(
        top_k,
        start=1
    ):

        if property_id in relevant_ids:

            dcg += (
                1.0
                / np.log2(rank + 1)
            )

    ideal_hits = min(
        len(relevant_ids),
        k
    )

    idcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1
        )
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


# ============================================================
# VALIDATION EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_validation(
    model,
    benchmark,
    val_query_ids,
    val_interactions,
    property_ids,
    property_to_idx,
    adjacency,
    processor
):

    if len(val_query_ids) == 0:
        return 0.0

    property_embeddings = (
        get_property_embeddings(
            model,
            benchmark,
            TRAIN_QUERY_IDS_GLOBAL,
            adjacency,
            processor
        )
    )

    idx_to_property = {
        idx: property_id
        for property_id, idx
        in property_to_idx.items()
    }

    ndcg_values = []

    for query_id in val_query_ids:

        query = get_query(
            benchmark,
            query_id
        )

        relevant_ids = set(
            val_interactions[
                val_interactions["query_id"]
                .astype(str)
                == str(query_id)
            ]["property_id"]
            .astype(int)
            .tolist()
        )

        if len(relevant_ids) == 0 or str(query_id) not in RANKING_QUERY_IDS:
            continue

        ranked = rank_properties(
            model,
            query,
            property_embeddings,
            idx_to_property,
            processor
        )

        ranked_ids = [
            x[0]
            for x in ranked
        ]

        score = ndcg_at_k(
            ranked_ids,
            relevant_ids,
            10
        )

        if score is not None:

            ndcg_values.append(
                score
            )

    if len(ndcg_values) == 0:
        return 0.0

    return float(
        np.mean(ndcg_values)
    )


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_test(
    model,
    benchmark,
    raw_properties,
    test_query_ids,
    test_interactions,
    property_ids,
    property_to_idx,
    adjacency,
    processor,
    train_query_ids
):

    print("\n===== FINAL TEST EVALUATION =====")

    property_embeddings = (
        get_property_embeddings(
            model,
            benchmark,
            train_query_ids,
            adjacency,
            processor
        )
    )

    idx_to_property = {
        idx: property_id
        for property_id, idx
        in property_to_idx.items()
    }

    p5_values = []
    p10_values = []
    recall10_values = []
    ndcg10_values = []
    mrr_values = []

    evaluated_queries = []

    recommended_property_ids = set()

    total_response_time = 0.0

    for query_id in test_query_ids:

        query = get_query(
            benchmark,
            query_id
        )

        relevant_ids = set(
            test_interactions[
                test_interactions["query_id"]
                .astype(str)
                == str(query_id)
            ]["property_id"]
            .astype(int)
            .tolist()
        )

        # ----------------------------------------------------
        # Only "ranking" queries (>= 5 eligible properties)
        # enter ranking metrics (eligibility rule ELIG-v1).
        # ----------------------------------------------------

        if len(relevant_ids) == 0 or str(query_id) not in RANKING_QUERY_IDS:

            continue

        start_time = time.perf_counter()

        ranked = rank_properties(
            model,
            query,
            property_embeddings,
            idx_to_property,
            processor
        )

        end_time = time.perf_counter()

        total_response_time += (
            end_time - start_time
        )

        ranked_ids = [
            x[0]
            for x in ranked
        ]

        p5 = precision_at_k(
            ranked_ids,
            relevant_ids,
            5
        )

        p10 = precision_at_k(
            ranked_ids,
            relevant_ids,
            10
        )

        recall10 = recall_at_k(
            ranked_ids,
            relevant_ids,
            10
        )

        ndcg10 = ndcg_at_k(
            ranked_ids,
            relevant_ids,
            10
        )

        mrr = reciprocal_rank(
            ranked_ids,
            relevant_ids
        )

        p5_values.append(p5)
        p10_values.append(p10)

        if recall10 is not None:
            recall10_values.append(
                recall10
            )

        if ndcg10 is not None:
            ndcg10_values.append(
                ndcg10
            )

        mrr_values.append(mrr)

        recommended_property_ids.update(
            ranked_ids[:10]
        )

        evaluated_queries.append(
            query_id
        )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    if len(property_ids) > 0:

        coverage10 = (
            len(recommended_property_ids)
            /
            len(property_ids)
        )

    else:

        coverage10 = 0.0

    # --------------------------------------------------------
    # Response time
    # --------------------------------------------------------

    if len(evaluated_queries) > 0:

        avg_response_time_ms = (
            total_response_time
            /
            len(evaluated_queries)
            * 1000
        )

    else:

        avg_response_time_ms = 0.0

    results = {

        "P@5": (
            float(np.mean(p5_values))
            if p5_values
            else 0.0
        ),

        "P@10": (
            float(np.mean(p10_values))
            if p10_values
            else 0.0
        ),

        "Recall@10": (
            float(np.mean(recall10_values))
            if recall10_values
            else 0.0
        ),

        "NDCG@10": (
            float(np.mean(ndcg10_values))
            if ndcg10_values
            else 0.0
        ),

        "MRR": (
            float(np.mean(mrr_values))
            if mrr_values
            else 0.0
        ),

        "Coverage@10": float(
            coverage10
        ),

        "ResponseTime_ms": float(
            avg_response_time_ms
        ),

        "EvaluatedQueries": len(
            evaluated_queries
        ),

        "TestQueries": len(
            test_query_ids
        ),
    }

    print(
        f"Evaluated queries: "
        f"{results['EvaluatedQueries']}/"
        f"{results['TestQueries']}"
    )

    print(
        f"P@5        : {results['P@5']:.6f}"
    )

    print(
        f"P@10       : {results['P@10']:.6f}"
    )

    print(
        f"Recall@10  : {results['Recall@10']:.6f}"
    )

    print(
        f"NDCG@10    : {results['NDCG@10']:.6f}"
    )

    print(
        f"MRR        : {results['MRR']:.6f}"
    )

    print(
        f"Coverage@10: {results['Coverage@10']:.6f}"
    )

    print(
        f"Response   : "
        f"{results['ResponseTime_ms']:.4f} ms"
    )

    return results


# ============================================================
# Q001 DEMONSTRATION
# ============================================================

@torch.no_grad()
def show_q001_recommendations(
    model,
    benchmark,
    raw_properties,
    property_embeddings,
    property_to_idx,
    top_k=15
):

    query_id = "Q001"

    query = get_query(
        benchmark,
        query_id
    )

    idx_to_property = {
        idx: property_id
        for property_id, idx
        in property_to_idx.items()
    }

    ranked = rank_properties(
        model,
        query,
        property_embeddings,
        idx_to_property,
        PROCESSOR_GLOBAL
    )

    top_results = ranked[
        :top_k
    ]

    result_rows = []

    for rank, (
        property_id,
        score
    ) in enumerate(
        top_results,
        start=1
    ):

        property_rows = raw_properties[
            raw_properties["property_id"]
            .astype(int)
            == int(property_id)
        ]

        if len(property_rows) == 0:
            continue

        property_row = (
            property_rows.iloc[0]
        )

        result_rows.append({

            "rank": rank,

            "property_id": int(
                property_id
            ),

            "score": float(
                score
            ),

            "price": property_row.get(
                "price",
                np.nan
            ),

            "property_type":
                property_row.get(
                    "property_type",
                    ""
                ),

            "state":
                property_row.get(
                    "state",
                    ""
                ),

            "district":
                property_row.get(
                    "district",
                    ""
                ),

            "bedroom":
                property_row.get(
                    "bedroom",
                    np.nan
                ),

            "bathroom":
                property_row.get(
                    "bathroom",
                    np.nan
                ),
        })

    result_df = pd.DataFrame(
        result_rows
    )

    result_df.to_csv(
        RECOMMENDATION_PATH,
        index=False
    )

    print("\n===== Q001 RECOMMENDATIONS =====")

    print(
        result_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved to: "
        f"{RECOMMENDATION_PATH}"
    )

    return result_df


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():

    global TRAIN_QUERY_IDS_GLOBAL
    global PROCESSOR_GLOBAL
    global RANKING_QUERY_IDS

    RANKING_QUERY_IDS = load_ranking_query_ids()
    print(f"Ranking-eligible queries (ELIG-v1): {len(RANKING_QUERY_IDS)}")

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    (
        benchmark,
        properties,
        raw_properties
    ) = load_data()

    # --------------------------------------------------------
    # Build all relevance labels
    # --------------------------------------------------------

    interactions = build_interactions(
        benchmark,
        raw_properties
    )

    # --------------------------------------------------------
    # Property mapping
    # --------------------------------------------------------

    (
        property_to_idx,
        idx_to_property
    ) = create_property_mappings(
        properties
    )

    property_ids = list(
        property_to_idx.keys()
    )

    all_seed_results = []

    # ========================================================
    # RUN EACH RANDOM SEED
    # ========================================================

    for seed in SEEDS:

        print("\n")
        print("=" * 70)
        print(
            f"RUNNING LIGHTGCN - SEED {seed}"
        )
        print("=" * 70)

        set_seed(seed)

        # ----------------------------------------------------
        # Scenario-level split
        # ----------------------------------------------------

        (
            train_query_ids,
            val_query_ids,
            test_query_ids
        ) = split_scenarios(
            benchmark,
            seed
        )

        TRAIN_QUERY_IDS_GLOBAL = (
            train_query_ids
        )

        # ----------------------------------------------------
        # Split interactions according to scenarios
        # ----------------------------------------------------

        (
            train_interactions,
            val_interactions,
            test_interactions
        ) = split_interactions_by_scenario(
            interactions,
            train_query_ids,
            val_query_ids,
            test_query_ids
        )

        # ----------------------------------------------------
        # Build positive dictionary
        # ----------------------------------------------------

        positive_dict = (
            build_positive_dict(
                train_interactions
            )
        )

        # ----------------------------------------------------
        # Query feature processor
        #
        # IMPORTANT:
        # Fit only on TRAIN scenarios.
        # ----------------------------------------------------

        train_benchmark = benchmark[
            benchmark["query_id"]
            .astype(str)
            .isin(train_query_ids)
        ].copy()

        processor = (
            QueryFeatureProcessor()
        )

        processor.fit(
            train_benchmark
        )

        PROCESSOR_GLOBAL = (
            processor
        )

        # ----------------------------------------------------
        # Training graph
        # ----------------------------------------------------

        (
            adjacency,
            train_query_to_idx
        ) = build_training_adjacency(
            train_query_ids,
            train_interactions,
            property_to_idx
        )

        # ----------------------------------------------------
        # Query encoder
        #
        # +1 for unknown category index 0
        # ----------------------------------------------------

        query_encoder = QueryEncoder(
            num_states=
                len(
                    processor.state_mapping
                ) + 1,

            num_districts=
                len(
                    processor.district_mapping
                ) + 1,

            num_property_types=
                len(
                    processor.property_type_mapping
                ) + 1,

            embedding_dim=
                EMBEDDING_DIM
        )

        # ----------------------------------------------------
        # LightGCN
        # ----------------------------------------------------

        model = LightGCN(
            num_train_queries=
                len(train_query_ids),

            num_properties=
                len(property_ids),

            query_encoder=
                query_encoder,

            embedding_dim=
                EMBEDDING_DIM,

            n_layers=
                N_LAYERS
        ).to(DEVICE)

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model = train_model(
            model=model,
            benchmark=benchmark,
            train_query_ids=train_query_ids,
            train_interactions=train_interactions,
            positive_dict=positive_dict,
            property_ids=property_ids,
            property_to_idx=property_to_idx,
            adjacency=adjacency,
            processor=processor,
            val_query_ids=val_query_ids,
            val_interactions=val_interactions
        )

        # ----------------------------------------------------
        # Final test evaluation
        # ----------------------------------------------------

        test_results = evaluate_test(
            model=model,
            benchmark=benchmark,
            raw_properties=raw_properties,
            test_query_ids=test_query_ids,
            test_interactions=test_interactions,
            property_ids=property_ids,
            property_to_idx=property_to_idx,
            adjacency=adjacency,
            processor=processor,
            train_query_ids=train_query_ids
        )

        test_results["seed"] = seed

        all_seed_results.append(
            test_results
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            f"\nModel saved to: "
            f"{MODEL_PATH}"
        )

        # ----------------------------------------------------
        # Q001 demonstration
        #
        # This is demonstration only, NOT test evidence.
        # ----------------------------------------------------

        property_embeddings = (
            get_property_embeddings(
                model,
                benchmark,
                train_query_ids,
                adjacency,
                processor
            )
        )

        show_q001_recommendations(
            model=model,
            benchmark=benchmark,
            raw_properties=raw_properties,
            property_embeddings=
                property_embeddings,
            property_to_idx=
                property_to_idx,
            top_k=15
        )

    # ========================================================
    # MULTI-SEED SUMMARY
    # ========================================================

    results_df = pd.DataFrame(
        all_seed_results
    )

    results_df.to_csv(
        RESULT_PATH,
        index=False
    )

    metric_columns = [
        "P@5",
        "P@10",
        "Recall@10",
        "NDCG@10",
        "MRR",
        "Coverage@10",
        "ResponseTime_ms",
    ]

    summary_rows = []

    for metric in metric_columns:

        summary_rows.append({

            "metric": metric,

            "mean": results_df[
                metric
            ].mean(),

            "std": results_df[
                metric
            ].std(),

            "min": results_df[
                metric
            ].min(),

            "max": results_df[
                metric
            ].max(),
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "lightgcn_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("FINAL MULTI-SEED SUMMARY")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        f"\nDetailed results saved to:"
        f"\n{RESULT_PATH}"
    )

    print(
        f"\nSummary saved to:"
        f"\n{summary_path}"
    )


# ============================================================
# DEBUG INFORMATION
# ============================================================

def debug_dataset(
    benchmark,
    properties,
    raw_properties,
    interactions
):

    print("\n")
    print("=" * 70)
    print("DATASET DEBUG")
    print("=" * 70)

    print(
        f"Benchmark queries: "
        f"{benchmark['query_id'].nunique()}"
    )

    print(
        f"Processed properties: "
        f"{properties['property_id'].nunique()}"
    )

    print(
        f"Raw properties: "
        f"{raw_properties['property_id'].nunique()}"
    )

    processed_ids = set(
        properties["property_id"]
        .astype(int)
    )

    raw_ids = set(
        raw_properties["property_id"]
        .astype(int)
    )

    print(
        f"Common property IDs: "
        f"{len(processed_ids & raw_ids)}"
    )

    print(
        f"Missing processed IDs: "
        f"{len(raw_ids - processed_ids)}"
    )

    print(
        f"Missing raw IDs: "
        f"{len(processed_ids - raw_ids)}"
    )

    print(
        f"Positive interactions: "
        f"{len(interactions)}"
    )

    q001 = interactions[
        interactions["query_id"]
        .astype(str)
        == "Q001"
    ]

    print(
        f"Q001 positive properties: "
        f"{len(q001)}"
    )

    print(
        "Q001 property IDs:"
    )

    print(
        q001["property_id"]
        .astype(int)
        .tolist()
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print(
        "Starting LightGCN experiment..."
    )

    run_experiment()