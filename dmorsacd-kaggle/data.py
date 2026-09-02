"""Dataset loading & preprocessing for DMoR-SACD.

Supports the three datasets used in the paper plus RetailRocket:
    * MovieLens 100K  (http://files.grouplens.org/datasets/movielens/ml-100k.zip)
    * MovieLens 1M    (http://files.grouplens.org/datasets/movielens/ml-1m.zip)
    * LastFM / HetRec (https://grouplens.org/datasets/hetrec-2011/)
    * RetailRocket    (https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

RetailRocket is implicit feedback (view / addtocart / transaction). The
loader converts it to pseudo-ratings with the graded mapping
`view=1, addtocart=2, transaction=3` (max per user-item pair), so the
paper's `positive_threshold=3.0` still means "only purchases are positive"
and normalised ratings land on {view: -1, addtocart: 0, transaction: +1}.
Because the full item space has ~235K items, `max_items` keeps only the
most frequent items as the candidate set (config default 10,000).

Users are split 80/10/10 (train/val/test); users with fewer than
`min_interactions` interactions are discarded (paper, Section 4.1).
Every user keeps their full interaction record -- the offline simulator
uses those records as environment feedback.

Item-similarity vectors (used by the diversity reward and the D@K metric):
    * MovieLens: genre vectors (content-based diversity, as the paper
      motivates diversity through e.g. "movie genres").
    * LastFM:    play-count rating column (normalised), since no genre
      table ships with the HetRec dataset.
    * RetailRocket: one-hot category vectors built from `item_properties`
      (property == categoryid). The generic rating-column fallback is NOT
      usable here: 235K items x 1.4M users would be ~1.3 TB.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import LASTFM, ML_100K, ML_1M, RETAILROCKET


# ---------------------------------------------------------------------------
# Raw loaders
# ---------------------------------------------------------------------------
def _load_ml_100k(data_dir: str) -> pd.DataFrame:
    base = os.path.join(data_dir, "ml-100k")
    ratings = pd.read_csv(
        os.path.join(base, "u.data"), sep="\t", header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    # u.item : item_id|movie title|release date|video date|IMDb URL|genres(19)
    items = pd.read_csv(
        os.path.join(base, "u.item"), sep="|", header=None, encoding="latin-1",
        usecols=range(24),
        names=["item_id", "title", "date", "video", "imdb"]
               + [f"g{i}" for i in range(19)],
    )
    return ratings, items


def _load_ml_1m(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = os.path.join(data_dir, "ml-1m")
    ratings = pd.read_csv(
        os.path.join(base, "ratings.dat"), sep="::", header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    # movies.dat : item_id|title|genres  (latin-1 encoded titles)
    items = pd.read_csv(
        os.path.join(base, "movies.dat"), sep="::", header=None,
        names=["item_id", "title", "genres"], engine="python",
        encoding="latin-1",
    )
    return ratings, items


def _load_lastfm(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = os.path.join(data_dir, "lastfm")
    # user_artists.dat : userID::artistID::weight (play count)
    ratings = pd.read_csv(
        os.path.join(base, "user_artists.dat"), sep="\t", header=0,
        names=["user_id", "item_id", "rating"],
    )
    items = None
    return ratings, items


def _load_retailrocket(data_dir: str,
                       max_items: Optional[int] = None
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load RetailRocket as pseudo-ratings + item category table.

    Expected files under ``data_dir/retailrocket/``:
        events.csv            (timestamp, visitorid, event, itemid, ...)
        item_categories.csv   (itemid, categoryid)  -- compact, preferred
        item_properties_part1/2.csv                  -- fallback (raw 900 MB)

    Graded mapping: view=1, addtocart=2, transaction=3 (max per pair).
    When `max_items` is set, only the most frequent items are kept.
    """
    # Path to the RetailRocket data directory (data_dir/retailrocket)
    base = os.path.join(data_dir, "RetailRocketDataset")
    # Read the raw events log: one row per user action, columns include
    # timestamp, visitorid, event, itemid, transactionid
    events = pd.read_csv(os.path.join(base, "events.csv"))
    # Keep only the three implicit-feedback event types used by the paper
    events = events[events["event"].isin(("view", "addtocart", "transaction"))]
    # Convert each event type into a pseudo-rating: view=1, addtocart=2, transaction=3
    events["rating"] = events["event"].map(
        {"view": 1.0, "addtocart": 2.0, "transaction": 3.0}
    )
    # Collapse to one row per (visitor, item), keeping the *max* pseudo-rating
    # so a transaction (3) overrides an earlier view (1) of the same item
    ratings = (events.groupby(["visitorid", "itemid"], as_index=False)
               ["rating"].max()
               # Rename to the generic user_id / item_id convention used elsewhere
               .rename(columns={"visitorid": "user_id", "itemid": "item_id"}))

    # If a cap on the candidate item set was requested (config default 10,000)...
    if max_items and max_items > 0:
        # Count interactions per item and sort descending (most frequent first)
        freq = ratings.groupby("item_id")["rating"].count()\
            .sort_values(ascending=False)
        # Keep only the ids of the top-K most frequent items
        keep = set(freq.head(max_items).index)
        # Drop every rating row whose item is outside that candidate set
        ratings = ratings[ratings["item_id"].isin(keep)]

    # item -> leaf category (compact preprocessed file preferred)
    # Path to the compact item-to-category lookup file
    cat_file = os.path.join(base, "item_categories.csv")
    # Prefer the compact file when it is available...
    if os.path.exists(cat_file):
        cat = pd.read_csv(cat_file)
    else:
        # Fallback: read both raw item_properties parts (~900 MB combined)
        props = pd.concat([
            pd.read_csv(os.path.join(base, "item_properties_part1.csv"),
                        usecols=["itemid", "property", "value"]),
            pd.read_csv(os.path.join(base, "item_properties_part2.csv"),
                        usecols=["itemid", "property", "value"]),
        ], ignore_index=True)
        # Keep only rows whose property == "categoryid" and rename value -> categoryid
        cat = (props[props["property"] == "categoryid"][["itemid", "value"]]
               .rename(columns={"value": "categoryid"}))
    # shared defensive coercion (a hand-made compact file may contain NaN)
    # Coerce both id columns to numeric, turning any bad entry into NaN
    cat["itemid"] = pd.to_numeric(cat["itemid"], errors="coerce")
    cat["categoryid"] = pd.to_numeric(cat["categoryid"], errors="coerce")
    # Drop NaN rows and dedupe to one category per item (last one wins)
    cat = cat.dropna().drop_duplicates("itemid", keep="last")
    # Rename itemid -> item_id to match the ratings table's column name
    items = cat.rename(columns={"itemid": "item_id"})
    # Cast to int64 so item ids align with the (remapped) ratings table
    items["item_id"] = items["item_id"].astype("int64")
    items["categoryid"] = items["categoryid"].astype("int64")
    # Return the pseudo-ratings table and the item -> category table
    return ratings, items


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class RatingData:
    """Encapsulates everything the simulator needs for one dataset."""

    name: str
    num_users: int
    num_items: int
    rmin: float
    rmax: float
    train_users: np.ndarray
    val_users: np.ndarray
    test_users: np.ndarray
    # user_id -> {item_id: raw rating}
    user_ratings: Dict[int, Dict[int, float]] = field(default_factory=dict)
    # item_id -> L2-normalised vector used for cosine similarity
    item_vectors: Optional[np.ndarray] = None
    # dense user-item rating matrix (num_users+1 x num_items+1, 0 = unrated)
    rating_matrix: Optional[np.ndarray] = None

    def rating(self, user: int, item: int) -> float:
        return self.user_ratings[user].get(item, 0.0)

    def normalize_rating(self, r: float) -> float:
        """Eq. 5: linear transform of a raw rating to [-1, 1]."""
        if self.rmax == self.rmin:
            return 0.0
        return (2.0 * r - self.rmax - self.rmin) / (self.rmax - self.rmin)

    def positive(self, r: float) -> bool:
        return r >= 3.0

    def item_sim(self, a: int, b: int) -> float:
        va, vb = self.item_vectors[a], self.item_vectors[b]
        return float(np.dot(va, vb))


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def _genre_matrix_100k(items: pd.DataFrame, item_map: Dict[int, int],
                        num_items: int) -> np.ndarray:
    """19 genre columns of u.item, indexed by the *remapped* item ids."""
    vec = np.zeros((num_items + 1, 19), dtype=np.float32)
    genre_cols = [f"g{i}" for i in range(19)]
    for _, row in items.iterrows():
        new_id = item_map.get(int(row["item_id"]))
        if new_id is None:
            continue
        vec[new_id] = row[genre_cols].values.astype(np.float32)
    return vec


def _genre_matrix_1m(items: pd.DataFrame, item_map: Dict[int, int],
                     num_items: int) -> np.ndarray:
    vec = np.zeros((num_items + 1, 18), dtype=np.float32)
    genres = ["Action", "Adventure", "Animation", "Children's", "Comedy",
              "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
              "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
              "War", "Western"]
    genre_idx = {g: i for i, g in enumerate(genres)}
    for _, row in items.iterrows():
        new_id = item_map.get(int(row["item_id"]))
        if new_id is None:
            continue
        for g in str(row["genres"]).split("|"):
            if g in genre_idx:
                vec[new_id, genre_idx[g]] = 1.0
    return vec


def _category_vectors(items: pd.DataFrame, item_map: Dict[int, int],
                      num_items: int) -> np.ndarray:
    """One-hot leaf-category vectors, indexed by the *remapped* item ids.

    Used for RetailRocket (content diversity via item category, mirroring
    the genre vectors used for MovieLens). Items without a category get a
    zero vector and therefore neutral cosine similarity.
    """
    cats = np.sort(items["categoryid"].unique())
    cat_idx = {int(c): i for i, c in enumerate(cats)}
    vec = np.zeros((num_items + 1, len(cats)), dtype=np.float32)
    kept = items[items["item_id"].isin(item_map)]
    if len(kept):
        rows = kept["item_id"].map(item_map).to_numpy(dtype=np.int64)
        cols = kept["categoryid"].map(cat_idx).to_numpy(dtype=np.int64)
        np.add.at(vec, (rows, cols), 1.0)
    return vec


def _column_vectors(ratings: pd.DataFrame, num_items: int,
                    num_users: int) -> np.ndarray:
    """Item vector = L2-normalised rating/play-count column.

    Used for LastFM (which has no genre table): the similarity of two
    artists reflects how similarly users listened to them.
    """
    mat = np.zeros((num_items + 1, num_users + 1), dtype=np.float32)
    for row in ratings.itertuples(index=False):
        mat[int(row.item_id), int(row.user_id)] = float(row.rating)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def prepare_data(dataset: str, data_dir: str,
                 seed: int = 42, min_interactions: int = 20,
                 train_ratio: float = 0.8, val_ratio: float = 0.1,
                 max_items: Optional[int] = None) -> RatingData:
    """Load, clean and split one of the supported datasets.

    `max_items` only affects RetailRocket: keep the top-K most frequent
    items as the candidate set (config default 10,000; None/0 = keep all).
    """
    rng = np.random.default_rng(seed)

    if dataset == ML_100K:
        ratings, items = _load_ml_100k(data_dir)
    elif dataset == ML_1M:
        ratings, items = _load_ml_1m(data_dir)
    elif dataset == LASTFM:
        ratings, items = _load_lastfm(data_dir)
    elif dataset == RETAILROCKET:
        ratings, items = _load_retailrocket(data_dir, max_items=max_items)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")

    ratings = ratings[["user_id", "item_id", "rating"]].copy()
    ratings = ratings[ratings["rating"] > 0]  # drop invalid entries

    # discard users with too few interactions (paper: >= 20)
    counts = ratings.groupby("user_id")["item_id"].count()
    keep = counts[counts >= min_interactions].index
    ratings = ratings[ratings["user_id"].isin(keep)]

    # remap ids to 1..N (0 reserved for padding)
    users = sorted(ratings["user_id"].unique())
    items_ = sorted(ratings["item_id"].unique())
    user_map = {u: i + 1 for i, u in enumerate(users)}
    item_map = {i: j + 1 for j, i in enumerate(items_)}
    ratings["user_id"] = ratings["user_id"].map(user_map)
    ratings["item_id"] = ratings["item_id"].map(item_map)
    num_users, num_items = len(users), len(items_)

    # split users 80 / 10 / 10
    perm = rng.permutation(num_users) + 1
    n_train = int(num_users * train_ratio)
    n_val = int(num_users * val_ratio)
    train_users = perm[:n_train]
    val_users = perm[n_train:n_train + n_val]
    test_users = perm[n_train + n_val:]

    # feedback maps
    user_ratings: Dict[int, Dict[int, float]] = {
        u: {} for u in range(1, num_users + 1)
    }
    for row in ratings.itertuples(index=False):
        user_ratings[int(row.user_id)][int(row.item_id)] = float(row.rating)

    rmin, rmax = float(ratings["rating"].min()), float(ratings["rating"].max())

    # item similarity vectors (indexed by the remapped item ids)
    if dataset == ML_100K:
        vec = _genre_matrix_100k(items, item_map, num_items)
    elif dataset == ML_1M:
        vec = _genre_matrix_1m(items, item_map, num_items)
    elif dataset == RETAILROCKET:
        vec = _category_vectors(items, item_map, num_items)
    else:
        vec = _column_vectors(ratings, num_items, num_users)
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vec /= norms

    return RatingData(
        name=dataset,
        num_users=num_users,
        num_items=num_items,
        rmin=rmin,
        rmax=rmax,
        train_users=np.asarray(train_users, dtype=np.int64),
        val_users=np.asarray(val_users, dtype=np.int64),
        test_users=np.asarray(test_users, dtype=np.int64),
        user_ratings=user_ratings,
        item_vectors=vec,
    )
