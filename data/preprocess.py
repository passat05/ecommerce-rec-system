import json
import logging
from pathlib import Path

import duckdb
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Config
def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

class RetailrocketPreprocessor:
    """End-to-end DuckDB preprocessing pipeline for the Retailrocket dataset."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.con = duckdb.connect()          # in-memory; DuckDB streams files
        self.out = Path(cfg["data"]["processed_path"])
        self.out.mkdir(parents=True, exist_ok=True)

    # Step 1: Load events

    def load_events(self) -> None:
        """
        Read events.csv into a DuckDB table.
        Timestamp is converted from milliseconds to TIMESTAMPTZ.
        event_weight encodes the signal strength (view < addtocart < transaction).
        """
        path = self.cfg["data"]["events_path"]
        log.info("Loading events via DuckDB: %s", path)

        self.con.execute(f"""
            CREATE OR REPLACE TABLE events AS
            SELECT
                visitorid,
                itemid,
                event,
                transactionid,
                to_timestamp(CAST(timestamp AS DOUBLE) / 1000.0) AS timestamp,
                CASE event
                    WHEN 'view'        THEN 1
                    WHEN 'addtocart'   THEN 2
                    WHEN 'transaction' THEN 3
                    ELSE 1
                END AS event_weight,
                CASE event
                    WHEN 'view'        THEN 0
                    WHEN 'addtocart'   THEN 1
                    WHEN 'transaction' THEN 2
                    ELSE 0
                END AS event_type,
                CASE WHEN event IN ('addtocart', 'transaction') THEN 1
                     ELSE 0
                END AS label
            FROM read_csv_auto('{path}')
        """)

        n = self.con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        log.info("Events loaded: %d rows", n)

    # Step 2: Build item meta

    def build_item_meta(self) -> None:
        """
        Read both item_properties files, keep the latest snapshot per
        (itemid, property) via ROW_NUMBER(), then pivot categoryid and
        price_bucket.
        """
        part1 = self.cfg["data"]["item_props_part1_path"]
        part2 = self.cfg["data"]["item_props_part2_path"]
        log.info("Building item meta from property files …")

        # Union both files and keep latest snapshot per (itemid, property)
        self.con.execute(f"""
            CREATE OR REPLACE TABLE latest_props AS
            SELECT itemid, property, value
            FROM (
                SELECT
                    itemid,
                    property,
                    value,
                    ROW_NUMBER() OVER (
                        PARTITION BY itemid, property
                        ORDER BY timestamp DESC
                    ) AS rn
                FROM (
                    SELECT * FROM read_csv_auto('{part1}')
                    UNION ALL
                    SELECT * FROM read_csv_auto('{part2}')
                )
            )
            WHERE rn = 1
        """)

        log.info("Latest props computed.")

        # Pivot: categoryid and price_raw as separate columns per item
        self.con.execute("""
            CREATE OR REPLACE TABLE item_meta_raw AS
            SELECT
                itemid,
                MAX(CASE WHEN property = 'category' THEN value END)
                    AS categoryid_raw,
                MAX(CASE WHEN property = '790'
                         THEN TRY_CAST(REPLACE(value, 'n', '') AS DOUBLE) END)
                    AS price_raw
            FROM latest_props
            GROUP BY itemid
        """)

        # Map price into buckets and coalesce null values
        self.con.execute("""
            CREATE OR REPLACE TABLE item_meta_clean AS
            SELECT
                itemid,
                COALESCE(
                    TRY_CAST(categoryid_raw AS INTEGER),
                    -1
                ) AS categoryid,
                COALESCE(
                    NTILE(10) OVER (
                        ORDER BY price_raw NULLS LAST
                    ) - 1,
                    0
                ) AS price_bucket
            FROM item_meta_raw
        """)

        # Save item_meta.parquet
        df = self.con.execute(
            "SELECT itemid, categoryid, price_bucket FROM item_meta_clean"
        ).df()
        df["categoryid"]   = df["categoryid"].astype("int32")
        df["price_bucket"] = df["price_bucket"].fillna(0).astype("int8")
        df.set_index("itemid").to_parquet(self.cfg["data"]["item_meta_path"])
        log.info("item_meta.parquet saved: %d items", len(df))

    # Step 3: Build interactions

    def build_interactions(self) -> None:
        """
        Dedup, filter cold users, join item features, and compute recency.
        """
        min_n = self.cfg["data"]["min_interactions"]
        log.info("Building interactions (min_interactions=%d) …", min_n)

        # Dedup: keep highest-weight event per (visitor, item)
        self.con.execute("""
            CREATE OR REPLACE TABLE dedup_events AS
            SELECT *
            FROM (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY visitorid, itemid
                        ORDER BY event_weight DESC
                    ) AS rn
                FROM events
            )
            WHERE rn = 1
        """)

        # Encode user IDs (sorted for determinism)
        self.con.execute("""
            CREATE OR REPLACE TABLE user_map AS
            SELECT
                visitorid,
                ROW_NUMBER() OVER (ORDER BY visitorid) - 1 AS user_id
            FROM (
                SELECT DISTINCT visitorid
                FROM dedup_events
                WHERE visitorid IN (
                    SELECT visitorid
                    FROM dedup_events
                    GROUP BY visitorid
                    HAVING COUNT(*) >= {min_n}
                )
            )
        """.format(min_n=min_n))

        # Encode item IDs (sorted for determinism)
        self.con.execute("""
            CREATE OR REPLACE TABLE item_map AS
            SELECT
                itemid,
                ROW_NUMBER() OVER (ORDER BY itemid) - 1 AS item_id
            FROM (SELECT DISTINCT itemid FROM dedup_events)
        """)

        n_users = self.con.execute("SELECT COUNT(*) FROM user_map").fetchone()[0]
        n_items = self.con.execute("SELECT COUNT(*) FROM item_map").fetchone()[0]
        log.info("Users: %d  |  Items: %d", n_users, n_items)

        # Featuring interaction data
        self.con.execute("""
            CREATE OR REPLACE TABLE interactions AS
            SELECT
                u.user_id,
                i.item_id,
                e.label,
                e.timestamp,
                COALESCE(m.categoryid,   -1) AS categoryid,
                COALESCE(m.price_bucket,  0) AS price_bucket,
                e.event_type,
                ROUND(
                    EXTRACT(EPOCH FROM (max_ts.ts - e.timestamp)) / 86400.0,
                    1
                ) AS recency_days
            FROM dedup_events e
            JOIN user_map u ON e.visitorid = u.visitorid
            JOIN item_map i ON e.itemid    = i.itemid
            LEFT JOIN item_meta_clean m ON e.itemid = m.itemid
            CROSS JOIN (
                SELECT MAX(timestamp) AS ts FROM dedup_events
            ) AS max_ts
        """)

        n = self.con.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
        log.info("Interactions after cold-user filter: %d", n)

    # Step 4: Encode categories

    def encode_categories(self) -> dict:
        """
        Map raw categoryid values to sequential integers 0..N.
        """
        log.info("Encoding category IDs …")

        self.con.execute("""
            CREATE OR REPLACE TABLE cat_map AS
            SELECT
                categoryid AS raw_cat,
                CAST(DENSE_RANK() OVER (ORDER BY categoryid) - 1 AS INTEGER)
                    AS category_id
            FROM (SELECT DISTINCT categoryid FROM interactions)
        """)

        # Apply encoding back to interactions
        self.con.execute("""
            CREATE OR REPLACE TABLE interactions AS
            SELECT
                i.user_id,
                i.item_id,
                i.label,
                i.timestamp,
                c.category_id,
                i.price_bucket,
                i.event_type,
                i.recency_days
            FROM interactions i
            LEFT JOIN cat_map c ON i.categoryid = c.raw_cat
        """)

        cat_rows = self.con.execute(
            "SELECT raw_cat, category_id FROM cat_map ORDER BY category_id"
        ).fetchall()
        cat2idx = {int(raw): int(idx) for raw, idx in cat_rows}

        n_cats = len(cat2idx)
        log.info("Categories encoded: %d unique values → 0..%d", n_cats, n_cats - 1)
        return cat2idx

    # Step 5: User history

    def build_user_history(self) -> pd.DataFrame:
        """
        Build last-K item sequence per user.
        """
        k = self.cfg["data"]["history_len"]
        log.info("Building user history (last K=%d, oldest→newest) …", k)

        df = self.con.execute(f"""
            SELECT
                user_id,
                LIST(item_id ORDER BY timestamp ASC)[-{k}:] AS history
            FROM interactions
            GROUP BY user_id
        """).df()

        log.info("User history built: %d users", len(df))
        return df

    # Step 6: Time-based split

    def split(self) -> tuple:
        """Chronological 70/15/15 split. Popularity computed from train only."""
        log.info("Splitting 70/15/15 by timestamp …")

        df = self.con.execute("""
            SELECT
                user_id, item_id, label, timestamp,
                category_id, price_bucket, event_type, recency_days
            FROM interactions
            ORDER BY timestamp
        """).df()

        n  = len(df)
        t1 = int(n * 0.70)
        t2 = int(n * 0.85)

        train = df.iloc[:t1].copy()
        val   = df.iloc[t1:t2].copy()
        test  = df.iloc[t2:].copy()

        # Popularity from train only (avoids leakage into val/test)
        pop = train.groupby("item_id").size().rename("n_views").reset_index()
        train = train.merge(pop, on="item_id", how="left")
        val   = val.merge(pop,   on="item_id", how="left")
        test  = test.merge(pop,  on="item_id", how="left")
        for df_ in (train, val, test):
            df_["n_views"] = df_["n_views"].fillna(0).astype("int32")

        log.info("Train: %d  Val: %d  Test: %d", len(train), len(val), len(test))
        return train, val, test

    # Step 7: Save ID maps

    def save_id_maps(self, cat2idx: dict) -> None:
        """
        Save the three ID maps that downstream scripts expect.
        Keys are stored as strings for JSON compatibility.
        """
        # user_id_map: original visitorid (int) → integer index
        user_rows = self.con.execute(
            "SELECT visitorid, user_id FROM user_map ORDER BY user_id"
        ).fetchall()
        user2idx = {str(v): int(u) for v, u in user_rows}

        # item_id_map: original itemid (int) → integer index
        item_rows = self.con.execute(
            "SELECT itemid, item_id FROM item_map ORDER BY item_id"
        ).fetchall()
        item2idx = {str(i): int(u) for i, u in item_rows}

        with open(self.cfg["data"]["user_id_map_path"], "w") as f:
            json.dump(user2idx, f)
        with open(self.cfg["data"]["item_id_map_path"], "w") as f:
            json.dump(item2idx, f)
        with open(self.cfg["data"]["category_map_path"], "w") as f:
            # FIX BUG4: category_map.json now saved
            json.dump({str(k): v for k, v in cat2idx.items()}, f)

        log.info(
            "ID maps saved — users: %d  items: %d  categories: %d",
            len(user2idx), len(item2idx), len(cat2idx),
        )

    # Main pipeline

    def run(self):
        """Execute all preprocessing steps in order."""

        # Step 1 — events
        self.load_events()

        # Step 2 — item meta (saved to parquet immediately)
        self.build_item_meta()

        # Step 3 — interactions (dedup + filter + join + recency inline)
        self.build_interactions()

        # Step 4 — encode categories
        cat2idx = self.encode_categories()

        # Step 5 — user history
        history = self.build_user_history()

        # Step 6 — chronological split
        train, val, test = self.split()

        # Persist splits
        train.to_parquet(self.cfg["data"]["train_path"], index=False)
        val.to_parquet(  self.cfg["data"]["val_path"], index=False)
        test.to_parquet( self.cfg["data"]["test_path"], index=False)

        # Persist user history
        history.to_parquet(self.cfg["data"]["user_history_path"], index=False)

        # Step 7 — ID maps (FIX BUG4: category_map.json included)
        self.save_id_maps(cat2idx)

        log.info("Preprocessing complete. Artefacts → %s", self.out)
        return train, val, test

if __name__ == "__main__":
    cfg = load_config()
    RetailrocketPreprocessor(cfg).run()