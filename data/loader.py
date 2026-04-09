"""
STPA - Data Loader
===================
Loads and preprocesses credit datasets for STPA calibration.

Supported datasets:
    - UCI Credit Card Default (Taiwan dataset)
    - Lending Club loan data
    - Synthetic data generator (for testing)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple


DATA_DIR = Path(__file__).parent / "raw"

# ── Column mappings ────────────────────────────────────────────────────────────

UCI_COLUMN_MAP = {
    "LIMIT_BAL": "credit_limit",
    "SEX": "gender",
    "EDUCATION": "education",
    "MARRIAGE": "marital_status",
    "AGE": "age",
    "PAY_0": "pay_status_sep",
    "PAY_2": "pay_status_aug",
    "PAY_3": "pay_status_jul",
    "PAY_4": "pay_status_jun",
    "PAY_5": "pay_status_may",
    "PAY_6": "pay_status_apr",
    "BILL_AMT1": "bill_sep",
    "BILL_AMT2": "bill_aug",
    "BILL_AMT3": "bill_jul",
    "BILL_AMT4": "bill_jun",
    "BILL_AMT5": "bill_may",
    "BILL_AMT6": "bill_apr",
    "PAY_AMT1": "paid_sep",
    "PAY_AMT2": "paid_aug",
    "PAY_AMT3": "paid_jul",
    "PAY_AMT4": "paid_jun",
    "PAY_AMT5": "paid_may",
    "PAY_AMT6": "paid_apr",
    "default.payment.next.month": "default",
}


class DataLoader:
    """
    Loads and preprocesses credit data for STPA parameter calibration.

    Usage:
        loader = DataLoader()
        df = loader.load_uci()
        train, test = loader.split(df)
    """

    def load_uci(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load UCI Credit Card Default dataset.
        Download from: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
        """
        path = filepath or DATA_DIR / "UCI_Credit_Card.csv"
        df = pd.read_csv(path)
        df = df.rename(columns=UCI_COLUMN_MAP)
        df = self._clean_uci(df)
        return df

    def _clean_uci(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and feature-engineer the UCI dataset."""
        # Utilization ratio
        bill_cols = ["bill_sep", "bill_aug", "bill_jul", "bill_jun", "bill_may", "bill_apr"]
        df["avg_bill"] = df[bill_cols].mean(axis=1)
        df["utilization"] = (df["avg_bill"] / df["credit_limit"].replace(0, np.nan)).clip(0, 1)

        # Payment behavior score
        pay_cols = ["pay_status_sep", "pay_status_aug", "pay_status_jul",
                    "pay_status_jun", "pay_status_may", "pay_status_apr"]
        df["avg_pay_delay"] = df[pay_cols].clip(-1, 8).mean(axis=1)

        # Compute a synthetic health score (0–100)
        df["health_score"] = (
            100
            - (df["utilization"] * 30)
            - (df["avg_pay_delay"].clip(0, 5) * 8)
            - (df["default"] * 20)
        ).clip(0, 100)

        return df.dropna()

    def load_lending_club(self, filepath: str) -> pd.DataFrame:
        """Load Lending Club loan data (requires manual download)."""
        df = pd.read_csv(filepath, low_memory=False)
        df["default"] = (df["loan_status"].isin(["Charged Off", "Default"])).astype(int)
        df["health_score"] = pd.to_numeric(df["fico_range_low"], errors="coerce").fillna(600)
        df["health_score"] = ((df["health_score"] - 300) / 550 * 100).clip(0, 100)
        return df

    def generate_synthetic(
        self,
        n_samples: int = 5000,
        default_rate: float = 0.22,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate synthetic credit data for testing STPA pipelines.
        Mimics UCI-style distributions.
        """
        np.random.seed(seed)

        health_scores = np.random.beta(5, 3, n_samples) * 100
        defaults = (health_scores < np.percentile(health_scores, default_rate * 100)).astype(int)

        return pd.DataFrame({
            "borrower_id": [f"B{i:05d}" for i in range(n_samples)],
            "health_score": health_scores.round(2),
            "credit_limit": np.random.lognormal(10.5, 0.8, n_samples).round(-2),
            "age": np.random.randint(22, 65, n_samples),
            "utilization": np.random.beta(2, 5, n_samples).round(4),
            "avg_pay_delay": np.random.exponential(0.5, n_samples).round(2),
            "default": defaults,
        })

    def split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split into train/test sets."""
        np.random.seed(seed)
        idx = np.random.permutation(len(df))
        split = int(len(df) * (1 - test_size))
        train = df.iloc[idx[:split]].reset_index(drop=True)
        test = df.iloc[idx[split:]].reset_index(drop=True)
        return train, test
