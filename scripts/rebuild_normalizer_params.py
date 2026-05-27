#!/usr/bin/env python3
"""Rebuild normalizer_params.pkl with increment_scalers enabled.

Use this after applying the project-level normalization fixes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PI_PHM_Config
from data.dataset import PhysicsAwareNormalizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_final_correct.yaml')
    parser.add_argument('--features-raw', type=str, default='outputs/data/features_raw.parquet')
    parser.add_argument('--output', type=str, default='outputs/normalizer_params.pkl')
    args = parser.parse_args()

    config = PI_PHM_Config.from_yaml(args.config)
    features_path = PROJECT_ROOT / args.features_raw
    output_path = PROJECT_ROOT / args.output

    df_features = pd.read_parquet(features_path)
    train_end = pd.to_datetime(config.data.train_end)
    train_mask = df_features.index <= train_end

    normalizer = PhysicsAwareNormalizer(config)
    normalizer.fit(df_features, train_mask=train_mask)
    normalizer.save(str(output_path))

    print(f'Rebuilt normalizer params with {len(normalizer.increment_scalers)} increment scalers -> {output_path}')


if __name__ == '__main__':
    main()
