#!/usr/bin/env python3
"""Verify whether the project-level fixes were correctly synced/applied.

Checks:
1. trainer contains scheduler/monitor/weighted-loss instrumentation
2. dataset contains normalizer pass-through + increment scaler serialization
3. main / group1 runner use normalizer correctly
4. normalizer_params.pkl contains increment_scalers
5. event catalog loads Table_S2.csv
"""

from __future__ import annotations

import inspect
import pickle
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_code_tokens(path: Path, tokens: list[str]):
    text = path.read_text(encoding='utf-8')
    return {tok: (tok in text) for tok in tokens}


def main():
    trainer_tokens = [
        'val/monitor_score',
        'train/weighted_loss_event',
        'train/phase',
        'def _step_learning_rate',
        "val_monitor_score = score_multi",
        "self.early_stopping(val_metrics['val_monitor_score'])",
    ]
    dataset_tokens = [
        'increment_scalers',
        'inverse_transform_single_feature',
        'normalizer=normalizer',
        'def create_dataloaders(',
    ]
    main_tokens = [
        'features_raw.parquet',
        'checkpoint_path=',
        'normalizer=normalizer',
    ]
    runner_tokens = ['normalizer=normalizer']

    print('=== code checks ===')
    for name, path, tokens in [
        ('trainer.py', ROOT / 'training' / 'trainer.py', trainer_tokens),
        ('dataset.py', ROOT / 'data' / 'dataset.py', dataset_tokens),
        ('main.py', ROOT / 'main.py', main_tokens),
        ('run_group1_backbone.py', ROOT / 'experiments' / 'run_group1_backbone.py', runner_tokens),
    ]:
        print(f'[{name}]')
        result = check_code_tokens(path, tokens)
        for k, v in result.items():
            print(f'  {k}: {v}')

    print('\n=== runtime import path checks ===')
    import training.trainer as trainer_mod
    import data.dataset as dataset_mod
    import data.event_catalog as event_mod
    print('trainer module path:', trainer_mod.__file__)
    print('dataset module path:', dataset_mod.__file__)
    print('event_catalog module path:', event_mod.__file__)

    print('\n=== normalizer pickle check ===')
    pkl_path = ROOT / 'outputs' / 'normalizer_params.pkl'
    if not pkl_path.exists():
        print('normalizer_params.pkl: MISSING')
    else:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        keys = sorted(data.keys())
        print('keys:', keys)
        inc = data.get('increment_scalers', {})
        print('increment_scalers count:', len(inc))

    print('\n=== event catalog check ===')
    from data.event_catalog import CreepBurstCatalog
    cat = CreepBurstCatalog()
    print('loaded events:', len(cat.events))
    stats = cat.get_statistics()
    print('severity counts:', stats.get('severity_counts', {}))


if __name__ == '__main__':
    main()
