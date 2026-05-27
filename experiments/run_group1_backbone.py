#!/usr/bin/env python3
"""AAAI GROUP1 backbone comparison runner for PI-PHM.

This script re-implements GROUP1 with a protocol-aligned setup:
- same processed input features
- same split
- same training pipeline
- same threshold calibration / evaluator
- fair backbone swap only

Outputs:
- per-backbone checkpoint table
- selected-checkpoint summary table
- CSV files saved under outputs/group1_reimpl/
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PI_PHM_Config
from data.dataset import PhysicsAwareNormalizer, create_dataloaders
from evaluation.evaluator import PIPHMEvaluator
from models.ablation_patchtst_lstm import PIPHM_LSTM
from models.ablation_patchtst_transformer import PIPHM_Transformer
from models.group1_models import PIPHM_GRU_Full, PIPHM_PatchTSTOnly_Full
from models.pi_phm import PIPHM
from training.trainer import PI_PHM_Trainer
from utils.utils import set_seed


BACKBONES = {
    'PatchTST+LSTM': PIPHM_LSTM,
    'PatchTST+GRU': PIPHM_GRU_Full,
    'PatchTST+Mamba': PIPHM,
    'PatchTST+Transformer': PIPHM_Transformer,
    'PatchTST-only': PIPHM_PatchTSTOnly_Full,
}

CHECKPOINT_PRIORITY = ['best_event', 'best_multi', 'best_disp', 'last']


@dataclass
class RunArtifacts:
    workspace: Path
    outputs_dir: Path
    checkpoints_dir: Path


@contextlib.contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def ensure_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    os.symlink(src, dst, target_is_directory=src.is_dir())


def prepare_workspace(root: Path, backbone: str, clean: bool) -> RunArtifacts:
    safe_name = backbone.replace('+', '_plus_').replace(' ', '_').replace('-', '_')
    workspace = root / 'outputs' / 'group1_reimpl' / 'workspaces' / safe_name
    outputs_dir = workspace / 'outputs'
    checkpoints_dir = outputs_dir / 'checkpoints'

    if clean and workspace.exists():
        shutil.rmtree(workspace)

    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / 'group1').mkdir(parents=True, exist_ok=True)

    ensure_symlink(root / 'outputs' / 'data', outputs_dir / 'data')
    ensure_symlink(root / 'outputs' / 'normalizer_params.pkl', outputs_dir / 'normalizer_params.pkl')

    return RunArtifacts(workspace=workspace, outputs_dir=outputs_dir, checkpoints_dir=checkpoints_dir)


def load_processed_data(config: PI_PHM_Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PhysicsAwareNormalizer]:
    data_dir = Path('outputs/data')
    df_features = pd.read_parquet(data_dir / 'features_normalized.parquet')
    labels = pd.read_csv(data_dir / 'risk_labels.csv', index_col=0, parse_dates=True)

    quality_path = data_dir / 'quality_extended.parquet'
    if not quality_path.exists():
        quality_path = data_dir / 'quality_flags.parquet'
    df_quality = pd.read_parquet(quality_path)

    common_index = df_features.index.intersection(labels.index).intersection(df_quality.index)
    df_features = df_features.loc[common_index]
    labels = labels.loc[common_index]
    df_quality = df_quality.loc[common_index]

    normalizer = PhysicsAwareNormalizer(config)
    normalizer.load('outputs/normalizer_params.pkl')
    return df_features, df_quality, labels, normalizer


def build_feature_index_map(df_features_norm: pd.DataFrame) -> Dict[str, int | List[int]]:
    cols = df_features_norm.columns

    def idx(name: str) -> int:
        loc = cols.get_loc(name)
        if hasattr(loc, '__len__'):
            return int(loc[0])
        return int(loc)

    required = [
        'GNSS_12H_velocity', 'GNSS_12H_acceleration', 'GNSS_12H_inverse_velocity',
        'KH0206_Piezometer_rate', 'KH0112_Piezometer_rate', 'KH0117_Piezometer_rate',
        'KH0118_Piezometer_rate', 'KH0217_Piezometer_rate', 'KH0218_Piezometer_rate',
        'KH0306_Piezometer_rate', 'seismic_total_rate', 'rain_7d'
    ]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f'Missing required columns for feature_index_map: {missing}')

    return {
        'velocity_indices': idx('GNSS_12H_velocity'),
        'acceleration_indices': idx('GNSS_12H_acceleration'),
        'inverse_velocity_indices': idx('GNSS_12H_inverse_velocity'),
        'piezometer_rate_indices': [
            idx('KH0206_Piezometer_rate'), idx('KH0112_Piezometer_rate'), idx('KH0117_Piezometer_rate'),
            idx('KH0118_Piezometer_rate'), idx('KH0217_Piezometer_rate'), idx('KH0218_Piezometer_rate'),
            idx('KH0306_Piezometer_rate')
        ],
        'seismic_rate_indices': idx('seismic_total_rate'),
        'rain_7d_index': idx('rain_7d'),
    }


def build_model(backbone: str, config: PI_PHM_Config, feature_index_map: Dict, input_channels: int):
    model_cls = BACKBONES[backbone]
    return model_cls.from_config(config, feature_index_map, input_channels=input_channels)


def extract_metrics(eval_results: Dict, checkpoint_name: str, backbone: str, params: int) -> Dict:
    metrics = eval_results['metrics']
    disp = metrics['displacement']
    event = metrics['creep_burst_event']

    if 'detection_rate_f2' in event:
        detection_rate = float(event['detection_rate_f2'])
        lead = float(event['mean_lead_time_f2'])
        operating_threshold = float(event['operating_threshold_f2'])
    else:
        detection_rate = float(event.get('detection_rate', 0.0))
        lead = float(event.get('mean_lead_time', 0.0))
        operating_threshold = float(event.get('operating_threshold', 0.5))

    return {
        'backbone': backbone,
        'selected_checkpoint': checkpoint_name,
        'disp_MAE': float(disp['overall_mae']),
        'event_PR_AUC': float(event['pr_auc']),
        'event_detection_rate': detection_rate,
        'strict_FPR': float(event['strict_fpr']),
        'mean_lead_days': lead,
        'operating_threshold': operating_threshold,
        'parameters': int(params),
    }


def train_and_evaluate_backbone(backbone: str, config: PI_PHM_Config, report_checkpoint: str, device: str):
    artifacts = prepare_workspace(PROJECT_ROOT, backbone, clean=True)

    with pushd(artifacts.workspace):
        set_seed(getattr(config, 'seed', 42))
        df_features, df_quality, labels, normalizer = load_processed_data(config)
        feature_index_map = build_feature_index_map(df_features)
        train_loader, val_loader, test_loader = create_dataloaders(
            df_features, df_quality, labels, config, getattr(config, 'event_aware_split', False), normalizer=normalizer
        )

        model = build_model(backbone, config, feature_index_map, input_channels=df_features.shape[1])
        model = model.to(device)
        params = sum(p.numel() for p in model.parameters())

        trainer = PI_PHM_Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=torch.device(device),
            feature_index_map=feature_index_map,
            normalizer=normalizer,
        )
        history = trainer.fit()

        evaluator = PIPHMEvaluator(config)
        rows = []
        checkpoint_rows = []
        for ckpt in CHECKPOINT_PRIORITY:
            ckpt_path = Path('outputs/checkpoints') / f'{ckpt}.pth'
            if not ckpt_path.exists():
                continue
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(state['model_state_dict'])
            eval_results = evaluator.evaluate(model, test_loader, normalizer, device=device, checkpoint_path=str(ckpt_path))
            row = extract_metrics(eval_results, ckpt, backbone, params)
            checkpoint_rows.append(row)
            if ckpt == report_checkpoint:
                rows.append(row)

        if not rows and checkpoint_rows:
            rows = [checkpoint_rows[0]]

        pd.DataFrame(checkpoint_rows).to_csv(Path('outputs/group1') / 'checkpoint_metrics.csv', index=False)
        with open(Path('outputs/group1') / 'train_history_meta.json', 'w', encoding='utf-8') as f:
            json.dump({'epochs_ran': len(history['epochs']), 'report_checkpoint': report_checkpoint}, f, indent=2)

        return rows[0] if rows else None, checkpoint_rows, artifacts.workspace


def main():
    parser = argparse.ArgumentParser(description='Re-implement GROUP1 backbone comparison for PI-PHM')
    parser.add_argument('--config', type=str, default='config_final_correct.yaml')
    parser.add_argument('--backbones', nargs='+', default=list(BACKBONES.keys()), choices=list(BACKBONES.keys()))
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--report-checkpoint', type=str, default='best_event', choices=CHECKPOINT_PRIORITY)
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--min-epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    args = parser.parse_args()

    config = PI_PHM_Config.from_yaml(args.config)
    if args.max_epochs is not None:
        config.training.max_epochs = args.max_epochs
    if args.min_epochs is not None:
        config.training.min_epochs = args.min_epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size

    summary_rows: List[Dict] = []
    detailed_rows: List[Dict] = []
    workspace_map: Dict[str, str] = {}

    for backbone in args.backbones:
        row, ckpt_rows, workspace = train_and_evaluate_backbone(
            backbone=backbone,
            config=config,
            report_checkpoint=args.report_checkpoint,
            device=args.device,
        )
        if row is not None:
            summary_rows.append(row)
        detailed_rows.extend(ckpt_rows)
        workspace_map[backbone] = str(workspace)

    out_root = PROJECT_ROOT / 'outputs' / 'group1_reimpl'
    out_root.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    detailed_df = pd.DataFrame(detailed_rows)

    if not summary_df.empty:
        summary_df.to_csv(out_root / f'group1_summary_{args.report_checkpoint}.csv', index=False)
        print('\n=== GROUP1 summary ===')
        print(summary_df.to_string(index=False))
    if not detailed_df.empty:
        detailed_df.to_csv(out_root / 'group1_all_checkpoints.csv', index=False)

    with open(out_root / 'workspace_map.json', 'w', encoding='utf-8') as f:
        json.dump(workspace_map, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
