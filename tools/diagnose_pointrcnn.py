#!/usr/bin/env python3

import sys
import os
sys.path.append('/root/autodl-tmp/OpenPCDet')

import torch
import numpy as np
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

def diagnose_pointrcnn_test():
    # 加载配置
    config_file = '/root/autodl-tmp/OpenPCDet/tools/cfgs/nuscenes_models/pointrcnn.yaml'
    cfg_from_yaml_file(config_file, cfg)
    
    print("=== Configuration Diagnosis ===")
    print(f"Model Name: {cfg.MODEL.NAME}")
    print(f"Class Names: {cfg.CLASS_NAMES}")
    print(f"Score Threshold: {cfg.MODEL.POST_PROCESSING.SCORE_THRESH}")
    print(f"NMS Threshold: {cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH}")
    
    # 检查数据集
    print("\n=== Dataset Diagnosis ===")
    try:
        test_set, test_loader, sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=1,
            dist=False, workers=1, training=False
        )
        print(f"Test dataset size: {len(test_set)}")
        print(f"First sample keys: {list(test_set[0].keys()) if len(test_set) > 0 else 'No samples'}")
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        return
    
    # 检查模型
    print("\n=== Model Diagnosis ===")
    try:
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
        print(f"Model created successfully: {type(model)}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    except Exception as e:
        print(f"Model creation failed: {e}")
        return
    
    # 检查一个样本的前向传播
    print("\n=== Forward Pass Diagnosis ===")
    try:
        model.eval()
        sample = test_set[0]
        batch_dict = test_set.collate_batch([sample])
        
        print(f"Input batch keys: {list(batch_dict.keys())}")
        print(f"Points shape: {batch_dict.get('points', 'No points').shape if 'points' in batch_dict else 'No points'}")
        
        with torch.no_grad():
            pred_dicts, _ = model(batch_dict)
        
        if pred_dicts and len(pred_dicts) > 0:
            pred = pred_dicts[0]
            print(f"Prediction keys: {list(pred.keys())}")
            if 'pred_boxes' in pred:
                print(f"Predicted boxes shape: {pred['pred_boxes'].shape}")
                print(f"Number of predictions: {pred['pred_boxes'].shape[0]}")
            if 'pred_scores' in pred:
                print(f"Score range: [{pred['pred_scores'].min():.4f}, {pred['pred_scores'].max():.4f}]")
                high_score_count = (pred['pred_scores'] > cfg.MODEL.POST_PROCESSING.SCORE_THRESH).sum()
                print(f"Predictions above threshold ({cfg.MODEL.POST_PROCESSING.SCORE_THRESH}): {high_score_count}")
        else:
            print("No predictions returned!")
            
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_pointrcnn_test()
