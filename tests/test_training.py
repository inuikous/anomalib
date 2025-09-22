#!/usr/bin/env python3
"""学習テストスクリプト"""

import torch
from pathlib import Path
from . import *
from training_app.core.dataset_manager import DatasetManager
from training_app.core.training_manager import TrainingManager
from training_app.core.model_manager import ModelManager

def test_training():
    """学習テスト実行"""
    print("=== 学習テスト開始 ===")
    
    # 1. データセット管理初期化
    print("\n1. データセット管理初期化...")
    dm = DatasetManager()
    
    # データセット情報確認
    info = dm.get_dataset_info()
    if not info or not info.is_valid:
        print("データセット検証失敗")
        return False
    
    print(f"データセット情報: {info.get_summary()}")
    
    # 2. 学習管理初期化
    print("\n2. 学習管理初期化...")
    tm = TrainingManager()
    
    # 3. モデル管理初期化
    print("\n3. モデル管理初期化...")
    mm = ModelManager()
    
    # 4. モデル作成テスト
    print("\n4. モデル作成テスト...")
    try:
        model = tm.create_model()
        print(f"モデル作成成功: {type(model).__name__}")
    except Exception as e:
        print(f"モデル作成失敗: {e}")
        return False
    
    # 5. データセット作成テスト
    print("\n5. データセット作成テスト...")
    try:
        datamodule = tm.create_datamodule()
        print(f"データモジュール作成成功: {type(datamodule).__name__}")
    except Exception as e:
        print(f"データモジュール作成失敗: {e}")
        return False
    
    # 6. 軽量学習テスト（基本テストのみ）
    print("\n6. 学習開始テスト...")
    try:
        # 学習開始テスト（実際の学習は行わない）
        print("学習開始可能性チェック...")
        
        # 学習に必要なコンポーネントが正常に初期化されているかテスト
        if tm.is_training:
            print("学習状態エラー: 既に学習中")
            return False
        
        print("学習コンポーネント準備完了")
        print("軽量学習テスト成功（実際の学習はスキップ）")
        return True
            
    except Exception as e:
        print(f"学習テストエラー: {e}")
        return False

if __name__ == "__main__":
    success = test_training()
    print(f"\n=== 学習テスト結果: {'成功' if success else '失敗'} ===")