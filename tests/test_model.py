#!/usr/bin/env python3
"""モデル作成テスト"""

import torch
from . import *
from training_app.core.training_manager import TrainingManager

def test_model_creation():
    """モデル作成のみをテスト"""
    print("=== モデル作成テスト ===")
    
    try:
        tm = TrainingManager()
        print("TrainingManager初期化完了")
        
        print("モデル作成中...")
        model = tm.create_model()
        print(f"モデル作成成功: {type(model).__name__}")
        
        # モデルの基本情報
        total_params = sum(p.numel() for p in model.parameters())
        print(f"パラメータ数: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"モデル作成エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_creation()
    print(f"結果: {'成功' if success else '失敗'}")