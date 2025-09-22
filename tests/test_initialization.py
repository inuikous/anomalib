#!/usr/bin/env python3
"""システム初期化テスト"""

import sys
from pathlib import Path

# テストパッケージの初期化
from . import *

from training_app.core.dataset_manager import DatasetManager
from training_app.core.training_manager import TrainingManager

def test_initialization():
    """初期化テスト"""
    print("=== システム初期化テスト ===")
    
    # データセット管理初期化
    print("データセット管理を初期化中...")
    dm = DatasetManager()
    
    # データセット情報取得
    print("データセット情報を取得中...")
    info = dm.get_dataset_info()
    
    if info and info.is_valid:
        print(f"データセット情報: {info.get_summary()}")
    else:
        print("データセット情報の取得に失敗しました")
        return False
    
    # 学習管理初期化
    print("学習管理を初期化中...")
    tm = TrainingManager()
    print("学習管理初期化完了")
    
    return True

if __name__ == "__main__":
    success = test_initialization()
    print(f"初期化テスト: {'成功' if success else '失敗'}")