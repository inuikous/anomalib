#!/usr/bin/env python3
"""統合テストランナー"""

import sys
from pathlib import Path
from . import *

def run_all_tests():
    """すべてのテストを実行"""
    print("="*60)
    print(" AI異常検知システム - 統合テストスイート")
    print("="*60)
    
    test_results = {}
    
    # 1. 初期化テスト
    print("\n1. 初期化テスト実行中...")
    try:
        from .test_initialization import test_initialization
        test_results['initialization'] = test_initialization()
    except Exception as e:
        print(f"初期化テストエラー: {e}")
        test_results['initialization'] = False
    
    # 2. データセットテスト
    print("\n2. データセットテスト実行中...")
    try:
        from .test_dataset import test_dataset_debug
        test_results['dataset'] = test_dataset_debug()
    except Exception as e:
        print(f"データセットテストエラー: {e}")
        test_results['dataset'] = False
    
    # 3. モデル作成テスト
    print("\n3. モデル作成テスト実行中...")
    try:
        from .test_model import test_model_creation
        test_results['model'] = test_model_creation()
    except Exception as e:
        print(f"モデル作成テストエラー: {e}")
        test_results['model'] = False
    
    # 4. 学習テスト（軽量版）
    print("\n4. 学習テスト実行中...")
    try:
        from .test_training import test_training
        test_results['training'] = test_training()
    except Exception as e:
        print(f"学習テストエラー: {e}")
        test_results['training'] = False
    
    # 5. GUIテスト
    print("\n5. GUIテスト実行中...")
    try:
        from .test_gui import test_gui_applications
        test_results['gui'] = test_gui_applications()
    except Exception as e:
        print(f"GUIテストエラー: {e}")
        test_results['gui'] = False
    
    # 結果サマリー
    print("\n" + "="*60)
    print(" テスト結果サマリー")
    print("="*60)
    
    all_passed = True
    for test_name, result in test_results.items():
        status = "成功" if result else "失敗"
        print(f"{test_name:15}: {status}")
        if not result:
            all_passed = False
    
    print("-"*60)
    print(f"総合結果: {'全テスト成功' if all_passed else '一部テスト失敗'}")
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)