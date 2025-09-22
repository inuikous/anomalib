#!/usr/bin/env python3
"""データセット管理デバッグテスト"""

from pathlib import Path
from . import *
from training_app.core.dataset_manager import DatasetManager

def test_dataset_debug():
    """データセット情報取得をデバッグ"""
    print("=== データセット管理デバッグテスト ===")
    
    print("DatasetManager初期化中...")
    dm = DatasetManager()
    
    print(f"データセットタイプ: {dm.dataset_type}")
    print(f"カテゴリ: {dm.category}")
    print(f"ベースパス: {dm.base_path}")
    print(f"カテゴリパス: {dm.category_path}")
    
    # パスの存在確認
    print(f"ベースパス存在: {dm.base_path.exists()}")
    print(f"カテゴリパス存在: {dm.category_path.exists()}")
    
    if dm.category_path.exists():
        print("カテゴリパス内容:")
        for item in dm.category_path.iterdir():
            print(f"  {item.name}")
    
    print("学習画像取得中...")
    try:
        train_images = dm._get_train_images()
        print(f"学習画像数: {len(train_images)}")
    except Exception as e:
        print(f"学習画像取得エラー: {e}")
        return False
    
    print("テスト用正常画像取得中...")
    try:
        test_good_images = dm._get_test_good_images()
        print(f"テスト用正常画像数: {len(test_good_images)}")
    except Exception as e:
        print(f"テスト用正常画像取得エラー: {e}")
        return False
    
    print("テスト用異常画像取得中...")
    try:
        test_defect_images, defect_types = dm._get_test_defect_images()
        print(f"テスト用異常画像数: {len(test_defect_images)}")
        print(f"異常タイプ: {defect_types}")
    except Exception as e:
        print(f"テスト用異常画像取得エラー: {e}")
        return False
    
    print("データセット情報取得中...")
    try:
        info = dm.get_dataset_info()
        print(f"データセット情報: {info.get_summary()}")
        return True
    except Exception as e:
        print(f"データセット情報取得エラー: {e}")
        return False

if __name__ == "__main__":
    success = test_dataset_debug()
    print(f"テスト結果: {'成功' if success else '失敗'}")