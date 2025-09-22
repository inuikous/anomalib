#!/usr/bin/env python3
"""GUIアプリケーション起動テスト"""

import sys
from pathlib import Path
import tkinter as tk
from . import *

def test_training_gui():
    """学習GUIのテスト（初期化のみ）"""
    print("学習GUI初期化テスト開始...")
    
    try:
        from training_app.main import TrainingApp
        
        print("TrainingAppクラスの初期化...")
        app = TrainingApp()
        
        print("学習GUI初期化成功")
        
        # コンポーネントが正しく初期化されているかチェック
        if (hasattr(app, 'dataset_manager') and app.dataset_manager and
            hasattr(app, 'training_manager') and app.training_manager and
            hasattr(app, 'model_manager') and app.model_manager):
            print("学習GUIコンポーネント初期化確認完了")
            return True
        else:
            print("学習GUIコンポーネント初期化失敗")
            return False
        
    except Exception as e:
        print(f"学習GUI初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inference_gui():
    """推論GUIのテスト（初期化のみ）"""
    print("推論GUI初期化テスト開始...")
    
    try:
        from inference_app.main import InferenceApp
        
        print("InferenceAppクラスの初期化...")
        app = InferenceApp()
        
        print("推論GUI初期化成功")
        
        # コンポーネントが正しく初期化されているかチェック
        if (hasattr(app, 'image_manager') and app.image_manager and
            hasattr(app, 'anomaly_detector') and app.anomaly_detector and
            hasattr(app, 'result_manager') and app.result_manager):
            print("推論GUIコンポーネント初期化確認完了")
            return True
        else:
            print("推論GUIコンポーネント初期化失敗")
            return False
        
    except Exception as e:
        print(f"推論GUI初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_applications():
    """GUIアプリケーション統合テスト"""
    print("=== GUIアプリケーション初期化テスト ===")
    
    # 学習GUIテスト
    training_success = test_training_gui()
    print(f"学習GUI: {'成功' if training_success else '失敗'}")
    
    print("\n" + "="*50 + "\n")
    
    # 推論GUIテスト
    inference_success = test_inference_gui()
    print(f"推論GUI: {'成功' if inference_success else '失敗'}")
    
    print(f"\n=== 最終結果 ===")
    print(f"学習GUI: {'成功' if training_success else '失敗'}")
    print(f"推論GUI: {'成功' if inference_success else '失敗'}")
    print(f"総合: {'成功' if training_success and inference_success else '失敗'}")
    
    return training_success and inference_success

if __name__ == "__main__":
    success = test_gui_applications()
    sys.exit(0 if success else 1)