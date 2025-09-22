"""学習アプリメイン"""

import sys
import os
from pathlib import Path

# Windows対応: multiprocessing設定を最初に設定
import multiprocessing
if sys.platform.startswith('win'):
    multiprocessing.set_start_method('spawn', force=True)

# PyTorch DLL問題対応
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# パス設定
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "shared"))

import tkinter as tk
from tkinter import messagebox

from shared.utils import setup_logger
from training_app.gui.main_window import TrainingMainWindow
from training_app.core.dataset_manager import DatasetManager
from training_app.core.training_manager import TrainingManager
from training_app.core.model_manager import ModelManager


class TrainingApp:
    """学習アプリケーション"""
    
    def __init__(self):
        """初期化"""
        self.logger = setup_logger("training_app")
        
        # GUI
        self.root = None
        self.main_window = None
        
        # コアコンポーネント
        self.dataset_manager = None
        self.training_manager = None
        self.model_manager = None
        
        self.setup_components()
    
    def setup_components(self):
        """コンポーネント初期化"""
        try:
            # コアコンポーネント初期化
            self.dataset_manager = DatasetManager()
            self.training_manager = TrainingManager()
            self.model_manager = ModelManager()
            
            self.logger.info("学習アプリ初期化完了")
            
        except Exception as e:
            self.logger.error(f"学習アプリ初期化エラー: {e}")
            raise
    
    def run(self):
        """アプリケーション実行"""
        try:
            # Tkinterルート作成
            self.root = tk.Tk()
            self.root.title("異常検出システム - 学習")
            self.root.geometry("1000x700")
            
            # アイコン設定（存在する場合）
            try:
                self.root.iconbitmap(default="app.ico")
            except:
                pass  # アイコンがない場合は無視
            
            # メインウィンドウ作成
            self.main_window = TrainingMainWindow(
                self.root,
                self.dataset_manager,
                self.training_manager,
                self.model_manager
            )
            
            # 終了時処理設定
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            self.logger.info("学習アプリ開始")
            
            # イベントループ開始
            self.root.mainloop()
            
        except Exception as e:
            self.logger.error(f"学習アプリ実行エラー: {e}")
            messagebox.showerror("エラー", f"アプリケーション実行エラー:\n{e}")
        finally:
            self.cleanup()
    
    def on_closing(self):
        """終了時処理"""
        try:
            # 学習中チェック
            if hasattr(self.main_window, 'is_training') and self.main_window.is_training:
                if not messagebox.askokcancel("終了確認", "学習が実行中です。終了しますか？"):
                    return
            
            # 確認ダイアログ
            if messagebox.askokcancel("終了確認", "アプリケーションを終了しますか？"):
                self.logger.info("学習アプリ終了")
                self.root.destroy()
        except Exception as e:
            self.logger.error(f"終了処理エラー: {e}")
            self.root.destroy()
    
    def cleanup(self):
        """クリーンアップ処理"""
        try:
            # リソース解放
            if self.training_manager:
                # 学習停止処理があれば実行
                pass
            
            self.logger.info("学習アプリクリーンアップ完了")
            
        except Exception as e:
            self.logger.error(f"クリーンアップエラー: {e}")


def main():
    """メイン関数"""
    try:
        # ログ設定
        logger = setup_logger("main")
        logger.info("=== 学習アプリケーション開始 ===")
        
        # アプリ実行
        app = TrainingApp()
        app.run()
        
    except Exception as e:
        # 最上位エラーハンドリング
        logger = setup_logger("main")
        logger.error(f"アプリケーション開始エラー: {e}")
        
        # エラーダイアログ表示
        root = tk.Tk()
        root.withdraw()  # ルートウィンドウを隠す
        messagebox.showerror(
            "起動エラー",
            f"アプリケーションの開始に失敗しました:\n\n{e}\n\n詳細はログファイルを確認してください。"
        )
        root.destroy()
        
        sys.exit(1)


if __name__ == "__main__":
    main()