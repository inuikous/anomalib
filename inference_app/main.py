"""推論アプリメイン"""

import sys
from pathlib import Path

# パス設定
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "shared"))

import tkinter as tk
from tkinter import messagebox

from shared.utils import setup_logger
from gui.main_window import InferenceMainWindow
from core.image_manager import ImageManager
from core.anomaly_detector import AnomalyDetector
from core.result_manager import ResultManager


class InferenceApp:
    """推論アプリケーション"""
    
    def __init__(self):
        """初期化"""
        self.logger = setup_logger("inference_app")
        
        # GUI
        self.root = None
        self.main_window = None
        
        # コアコンポーネント
        self.image_manager = None
        self.anomaly_detector = None
        self.result_manager = None
        
        self.setup_components()
    
    def setup_components(self):
        """コンポーネント初期化"""
        try:
            # コアコンポーネント初期化
            self.image_manager = ImageManager()
            self.anomaly_detector = AnomalyDetector()
            self.result_manager = ResultManager()
            
            self.logger.info("推論アプリ初期化完了")
            
        except Exception as e:
            self.logger.error(f"推論アプリ初期化エラー: {e}")
            raise
    
    def run(self):
        """アプリケーション実行"""
        try:
            # Tkinterルート作成
            self.root = tk.Tk()
            self.root.title("異常検出システム - 推論")
            self.root.geometry("1200x800")
            
            # アイコン設定（存在する場合）
            try:
                self.root.iconbitmap(default="app.ico")
            except:
                pass  # アイコンがない場合は無視
            
            # メインウィンドウ作成
            self.main_window = InferenceMainWindow(
                self.root,
                self.image_manager,
                self.anomaly_detector,
                self.result_manager
            )
            
            # 終了時処理設定
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            self.logger.info("推論アプリ開始")
            
            # イベントループ開始
            self.root.mainloop()
            
        except Exception as e:
            self.logger.error(f"推論アプリ実行エラー: {e}")
            messagebox.showerror("エラー", f"アプリケーション実行エラー:\n{e}")
        finally:
            self.cleanup()
    
    def on_closing(self):
        """終了時処理"""
        try:
            # 確認ダイアログ
            if messagebox.askokcancel("終了確認", "アプリケーションを終了しますか？"):
                self.logger.info("推論アプリ終了")
                self.root.destroy()
        except Exception as e:
            self.logger.error(f"終了処理エラー: {e}")
            self.root.destroy()
    
    def cleanup(self):
        """クリーンアップ処理"""
        try:
            # リソース解放
            if self.anomaly_detector:
                # モデル解放処理があれば実行
                pass
            
            self.logger.info("推論アプリクリーンアップ完了")
            
        except Exception as e:
            self.logger.error(f"クリーンアップエラー: {e}")


def main():
    """メイン関数"""
    try:
        # ログ設定
        logger = setup_logger("main")
        logger.info("=== 推論アプリケーション開始 ===")
        
        # アプリ実行
        app = InferenceApp()
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