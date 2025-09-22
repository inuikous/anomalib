"""結果表示UI"""

import sys
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageTk, ImageDraw
from pathlib import Path
from typing import Optional, Dict, Any
import json

# パス設定
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

try:
    from shared.domain.detection_result import DetectionResult
except ImportError:
    # フォールバック用の簡易クラス
    class DetectionResult:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        
        def to_dict(self):
            return self.__dict__

from shared.utils import setup_logger


class ResultView:
    """結果表示ビュー"""
    
    def __init__(self, parent):
        """
        初期化
        
        Args:
            parent: 親ウィジェット
        """
        self.parent = parent
        self.logger = setup_logger("result_view")
        
        # 表示状態
        self.current_result = None
        self.result_photo = None
        
        # UI要素
        self.frame = None
        self.canvas = None
        self.details_frame = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """UI構築"""
        # メインフレーム
        self.frame = ttk.LabelFrame(self.parent, text="検出結果", padding="10")
        
        # 上部: 結果画像表示
        self.setup_result_canvas()
        
        # 下部: 詳細情報
        self.setup_details_frame()
        
        # 初期状態
        self.show_no_result()
    
    def setup_result_canvas(self):
        """結果画像キャンバス設定"""
        canvas_frame = ttk.Frame(self.frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # キャンバス
        self.canvas = tk.Canvas(canvas_frame, bg="white", relief=tk.SUNKEN, borderwidth=2, height=200)
        
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # グリッド配置
        self.canvas.grid(row=0, column=0, sticky="nsew")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
    
    def setup_details_frame(self):
        """詳細情報フレーム設定"""
        self.details_frame = ttk.Frame(self.frame)
        self.details_frame.pack(fill=tk.X)
        
        # 結果情報
        info_frame = ttk.LabelFrame(self.details_frame, text="検出情報", padding="5")
        info_frame.pack(fill=tk.X, pady=(0, 5))
        
        # 判定結果
        ttk.Label(info_frame, text="判定:").grid(row=0, column=0, sticky="w")
        self.judgment_var = tk.StringVar(value="-")
        self.judgment_label = ttk.Label(info_frame, textvariable=self.judgment_var, font=("Arial", 10, "bold"))
        self.judgment_label.grid(row=0, column=1, sticky="w", padx=(5, 0))
        
        # 異常スコア
        ttk.Label(info_frame, text="異常スコア:").grid(row=1, column=0, sticky="w")
        self.score_var = tk.StringVar(value="-")
        ttk.Label(info_frame, textvariable=self.score_var).grid(row=1, column=1, sticky="w", padx=(5, 0))
        
        # 閾値
        ttk.Label(info_frame, text="閾値:").grid(row=2, column=0, sticky="w")
        self.threshold_var = tk.StringVar(value="-")
        ttk.Label(info_frame, textvariable=self.threshold_var).grid(row=2, column=1, sticky="w", padx=(5, 0))
        
        # 処理時間
        ttk.Label(info_frame, text="処理時間:").grid(row=3, column=0, sticky="w")
        self.inference_time_var = tk.StringVar(value="-")
        ttk.Label(info_frame, textvariable=self.inference_time_var).grid(row=3, column=1, sticky="w", padx=(5, 0))
        
        # 詳細データ表示
        details_data_frame = ttk.LabelFrame(self.details_frame, text="詳細データ", padding="5")
        details_data_frame.pack(fill=tk.X)
        
        # JSONテキスト表示
        self.details_text = tk.Text(details_data_frame, height=6, wrap=tk.WORD)
        details_scrollbar = ttk.Scrollbar(details_data_frame, orient=tk.VERTICAL, command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=details_scrollbar.set)
        
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def display_result(self, result: DetectionResult, original_image: Optional[np.ndarray] = None):
        """
        結果表示
        
        Args:
            result: 検出結果
            original_image: 元画像（オプション）
        """
        try:
            self.current_result = result
            
            # 結果画像表示
            self.display_result_image(result, original_image)
            
            # 詳細情報表示
            self.update_result_info(result)
            
            self.logger.info(f"結果表示完了: {result.is_anomaly}")
            
        except Exception as e:
            self.logger.error(f"結果表示エラー: {e}")
            self.show_error_message(f"結果表示エラー: {e}")
    
    def display_result_image(self, result: DetectionResult, original_image: Optional[np.ndarray] = None):
        """結果画像表示"""
        try:
            # 結果画像作成
            if original_image is not None and hasattr(result, 'anomaly_map') and result.anomaly_map is not None:
                # 元画像に異常マップをオーバーレイ
                result_image = self.create_overlay_image(original_image, result.anomaly_map, result.is_anomaly)
            elif original_image is not None:
                # 元画像のみ（判定結果をテキストで表示）
                result_image = self.add_result_text(original_image, result.is_anomaly, result.anomaly_score)
            else:
                # 結果画像がない場合はプレースホルダー
                result_image = self.create_result_placeholder(result.is_anomaly, result.anomaly_score)
            
            # PIL画像に変換して表示
            if result_image.dtype == np.float32 or result_image.dtype == np.float64:
                if result_image.max() <= 1.0:
                    result_image = (result_image * 255).astype(np.uint8)
                else:
                    result_image = np.clip(result_image, 0, 255).astype(np.uint8)
            
            if len(result_image.shape) == 3:
                pil_image = Image.fromarray(result_image, 'RGB')
            else:
                pil_image = Image.fromarray(result_image, 'L')
            
            # 表示サイズ調整
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                max_size = min(canvas_width - 20, canvas_height - 20, 400)
                if max(pil_image.size) > max_size:
                    ratio = max_size / max(pil_image.size)
                    new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
                    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Tkinter用に変換
            self.result_photo = ImageTk.PhotoImage(pil_image)
            
            # キャンバス表示
            self.canvas.delete("all")
            self.canvas.create_image(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                anchor=tk.CENTER,
                image=self.result_photo
            )
            
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
        except Exception as e:
            self.logger.error(f"結果画像表示エラー: {e}")
            self.show_canvas_error(f"画像表示エラー: {e}")
    
    def create_overlay_image(self, original: np.ndarray, anomaly_map: np.ndarray, is_anomaly: bool) -> np.ndarray:
        """異常マップオーバーレイ画像作成"""
        try:
            # 元画像をコピー
            if len(original.shape) == 3:
                overlay_image = original.copy()
            else:
                overlay_image = np.stack([original] * 3, axis=-1)
            
            # 異常マップを正規化
            if anomaly_map.max() > 0:
                normalized_map = anomaly_map / anomaly_map.max()
            else:
                normalized_map = anomaly_map
            
            # 異常領域をハイライト（赤色）
            if is_anomaly:
                red_overlay = normalized_map > 0.5  # 閾値は調整可能
                overlay_image[red_overlay, 0] = np.minimum(255, overlay_image[red_overlay, 0] + 100)  # 赤チャンネル強調
            
            return overlay_image
            
        except Exception as e:
            self.logger.error(f"オーバーレイ画像作成エラー: {e}")
            return original
    
    def add_result_text(self, image: np.ndarray, is_anomaly: bool, score: float) -> np.ndarray:
        """結果テキスト追加"""
        try:
            # PIL画像に変換
            if len(image.shape) == 3:
                pil_image = Image.fromarray(image.astype(np.uint8), 'RGB')
            else:
                pil_image = Image.fromarray(image.astype(np.uint8), 'L').convert('RGB')
            
            # 描画オブジェクト作成
            draw = ImageDraw.Draw(pil_image)
            
            # 結果テキスト
            result_text = "異常" if is_anomaly else "正常"
            color = "red" if is_anomaly else "green"
            
            # テキスト描画（左上）
            draw.text((10, 10), f"判定: {result_text}", fill=color)
            draw.text((10, 30), f"スコア: {score:.3f}", fill=color)
            
            return np.array(pil_image)
            
        except Exception as e:
            self.logger.error(f"結果テキスト追加エラー: {e}")
            return image
    
    def create_result_placeholder(self, is_anomaly: bool, score: float) -> np.ndarray:
        """結果プレースホルダー作成"""
        # 200x200の白画像作成
        placeholder = np.ones((200, 200, 3), dtype=np.uint8) * 255
        
        try:
            pil_image = Image.fromarray(placeholder, 'RGB')
            draw = ImageDraw.Draw(pil_image)
            
            # 中央にテキスト表示
            result_text = "異常検出" if is_anomaly else "正常"
            color = "red" if is_anomaly else "green"
            
            draw.text((70, 90), result_text, fill=color)
            draw.text((60, 110), f"スコア: {score:.3f}", fill=color)
            
            return np.array(pil_image)
            
        except Exception as e:
            self.logger.error(f"プレースホルダー作成エラー: {e}")
            return placeholder
    
    def update_result_info(self, result: DetectionResult):
        """結果情報更新"""
        try:
            # 判定結果
            judgment = "異常" if result.is_anomaly else "正常"
            self.judgment_var.set(judgment)
            
            # 色設定
            color = "red" if result.is_anomaly else "green"
            self.judgment_label.configure(foreground=color)
            
            # 異常スコア
            self.score_var.set(f"{result.anomaly_score:.3f}")
            
            # 閾値
            self.threshold_var.set(f"{result.threshold:.3f}")
            
            # 処理時間
            if hasattr(result, 'inference_time') and result.inference_time is not None:
                self.inference_time_var.set(f"{result.inference_time:.3f}ms")
            else:
                self.inference_time_var.set("-")
            
            # 詳細データ（JSON形式）
            self.update_details_data(result)
            
        except Exception as e:
            self.logger.error(f"結果情報更新エラー: {e}")
            self.show_info_error()
    
    def update_details_data(self, result: DetectionResult):
        """詳細データ更新"""
        try:
            # 結果をJSON形式で表示
            result_dict = result.to_dict()
            
            # 表示用に整形（大きなデータは省略）
            display_dict = {}
            for key, value in result_dict.items():
                if key in ['anomaly_map']:  # 大きなデータは省略
                    if value is not None:
                        display_dict[key] = f"<配列: shape={getattr(value, 'shape', 'unknown')}>"
                    else:
                        display_dict[key] = None
                else:
                    display_dict[key] = value
            
            # JSONテキスト作成
            json_text = json.dumps(display_dict, ensure_ascii=False, indent=2)
            
            # テキスト表示
            self.details_text.delete(1.0, tk.END)
            self.details_text.insert(1.0, json_text)
            
        except Exception as e:
            self.logger.error(f"詳細データ更新エラー: {e}")
            self.details_text.delete(1.0, tk.END)
            self.details_text.insert(1.0, f"詳細データ表示エラー: {e}")
    
    def show_no_result(self):
        """結果なし表示"""
        # キャンバスクリア
        self.canvas.delete("all")
        self.canvas.create_text(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            text="検出結果なし\n\n異常検出を実行してください",
            anchor=tk.CENTER,
            fill="gray",
            font=("Arial", 10),
            justify=tk.CENTER
        )
        
        # 情報クリア
        self.judgment_var.set("-")
        self.score_var.set("-")
        self.threshold_var.set("-")
        self.inference_time_var.set("-")
        
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(1.0, "検出結果なし")
    
    def show_error_message(self, message: str):
        """エラーメッセージ表示"""
        self.show_canvas_error(message)
        self.show_info_error()
    
    def show_canvas_error(self, message: str):
        """キャンバスエラー表示"""
        self.canvas.delete("all")
        self.canvas.create_text(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            text=f"エラー:\n{message}",
            anchor=tk.CENTER,
            fill="red",
            font=("Arial", 10),
            justify=tk.CENTER
        )
    
    def show_info_error(self):
        """情報エラー表示"""
        self.judgment_var.set("エラー")
        self.judgment_label.configure(foreground="red")
        self.score_var.set("エラー")
        self.threshold_var.set("エラー")
        self.inference_time_var.set("エラー")
    
    def clear_result(self):
        """結果クリア"""
        self.current_result = None
        self.result_photo = None
        self.show_no_result()
        self.logger.debug("結果表示クリア")
    
    def get_current_result(self) -> Optional[DetectionResult]:
        """現在の結果取得"""
        return self.current_result