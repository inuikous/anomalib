"""画像表示UI"""

import sys
import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
from typing import Optional

# パス設定
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from shared.utils import setup_logger


class ImageView:
    """画像表示ビュー"""
    
    def __init__(self, parent, image_manager):
        """
        初期化
        
        Args:
            parent: 親ウィジェット
            image_manager: 画像管理インスタンス
        """
        self.parent = parent
        self.image_manager = image_manager
        self.logger = setup_logger("image_view")
        
        # 表示状態
        self.current_image = None
        self.current_photo = None
        self.current_file_path = None
        
        # UI要素
        self.frame = None
        self.canvas = None
        self.info_frame = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """UI構築"""
        # メインフレーム
        self.frame = ttk.LabelFrame(self.parent, text="画像表示", padding="10")
        
        # 画像表示エリア
        canvas_frame = ttk.Frame(self.frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # キャンバス（スクロール対応）
        self.canvas = tk.Canvas(canvas_frame, bg="white", relief=tk.SUNKEN, borderwidth=2)
        
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # グリッド配置
        self.canvas.grid(row=0, column=0, sticky="nsew")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        # 画像情報表示
        self.setup_info_frame()
        
        # ドラッグ&ドロップ対応
        self.setup_drag_drop()
        
        # 初期表示
        self.show_placeholder()
    
    def setup_info_frame(self):
        """情報表示フレーム設定"""
        self.info_frame = ttk.Frame(self.frame)
        self.info_frame.pack(fill=tk.X)
        
        # ファイル名
        self.filename_var = tk.StringVar(value="画像未選択")
        ttk.Label(self.info_frame, text="ファイル:").grid(row=0, column=0, sticky="w")
        ttk.Label(self.info_frame, textvariable=self.filename_var).grid(row=0, column=1, sticky="w", padx=(5, 0))
        
        # 画像サイズ
        self.size_var = tk.StringVar(value="-")
        ttk.Label(self.info_frame, text="サイズ:").grid(row=1, column=0, sticky="w")
        ttk.Label(self.info_frame, textvariable=self.size_var).grid(row=1, column=1, sticky="w", padx=(5, 0))
        
        # ファイルサイズ
        self.filesize_var = tk.StringVar(value="-")
        ttk.Label(self.info_frame, text="ファイルサイズ:").grid(row=2, column=0, sticky="w")
        ttk.Label(self.info_frame, textvariable=self.filesize_var).grid(row=2, column=1, sticky="w", padx=(5, 0))
    
    def setup_drag_drop(self):
        """ドラッグ&ドロップ設定"""
        # 基本的なドラッグ&ドロップ（簡易版）
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # プレースホルダーテキスト表示用
        self.placeholder_text = None
    
    def display_image(self, image: np.ndarray, file_path: str):
        """
        画像表示
        
        Args:
            image: 表示する画像配列
            file_path: ファイルパス
        """
        try:
            self.current_image = image
            self.current_file_path = file_path
            
            # PIL画像に変換
            if image.dtype == np.float32 or image.dtype == np.float64:
                # 0-1範囲の場合は255倍
                if image.max() <= 1.0:
                    image_uint8 = (image * 255).astype(np.uint8)
                else:
                    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
            else:
                image_uint8 = image
            
            # PIL画像作成
            if len(image_uint8.shape) == 3:
                pil_image = Image.fromarray(image_uint8, 'RGB')
            else:
                pil_image = Image.fromarray(image_uint8, 'L')
            
            # 表示サイズ調整（大きすぎる場合は縮小）
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:  # ウィンドウが初期化済み
                max_size = min(canvas_width - 20, canvas_height - 20, 800)
                if max(pil_image.size) > max_size:
                    ratio = max_size / max(pil_image.size)
                    new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
                    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Tkinter用に変換
            self.current_photo = ImageTk.PhotoImage(pil_image)
            
            # キャンバスクリア
            self.canvas.delete("all")
            
            # 画像表示
            self.canvas.create_image(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                anchor=tk.CENTER,
                image=self.current_photo
            )
            
            # スクロール領域設定
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
            # 情報表示更新
            self.update_image_info(file_path)
            
            self.logger.info(f"画像表示完了: {Path(file_path).name}")
            
        except Exception as e:
            self.logger.error(f"画像表示エラー: {e}")
            self.show_error_message(f"画像表示エラー: {e}")
    
    def update_image_info(self, file_path: str):
        """画像情報更新"""
        try:
            file_path_obj = Path(file_path)
            
            # ファイル名
            self.filename_var.set(file_path_obj.name)
            
            # 画像情報取得
            info = self.image_manager.get_image_info(file_path)
            
            if info:
                # サイズ
                self.size_var.set(f"{info['width']} x {info['height']}")
                
                # ファイルサイズ
                if 'file_size_mb' in info:
                    self.filesize_var.set(f"{info['file_size_mb']:.2f} MB")
                else:
                    self.filesize_var.set(f"{info['file_size']} bytes")
            else:
                self.size_var.set("情報取得失敗")
                self.filesize_var.set("-")
                
        except Exception as e:
            self.logger.error(f"画像情報更新エラー: {e}")
            self.size_var.set("エラー")
            self.filesize_var.set("エラー")
    
    def show_placeholder(self):
        """プレースホルダー表示"""
        self.canvas.delete("all")
        
        # プレースホルダーテキスト
        self.placeholder_text = self.canvas.create_text(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            text="画像をドラッグ&ドロップ\nまたは「画像選択」ボタンから\n画像を選択してください",
            anchor=tk.CENTER,
            fill="gray",
            font=("Arial", 12),
            justify=tk.CENTER
        )
        
        # 情報クリア
        self.filename_var.set("画像未選択")
        self.size_var.set("-")
        self.filesize_var.set("-")
    
    def show_error_message(self, message: str):
        """エラーメッセージ表示"""
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
    
    def clear_image(self):
        """画像クリア"""
        self.current_image = None
        self.current_photo = None
        self.current_file_path = None
        
        self.show_placeholder()
        self.logger.debug("画像表示クリア")
    
    def on_canvas_click(self, event):
        """キャンバスクリック処理"""
        # 現在は基本的なクリック処理のみ
        pass
    
    def get_current_image(self) -> Optional[np.ndarray]:
        """現在の画像取得"""
        return self.current_image
    
    def get_current_file_path(self) -> Optional[str]:
        """現在のファイルパス取得"""
        return self.current_file_path