"""推論アプリ メインウィンドウ"""

import os
import csv
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional
import threading

# パス設定
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from shared.config import get_config_manager
from shared.utils import setup_logger
from inference_app.core.image_manager import ImageManager
from inference_app.core.anomaly_detector import AnomalyDetector
from inference_app.core.result_manager import ResultManager
from inference_app.core.model_manager import InferenceModelManager
from inference_app.core.dataset_manager import InferenceDatasetManager
from .image_view import ImageView
from .result_view import ResultView


class InferenceMainWindow:
    """推論アプリケーション メインウィンドウ"""
    
    def __init__(self, config_manager=None):
        """
        初期化
        
        Args:
            config_manager: 設定管理インスタンス
        """
        self.config_manager = config_manager or get_config_manager()
        self.config = self.config_manager.get_config()
        self.logger = setup_logger("inference_main_window")
        
        # コアマネージャー初期化
        self.image_manager = ImageManager(self.config_manager)
        self.anomaly_detector = AnomalyDetector(self.config_manager)
        self.result_manager = ResultManager(self.config_manager)
        self.model_manager = InferenceModelManager(self.config_manager)
        self.dataset_manager = InferenceDatasetManager(self.config_manager)
        
        # GUI要素
        self.root = None
        self.image_view = None
        self.result_view = None
        
        # 状態管理
        self.model_loaded = False
        self.current_model = None
        self.current_category = None
        self.test_images = []
        self.test_image_data = []
        self.inference_results = []
        self.selected_image_index = -1
        
        self.logger.info("InferenceMainWindow初期化完了")
    
    def create_window(self):
        """ウィンドウ作成"""
        self.root = tk.Tk()
        self.root.title("AI異常検知システム - 推論アプリ (Phase1)")
        
        # ウィンドウサイズ設定
        window_size = self.config.get('ui.window_size', {'width': 1024, 'height': 768})
        self.root.geometry(f"{window_size['width']}x{window_size['height']}")
        
        # アイコン設定（あれば）
        try:
            # self.root.iconbitmap("icon.ico")
            pass
        except:
            pass
        
        self.setup_ui()
        self.setup_menu()
        
        # 終了処理
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.logger.info("ウィンドウ作成完了")
    
    def setup_ui(self):
        """UI構築"""
        # メインフレーム
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 上部フレーム（モデル選択・制御）
        self.setup_control_frame(main_frame)
        
        # 中央フレーム（画像表示・結果表示）
        self.setup_main_content_frame(main_frame)
        
        # 下部フレーム（ステータス・ログ）
        self.setup_status_frame(main_frame)
    
    def setup_control_frame(self, parent):
        """制御フレーム設定"""
        control_frame = ttk.LabelFrame(parent, text="モデル・データセット選択", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 上段: モデル選択とカテゴリ選択
        selection_frame = ttk.Frame(control_frame)
        selection_frame.pack(fill=tk.X, pady=(0, 10))
        
        # モデル選択
        ttk.Label(selection_frame, text="モデル:").pack(side=tk.LEFT)
        
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            selection_frame, 
            textvariable=self.model_var,
            state="readonly",
            width=35
        )
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_selected)
        self.model_combo.pack(side=tk.LEFT, padx=(5, 15))
        
        # カテゴリ選択
        ttk.Label(selection_frame, text="カテゴリ:").pack(side=tk.LEFT)
        
        self.category_var = tk.StringVar()
        self.category_combo = ttk.Combobox(
            selection_frame,
            textvariable=self.category_var,
            state="readonly",
            width=20
        )
        self.category_combo.bind("<<ComboboxSelected>>", self.on_category_selected)
        self.category_combo.pack(side=tk.LEFT, padx=(5, 15))
        
        # バッチ処理ボタン
        self.batch_button = ttk.Button(
            selection_frame, 
            text="バッチ処理実行", 
            command=self.run_batch_inference,
            state=tk.DISABLED
        )
        self.batch_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # CSV出力ボタン
        self.export_button = ttk.Button(
            selection_frame, 
            text="CSV出力", 
            command=self.export_results,
            state=tk.DISABLED
        )
        self.export_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # 下段: 閾値設定と状態表示
        info_frame = ttk.Frame(control_frame)
        info_frame.pack(fill=tk.X, pady=(0, 5))
        
        # 閾値設定
        ttk.Label(info_frame, text="閾値:").pack(side=tk.LEFT)
        
        self.threshold_var = tk.DoubleVar(value=self.config.get('inference.confidence_threshold', 0.5))
        threshold_scale = ttk.Scale(info_frame, from_=0.0, to=1.0, 
                                   variable=self.threshold_var, orient=tk.HORIZONTAL,
                                   command=self.on_threshold_changed, length=150)
        threshold_scale.pack(side=tk.LEFT, padx=(5, 5))
        
        self.threshold_label = ttk.Label(info_frame, text=f"{self.threshold_var.get():.2f}")
        self.threshold_label.pack(side=tk.LEFT, padx=(0, 15))
        
        # 処理対象画像数表示
        self.image_count_label = ttk.Label(info_frame, text="処理対象: 0枚")
        self.image_count_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # 処理状況表示
        self.processing_status_label = ttk.Label(info_frame, text="準備完了")
        self.processing_status_label.pack(side=tk.LEFT)
        
        # 初期化処理
        self.load_models()
        self.load_categories()
        
        # 初期選択を実行
        self.root.after(100, self.initialize_default_selections)
    
    def setup_main_content_frame(self, parent):
        """メインコンテンツフレーム設定"""
        content_frame = ttk.Frame(parent)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 左側：画像リストと選択画像表示
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # 画像リスト
        list_frame = ttk.LabelFrame(left_frame, text="処理対象画像", padding="5")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # リストボックスとスクロールバー
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        self.image_listbox = tk.Listbox(list_container, selectmode=tk.SINGLE)
        scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.image_listbox.yview)
        self.image_listbox.config(yscrollcommand=scrollbar.set)
        
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.image_listbox.bind("<<ListboxSelect>>", self.on_image_selected)
        
        # 選択画像表示
        self.image_view = ImageView(left_frame, self.image_manager)
        self.image_view.frame.pack(fill=tk.X, pady=(5, 0))
        
        # 右側：結果表示
        self.result_view = ResultView(content_frame)
        self.result_view.set_result_manager(self.result_manager)
        self.result_view.frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
    
    def setup_status_frame(self, parent):
        """ステータスフレーム設定"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X)
        
        # ステータスバー
        self.status_var = tk.StringVar(value="準備完了")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # モデル状態表示
        self.model_status_var = tk.StringVar(value="モデル未読み込み")
        model_status_label = ttk.Label(status_frame, textvariable=self.model_status_var,
                                      relief=tk.SUNKEN, anchor=tk.W)
        model_status_label.pack(side=tk.RIGHT, padx=(5, 0))
    
    def setup_menu(self):
        """メニュー設定"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # ファイルメニュー
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ファイル", menu=file_menu)
        file_menu.add_command(label="バッチ処理実行", command=self.run_batch_inference)
        file_menu.add_separator()
        file_menu.add_command(label="結果をCSVエクスポート", command=self.export_results)
        file_menu.add_command(label="レポート生成", command=self.generate_report)
        file_menu.add_separator()
        file_menu.add_command(label="終了", command=self.on_closing)
        
        # データメニュー
        data_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="データ", menu=data_menu)
        data_menu.add_command(label="モデル一覧更新", command=self.load_models)
        data_menu.add_command(label="カテゴリ一覧更新", command=self.load_categories)
        data_menu.add_command(label="モデル情報", command=self.show_model_info)
        
        # ヘルプメニュー
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ヘルプ", menu=help_menu)
        help_menu.add_command(label="使い方", command=self.show_help)
        help_menu.add_command(label="バージョン情報", command=self.show_about)
    
    def select_model(self):
        """旧式モデル選択（新システムでは使用しない）"""
        pass
    
    def load_model(self):
        """旧式モデル読み込み（新システムでは使用しない）"""
        pass
    
    def _on_model_loaded(self, success: bool):
        """旧式モデル読み込み完了処理（新システムでは使用しない）"""
        pass
    
    def select_image(self):
        """旧式画像選択（新システムでは使用しない）"""
        pass
    
    def load_image(self, file_path: str):
        """旧式画像読み込み（新システムでは使用しない）"""
        pass
    
    def process_image(self):
        """旧式画像処理（新システムでは使用しない）"""
        pass
    
    def _on_process_completed(self, result):
        """旧式処理完了時の処理（新システムでは使用しない）"""
        pass
    
    def _on_process_error(self, error_msg: str):
        """旧式処理エラー時の処理（新システムでは使用しない）"""
        pass
    
    def select_directory(self):
        """旧式ディレクトリ選択（新システムでは使用しない）"""
        pass
    

    
    def _on_batch_completed(self, results: list):
        """旧式一括処理完了時の処理（新システムでは使用しない）"""
        pass
    
    def _on_batch_error(self, error_msg: str):
        """旧式一括処理エラー時の処理（新システムでは使用しない）"""
        pass
    
    def export_batch_csv(self):
        """旧式一括結果CSV エクスポート（新システムでは使用しない）"""
        pass
    
    def on_threshold_changed(self, value):
        """閾値変更時の処理"""
        threshold = float(value)
        self.threshold_label.config(text=f"{threshold:.2f}")
        self.logger.debug(f"閾値変更: {threshold:.2f}")
    
    def export_csv(self):
        """旧式CSV エクスポート（新システムでは使用しない）"""
        pass
    
    def generate_report(self):
        """レポート生成"""
        try:
            report = self.result_manager.generate_simple_report()
            
            # レポート表示ウィンドウ
            self._show_report_window(report)
            
        except Exception as e:
            self.logger.error(f"レポート生成エラー: {e}")
            messagebox.showerror("エラー", f"レポート生成中にエラーが発生しました: {e}")
    
    def _show_report_window(self, report: dict):
        """レポート表示ウィンドウ"""
        report_window = tk.Toplevel(self.root)
        report_window.title("検知結果レポート")
        report_window.geometry("600x400")
        
        # テキストエリア
        text_frame = ttk.Frame(report_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # レポート内容
        report_text = f"""# 異常検知結果レポート

## 概要
{report.get('summary', 'データなし')}

## 詳細統計
- 総検知数: {report.get('total_count', 0)}件
- 正常: {report.get('normal_count', 0)}件
- 異常: {report.get('anomaly_count', 0)}件
- 異常率: {report.get('anomaly_rate', 0.0):.1f}%

## パフォーマンス
- 平均処理時間: {report.get('average_processing_time_ms', 0.0):.2f}ms
- 平均信頼度: {report.get('average_confidence', 0.0):.3f}

## モデル情報
- 使用モデル: {', '.join(report.get('model_versions', []))}

## 生成日時
{report.get('generated_at', 'unknown')}
"""
        
        text_widget.insert(tk.END, report_text)
        text_widget.config(state=tk.DISABLED)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 保存ボタン
        button_frame = ttk.Frame(report_window)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Button(button_frame, text="レポート保存", 
                  command=lambda: self.result_manager.save_report(report)).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="閉じる", 
                  command=report_window.destroy).pack(side=tk.RIGHT)
    
    def show_model_info(self):
        """モデル情報表示"""
        try:
            model_info = self.anomaly_detector.get_model_info()
            stats = self.anomaly_detector.get_inference_stats()
            
            info_text = f"""モデル情報:
・読み込み状態: {'読み込み済み' if model_info['model_loaded'] else '未読み込み'}
・モデルパス: {model_info.get('model_path', 'なし')}
・バージョン: {model_info.get('model_version', 'unknown')}
・処理デバイス: {model_info.get('device', 'unknown')}
・信頼度閾値: {model_info.get('threshold', 0.0):.2f}

推論統計:
・実行回数: {stats['inference_count']}回
・平均処理時間: {stats['average_time_ms']:.2f}ms
・FPS: {stats['fps']:.1f}
"""
            
            messagebox.showinfo("モデル情報", info_text)
            
        except Exception as e:
            self.logger.error(f"モデル情報表示エラー: {e}")
            messagebox.showerror("エラー", f"モデル情報取得中にエラーが発生しました: {e}")
    
    def show_help(self):
        """ヘルプ表示"""
        help_text = """AI異常検知システム - 推論アプリ (Enhanced Edition)

使い方:
1. [モデル] ドロップダウンから学習済みモデルを選択
2. [カテゴリ] ドロップダウンからテストデータセットのカテゴリを選択
3. [バッチ処理実行] ボタンで選択カテゴリの全テスト画像を一括処理
4. 画像リストから個別画像を選択して結果を詳細表示
5. [CSV出力] ボタンで結果をCSVファイルとしてエクスポート

機能:
・自動モデル検出: models/openvino/ 配下の学習済みモデルを自動発見
・自動データセット管理: MVTecADテストデータを自動カテゴリ分類
・バッチ処理: カテゴリ全体の画像を一括で異常検知処理
・閾値調整: スライダーで異常判定の閾値をリアルタイム調整
・結果表示: 画像リストで処理状況を確認、個別選択で詳細表示
・CSV出力: 処理結果を詳細な統計情報付きでCSV出力

対応画像形式: PNG, JPEG, BMP, TIFF
Phase1では bottle カテゴリのみ対応
"""
        messagebox.showinfo("使い方", help_text)
    
    def show_about(self):
        """バージョン情報表示"""
        about_text = f"""AI異常検知システム
Phase1 MVP版

バージョン: {self.config.get('app.version', '1.0.0')}
対象カテゴリ: bottle
技術スタック: anomalib + OpenVINO + Tkinter

開発: AI画像解析プロジェクトチーム
"""
        messagebox.showinfo("バージョン情報", about_text)
    
    def load_models(self):
        """利用可能なモデル一覧を読み込み"""
        try:
            models = self.model_manager.get_available_models()
            model_options = []
            
            for model_info in models:
                display_name = f"{model_info['category']} - {model_info['name']}"
                model_options.append(display_name)
            
            self.model_combo['values'] = model_options
            
            if model_options:
                self.model_combo.set(model_options[0])
                self.logger.info(f"モデル一覧読み込み完了: {len(model_options)}個")
            else:
                self.logger.warning("利用可能なモデルが見つかりません")
                
        except Exception as e:
            self.logger.error(f"モデル一覧読み込みエラー: {e}")

    def load_categories(self):
        """データセットカテゴリ一覧を読み込み"""
        try:
            categories = self.dataset_manager.get_available_categories()
            self.category_combo['values'] = categories
            
            if categories:
                self.category_combo.set(categories[0])
                self.logger.info(f"カテゴリ一覧読み込み完了: {len(categories)}個")
            else:
                self.logger.warning("利用可能なカテゴリが見つかりません")
                
        except Exception as e:
            self.logger.error(f"カテゴリ一覧読み込みエラー: {e}")

    def initialize_default_selections(self):
        """初期選択の自動実行"""
        try:
            # デフォルトモデル選択を実行
            if self.model_combo.get():
                self.on_model_selected()
            
            # デフォルトカテゴリ選択を実行
            if self.category_combo.get():
                self.on_category_selected()
                
        except Exception as e:
            self.logger.error(f"初期選択エラー: {e}")

    def on_model_selected(self, event=None):
        """モデル選択時の処理"""
        selected = self.model_var.get()
        if selected:
            try:
                models = self.model_manager.get_available_models()
                selected_model = None
                
                for model_info in models:
                    display_name = f"{model_info['category']} - {model_info['name']}"
                    if display_name == selected:
                        selected_model = model_info
                        break
                
                if selected_model:
                    self.current_model = selected_model
                    self.logger.info(f"モデル選択: {selected_model['name']}")
                    self.processing_status_label.config(text=f"モデル: {selected_model['name']}")
                    self.update_batch_button_state()
                    
            except Exception as e:
                self.logger.error(f"モデル選択エラー: {e}")

    def on_category_selected(self, event=None):
        """カテゴリ選択時の処理"""
        selected_category = self.category_var.get()
        if selected_category:
            try:
                self.current_category = selected_category
                image_data_list = self.dataset_manager.get_test_images(selected_category)
                
                # パスのリストを取得
                self.test_images = [img_data['path'] for img_data in image_data_list]
                self.test_image_data = image_data_list  # 詳細情報も保存
                
                # 画像リストを更新
                self.update_image_list()
                
                self.logger.info(f"カテゴリ選択: {selected_category}, 画像数: {len(self.test_images)}")
                self.image_count_label.config(text=f"処理対象: {len(self.test_images)}枚")
                self.update_batch_button_state()
                
            except Exception as e:
                self.logger.error(f"カテゴリ選択エラー: {e}")

    def update_image_list(self):
        """画像リストを更新"""
        self.image_listbox.delete(0, tk.END)
        
        for i, image_path in enumerate(self.test_images):
            # 詳細情報がある場合は表示名を拡張
            if i < len(self.test_image_data):
                img_data = self.test_image_data[i]
                display_name = f"[{img_data['type']}] {img_data['filename']}"
            else:
                display_name = os.path.basename(image_path)
            
            self.image_listbox.insert(tk.END, display_name)
        
        if self.test_images:
            self.image_listbox.selection_set(0)
            self.on_image_selected()

    def on_image_selected(self, event=None):
        """画像リストから画像選択時の処理"""
        selection = self.image_listbox.curselection()
        if selection:
            self.selected_image_index = selection[0]
            image_path = self.test_images[self.selected_image_index]
            
            # 画像を表示
            self.image_manager.load_image(image_path)
            # ImageViewに画像を表示（update_displayメソッドがない場合の対応）
            if hasattr(self.image_view, 'update_display'):
                self.image_view.update_display()
            elif hasattr(self.image_view, 'display_current_image'):
                self.image_view.display_current_image()
            else:
                # 画像表示の代替処理
                self.logger.debug("画像表示メソッドが見つかりません")
            
            # 結果が既にある場合は表示
            if self.selected_image_index < len(self.inference_results):
                result = self.inference_results[self.selected_image_index]
                self.result_view.display_result(result)
            else:
                self.result_view.clear_result()

    def update_batch_button_state(self):
        """バッチ処理ボタンの状態を更新"""
        if self.current_model and self.current_category and self.test_images:
            self.batch_button.config(state=tk.NORMAL)
        else:
            self.batch_button.config(state=tk.DISABLED)

    def run_batch_inference(self):
        """バッチ推論実行"""
        if not self.current_model or not self.test_images:
            messagebox.showerror("エラー", "モデルまたは処理対象画像が選択されていません")
            return
        
        try:
            self.processing_status_label.config(text="バッチ処理実行中...")
            self.batch_button.config(state=tk.DISABLED)
            
            # 異常検知器初期化とモデル読み込み
            detector = AnomalyDetector()
            detector.load_model(self.current_model['path'])
            
            # 画像データを準備（batch_detect_anomalyが期待する形式）
            images_batch_data = []
            total_images = len(self.test_image_data)
            
            for i, img_data in enumerate(self.test_image_data):
                image_path = img_data['path']
                try:
                    # 進行状況を更新
                    self.processing_status_label.config(
                        text=f"画像読み込み中... ({i+1}/{total_images})"
                    )
                    self.root.update()
                    
                    # 画像を個別に読み込み・前処理
                    image_array = self.image_manager.load_image(image_path)
                    if image_array is not None:
                        # 前処理（リサイズ等）を実行
                        preprocessed_image = self.image_manager.preprocess_image()
                        if preprocessed_image is not None:
                            images_batch_data.append((image_path, preprocessed_image, True))
                        else:
                            self.logger.warning(f"前処理失敗: {image_path}")
                            images_batch_data.append((image_path, None, False))
                    else:
                        self.logger.warning(f"読み込み失敗: {image_path}")
                        images_batch_data.append((image_path, None, False))
                except Exception as e:
                    self.logger.error(f"画像処理エラー: {image_path}, {e}")
                    images_batch_data.append((image_path, None, False))
            
            # 処理状況を更新
            self.processing_status_label.config(text="異常検知実行中...")
            self.root.update()
            
            # プログレスコールバック定義
            def update_progress(current, total, filename):
                self.processing_status_label.config(
                    text=f"異常検知実行中... ({current}/{total}) - {filename}"
                )
                self.root.update()
            
            # バッチ処理実行
            results = detector.batch_detect_anomaly(images_batch_data, update_progress)
            
            self.inference_results = results
            
            # 現在選択中の画像の結果を表示
            if self.selected_image_index >= 0 and self.selected_image_index < len(results):
                self.result_view.display_result(results[self.selected_image_index])
            
            # 結果サマリーを表示
            normal_count = sum(1 for r in results if not getattr(r, 'is_anomaly', False))
            anomaly_count = len(results) - normal_count
            
            self.processing_status_label.config(
                text=f"処理完了 - 正常: {normal_count}枚, 異常: {anomaly_count}枚"
            )
            
            self.export_button.config(state=tk.NORMAL)
            self.logger.info(f"バッチ処理完了: 処理枚数={len(results)}")
            
        except Exception as e:
            self.logger.error(f"バッチ処理エラー: {e}")
            messagebox.showerror("エラー", f"バッチ処理中にエラーが発生しました:\n{e}")
        finally:
            self.batch_button.config(state=tk.NORMAL)

    def export_results(self):
        """結果をCSV出力"""
        if not self.inference_results:
            messagebox.showwarning("警告", "出力する結果がありません")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                title="CSV保存",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['画像パス', 'ファイル名', 'カテゴリ', 'サブタイプ', '期待結果', '異常スコア', '閾値', '判定結果', '処理時間']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    writer.writeheader()
                    for i, result in enumerate(self.inference_results):
                        # 詳細情報を取得
                        img_data = self.test_image_data[i] if i < len(self.test_image_data) else {}
                        
                        writer.writerow({
                            '画像パス': self.test_images[i],
                            'ファイル名': os.path.basename(self.test_images[i]),
                            'カテゴリ': img_data.get('category', 'unknown'),
                            'サブタイプ': img_data.get('type', 'unknown'),
                            '期待結果': 'anomaly' if img_data.get('is_anomaly_expected', False) else 'normal',
                            '異常スコア': getattr(result, 'confidence_score', 0.0),
                            '閾値': 0.5,  # 設定から取得
                            '判定結果': 'anomaly' if getattr(result, 'is_anomaly', False) else 'normal',
                            '処理時間': getattr(result, 'processing_time_ms', 0.0)
                        })
                
                messagebox.showinfo("完了", f"結果をCSVファイルに出力しました:\n{filename}")
                self.logger.info(f"CSV出力完了: {filename}")
                
        except Exception as e:
            self.logger.error(f"CSV出力エラー: {e}")
            messagebox.showerror("エラー", f"CSV出力中にエラーが発生しました:\n{e}")

    def on_closing(self):
        """ウィンドウ終了処理"""
        try:
            # リソースクリーンアップ
            if hasattr(self.anomaly_detector, 'cleanup'):
                self.anomaly_detector.cleanup()
            
            self.logger.info("アプリケーション終了")
            self.root.destroy()
            
        except Exception as e:
            self.logger.error(f"終了処理エラー: {e}")
            self.root.destroy()
    
    def run(self):
        """アプリケーション実行"""
        self.create_window()
        self.root.mainloop()