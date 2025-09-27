"""推論アプリ メインウィンドウ"""

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
        
        # GUI要素
        self.root = None
        self.image_view = None
        self.result_view = None
        
        # 状態管理
        self.model_loaded = False
        self.current_model_path = None
        self.current_directory = None
        self.batch_results = []
        
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
        control_frame = ttk.LabelFrame(parent, text="モデル・制御", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # モデル選択
        model_frame = ttk.Frame(control_frame)
        model_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(model_frame, text="モデル:").pack(side=tk.LEFT)
        
        self.model_path_var = tk.StringVar()
        model_entry = ttk.Entry(model_frame, textvariable=self.model_path_var, 
                               state="readonly", width=50)
        model_entry.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        
        ttk.Button(model_frame, text="モデル選択", 
                  command=self.select_model).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(model_frame, text="モデル読み込み", 
                  command=self.load_model).pack(side=tk.LEFT)
        
        # 画像処理制御
        process_frame = ttk.Frame(control_frame)
        process_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(process_frame, text="画像選択", 
                  command=self.select_image).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(process_frame, text="ディレクトリ選択", 
                  command=self.select_directory).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(process_frame, text="異常検知実行", 
                  command=self.process_image, 
                  state=tk.DISABLED).pack(side=tk.LEFT, padx=(0, 5))
        
        self.batch_button = ttk.Button(process_frame, text="一括処理実行", 
                                      command=self.process_batch, 
                                      state=tk.DISABLED)
        self.batch_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # 閾値設定
        threshold_frame = ttk.Frame(control_frame)
        threshold_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(threshold_frame, text="閾値:").pack(side=tk.LEFT)
        
        self.threshold_var = tk.DoubleVar(value=self.config.get('inference.confidence_threshold', 0.5))
        threshold_scale = ttk.Scale(threshold_frame, from_=0.0, to=1.0, 
                                   variable=self.threshold_var, orient=tk.HORIZONTAL,
                                   command=self.on_threshold_changed)
        threshold_scale.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        
        self.threshold_label = ttk.Label(threshold_frame, text=f"{self.threshold_var.get():.2f}")
        self.threshold_label.pack(side=tk.LEFT)
        
        # ボタン状態管理用参照保存
        self.process_button = process_frame.winfo_children()[2]
    
    def setup_main_content_frame(self, parent):
        """メインコンテンツフレーム設定"""
        content_frame = ttk.Frame(parent)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 左側：画像表示
        self.image_view = ImageView(content_frame, self.image_manager)
        self.image_view.frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
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
        file_menu.add_command(label="画像を開く", command=self.select_image)
        file_menu.add_command(label="ディレクトリを開く", command=self.select_directory)
        file_menu.add_separator()
        file_menu.add_command(label="結果をCSVエクスポート", command=self.export_csv)
        file_menu.add_command(label="一括結果をCSVエクスポート", command=self.export_batch_csv)
        file_menu.add_command(label="レポート生成", command=self.generate_report)
        file_menu.add_separator()
        file_menu.add_command(label="終了", command=self.on_closing)
        
        # モデルメニュー
        model_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="モデル", menu=model_menu)
        model_menu.add_command(label="モデル選択", command=self.select_model)
        model_menu.add_command(label="モデル読み込み", command=self.load_model)
        model_menu.add_command(label="モデル情報", command=self.show_model_info)
        
        # ヘルプメニュー
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ヘルプ", menu=help_menu)
        help_menu.add_command(label="使い方", command=self.show_help)
        help_menu.add_command(label="バージョン情報", command=self.show_about)
    
    def select_model(self):
        """モデル選択"""
        try:
            # デフォルトディレクトリを設定
            openvino_dir = self.config.get('models.openvino_path', './models/openvino')
            if not Path(openvino_dir).exists():
                openvino_dir = "."
                
            file_path = filedialog.askopenfilename(
                title="OpenVINOモデルファイル(.xml)を選択",
                initialdir=openvino_dir,
                filetypes=[
                    ("OpenVINO IR", "*.xml"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                self.model_path_var.set(file_path)
                self.current_model_path = file_path
                self.logger.info(f"モデル選択: {file_path}")
                
        except Exception as e:
            self.logger.error(f"モデル選択エラー: {e}")
            messagebox.showerror("エラー", f"モデル選択中にエラーが発生しました: {e}")
    
    def load_model(self):
        """モデル読み込み"""
        if not self.current_model_path:
            messagebox.showwarning("警告", "先にモデルを選択してください")
            return
        
        try:
            self.status_var.set("モデル読み込み中...")
            self.root.update()
            
            # バックグラウンドで読み込み
            def load_worker():
                success = self.anomaly_detector.load_model(self.current_model_path)
                
                # UIスレッドで結果処理
                self.root.after(0, lambda: self._on_model_loaded(success))
            
            threading.Thread(target=load_worker, daemon=True).start()
            
        except Exception as e:
            self.logger.error(f"モデル読み込みエラー: {e}")
            messagebox.showerror("エラー", f"モデル読み込み中にエラーが発生しました: {e}")
            self.status_var.set("エラー")
    
    def _on_model_loaded(self, success: bool):
        """モデル読み込み完了処理"""
        if success:
            self.model_loaded = True
            self.model_status_var.set("モデル読み込み済み")
            self.status_var.set("モデル読み込み完了")
            self.process_button.config(state=tk.NORMAL)
            
            if self.current_directory:
                self.batch_button.config(state=tk.NORMAL)
            
            self.logger.info("モデル読み込み成功")
        else:
            self.model_loaded = False
            self.model_status_var.set("モデル読み込み失敗")
            self.status_var.set("モデル読み込み失敗")
            self.process_button.config(state=tk.DISABLED)
            self.batch_button.config(state=tk.DISABLED)
            messagebox.showerror("エラー", "モデルの読み込みに失敗しました")
    
    def select_image(self):
        """画像選択"""
        try:
            # デフォルトディレクトリをMVTecテストデータに設定
            initial_dir = "./datasets/development/mvtec_anomaly_detection"
            if not Path(initial_dir).exists():
                initial_dir = "."
            
            file_path = filedialog.askopenfilename(
                title="画像ファイルを選択",
                initialdir=initial_dir,
                filetypes=[
                    ("画像ファイル", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                    ("PNG", "*.png"),
                    ("JPEG", "*.jpg *.jpeg"),
                    ("BMP", "*.bmp"),
                    ("TIFF", "*.tiff *.tif"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                self.load_image(file_path)
                
        except Exception as e:
            self.logger.error(f"画像選択エラー: {e}")
            messagebox.showerror("エラー", f"画像選択中にエラーが発生しました: {e}")
    
    def load_image(self, file_path: str):
        """画像読み込み"""
        try:
            self.status_var.set("画像読み込み中...")
            self.root.update()
            
            # 画像読み込み
            image = self.image_manager.load_image(file_path)
            
            if image is not None:
                # 画像表示
                self.image_view.display_image(image, file_path)
                self.status_var.set(f"画像読み込み完了: {Path(file_path).name}")
                self.logger.info(f"画像読み込み成功: {file_path}")
            else:
                self.status_var.set("画像読み込み失敗")
                messagebox.showerror("エラー", "画像の読み込みに失敗しました")
                
        except Exception as e:
            self.logger.error(f"画像読み込みエラー: {e}")
            messagebox.showerror("エラー", f"画像読み込み中にエラーが発生しました: {e}")
            self.status_var.set("エラー")
    
    def process_image(self):
        """画像処理（異常検知）実行"""
        if not self.model_loaded:
            messagebox.showwarning("警告", "先にモデルを読み込んでください")
            return
        
        if not self.image_manager.is_image_loaded():
            messagebox.showwarning("警告", "先に画像を選択してください")
            return
        
        try:
            self.status_var.set("異常検知実行中...")
            self.root.update()
            
            # バックグラウンドで処理
            def process_worker():
                # 前処理
                preprocessed = self.image_manager.preprocess_image()
                if preprocessed is None:
                    self.root.after(0, lambda: self._on_process_error("前処理失敗"))
                    return
                
                # 異常検知実行
                result = self.anomaly_detector.detect_anomaly(preprocessed)
                if result is None:
                    self.root.after(0, lambda: self._on_process_error("異常検知失敗"))
                    return
                
                # 結果処理
                result.image_path = self.image_manager.get_current_image_path()
                self.root.after(0, lambda: self._on_process_completed(result))
            
            threading.Thread(target=process_worker, daemon=True).start()
            
        except Exception as e:
            self.logger.error(f"画像処理エラー: {e}")
            messagebox.showerror("エラー", f"画像処理中にエラーが発生しました: {e}")
            self.status_var.set("エラー")
    
    def _on_process_completed(self, result):
        """処理完了時の処理"""
        try:
            # 結果保存
            self.result_manager.save_result(result, result.image_path)
            
            # 結果表示
            self.result_view.display_result(result)
            
            # ステータス更新
            self.status_var.set(f"異常検知完了: {result.status_text} "
                              f"({result.confidence_percentage}%, "
                              f"{result.processing_time_ms:.1f}ms)")
            
            self.logger.info(f"異常検知完了: {result.status_text}")
            
        except Exception as e:
            self.logger.error(f"処理完了時エラー: {e}")
            self.status_var.set("結果処理エラー")
    
    def _on_process_error(self, error_msg: str):
        """処理エラー時の処理"""
        self.status_var.set(f"エラー: {error_msg}")
        messagebox.showerror("処理エラー", error_msg)
    
    def select_directory(self):
        """ディレクトリ選択"""
        try:
            initial_dir = "./datasets/development/mvtec_anomaly_detection"
            if not Path(initial_dir).exists():
                initial_dir = "."
            
            directory_path = filedialog.askdirectory(
                title="画像ディレクトリを選択",
                initialdir=initial_dir
            )
            
            if directory_path:
                self.current_directory = directory_path
                image_files = self.image_manager.find_images_in_directory(directory_path)
                
                if image_files:
                    self.batch_button.config(state=tk.NORMAL if self.model_loaded else tk.DISABLED)
                    self.status_var.set(f"ディレクトリ選択完了: {len(image_files)}件の画像を発見")
                    self.logger.info(f"ディレクトリ選択: {directory_path} ({len(image_files)}件)")
                else:
                    self.status_var.set("選択したディレクトリに画像が見つかりませんでした")
                    messagebox.showwarning("警告", "選択したディレクトリに画像ファイルが見つかりませんでした")
                    
        except Exception as e:
            self.logger.error(f"ディレクトリ選択エラー: {e}")
            messagebox.showerror("エラー", f"ディレクトリ選択中にエラーが発生しました: {e}")
    
    def process_batch(self):
        """一括処理実行"""
        if not self.model_loaded:
            messagebox.showwarning("警告", "先にモデルを読み込んでください")
            return
        
        if not self.current_directory:
            messagebox.showwarning("警告", "先にディレクトリを選択してください")
            return
        
        try:
            self.status_var.set("一括処理開始...")
            self.root.update()
            
            def batch_worker():
                image_files = self.image_manager.find_images_in_directory(self.current_directory)
                
                if not image_files:
                    self.root.after(0, lambda: self._on_batch_error("ディレクトリに画像が見つかりません"))
                    return
                
                images_data = self.image_manager.batch_load_images(image_files)
                
                if not images_data:
                    self.root.after(0, lambda: self._on_batch_error("画像の読み込みに失敗しました"))
                    return
                
                results = self.anomaly_detector.batch_detect_anomaly(images_data)
                
                if results:
                    self.batch_results = results
                    self.root.after(0, lambda: self._on_batch_completed(results))
                else:
                    self.root.after(0, lambda: self._on_batch_error("一括処理で結果が生成されませんでした"))
            
            threading.Thread(target=batch_worker, daemon=True).start()
            
        except Exception as e:
            self.logger.error(f"一括処理エラー: {e}")
            messagebox.showerror("エラー", f"一括処理中にエラーが発生しました: {e}")
            self.status_var.set("エラー")
    
    def _on_batch_completed(self, results: list):
        """一括処理完了時の処理"""
        try:
            self.result_manager.save_batch_results(results)
            
            anomaly_count = sum(1 for r in results if r.is_anomaly)
            normal_count = len(results) - anomaly_count
            
            self.status_var.set(f"一括処理完了: {len(results)}件処理 "
                              f"(正常:{normal_count}件, 異常:{anomaly_count}件)")
            
            self.logger.info(f"一括処理完了: {len(results)}件の結果を生成")
            
            result_text = f"一括処理が完了しました。\n\n"
            result_text += f"処理件数: {len(results)}件\n"
            result_text += f"正常: {normal_count}件\n"
            result_text += f"異常: {anomaly_count}件\n\n"
            result_text += "結果をCSVでエクスポートしますか？"
            
            if messagebox.askyesno("一括処理完了", result_text):
                self.export_batch_csv()
                
        except Exception as e:
            self.logger.error(f"一括処理完了時エラー: {e}")
            self.status_var.set("一括処理結果保存エラー")
    
    def _on_batch_error(self, error_msg: str):
        """一括処理エラー時の処理"""
        self.status_var.set(f"一括処理エラー: {error_msg}")
        messagebox.showerror("一括処理エラー", error_msg)
    
    def export_batch_csv(self):
        """一括結果CSV エクスポート"""
        try:
            if not self.batch_results:
                messagebox.showinfo("情報", "エクスポートする一括結果がありません")
                return
            
            file_path = filedialog.asksaveasfilename(
                title="一括結果CSV エクスポート先を選択",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")]
            )
            
            if file_path:
                success = self.result_manager.export_batch_results_csv(self.batch_results, file_path)
                if success:
                    messagebox.showinfo("完了", f"一括結果CSV エクスポート完了: {file_path}")
                else:
                    messagebox.showerror("エラー", "一括結果CSV エクスポートに失敗しました")
                    
        except Exception as e:
            self.logger.error(f"一括結果CSV エクスポートエラー: {e}")
            messagebox.showerror("エラー", f"一括結果CSV エクスポート中にエラーが発生しました: {e}")
    
    def on_threshold_changed(self, value):
        """閾値変更時の処理"""
        threshold = float(value)
        self.threshold_label.config(text=f"{threshold:.2f}")
        self.anomaly_detector.set_threshold(threshold)
    
    def export_csv(self):
        """CSV エクスポート"""
        try:
            file_path = filedialog.asksaveasfilename(
                title="CSV エクスポート先を選択",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")]
            )
            
            if file_path:
                success = self.result_manager.export_results_csv(file_path)
                if success:
                    messagebox.showinfo("完了", f"CSV エクスポート完了: {file_path}")
                else:
                    messagebox.showerror("エラー", "CSV エクスポートに失敗しました")
                    
        except Exception as e:
            self.logger.error(f"CSV エクスポートエラー: {e}")
            messagebox.showerror("エラー", f"CSV エクスポート中にエラーが発生しました: {e}")
    
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
        help_text = """AI異常検知システム - 推論アプリ

使い方:
1. [モデル選択] ボタンでOpenVINOモデルファイル(.xml)を選択
2. [モデル読み込み] ボタンでモデルを読み込み
3. [画像選択] ボタンで検知対象画像を選択
4. [異常検知実行] ボタンで異常検知を実行

機能:
・閾値スライダーで異常判定の閾値を調整可能
・結果はCSV形式でエクスポート可能
・レポート機能で統計情報を確認可能

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