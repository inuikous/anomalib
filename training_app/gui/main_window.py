"""学習アプリメインウィンドウ"""

import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
from typing import Optional

# パス設定
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from shared.utils import setup_logger
try:
    from shared.domain.dataset import DatasetInfo
    from shared.domain.model import ModelInfo
except ImportError:
    # フォールバック用の簡易クラス
    class DatasetInfo:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class ModelInfo:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


class TrainingMainWindow:
    """学習アプリメインウィンドウ（汎用データセット対応）"""
    
    def __init__(self, root, dataset_manager, training_manager, model_manager):
        """
        初期化
        
        Args:
            root: Tkinterルート
            dataset_manager: データセット管理
            training_manager: 学習管理
            model_manager: モデル管理
        """
        self.root = root
        self.dataset_manager = dataset_manager
        self.training_manager = training_manager
        self.model_manager = model_manager
        self.logger = setup_logger("training_main_window")
        
        # 状態管理
        self.current_dataset_path = None
        self.current_dataset_info = None
        self.is_training = False
        self.training_thread = None
        
        # UI要素
        self.notebook = None
        self.dataset_frame = None
        self.training_frame = None
        self.model_frame = None
        
        # UI変数
        self.dataset_path_var = None
        self.dataset_type_var = None
        self.category_var = None
        self.training_progress_var = None
        self.training_status_var = None
        
        self.setup_ui()
        self.logger.info("学習アプリメインウィンドウ初期化完了")
    
    def setup_ui(self):
        """UI構築"""
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # タブノートブック
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # タブ作成
        self.setup_dataset_tab()
        self.setup_training_tab()
        self.setup_model_tab()
        
        # メニューバー
        self.setup_menu()
        
        # ステータスバー
        self.setup_status_bar(main_frame)
    
    def setup_dataset_tab(self):
        """データセットタブ設定"""
        self.dataset_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.dataset_frame, text="データセット管理")
        
        # データセットタイプ選択
        type_frame = ttk.LabelFrame(self.dataset_frame, text="データセットタイプ", padding="10")
        type_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(type_frame, text="データセットタイプ:").pack(anchor="w")
        
        type_select_frame = ttk.Frame(type_frame)
        type_select_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.dataset_type_var = tk.StringVar(value="mvtec")
        ttk.Radiobutton(type_select_frame, text="MVTec AD", variable=self.dataset_type_var, 
                       value="mvtec", command=self.on_dataset_type_change).pack(side=tk.LEFT)
        ttk.Radiobutton(type_select_frame, text="カスタム", variable=self.dataset_type_var, 
                       value="custom", command=self.on_dataset_type_change).pack(side=tk.LEFT, padx=(20, 0))
        
        # カテゴリ選択
        category_frame = ttk.LabelFrame(self.dataset_frame, text="カテゴリ選択", padding="10")
        category_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(category_frame, text="カテゴリ:").pack(anchor="w")
        
        category_select_frame = ttk.Frame(category_frame)
        category_select_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.category_var = tk.StringVar(value="bottle")
        self.category_combo = ttk.Combobox(category_select_frame, textvariable=self.category_var, 
                                          state="readonly", width=20)
        self.category_combo.pack(side=tk.LEFT)
        self.category_combo.bind("<<ComboboxSelected>>", self.on_category_change)
        
        ttk.Button(category_select_frame, text="更新", command=self.refresh_categories_full).pack(side=tk.LEFT, padx=(10, 0))
        
        # データセット選択
        dataset_select_frame = ttk.LabelFrame(self.dataset_frame, text="データセットパス", padding="10")
        dataset_select_frame.pack(fill=tk.X, pady=(0, 10))
        
        # パス表示
        ttk.Label(dataset_select_frame, text="データセットパス:").pack(anchor="w")
        
        path_frame = ttk.Frame(dataset_select_frame)
        path_frame.pack(fill=tk.X, pady=(5, 10))
        
        self.dataset_path_var = tk.StringVar(value="データセット未選択")
        ttk.Entry(path_frame, textvariable=self.dataset_path_var, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(path_frame, text="参照", command=self.select_dataset_folder).pack(side=tk.RIGHT, padx=(5, 0))
        
        # データセット情報
        info_frame = ttk.LabelFrame(self.dataset_frame, text="データセット情報", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        # 情報表示テキスト
        info_text_frame = ttk.Frame(info_frame)
        info_text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.dataset_info_text = tk.Text(info_text_frame, height=15, wrap=tk.WORD)
        info_scrollbar = ttk.Scrollbar(info_text_frame, orient=tk.VERTICAL, command=self.dataset_info_text.yview)
        self.dataset_info_text.configure(yscrollcommand=info_scrollbar.set)
        
        self.dataset_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # ボタン
        button_frame = ttk.Frame(info_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="データセット検証", command=self.validate_dataset).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="統計情報更新", command=self.update_dataset_stats).pack(side=tk.LEFT, padx=(5, 0))
        
        # 初期表示（軽量化）
        self.setup_initial_state()
        self.show_no_dataset_info()
    
    def on_dataset_type_change(self):
        """データセットタイプ変更時の処理"""
        dataset_type = self.dataset_type_var.get()
        self.dataset_manager.set_dataset_type(dataset_type)
        self.refresh_categories()
        self.logger.info(f"データセットタイプ変更: {dataset_type}")
    
    def on_category_change(self, event=None):
        """カテゴリ変更時の処理"""
        category = self.category_var.get()
        if category:
            self.dataset_manager.set_category(category)
            self.current_dataset_path = str(self.dataset_manager.category_path)
            self.dataset_path_var.set(self.current_dataset_path)
            
            # 必要な場合のみ検証実行（パフォーマンス向上）
            if hasattr(self, 'root') and self.root.winfo_exists():
                # ウィンドウが完全に初期化されてから検証
                self.root.after(100, self.validate_dataset)
            
            self.logger.info(f"カテゴリ変更: {category}, パス: {self.current_dataset_path}")
    
    def refresh_categories(self):
        """カテゴリ一覧更新"""
        try:
            categories = self.dataset_manager.get_available_categories()
            self.category_combo['values'] = categories
            
            if categories:
                if self.category_var.get() not in categories:
                    self.category_var.set(categories[0])
                    self.on_category_change()
            else:
                self.category_combo['values'] = []
                self.category_var.set("")
                
            self.logger.info(f"カテゴリ一覧更新: {len(categories)}件")
            
        except Exception as e:
            self.logger.error(f"カテゴリ一覧更新エラー: {e}")
            messagebox.showerror("エラー", f"カテゴリ一覧更新エラー:\n{e}")
    
    def refresh_categories_full(self):
        """完全なカテゴリ更新（ユーザー操作用）"""
        try:
            self.status_var.set("カテゴリ一覧更新中...")
            categories = self.dataset_manager.get_available_categories()
            self.category_combo['values'] = categories
            
            if categories:
                current = self.category_var.get()
                if current not in categories:
                    self.category_var.set(categories[0])
                    self.on_category_change()
                else:
                    # 現在のカテゴリを保持しつつパスを再設定
                    self.on_category_change()
            else:
                self.category_combo['values'] = []
                self.category_var.set("")
                
            self.status_var.set(f"カテゴリ一覧更新完了: {len(categories)}件")
            messagebox.showinfo("完了", f"カテゴリ一覧を更新しました（{len(categories)}件）")
            
        except Exception as e:
            self.logger.error(f"カテゴリ一覧更新エラー: {e}")
            self.status_var.set("エラー")
            messagebox.showerror("エラー", f"カテゴリ一覧更新エラー:\n{e}")
    
    def setup_initial_state(self):
        """軽量な初期状態設定（起動時間短縮）"""
        try:
            # デフォルトカテゴリの設定のみ（検証は後回し）
            self.category_combo['values'] = ["bottle", "cable", "capsule", "carpet", "grid", 
                                            "hazelnut", "leather", "metal_nut", "pill", "screw", 
                                            "tile", "toothbrush", "transistor", "wood", "zipper"]
            
            # config.yamlからアクティブカテゴリを取得
            active_category = self.dataset_manager.config.get('datasets', {}).get('mvtec', {}).get('active_category', 'bottle')
            
            # デフォルトカテゴリ設定を確実に反映
            self.category_var.set(active_category)
            self.dataset_manager.set_category(active_category)
            self.current_dataset_path = str(self.dataset_manager.category_path)
            self.dataset_path_var.set(self.current_dataset_path)
            
            # TrainingManagerにも同じカテゴリを設定
            self.training_manager.update_category(active_category)
            
            self.logger.info(f"初期状態設定完了: カテゴリ={active_category}, パス={self.current_dataset_path}")
            
        except Exception as e:
            self.logger.error(f"初期状態設定エラー: {e}")
            # フォールバック: 通常の更新処理を実行
            self.refresh_categories()
    
    def setup_training_tab(self):
        """学習タブ設定"""
        self.training_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.training_frame, text="モデル学習")
        
        # 学習設定
        config_frame = ttk.LabelFrame(self.training_frame, text="学習設定", padding="10")
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        # config.yamlからデフォルト値を取得（新しい設定を確実に読み込み）
        self.training_manager.config = self.training_manager.config_manager.get_config()
        default_params = self.training_manager.get_default_params()
        
        # モデル選択
        ttk.Label(config_frame, text="モデルタイプ:").grid(row=0, column=0, sticky="w")
        self.model_type_var = tk.StringVar(value="PaDiM")
        model_combo = ttk.Combobox(config_frame, textvariable=self.model_type_var, 
                                  values=["PaDiM", "PatchCore", "FastFlow"], state="readonly")
        model_combo.grid(row=0, column=1, sticky="w", padx=(5, 0))
        
        # エポック数
        ttk.Label(config_frame, text="エポック数:").grid(row=1, column=0, sticky="w")
        self.epochs_var = tk.StringVar(value=str(default_params['epochs']))
        ttk.Entry(config_frame, textvariable=self.epochs_var, width=10).grid(row=1, column=1, sticky="w", padx=(5, 0))
        
        # バッチサイズ
        ttk.Label(config_frame, text="バッチサイズ:").grid(row=2, column=0, sticky="w")
        self.batch_size_var = tk.StringVar(value=str(default_params['batch_size']))
        ttk.Entry(config_frame, textvariable=self.batch_size_var, width=10).grid(row=2, column=1, sticky="w", padx=(5, 0))
        
        # 学習率
        ttk.Label(config_frame, text="学習率:").grid(row=0, column=2, sticky="w", padx=(20, 0))
        self.learning_rate_var = tk.StringVar(value=str(default_params['learning_rate']))
        ttk.Entry(config_frame, textvariable=self.learning_rate_var, width=10).grid(row=0, column=3, sticky="w", padx=(5, 0))
        
        # デバイス選択
        ttk.Label(config_frame, text="デバイス:").grid(row=1, column=2, sticky="w", padx=(20, 0))
        self.device_var = tk.StringVar(value=default_params['device'])
        device_combo = ttk.Combobox(config_frame, textvariable=self.device_var,
                                   values=["auto", "cpu", "cuda"], state="readonly", width=8)
        device_combo.grid(row=1, column=3, sticky="w", padx=(5, 0))
        
        # 早期停止
        ttk.Label(config_frame, text="早期停止:").grid(row=2, column=2, sticky="w", padx=(20, 0))
        self.early_stopping_var = tk.StringVar(value=str(default_params['early_stopping_patience']))
        ttk.Entry(config_frame, textvariable=self.early_stopping_var, width=10).grid(row=2, column=3, sticky="w", padx=(5, 0))
        
        # リセットボタン
        ttk.Button(config_frame, text="デフォルト値に戻す", 
                  command=self.reset_training_params).grid(row=3, column=0, columnspan=2, pady=(10, 0), sticky="w")
        
        # 学習実行
        execution_frame = ttk.LabelFrame(self.training_frame, text="学習実行", padding="10")
        execution_frame.pack(fill=tk.X, pady=(0, 10))
        
        # プログレスバー
        ttk.Label(execution_frame, text="進行状況:").pack(anchor="w")
        self.training_progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(execution_frame, variable=self.training_progress_var, 
                                          maximum=100, length=400)
        self.progress_bar.pack(fill=tk.X, pady=(5, 10))
        
        # ステータス
        self.training_status_var = tk.StringVar(value="学習未開始")
        ttk.Label(execution_frame, textvariable=self.training_status_var).pack(anchor="w")
        
        # ボタン
        button_frame = ttk.Frame(execution_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.start_button = ttk.Button(button_frame, text="学習開始", command=self.start_training)
        self.start_button.pack(side=tk.LEFT)
        
        self.stop_button = ttk.Button(button_frame, text="学習停止", command=self.stop_training, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=(5, 0))
        
        # 学習ログ
        log_frame = ttk.LabelFrame(self.training_frame, text="学習ログ", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        log_text_frame = ttk.Frame(log_frame)
        log_text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.training_log_text = tk.Text(log_text_frame, height=10, wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(log_text_frame, orient=tk.VERTICAL, command=self.training_log_text.yview)
        self.training_log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.training_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_model_tab(self):
        """モデルタブ設定"""
        self.model_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.model_frame, text="モデル管理")
        
        # モデル一覧
        list_frame = ttk.LabelFrame(self.model_frame, text="学習済みモデル", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # ツリービュー
        tree_frame = ttk.Frame(list_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.model_tree = ttk.Treeview(tree_frame, columns=("name", "type", "created", "accuracy"), show="headings")
        self.model_tree.heading("name", text="モデル名")
        self.model_tree.heading("type", text="タイプ")
        self.model_tree.heading("created", text="作成日時")
        self.model_tree.heading("accuracy", text="精度")
        
        self.model_tree.column("name", width=200)
        self.model_tree.column("type", width=100)
        self.model_tree.column("created", width=150)
        self.model_tree.column("accuracy", width=100)
        
        # モデルIDを保存するための辞書
        self.model_id_map = {}  # {tree_item_id: model_id}
        
        model_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.model_tree.yview)
        self.model_tree.configure(yscrollcommand=model_scrollbar.set)
        
        self.model_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        model_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # モデル操作
        operation_frame = ttk.LabelFrame(self.model_frame, text="モデル操作", padding="10")
        operation_frame.pack(fill=tk.X)
        
        operation_button_frame = ttk.Frame(operation_frame)
        operation_button_frame.pack(fill=tk.X)
        
        ttk.Button(operation_button_frame, text="リスト更新", command=self.refresh_model_list).pack(side=tk.LEFT)
        ttk.Button(operation_button_frame, text="OpenVINO変換", command=self.convert_to_openvino).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(operation_button_frame, text="モデル削除", command=self.delete_model).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(operation_button_frame, text="推論用エクスポート", command=self.export_for_inference).pack(side=tk.LEFT, padx=(5, 0))
    
    def setup_menu(self):
        """メニューバー設定"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # ファイルメニュー
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ファイル", menu=file_menu)
        file_menu.add_command(label="設定", command=self.open_settings)
        file_menu.add_separator()
        file_menu.add_command(label="終了", command=self.root.quit)
        
        # ヘルプメニュー
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ヘルプ", menu=help_menu)
        help_menu.add_command(label="使い方", command=self.show_usage)
        help_menu.add_command(label="バージョン情報", command=self.show_about)
    
    def setup_status_bar(self, parent):
        """ステータスバー設定"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_var = tk.StringVar(value="準備完了")
        ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN).pack(fill=tk.X)
    
    # データセット操作
    def select_dataset_folder(self):
        """データセットフォルダ選択"""
        if self.dataset_type_var.get() == "mvtec":
            title = "MVTecデータセットのルートフォルダを選択"
        else:
            title = "カスタムデータセットのカテゴリフォルダを選択"
            
        folder_path = filedialog.askdirectory(title=title)
        if folder_path:
            if self.dataset_type_var.get() == "mvtec":
                # MVTecの場合はルートフォルダを選択
                self.dataset_manager.base_path = Path(folder_path)
                self.dataset_manager.category_path = self.dataset_manager.base_path / self.dataset_manager.category
                self.current_dataset_path = str(self.dataset_manager.category_path)
            else:
                # カスタムの場合はカテゴリフォルダを直接選択
                self.current_dataset_path = folder_path
                
            self.dataset_path_var.set(self.current_dataset_path)
            self.validate_dataset()
            self.logger.info(f"データセットフォルダ選択: {folder_path}")
    
    def validate_dataset(self):
        """データセット検証"""
        if not self.current_dataset_path:
            messagebox.showwarning("警告", "データセットパスが選択されていません")
            return
        
        try:
            self.status_var.set("データセット検証中...")
            
            # データセット検証
            is_valid = self.dataset_manager.validate_dataset(self.current_dataset_path)
            
            if is_valid:
                # データセット情報取得
                self.current_dataset_info = self.dataset_manager.get_dataset_info(self.current_dataset_path)
                self.display_dataset_info(self.current_dataset_info)
                self.status_var.set("データセット検証完了")
                messagebox.showinfo("成功", "データセットの検証が完了しました")
            else:
                self.show_no_dataset_info()
                self.status_var.set("データセット検証失敗")
                messagebox.showerror("エラー", "無効なデータセット構造です")
                
        except Exception as e:
            self.logger.error(f"データセット検証エラー: {e}")
            self.status_var.set("エラー")
            messagebox.showerror("エラー", f"データセット検証エラー:\n{e}")
    
    def update_dataset_stats(self):
        """データセット統計更新"""
        if not self.current_dataset_info:
            messagebox.showwarning("警告", "データセットが選択されていません")
            return
        
        try:
            self.status_var.set("統計情報更新中...")
            
            # 統計情報更新
            self.current_dataset_info = self.dataset_manager.get_dataset_info(self.current_dataset_path)
            self.display_dataset_info(self.current_dataset_info)
            
            self.status_var.set("統計情報更新完了")
            
        except Exception as e:
            self.logger.error(f"統計情報更新エラー: {e}")
            self.status_var.set("エラー")
            messagebox.showerror("エラー", f"統計情報更新エラー:\n{e}")
    
    def display_dataset_info(self, dataset_info: DatasetInfo):
        """データセット情報表示"""
        try:
            # dataset_pathがない場合はcurrent_dataset_pathを使用
            dataset_path = getattr(dataset_info, 'dataset_path', None) or self.current_dataset_path or 'N/A'
            
            info_text = f"""データセット情報:
カテゴリ: {dataset_info.category}
データセットパス: {dataset_path}

画像統計:
- 学習画像数: {dataset_info.train_count}
- テスト正常画像数: {dataset_info.test_normal_count}
- テスト異常画像数: {dataset_info.test_defect_count}
- 合計画像数: {dataset_info.total_count}

異常タイプ:
- 種類数: {len(dataset_info.defect_types)}
- タイプ: {', '.join(dataset_info.defect_types)}

検証状況:
- 検証ステータス: {"✓ 有効" if dataset_info.is_valid else "✗ 無効"}
- エラー数: {len(dataset_info.validation_errors)}
"""
            
            self.dataset_info_text.delete(1.0, tk.END)
            self.dataset_info_text.insert(1.0, info_text)
            
        except Exception as e:
            self.logger.error(f"データセット情報表示エラー: {e}")
    
    def show_no_dataset_info(self):
        """データセット情報なし表示"""
        self.dataset_info_text.delete(1.0, tk.END)
        self.dataset_info_text.insert(1.0, "データセットが選択されていません\n\n「参照」ボタンからMVTecデータセットフォルダを選択してください")
    
    # 学習操作
    def start_training(self):
        """学習開始"""
        # データセットパスとカテゴリの再確認
        if not self.current_dataset_path:
            # パスが未設定の場合、現在のカテゴリから再設定
            category = self.category_var.get()
            if category:
                self.dataset_manager.set_category(category)
                self.current_dataset_path = str(self.dataset_manager.category_path)
                self.dataset_path_var.set(self.current_dataset_path)
        
        # データセット検証
        if not self.current_dataset_path:
            messagebox.showwarning("警告", "データセットパスが設定されていません。\n\n「データセット管理」タブでデータセットを選択してください。")
            return
            
        # パスの存在確認
        dataset_path = Path(self.current_dataset_path)
        if not dataset_path.exists():
            # 絶対パスでも確認
            abs_path = Path.cwd() / self.current_dataset_path
            self.logger.error(f"データセットパス不存在: 相対パス={self.current_dataset_path}, 絶対パス={abs_path}")
            messagebox.showwarning("警告", f"データセットパスが存在しません:\n相対パス: {self.current_dataset_path}\n絶対パス: {abs_path}\n\n正しいパスを設定してください。")
            return
            
        # データセット構造の検証
        try:
            is_valid = self.dataset_manager.validate_dataset(self.current_dataset_path)
            if not is_valid:
                # より詳細な検証情報を取得
                dataset_info = self.dataset_manager.get_dataset_info(self.current_dataset_path)
                error_msg = f"データセット構造が無効です:\n\nパス: {self.current_dataset_path}\nカテゴリ: {self.category_var.get()}\n"
                if hasattr(dataset_info, 'validation_errors') and dataset_info.validation_errors:
                    error_msg += f"\nエラー詳細:\n" + "\n".join(dataset_info.validation_errors[:3])  # 最初の3つのエラーのみ表示
                messagebox.showwarning("警告", error_msg)
                return
        except Exception as e:
            messagebox.showerror("エラー", f"データセット検証中にエラーが発生しました:\n{str(e)}")
            return
        
        # 二重チェック：GUIとTrainingManager両方で確認
        if self.is_training or self.training_manager.is_training:
            messagebox.showwarning("警告", "既に学習が実行中です")
            return
        
        try:
            # 学習パラメータ取得
            model_type = self.model_type_var.get()
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_size_var.get())
            learning_rate = float(self.learning_rate_var.get())
            device = self.device_var.get()
            early_stopping = int(self.early_stopping_var.get())
            
            # GUI設定を一時的に適用
            self.apply_gui_settings_to_config(model_type, epochs, batch_size, 
                                            learning_rate, device, early_stopping)
            
            # 学習状態を設定（GUI側とTrainingManager側両方）
            self.is_training = True
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            
            self.training_status_var.set("学習準備中...")
            self.training_progress_var.set(0)
            
            # 学習スレッド開始
            self.training_thread = threading.Thread(
                target=self.run_training,
                args=(model_type, epochs, batch_size),
                daemon=True
            )
            self.training_thread.start()
            
            self.logger.info(f"学習開始: {model_type}, epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, device={device}")
            
        except ValueError as e:
            messagebox.showerror("エラー", "学習パラメータが不正です")
        except Exception as e:
            self.logger.error(f"学習開始エラー: {e}")
            messagebox.showerror("エラー", f"学習開始エラー:\n{e}")
            self.reset_training_ui()
    
    def apply_gui_settings_to_config(self, model_type: str, epochs: int, batch_size: int, 
                                   learning_rate: float, device: str, early_stopping: int):
        """GUI設定をTrainingManagerに適用"""
        try:
            # TrainingManagerのGUI可変パラメータを設定
            self.training_manager.set_gui_params(
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                device=device,
                early_stopping_patience=early_stopping
            )
            
            # モデル名をconfig経由で設定
            config = self.training_manager.config_manager.get_config()
            if 'training' not in config:
                config['training'] = {}
            config['training']['model_name'] = model_type.lower()
            
            self.logger.info(f"GUI設定適用完了: モデル={model_type}, エポック={epochs}, バッチサイズ={batch_size}")
            
        except Exception as e:
            self.logger.error(f"GUI設定適用エラー: {e}")
            raise
    
    def reset_training_params(self):
        """学習パラメータをデフォルト値にリセット"""
        try:
            default_params = self.training_manager.get_default_params()
            
            self.epochs_var.set(str(default_params['epochs']))
            self.batch_size_var.set(str(default_params['batch_size']))
            self.learning_rate_var.set(str(default_params['learning_rate']))
            self.device_var.set(default_params['device'])
            self.early_stopping_var.set(str(default_params['early_stopping_patience']))
            
            self.logger.info("学習パラメータをデフォルト値にリセットしました")
            
        except Exception as e:
            self.logger.error(f"パラメータリセットエラー: {e}")
            messagebox.showerror("エラー", f"パラメータのリセットに失敗しました: {e}")
    
    def run_training(self, model_type: str, epochs: int, batch_size: int):
        """学習実行（バックグラウンド）"""
        try:
            # 現在選択されているカテゴリをTrainingManagerに設定
            current_category = self.current_dataset_info.category if self.current_dataset_info else "bottle"
            self.training_manager.update_category(current_category)
            
            # 進捗コールバック（TrainingManagerは2引数で呼び出す）
            def progress_callback(progress, message):
                # progress: 0-100の進捗率
                # message: 進捗メッセージ
                if progress >= 100:
                    # 学習完了
                    self.root.after(0, lambda: self.update_training_progress_simple(progress, message))
                    self.root.after(0, lambda: self.on_training_complete(True))
                else:
                    # 進捗更新
                    self.root.after(0, lambda: self.update_training_progress_simple(progress, message))
            
            # 学習実行
            result = self.training_manager.start_training(
                progress_callback=progress_callback
            )
            
            # 学習開始成功確認のみ（完了処理はprogress_callbackで実施）
            if not result:
                error_message = "学習開始に失敗しました"
                self.root.after(0, lambda: self.on_training_error(error_message))
            
        except Exception as e:
            self.logger.error(f"学習実行エラー: {e}")
            error_message = str(e)
            self.root.after(0, lambda: self.on_training_error(error_message))
    
    def update_training_progress(self, epoch: int, total_epochs: int, loss: float, progress: float):
        """学習進捗更新（詳細版）"""
        self.training_progress_var.set(progress)
        self.training_status_var.set(f"学習中... Epoch {epoch}/{total_epochs}, Loss: {loss:.4f}")
        
        # ログ更新
        log_message = f"Epoch {epoch}/{total_epochs}: Loss = {loss:.4f}\n"
        self.training_log_text.insert(tk.END, log_message)
        self.training_log_text.see(tk.END)
    
    def update_training_progress_simple(self, progress: float, message: str):
        """学習進捗更新（シンプル版）"""
        self.training_progress_var.set(progress)
        self.training_status_var.set(message)
        
        # ログ更新
        log_message = f"{message} ({progress:.1f}%)\n"
        self.training_log_text.insert(tk.END, log_message)
        self.training_log_text.see(tk.END)
    
    def on_training_complete(self, result):
        """学習完了処理"""
        self.training_progress_var.set(100)
        self.training_status_var.set("学習完了")
        
        # ログ更新（resultはbool型なので適切に処理）
        if isinstance(result, bool):
            if result:
                self.training_log_text.insert(tk.END, f"\n✅ 学習完了!\n")
            else:
                self.training_log_text.insert(tk.END, f"\n❌ 学習失敗\n")
        elif isinstance(result, dict):
            self.training_log_text.insert(tk.END, f"\n学習完了!\n最終精度: {result.get('accuracy', 'N/A')}\n")
        else:
            self.training_log_text.insert(tk.END, f"\n学習完了!\n結果: {result}\n")
        
        self.training_log_text.see(tk.END)
        
        # UI リセット
        self.reset_training_ui()
        
        # モデル一覧更新
        self.refresh_model_list()
        
        messagebox.showinfo("成功", "学習が完了しました")
        self.logger.info("学習完了")
    
    def on_training_error(self, error):
        """学習エラー処理"""
        self.training_status_var.set("学習エラー")
        
        # ログ更新
        self.training_log_text.insert(tk.END, f"\n学習エラー: {error}\n")
        self.training_log_text.see(tk.END)
        
        # UI リセット
        self.reset_training_ui()
        
        messagebox.showerror("エラー", f"学習エラー:\n{error}")
    
    def stop_training(self):
        """学習停止"""
        if not self.is_training:
            return
        
        if messagebox.askyesno("確認", "学習を停止しますか？"):
            self.training_status_var.set("学習停止中...")
            # 実際の停止処理は training_manager に依存
            self.reset_training_ui()
            self.logger.info("学習停止")
    
    def reset_training_ui(self):
        """学習UI リセット"""
        self.is_training = False
        # TrainingManagerの状態もリセット
        if hasattr(self.training_manager, 'is_training'):
            self.training_manager.is_training = False
        
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.training_thread = None
    
    # モデル操作
    def refresh_model_list(self):
        """モデル一覧更新"""
        try:
            # 既存項目クリア
            for item in self.model_tree.get_children():
                self.model_tree.delete(item)
            self.model_id_map.clear()
            
            # モデル一覧取得
            models = self.model_manager.list_models()
            
            for model in models:
                # ModelInfoオブジェクトの場合は属性アクセス、辞書の場合はget()
                if hasattr(model, 'name'):
                    # ModelInfoオブジェクト
                    model_id = model.model_id
                    name = model.name
                    # metadataからmodel_typeを取得、なければunknownから推測
                    model_type = model.metadata.get('model_type', 'Unknown')
                    if model_type == 'Unknown' and hasattr(model, 'model_id'):
                        # model_idから推測 (例: hazelnut_20250923_011324)
                        if 'padim' in name.lower():
                            model_type = 'PADIM'
                        elif 'patchcore' in name.lower():
                            model_type = 'PatchCore'
                        elif 'fastflow' in name.lower():
                            model_type = 'FastFlow'
                    
                    created_at = model.created_at.strftime('%Y-%m-%d %H:%M') if hasattr(model.created_at, 'strftime') else str(model.created_at)
                    
                    # 精度の表示改善
                    if model.accuracy and model.accuracy > 0:
                        accuracy = f"{model.accuracy:.3f}"
                    else:
                        accuracy = "N/A"
                else:
                    # 辞書形式（後方互換性）
                    model_id = model.get("model_id", "Unknown")
                    name = model.get("name", "Unknown")
                    model_type = model.get("model_type", "Unknown")
                    created_at = model.get("created_at", "Unknown")
                    accuracy = f"{model.get('accuracy', 0):.3f}" if model.get('accuracy') else "N/A"
                
                item_id = self.model_tree.insert("", "end", values=(
                    name, model_type, created_at, accuracy
                ))
                
                # model_idをマッピングに保存
                self.model_id_map[item_id] = model_id
            
            self.logger.info(f"モデル一覧更新: {len(models)}件")
            
        except Exception as e:
            self.logger.error(f"モデル一覧更新エラー: {e}")
            messagebox.showerror("エラー", f"モデル一覧更新エラー:\n{e}")
    
    def convert_to_openvino(self):
        """OpenVINO変換"""
        selection = self.model_tree.selection()
        if not selection:
            messagebox.showwarning("警告", "変換するモデルを選択してください")
            return
        
        try:
            selected_item = selection[0]
            model_id = self.model_id_map.get(selected_item)
            
            if not model_id:
                messagebox.showerror("エラー", "モデルIDが取得できませんでした")
                return
            
            # 変換実行
            result = self.model_manager.convert_to_openvino(model_id)
            
            if result:
                messagebox.showinfo("成功", f"OpenVINO変換が完了しました:\n{result}")
                self.refresh_model_list()
            else:
                messagebox.showerror("エラー", "OpenVINO変換に失敗しました")
                
        except Exception as e:
            self.logger.error(f"OpenVINO変換エラー: {e}")
            messagebox.showerror("エラー", f"OpenVINO変換エラー:\n{e}")
    
    def delete_model(self):
        """モデル削除"""
        selection = self.model_tree.selection()
        if not selection:
            messagebox.showwarning("警告", "削除するモデルを選択してください")
            return
        
        try:
            selected_item = selection[0]
            model_name = self.model_tree.item(selected_item)["values"][0]
            model_id = self.model_id_map.get(selected_item)
            
            if not model_id:
                messagebox.showerror("エラー", "モデルIDが取得できませんでした")
                return
        
            if messagebox.askyesno("確認", f"モデル '{model_name}' を削除しますか？"):
                result = self.model_manager.delete_model(model_id)
                
                if result:
                    messagebox.showinfo("成功", "モデルを削除しました")
                    self.refresh_model_list()
                else:
                    messagebox.showerror("エラー", "モデル削除に失敗しました")
                    
        except Exception as e:
            self.logger.error(f"モデル削除エラー: {e}")
            messagebox.showerror("エラー", f"モデル削除エラー:\n{e}")
    
    def export_for_inference(self):
        """推論用エクスポート"""
        selection = self.model_tree.selection()
        if not selection:
            messagebox.showwarning("警告", "エクスポートするモデルを選択してください")
            return
        
        try:
            selected_item = selection[0]
            model_name = self.model_tree.item(selected_item)["values"][0]
            model_id = self.model_id_map.get(selected_item)
            
            if not model_id:
                messagebox.showerror("エラー", "モデルIDが取得できませんでした")
                return
            
            # エクスポート先選択
            export_path = filedialog.askdirectory(title="エクスポート先フォルダを選択")
            if not export_path:
                return
            
            # エクスポート実行
            result = self.model_manager.export_for_inference(model_id, export_path)
            
            if result:
                messagebox.showinfo("成功", f"推論用エクスポートが完了しました:\n{result}")
            else:
                messagebox.showerror("エラー", "推論用エクスポートに失敗しました")
                
        except Exception as e:
            self.logger.error(f"推論用エクスポートエラー: {e}")
            messagebox.showerror("エラー", f"推論用エクスポートエラー:\n{e}")
    
    # メニュー操作
    def open_settings(self):
        """設定画面表示"""
        messagebox.showinfo("設定", "設定機能は今後のバージョンで実装予定です")
    
    def show_usage(self):
        """使い方表示"""
        usage_text = """異常検出システム - 学習アプリの使い方

1. データセット管理
   ■ データセットタイプ選択
   - MVTec AD: 標準的なMVTecデータセット形式
   - カスタム: 独自のデータセット形式
   
   ■ カテゴリ選択
   - MVTec: 15カテゴリから選択 (bottle, cable, etc.)
   - カスタム: フォルダ構造から自動検出
   
   ■ データセット検証
   - 「参照」ボタンでデータセットフォルダを選択
   - 自動的に構造検証と統計情報表示

2. データセット構造
   ■ MVTec形式:
   dataset_root/
   ├── category_name/
   │   ├── train/good/
   │   └── test/good/ + defect_types/
   
   ■ カスタム形式:
   category_folder/
   ├── train/class1/, class2/, ...
   └── test/class1/, class2/, ...

3. モデル学習
   - データセット選択後、学習設定を調整
   - 「学習開始」で学習を実行
   - 進捗とログをリアルタイムで確認

4. モデル管理
   - 学習済みモデルの一覧表示
   - OpenVINO形式への変換
   - 推論アプリ用のエクスポート

注意事項:
- 学習にはGPUの使用を推奨
- データセット構造は事前に確認してください
- モデルファイルは自動的に保存されます
"""
        
        # 使い方ウィンドウ表示
        usage_window = tk.Toplevel(self.root)
        usage_window.title("使い方")
        usage_window.geometry("500x400")
        
        text_widget = tk.Text(usage_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(1.0, usage_text)
        text_widget.configure(state="disabled")
        
        ttk.Button(usage_window, text="閉じる", command=usage_window.destroy).pack(pady=10)
    
    def show_about(self):
        """バージョン情報表示"""
        about_text = """異常検出システム - 学習アプリ

バージョン: 1.0.0 (Phase1 MVP)
開発: AI異常検出プロジェクト

機能:
- MVTec bottleデータセット対応
- 複数の異常検出モデル (PaDiM, PatchCore, FastFlow)
- OpenVINO最適化対応
- GUI学習環境

技術スタック:
- Python 3.8+
- Anomalib
- OpenVINO
- Tkinter

© 2024 AI Anomaly Detection Project
"""
        messagebox.showinfo("バージョン情報", about_text)