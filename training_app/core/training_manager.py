"""学習実行管理 - シンプル版"""

import os
import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime

import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore
from lightning.pytorch.callbacks import ModelCheckpoint

from shared.config import get_config_manager
from shared.utils import setup_logger, PerformanceTimer


class TrainingManager:
    """シンプルな学習実行管理"""
    
    def __init__(self, config_manager=None, dataset_manager=None, dataset_type: str = None, category: str = None):
        self.config_manager = config_manager or get_config_manager()
        self.config = self.config_manager.get_config()
        self.logger = setup_logger("training_manager")
        
        # 基本設定
        self.dataset_type = dataset_type or 'mvtec'
        self.category = category or 'hazelnut'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 学習状態
        self.is_training = False
        self.training_thread = None
        self.progress_callback = None
        self.training_results = {}
        
        self.logger.info(f"TrainingManager初期化: {self.dataset_type}/{self.category}, デバイス: {self.device}")
    
    def update_category(self, category: str):
        """カテゴリ更新"""
        self.category = category
        self.logger.info(f"学習カテゴリ更新: {category}")
    
    def create_model(self) -> torch.nn.Module:
        """モデル作成（ローカル事前学習済みモデル使用）"""
        model_name = self.config.get('training.model_name', 'padim')
        
        # 完全オフライン設定
        os.environ['TORCH_HUB_NO_DOWNLOAD'] = '1'
        os.environ['TIMM_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        try:
            if model_name.lower() == 'padim':
                model = Padim(backbone="resnet18")
            elif model_name.lower() == 'patchcore':
                model = Patchcore(
                    backbone="resnet18",
                    coreset_sampling_ratio=0.01,
                    num_neighbors=5
                )
            else:
                model = Padim(backbone="resnet18")
                model_name = 'padim'
            
            # ローカル事前学習済み重みをロード（一時的に無効化）
            # self._load_local_backbone(model)
            
            total_params = sum(p.numel() for p in model.parameters())
            self.logger.info(f"モデル作成完了: {model_name.upper()}, パラメータ数: {total_params:,}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"モデル作成エラー: {e}")
            raise RuntimeError(f"モデル作成に失敗: {e}")
    
    def _load_local_backbone(self, model):
        """ローカル保存されたResNet18重みをロード"""
        try:
            # ローカルモデルパス
            models_dir = Path("models/development/pretrained_models")
            model_path = models_dir / "resnet18_pretrained.pth"
            
            if not model_path.exists():
                raise FileNotFoundError(f"ローカル事前学習済みモデルが見つかりません: {model_path}")
            
            # ローカル重みをロード
            local_weights = torch.load(model_path, map_location='cpu')
            self.logger.info(f"ローカルモデルロード: {model_path}")
            
            # anomalibモデルのbackboneに重みを適用
            if hasattr(model, 'model') and hasattr(model.model, 'backbone'):
                backbone = model.model.backbone
                # ResNetのbackbone部分のみ重みをロード（分類層は除外）
                backbone_state_dict = {}
                for key, value in local_weights.items():
                    if not key.startswith('fc.'):  # 分類層は除外
                        backbone_state_dict[key] = value
                
                backbone.load_state_dict(backbone_state_dict, strict=False)
                self.logger.info("ローカルResNet18重みロード完了")
            
        except Exception as e:
            self.logger.error(f"ローカル重みロードエラー: {e}")
            raise RuntimeError(f"事前学習済みモデルの読み込みに失敗しました: {e}")
    
    def create_datamodule(self) -> MVTecAD:
        """データモジュール作成（setup_datasetのエイリアス）"""
        return self.setup_dataset()
    
    def setup_dataset(self) -> MVTecAD:
        """データセット準備"""
        self.logger.info(f"データセット準備開始: {self.category}")
        
        dataset_config = {
            "root": "./datasets/development/mvtec_anomaly_detection",
            "category": self.category,
            "train_batch_size": 16,
            "eval_batch_size": 16,
            "num_workers": 4,  # パフォーマンス向上
            "test_split_mode": "from_dir",
            "val_split_mode": "same_as_test",
            "val_split_ratio": 0.2,
        }
        
        dataset = MVTecAD(**dataset_config)
        dataset.setup()
        
        train_size = len(dataset.train_dataloader().dataset)
        val_size = len(dataset.val_dataloader().dataset) if dataset.val_dataloader() else 0
        test_size = len(dataset.test_dataloader().dataset)
        
        self.logger.info(f"データセット準備完了: 学習={train_size}, 検証={val_size}, テスト={test_size}")
        return dataset
    
    def start_training(self, progress_callback: Optional[Callable] = None) -> bool:
        """学習開始"""
        if self.is_training:
            self.logger.warning("既に学習実行中")
            return False
        
        self.progress_callback = progress_callback
        self.is_training = True
        
        self.training_thread = threading.Thread(target=self._train_worker)
        self.training_thread.start()
        
        self.logger.info("学習開始")
        return True
    
    def stop_training(self) -> bool:
        """学習停止"""
        self.is_training = False
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5.0)
        
        self.logger.info("学習停止")
        return True
    
    def _train_worker(self):
        """学習実行ワーカー"""
        try:
            with PerformanceTimer(self.logger, "training_execution"):
                self._execute_training()
        except Exception as e:
            self.logger.error(f"学習実行エラー: {e}")
            self.training_results = {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        finally:
            self.is_training = False
            if self.progress_callback:
                status = "学習完了" if "error" not in self.training_results else "学習エラー"
                self.progress_callback(100, status)
    
    def _execute_training(self):
        """実際の学習実行"""
        # データセット準備
        if self.progress_callback:
            self.progress_callback(10, "データセット準備中...")
        
        dataset = self.setup_dataset()
        
        # モデル作成
        if self.progress_callback:
            self.progress_callback(20, "モデル作成中...")
        
        model = self.create_model()
        
        # 学習設定
        if self.progress_callback:
            self.progress_callback(30, "学習設定中...")
        
        # コールバック設定
        callbacks = []
        if self.progress_callback:
            callbacks.append(self._create_progress_callback())

        # Engine作成（anomalib内部でModelCheckpointが自動設定される）
        checkpoint_dir = Path("./models/development/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        engine = Engine(
            logger=False,
            max_epochs=self.config.get('training.max_epochs', 10),
            accelerator="gpu" if self.device == "cuda" else "cpu",
            devices=1,
            callbacks=callbacks,
            enable_progress_bar=False,
            default_root_dir=str(checkpoint_dir.parent)
        )
        
        # 学習実行
        if self.progress_callback:
            self.progress_callback(40, "学習開始...")
        
        start_time = datetime.now()
        self.logger.info("学習開始")
        
        engine.fit(model=model, datamodule=dataset)
        
        # テスト実行
        if self.progress_callback:
            self.progress_callback(80, "テスト評価中...")
        
        test_results = engine.test(model=model, datamodule=dataset)
        
        # 結果保存
        if self.progress_callback:
            self.progress_callback(90, "結果保存中...")
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # 学習結果まとめ
        self.training_results = {
            "status": "success",
            "category": self.category,
            "model_name": self.config.get('training.model_name', 'padim'),
            "training_time_seconds": training_time,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "test_results": test_results if test_results else {},
            "config": {
                "max_epochs": self.config.get('training.max_epochs', 10),
                "device": self.device,
            }
        }
        
        # モデル保存
        try:
            model_path = self._save_model(model)
            self.training_results["model_path"] = str(model_path)
            self.logger.info(f"モデル保存完了: {model_path}")
        except Exception as e:
            self.logger.warning(f"モデル保存エラー: {e}")
        
        self.logger.info(f"学習完了: {training_time:.1f}秒")
    
    def _create_progress_callback(self):
        """進捗コールバック作成"""
        from lightning.pytorch.callbacks import Callback
        
        class ProgressCallback(Callback):
            def __init__(self, progress_func, total_epochs):
                self.progress_func = progress_func
                self.total_epochs = total_epochs
            
            def on_train_epoch_end(self, trainer, pl_module):
                if self.progress_func:
                    epoch = trainer.current_epoch + 1
                    progress = 40 + int((epoch / self.total_epochs) * 40)  # 40-80%
                    self.progress_func(progress, f"エポック {epoch}/{self.total_epochs}")
        
        max_epochs = self.config.get('training.max_epochs', 10)
        return ProgressCallback(self.progress_callback, max_epochs)
    
    def _save_model(self, model):
        """学習済みモデル保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = Path(f"./models/development/{self.category}_{timestamp}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # PyTorchモデルとして保存
        model_path = model_dir / "model.pth"
        torch.save(model.state_dict(), model_path)
        
        # メタデータ保存
        metadata = {
            "model_id": f"{self.category}_{timestamp}",
            "name": f"{self.config.get('training.model_name', 'PaDiM').upper()} ({self.category})",
            "category": self.category,
            "created_at": datetime.now().isoformat(),
            "training_results": self.training_results,
            "model_path": str(model_path),
        }
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return model_path
    
    def get_training_results(self) -> Dict[str, Any]:
        """学習結果取得"""
        return self.training_results.copy()
    
    def is_training_active(self) -> bool:
        """学習実行中確認"""
        return self.is_training
    
    def cleanup(self):
        """クリーンアップ"""
        self.is_training = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("TrainingManager クリーンアップ完了")