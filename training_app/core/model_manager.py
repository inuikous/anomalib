"""モデル管理 - 保存・変換・エクスポート"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

# OpenVINO imports
import openvino as ov

from shared.config import get_config_manager
from shared.domain import ModelInfo
from shared.utils import (
    setup_logger, log_function_call, safe_copy_file, 
    compress_directory, create_directory_if_not_exists, get_directory_size
)


class ModelManager:
    """学習済みモデル管理"""
    
    def __init__(self, config_manager=None):
        """
        初期化
        
        Args:
            config_manager: 設定管理インスタンス
        """
        self.config_manager = config_manager or get_config_manager()
        self.config = self.config_manager.get_config()
        self.logger = setup_logger("model_manager")
        
        # パス設定
        models_config = self.config.get('models', {})
        
        # モデル保存パス
        if models_config.get('save_path'):
            self.models_path = Path(models_config['save_path'])
        else:
            self.models_path = Path("./models")
            self.logger.warning(f"モデル保存パスが未設定のため、デフォルトを使用: {self.models_path}")
            
        # OpenVINOパス
        if models_config.get('openvino_path'):
            self.openvino_path = Path(models_config['openvino_path'])
        else:
            self.openvino_path = Path("./models/openvino")
            self.logger.warning(f"OpenVINOパスが未設定のため、デフォルトを使用: {self.openvino_path}")
            
        # エクスポートパス
        self.export_path = Path(models_config.get('export_path', './models/export'))
        
        # ディレクトリ作成
        create_directory_if_not_exists(self.models_path)
        create_directory_if_not_exists(self.openvino_path)
        create_directory_if_not_exists(self.export_path)
        
        self.logger.info(f"ModelManager初期化: models={self.models_path}")
    
    @log_function_call
    def save_model(self, model: Any, metadata: Dict[str, Any]) -> str:
        """
        モデル保存
        
        Args:
            model: 保存するモデル
            metadata: メタデータ
            
        Returns:
            モデルID
        """
        try:
            # モデルID生成
            model_id = metadata.get('model_id') or f"{metadata.get('category', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 保存ディレクトリ作成
            model_dir = self.models_path / model_id
            create_directory_if_not_exists(model_dir)
            
            # メタデータ保存
            full_metadata = {
                "model_id": model_id,
                "name": metadata.get('name', f"Model_{model_id}"),
                "category": metadata.get('category', 'bottle'),
                "created_at": datetime.now().isoformat(),
                "accuracy": metadata.get('accuracy', 0.0),
                "model_size_mb": 0.0,  # 後で更新
                "openvino_path": None,
                "status": "saved",
                "metadata": metadata
            }
            
            # モデルファイル保存
            model_file_path = model_dir / "model.pkl"  # 実際の形式に応じて調整
            
            if model is not None:
                # モデル保存処理（実際のモデル形式に応じて実装）
                # PyTorch, TensorFlow, scikit-learn等に対応
                self._save_model_file(model, model_file_path)
                
                # ファイルサイズ取得
                if model_file_path.exists():
                    full_metadata["model_size_mb"] = model_file_path.stat().st_size / (1024 * 1024)
            
            # メタデータファイル保存
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(full_metadata, f, indent=2, ensure_ascii=False)
            
            # README作成
            self._create_model_readme(model_dir, full_metadata)
            
            self.logger.info(f"モデル保存完了: {model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"モデル保存エラー: {e}")
            raise
    
    @log_function_call
    def convert_to_openvino(self, model_id: str) -> Optional[str]:
        """
        OpenVINO形式変換
        
        Args:
            model_id: モデルID
            
        Returns:
            OpenVINOモデルパス
        """
        try:
            # モデル情報取得
            model_info = self.get_model_info(model_id)
            if not model_info:
                raise ValueError(f"モデルが見つかりません: {model_id}")
            
            # 元モデルパス（複数の候補をチェック）
            model_dir = self.models_path / model_id
            model_file = None
            
            # 可能なモデルファイルを順番にチェック（実際のファイルのみ）
            possible_files = [
                model_dir / "model.ckpt", 
                model_dir / "model.pth",
                model_dir / "model.pt",
                model_dir / "model.onnx",
                model_dir / "best_model.pth"
            ]
            
            for file_path in possible_files:
                if file_path.exists():
                    model_file = file_path
                    self.logger.info(f"変換対象ファイル発見: {model_file}")
                    break
            
            if not model_file:
                raise FileNotFoundError(f"変換可能なモデルファイルが見つかりません: {model_dir}")
            
            # OpenVINO出力ディレクトリ
            openvino_dir = self.openvino_path / model_id
            create_directory_if_not_exists(openvino_dir)
            
            # OpenVINO変換実行
            ir_path = self._execute_openvino_conversion(model_file, openvino_dir)
            
            if ir_path:
                # メタデータ更新
                self._update_model_metadata(model_id, {"openvino_path": str(ir_path)})
                
                self.logger.info(f"OpenVINO変換完了: {model_id} -> {ir_path}")
                return str(ir_path)
            else:
                raise RuntimeError("OpenVINO変換失敗")
                
        except Exception as e:
            self.logger.error(f"OpenVINO変換エラー: {model_id}, エラー: {e}")
            return None
    
    @log_function_call
    def export_model_package(self, model_id: str, export_dir: Optional[str] = None) -> bool:
        """
        モデルパッケージエクスポート
        
        Args:
            model_id: モデルID
            export_dir: エクスポート先ディレクトリ
            
        Returns:
            エクスポート成功可否
        """
        try:
            # エクスポート先設定
            if export_dir:
                export_path = Path(export_dir)
            else:
                export_path = self.export_path
            
            create_directory_if_not_exists(export_path)
            
            # パッケージファイル名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            package_name = f"{model_id}_{timestamp}.zip"
            package_path = export_path / package_name
            
            # 一時ディレクトリでパッケージ作成
            temp_dir = export_path / f"temp_{model_id}"
            create_directory_if_not_exists(temp_dir)
            
            try:
                # モデルファイルコピー
                model_dir = self.models_path / model_id
                if model_dir.exists():
                    shutil.copytree(model_dir, temp_dir / "model")
                
                # OpenVINOファイルコピー
                openvino_dir = self.openvino_path / model_id
                if openvino_dir.exists():
                    shutil.copytree(openvino_dir, temp_dir / "openvino")
                
                # デプロイ用設定ファイル作成
                self._create_deployment_config(temp_dir, model_id)
                
                # ZIP圧縮
                success = compress_directory(temp_dir, package_path)
                
                if success:
                    self.logger.info(f"モデルパッケージエクスポート完了: {package_path}")
                    return True
                else:
                    raise RuntimeError("ZIP圧縮失敗")
                    
            finally:
                # 一時ディレクトリ削除
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    
        except Exception as e:
            self.logger.error(f"モデルパッケージエクスポートエラー: {model_id}, エラー: {e}")
            return False
    
    @log_function_call
    def save_trained_model(self, model: Any, model_name: str, training_results: Dict[str, Any]) -> Path:
        """
        anomalib学習済みモデルの保存
        
        Args:
            model: 学習済みanomalibモデル
            model_name: モデル名
            training_results: 学習結果
            
        Returns:
            保存されたモデルのパス
        """
        try:
            # モデルID生成
            model_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_dir = self.models_path / model_id
            create_directory_if_not_exists(model_dir)
            
            # PyTorchモデル保存
            model_path = model_dir / "model.pth"
            
            # anomalibモデルの状態辞書を保存
            if hasattr(model, 'state_dict'):
                import torch
                torch.save(model.state_dict(), model_path)
                self.logger.info(f"PyTorchモデル保存: {model_path}")
            
            # ONNXエクスポート（可能な場合）
            onnx_path = model_dir / "model.onnx"
            try:
                if hasattr(model, 'to_onnx'):
                    model.to_onnx(onnx_path)
                    self.logger.info(f"ONNXモデルエクスポート: {onnx_path}")
            except Exception as onnx_error:
                self.logger.warning(f"ONNXエクスポート失敗（続行）: {onnx_error}")
            
            # メタデータ作成
            metadata = {
                "model_id": model_id,
                "name": model_name,
                "category": training_results.get("category", "unknown"),
                "created_at": datetime.now().isoformat(),
                "model_type": training_results.get("model_name", "anomalib"),
                "model_size_mb": model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 0,
                "training_results": training_results,
                "accuracy": 0.0,  # テスト結果から取得
                "openvino_path": None,
                "status": "trained"
            }
            
            # テスト結果から精度を抽出
            test_results = training_results.get("test_results", {})
            if isinstance(test_results, dict):
                for key in ["AUROC", "auroc", "AUC", "auc"]:
                    if key in test_results:
                        metadata["accuracy"] = float(test_results[key])
                        break
            
            # メタデータ保存
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # README作成
            self._create_model_readme(model_dir, metadata)
            
            self.logger.info(f"学習済みモデル保存完了: {model_id}")
            return model_path
            
        except Exception as e:
            self.logger.error(f"学習済みモデル保存エラー: {e}")
            raise

    def list_models(self) -> List[ModelInfo]:
        """
        モデル一覧取得
        
        Returns:
            モデル情報リスト
        """
        models = []
        
        try:
            if not self.models_path.exists():
                return models
            
            for model_dir in self.models_path.iterdir():
                if model_dir.is_dir():
                    metadata_path = model_dir / "metadata.json"
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                            
                            # 必要なフィールドのデフォルト値設定
                            metadata.setdefault("accuracy", 0.0)
                            metadata.setdefault("model_size_mb", 0.0)
                            metadata.setdefault("openvino_path", None)
                            metadata.setdefault("metadata", {})
                            
                            # テスト結果から精度を取得
                            if "training_results" in metadata:
                                test_results = metadata["training_results"].get("test_results", {})
                                if "auroc" in test_results:
                                    metadata["accuracy"] = test_results["auroc"]
                            
                            model_info = ModelInfo.from_dict(metadata)
                            models.append(model_info)
                            
                        except Exception as e:
                            self.logger.warning(f"モデルメタデータ読み込みエラー: {metadata_path}, {e}")
            
            # 作成日時でソート
            models.sort(key=lambda x: x.created_at, reverse=True)
            
        except Exception as e:
            self.logger.error(f"モデル一覧取得エラー: {e}")
        
        return models
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        モデル詳細情報取得
        
        Args:
            model_id: モデルID
            
        Returns:
            モデル情報
        """
        try:
            metadata_path = self.models_path / model_id / "metadata.json"
            
            if not metadata_path.exists():
                self.logger.warning(f"モデルメタデータが見つかりません: {model_id}")
                return None
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 必要なフィールドのデフォルト値設定
            metadata.setdefault("accuracy", 0.0)
            metadata.setdefault("model_size_mb", 0.0)
            metadata.setdefault("openvino_path", None)
            metadata.setdefault("metadata", {})
            
            # テスト結果から精度を取得
            if "training_results" in metadata:
                test_results = metadata["training_results"].get("test_results", {})
                if "auroc" in test_results:
                    metadata["accuracy"] = test_results["auroc"]
            
            return ModelInfo.from_dict(metadata)
            
        except Exception as e:
            self.logger.error(f"モデル情報取得エラー: {model_id}, エラー: {e}")
            return None
    
    def delete_model(self, model_id: str) -> bool:
        """
        モデル削除
        
        Args:
            model_id: モデルID
            
        Returns:
            削除成功可否
        """
        try:
            # モデルディレクトリ削除
            model_dir = self.models_path / model_id
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            # OpenVINOディレクトリ削除
            openvino_dir = self.openvino_path / model_id
            if openvino_dir.exists():
                shutil.rmtree(openvino_dir)
            
            self.logger.info(f"モデル削除完了: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"モデル削除エラー: {model_id}, エラー: {e}")
            return False
    
    def _save_model_file(self, model: Any, file_path: Path):
        """モデルファイル保存（形式別処理）"""
        try:
            # PyTorchモデルの場合
            if hasattr(model, 'state_dict'):
                import torch
                torch.save(model, file_path)
                return
            
            # その他の場合はPickle
            import pickle
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
                
        except Exception as e:
            self.logger.error(f"モデルファイル保存エラー: {e}")
            # エラー情報をファイルに記録
            with open(file_path, 'w') as f:
                f.write(f"Model save error: {str(e)}\nTimestamp: {datetime.now().isoformat()}\n")
                f.write(f"Model type: {type(model).__name__}\n")
                f.write("This file indicates a model save failure. Please check the logs for details.")
    
    def _execute_openvino_conversion(self, model_file: Path, output_dir: Path) -> Optional[Path]:
        """OpenVINO変換実行（anomalib対応）"""
        try:
            # 変換設定
            ir_path = output_dir / "model.xml"
            
            # モデル形式に応じた変換
            if model_file.suffix in ['.onnx']:
                # ONNX形式の変換
                self.logger.info(f"ONNX → OpenVINO変換開始: {model_file}")
                ov_model = ov.convert_model(str(model_file))
                ov.save_model(ov_model, str(ir_path))
                self.logger.info(f"ONNX→OpenVINO変換完了: {ir_path}")
                
            elif model_file.suffix in ['.pth', '.pt']:
                # PyTorch形式の変換（ONNX経由）
                self.logger.info(f"PyTorch → ONNX → OpenVINO変換開始: {model_file}")
                
                # まずONNXに変換
                onnx_path = output_dir / "model.onnx"
                success = self._convert_pytorch_to_onnx(model_file, onnx_path)
                
                if success and onnx_path.exists():
                    # ONNXからOpenVINOに変換
                    ov_model = ov.convert_model(str(onnx_path))
                    ov.save_model(ov_model, str(ir_path))
                    self.logger.info(f"PyTorch→OpenVINO変換完了: {ir_path}")
                    
                    # 中間ONNXファイルは保持（デバッグ用）
                    self.logger.info(f"中間ONNXファイル保存: {onnx_path}")
                else:
                    raise RuntimeError("PyTorch → ONNX変換に失敗")
                
            elif model_file.suffix in ['.ckpt']:
                # Lightning checkpoint形式（anomalibでよく使用される）
                self.logger.info(f"Checkpoint → OpenVINO変換開始: {model_file}")
                
                # チェックポイントからONNXに変換
                onnx_path = output_dir / "model.onnx"
                success = self._convert_checkpoint_to_onnx(model_file, onnx_path)
                
                if success and onnx_path.exists():
                    # ONNXからOpenVINOに変換
                    ov_model = ov.convert_model(str(onnx_path))
                    ov.save_model(ov_model, str(ir_path))
                    self.logger.info(f"Checkpoint→OpenVINO変換完了: {ir_path}")
                else:
                    raise RuntimeError("Checkpoint → ONNX変換に失敗")
                
            else:
                raise ValueError(f"サポートされていないモデル形式: {model_file.suffix}")
            
            # 変換されたモデルファイルの検証
            if not ir_path.exists():
                raise RuntimeError(f"変換後のIRファイルが見つかりません: {ir_path}")
            
            # .binファイルも生成されているかチェック
            bin_path = ir_path.with_suffix('.bin')
            if not bin_path.exists():
                self.logger.warning(f"重みファイル(.bin)が見つかりません: {bin_path}")
            
            return ir_path
            
        except Exception as e:
            self.logger.error(f"OpenVINO変換実行エラー: {e}")
            raise
    
    def _update_model_metadata(self, model_id: str, updates: Dict[str, Any]):
        """モデルメタデータ更新"""
        try:
            metadata_path = self.models_path / model_id / "metadata.json"
            
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                metadata.update(updates)
                
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                    
        except Exception as e:
            self.logger.error(f"メタデータ更新エラー: {e}")
    
    def _create_model_readme(self, model_dir: Path, metadata: Dict[str, Any]):
        """モデルREADME作成"""
        readme_content = f"""# Model: {metadata['name']}

## Basic Information
- **Model ID**: {metadata['model_id']}
- **Category**: {metadata['category']}
- **Created**: {metadata['created_at']}
- **Accuracy**: {metadata['accuracy']:.3f}
- **Size**: {metadata['model_size_mb']:.2f} MB

## Files
- `model.pkl`: Trained model file
- `metadata.json`: Model metadata
- `README.md`: This file

## Usage
This model is trained for anomaly detection in {metadata['category']} category.

## OpenVINO Conversion
{f"✅ Converted to OpenVINO: {metadata.get('openvino_path', 'N/A')}" if metadata.get('openvino_path') else "❌ Not converted to OpenVINO"}

Generated by AI Anomaly Detection System Phase1
"""
        
        readme_path = model_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    def _create_deployment_config(self, temp_dir: Path, model_id: str):
        """デプロイ用設定ファイル作成"""
        model_info = self.get_model_info(model_id)
        
        deploy_config = {
            "model_id": model_id,
            "category": model_info.category if model_info else "bottle",
            "model_path": "model/model.pkl",
            "openvino_path": "openvino/model.xml",
            "confidence_threshold": self.config.get('inference.confidence_threshold', 0.5),
            "image_size": self.config.get('datasets.mvtec.image_size', [256, 256]),
            "deployment_info": {
                "created": datetime.now().isoformat(),
                "version": "1.0.0",
                "phase": "1"
            }
        }
        
        config_path = temp_dir / "deploy_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(deploy_config, f, indent=2, ensure_ascii=False)
    
    def export_for_inference(self, model_id: str, export_path: str) -> Optional[str]:
        """
        推論用モデルエクスポート
        
        Args:
            model_id: モデルID
            export_path: エクスポート先パス
            
        Returns:
            エクスポートされたファイルパス
        """
        try:
            export_dir = Path(export_path)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # モデル情報取得
            model_info = self.get_model_info(model_id)
            if not model_info:
                raise ValueError(f"モデルが見つかりません: {model_id}")
            
            # エクスポートファイル名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_file = export_dir / f"{model_id}_{timestamp}_inference.zip"
            
            # 一時ディレクトリでパッケージ作成
            temp_dir = export_dir / f"temp_{model_id}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # モデルファイルコピー
                model_dir = self.models_path / model_id
                if model_dir.exists():
                    shutil.copytree(model_dir, temp_dir / "model")
                
                # OpenVINOファイルコピー（存在する場合）
                openvino_dir = self.openvino_path / model_id
                if openvino_dir.exists():
                    shutil.copytree(openvino_dir, temp_dir / "openvino")
                
                # 推論用設定ファイル作成
                self._create_inference_config(temp_dir, model_id)
                
                # ZIP圧縮
                success = compress_directory(temp_dir, export_file)
                
                if success:
                    self.logger.info(f"推論用エクスポート完了: {export_file}")
                    return str(export_file)
                else:
                    raise RuntimeError("ZIP圧縮に失敗しました")
                    
            finally:
                # 一時ディレクトリ削除
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    
        except Exception as e:
            self.logger.error(f"推論用エクスポートエラー: {model_id}, エラー: {e}")
            return None
    
    def _create_inference_config(self, temp_dir: Path, model_id: str):
        """推論用設定ファイル作成"""
        model_info = self.get_model_info(model_id)
        
        inference_config = {
            "model_id": model_id,
            "category": model_info.category if model_info else "bottle",
            "model_path": "model/model.pkl",
            "openvino_path": "openvino/model.xml",
            "confidence_threshold": self.config.get('inference.confidence_threshold', 0.5),
            "image_size": self.config.get('datasets.mvtec.image_size', [256, 256]),
            "inference_info": {
                "created": datetime.now().isoformat(),
                "version": "1.0.0",
                "phase": "1",
                "accuracy": model_info.accuracy if model_info else 0.0
            }
        }
        
        config_path = temp_dir / "inference_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(inference_config, f, indent=2, ensure_ascii=False)
    
    def _convert_pytorch_to_onnx(self, model_file: Path, onnx_path: Path) -> bool:
        """
        PyTorchモデルをONNXに変換
        
        Args:
            model_file: PyTorchモデルファイル
            onnx_path: 出力ONNXパス
            
        Returns:
            変換成功可否
        """
        try:
            import torch
            
            self.logger.info(f"PyTorch → ONNX変換開始: {model_file}")
            
            # モデルを読み込み
            # 注意：実際のanomalibモデルの構造に応じて調整が必要
            model_state = torch.load(model_file, map_location='cpu')
            
            # ダミーの入力データを作成（モデル構造に応じて調整）
            # 一般的な異常検知モデルの入力サイズ
            dummy_input = torch.randn(1, 3, 256, 256)
            
            # ONNXエクスポート
            # 注意：実際のモデルオブジェクトが必要だが、状態辞書のみでは難しい
            # この実装は簡略化されており、実際の使用時は調整が必要
            self.logger.warning("PyTorch → ONNX変換は簡略実装です。完全な変換にはモデルオブジェクトが必要です。")
            
            # 代替として、直接的なONNX変換をスキップ
            return False
            
        except Exception as e:
            self.logger.error(f"PyTorch → ONNX変換エラー: {e}")
            return False
    
    def _convert_checkpoint_to_onnx(self, checkpoint_file: Path, onnx_path: Path) -> bool:
        """
        Lightning checkpointをONNXに変換
        
        Args:
            checkpoint_file: checkpointファイル
            onnx_path: 出力ONNXパス
            
        Returns:
            変換成功可否
        """
        try:
            self.logger.info(f"Checkpoint → ONNX変換開始: {checkpoint_file}")
            
            # Lightning checkpointからの変換
            # 注意：実際のanomalibモデルの構造に応じて調整が必要
            
            # 簡略実装：実際の使用時はanomalibのモデルクラスと組み合わせる必要がある
            self.logger.warning("Checkpoint → ONNX変換は簡略実装です。完全な変換にはモデルクラスが必要です。")
            
            # 代替として、直接的なONNX変換をスキップ
            return False
            
        except Exception as e:
            self.logger.error(f"Checkpoint → ONNX変換エラー: {e}")
            return False