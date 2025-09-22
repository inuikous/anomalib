"""異常検知実行エンジン - anomalib 2.1.0対応"""

import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# OpenVINO imports
import openvino as ov

# anomalib imports
from anomalib.deploy import OpenVINOInferencer

from shared.config import get_config_manager
from shared.domain import DetectionResult
from shared.utils import setup_logger, log_function_call, PerformanceTimer


class AnomalyDetector:
    """異常検知エンジン（anomalib 2.1.0 + OpenVINO対応）"""
    
    def __init__(self, config_manager=None):
        """
        初期化
        
        Args:
            config_manager: 設定管理インスタンス
        """
        self.config_manager = config_manager or get_config_manager()
        self.config = self.config_manager.get_config()
        self.logger = setup_logger("anomaly_detector")
        
        # anomalib OpenVINOInferencer
        self.inferencer = None
        self.model_path = None
        self.model_version = "unknown"
        
        # 推論設定
        self.confidence_threshold = self.config.get('inference.confidence_threshold', 0.5)
        self.processing_device = self.config.get('inference.processing_device', 'CPU')
        
        # 統計情報
        self.inference_count = 0
        self.total_inference_time = 0.0
        
        self.logger.info(f"AnomalyDetector初期化: デバイス={self.processing_device}, 閾値={self.confidence_threshold}")
        
        self._initialize_openvino()
    
    def _initialize_openvino(self):
        """OpenVINO初期化"""
        try:
            # OpenVINOコア情報を取得
            ie_core = ov.Core()
            available_devices = ie_core.available_devices
            self.logger.info(f"OpenVINO初期化完了: 利用可能デバイス={available_devices}")
        except Exception as e:
            self.logger.error(f"OpenVINO初期化エラー: {e}")
            raise
    
    @log_function_call
    def load_model(self, model_path: str) -> bool:
        """
        anomalib OpenVINOモデル読み込み
        
        Args:
            model_path: モデルファイルパス(.xml)
            
        Returns:
            読み込み成功可否
        """
        try:
            model_path_obj = Path(model_path)
            
            # ファイル存在確認
            if not model_path_obj.exists():
                self.logger.error(f"モデルファイルが見つかりません: {model_path}")
                return False
            
            # anomalib OpenVINOInferencerを使用
            with PerformanceTimer(self.logger, "model_loading"):
                self.inferencer = OpenVINOInferencer(
                    path=model_path,
                    device=self.processing_device
                )
                
                self.model_path = str(model_path_obj)
                self.model_version = self._extract_model_version(model_path_obj)
            
            self.logger.info(f"anomalib OpenVINOモデル読み込み完了: {model_path}")
            self.logger.info(f"モデルバージョン: {self.model_version}, デバイス: {self.processing_device}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"モデル読み込みエラー: {model_path}, エラー: {e}")
            self.inferencer = None
            return False
    
    @log_function_call
    def detect_anomaly(self, image: np.ndarray) -> Optional[DetectionResult]:
        """
        anomalib異常検知実行
        
        Args:
            image: 入力画像 (numpy配列、PIL.Image、またはパス)
            
        Returns:
            検知結果
        """
        if not self.is_model_loaded():
            self.logger.error("モデルが読み込まれていません")
            return None
        
        try:
            start_time = time.time()
            
            with PerformanceTimer(self.logger, "anomaly_detection"):
                # anomalib OpenVINOInferencerで推論実行
                prediction = self.inferencer.predict(image)
                
                # anomalibの結果を解析
                result = self._parse_anomalib_result(prediction)
            
            processing_time = (time.time() - start_time) * 1000  # ms
            
            # 統計更新
            self.inference_count += 1
            self.total_inference_time += processing_time
            
            # 結果作成
            detection_result = DetectionResult(
                image_path="",  # 画像パスは呼び出し元で設定
                is_anomaly=result['is_anomaly'],
                confidence_score=result['confidence_score'],
                processing_time_ms=processing_time,
                timestamp=datetime.now(),
                model_version=self.model_version,
                metadata={
                    "threshold": self.confidence_threshold,
                    "device": self.processing_device,
                    "raw_score": result.get('raw_score', result['confidence_score']),
                    "inference_count": self.inference_count,
                    "anomalib_prediction": prediction
                }
            )
            
            self.logger.info(f"異常検知完了: 結果={detection_result.status_text}, "
                           f"信頼度={detection_result.confidence_percentage}%, "
                           f"処理時間={processing_time:.1f}ms")
            
            return detection_result
            
        except Exception as e:
            self.logger.error(f"異常検知エラー: {e}")
            return None
    
    def is_model_loaded(self) -> bool:
        """モデル読み込み状態確認"""
        return self.inferencer is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報取得"""
        info = {
            "model_loaded": self.is_model_loaded(),
            "model_path": self.model_path,
            "model_version": self.model_version,
            "device": self.processing_device,
            "threshold": self.confidence_threshold,
            "openvino_available": True,
            "anomalib_version": "2.1.0"
        }
        
        if self.inferencer:
            info.update({
                "inferencer_type": "anomalib.OpenVINOInferencer",
                "model_loaded_successfully": True
            })
        
        return info
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """推論統計取得"""
        avg_time = (self.total_inference_time / self.inference_count 
                   if self.inference_count > 0 else 0.0)
        
        return {
            "inference_count": self.inference_count,
            "total_time_ms": self.total_inference_time,
            "average_time_ms": avg_time,
            "fps": 1000.0 / avg_time if avg_time > 0 else 0.0
        }
    
    def set_threshold(self, threshold: float) -> bool:
        """信頼度閾値設定"""
        try:
            if 0.0 <= threshold <= 1.0:
                self.confidence_threshold = threshold
                self.logger.info(f"閾値更新: {threshold}")
                return True
            else:
                self.logger.error(f"不正な閾値: {threshold}")
                return False
        except Exception as e:
            self.logger.error(f"閾値設定エラー: {e}")
            return False
    
    def _parse_anomalib_result(self, prediction) -> Dict[str, Any]:
        """anomalib推論結果の解析"""
        try:
            # anomalibの予測結果を解析
            # 結果の構造はanomalib 2.1.0の仕様に依存
            
            if hasattr(prediction, 'pred_score'):
                # 異常スコアの取得
                anomaly_score = float(prediction.pred_score.item())
            elif hasattr(prediction, 'anomaly_score'):
                anomaly_score = float(prediction.anomaly_score.item())
            elif isinstance(prediction, dict):
                # 辞書形式の場合
                anomaly_score = float(prediction.get('anomaly_score', prediction.get('pred_score', 0.0)))
            else:
                # フォールバック：最大値を使用
                if hasattr(prediction, 'max'):
                    anomaly_score = float(prediction.max().item())
                else:
                    anomaly_score = 0.0
            
            # 0-1範囲に正規化
            anomaly_score = np.clip(anomaly_score, 0.0, 1.0)
            
            return {
                "confidence_score": anomaly_score,
                "is_anomaly": anomaly_score > self.confidence_threshold,
                "raw_score": anomaly_score
            }
            
        except Exception as e:
            self.logger.error(f"anomalib結果解析エラー: {e}")
            # フォールバック
            return {
                "confidence_score": 0.0,
                "is_anomaly": False,
                "raw_score": 0.0
            }
    
    def _extract_model_version(self, model_path: Path) -> str:
        """モデルバージョン抽出"""
        try:
            # パスからタイムスタンプ等を抽出してバージョンとする
            parent_name = model_path.parent.name
            if "_" in parent_name:
                return parent_name
            else:
                return f"v1.0_{parent_name}"
        except:
            return "v1.0_unknown"
    
    def cleanup(self):
        """リソースクリーンアップ"""
        try:
            self.inferencer = None
            self.logger.info("AnomalyDetectorリソースクリーンアップ完了")
        except Exception as e:
            self.logger.error(f"クリーンアップエラー: {e}")