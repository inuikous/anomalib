"""検知結果ドメインモデル"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class DetectionResult:
    """異常検知結果"""
    image_path: str
    is_anomaly: bool
    confidence_score: float
    processing_time_ms: float
    timestamp: datetime
    model_version: str
    metadata: Dict[str, Any]
    
    @property
    def filename(self) -> str:
        """ファイル名"""
        return Path(self.image_path).name
    
    @property
    def status_text(self) -> str:
        """状態テキスト"""
        return "異常" if self.is_anomaly else "正常"
    
    @property
    def confidence_percentage(self) -> int:
        """信頼度パーセンテージ"""
        return int(self.confidence_score * 100)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式変換"""
        return {
            "image_path": self.image_path,
            "filename": self.filename,
            "is_anomaly": self.is_anomaly,
            "status": self.status_text,
            "confidence_score": self.confidence_score,
            "confidence_percentage": self.confidence_percentage,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "model_version": self.model_version,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DetectionResult':
        """辞書から生成"""
        return cls(
            image_path=data["image_path"],
            is_anomaly=data["is_anomaly"],
            confidence_score=data["confidence_score"],
            processing_time_ms=data["processing_time_ms"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            model_version=data["model_version"],
            metadata=data.get("metadata", {})
        )