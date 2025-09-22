"""モデル情報ドメインモデル"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class ModelInfo:
    """モデル情報"""
    model_id: str
    name: str
    category: str
    created_at: datetime
    accuracy: float
    model_size_mb: float
    openvino_path: Optional[str]
    metadata: Dict[str, Any]
    
    @property
    def display_name(self) -> str:
        """表示用名前"""
        return f"{self.name} ({self.category})"
    
    @property
    def accuracy_percentage(self) -> int:
        """精度パーセンテージ"""
        return int(self.accuracy * 100)
    
    @property
    def is_converted(self) -> bool:
        """OpenVINO変換済み確認"""
        return self.openvino_path is not None
    
    @property
    def age_days(self) -> int:
        """作成からの経過日数"""
        return (datetime.now() - self.created_at).days
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式変換"""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "category": self.category,
            "created_at": self.created_at.isoformat(),
            "accuracy": self.accuracy,
            "accuracy_percentage": self.accuracy_percentage,
            "model_size_mb": self.model_size_mb,
            "openvino_path": self.openvino_path,
            "is_converted": self.is_converted,
            "age_days": self.age_days,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfo':
        """辞書から生成"""
        return cls(
            model_id=data["model_id"],
            name=data["name"],
            category=data["category"],
            created_at=datetime.fromisoformat(data["created_at"]),
            accuracy=data["accuracy"],
            model_size_mb=data["model_size_mb"],
            openvino_path=data.get("openvino_path"),
            metadata=data.get("metadata", {})
        )