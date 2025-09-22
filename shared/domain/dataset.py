"""データセット関連ドメインモデル"""

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class DatasetInfo:
    """データセット情報"""
    category: str
    train_count: int
    test_normal_count: int
    test_defect_count: int
    defect_types: List[str]
    is_valid: bool
    validation_errors: List[str]
    base_path: Optional[str] = None
    dataset_path: Optional[str] = None  # データセットパス追加
    
    @property
    def total_count(self) -> int:
        """総画像数"""
        return self.train_count + self.test_normal_count + self.test_defect_count
    
    @property
    def has_defects(self) -> bool:
        """異常データ存在確認"""
        return self.test_defect_count > 0
    
    def get_summary(self) -> str:
        """サマリー文字列取得"""
        return (f"カテゴリ: {self.category}, "
                f"学習: {self.train_count}枚, "
                f"テスト正常: {self.test_normal_count}枚, "
                f"テスト異常: {self.test_defect_count}枚, "
                f"異常タイプ: {len(self.defect_types)}種類")


@dataclass
class ImageInfo:
    """画像情報"""
    file_path: Path
    category: str
    is_defect: bool
    defect_type: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    file_size: Optional[int] = None
    
    @property
    def filename(self) -> str:
        """ファイル名"""
        return self.file_path.name
    
    @property
    def status(self) -> str:
        """状態文字列"""
        if self.is_defect:
            return f"異常({self.defect_type})" if self.defect_type else "異常"
        return "正常"