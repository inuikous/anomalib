"""推論用データセット管理"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from shared.config import get_config_manager
from shared.utils import setup_logger


class InferenceDatasetManager:
    """推論用データセット管理"""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager or get_config_manager()
        self.config = self.config_manager.get_config()
        self.logger = setup_logger("inference_dataset_manager")
        
        self.datasets_dir = Path(self.config.get('datasets.mvtec.base_path', './datasets/development/mvtec_anomaly_detection'))
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
    def get_available_categories(self) -> List[str]:
        """利用可能なカテゴリ一覧取得"""
        try:
            categories = []
            
            if not self.datasets_dir.exists():
                self.logger.warning(f"データセットディレクトリが見つかりません: {self.datasets_dir}")
                return categories
            
            for category_dir in self.datasets_dir.iterdir():
                if category_dir.is_dir() and (category_dir / "test").exists():
                    categories.append(category_dir.name)
            
            categories.sort()
            self.logger.debug(f"利用可能なカテゴリ: {categories}")
            return categories
            
        except Exception as e:
            self.logger.error(f"カテゴリ一覧取得エラー: {e}")
            return []
    
    def get_test_images(self, category: str) -> List[Dict[str, Any]]:
        """テスト画像一覧取得"""
        try:
            test_images = []
            category_dir = self.datasets_dir / category / "test"
            
            if not category_dir.exists():
                self.logger.warning(f"テストディレクトリが見つかりません: {category_dir}")
                return test_images
            
            for subdir in category_dir.iterdir():
                if subdir.is_dir():
                    subdir_name = subdir.name
                    is_anomaly = subdir_name not in ['good', 'normal']
                    
                    for image_file in subdir.iterdir():
                        if image_file.suffix.lower() in self.supported_formats:
                            test_images.append({
                                'path': str(image_file),
                                'filename': image_file.name,
                                'category': category,
                                'type': subdir_name,
                                'is_anomaly_expected': is_anomaly,
                                'relative_path': str(image_file.relative_to(self.datasets_dir))
                            })
            
            test_images.sort(key=lambda x: (x['type'], x['filename']))
            self.logger.info(f"テスト画像取得完了 ({category}): {len(test_images)}件")
            return test_images
            
        except Exception as e:
            self.logger.error(f"テスト画像取得エラー ({category}): {e}")
            return []
    
    def get_test_image_summary(self, category: str) -> Dict[str, int]:
        """テスト画像サマリー取得"""
        try:
            images = self.get_test_images(category)
            summary = {'total': len(images), 'normal': 0, 'anomaly': 0}
            
            for image in images:
                if image['is_anomaly_expected']:
                    summary['anomaly'] += 1
                else:
                    summary['normal'] += 1
            
            return summary
            
        except Exception as e:
            self.logger.error(f"テスト画像サマリー取得エラー ({category}): {e}")
            return {'total': 0, 'normal': 0, 'anomaly': 0}