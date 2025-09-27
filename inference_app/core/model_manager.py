"""推論用モデル管理"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from shared.config import get_config_manager
from shared.utils import setup_logger


class InferenceModelManager:
    """推論用モデル管理"""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager or get_config_manager()
        self.config = self.config_manager.get_config()
        self.logger = setup_logger("inference_model_manager")
        
        self.models_dir = Path(self.config.get('models.openvino_path', './models/openvino'))
        
    def get_available_models(self) -> List[Dict[str, Any]]:
        """利用可能なモデル一覧取得"""
        try:
            models = []
            
            if not self.models_dir.exists():
                self.logger.warning(f"モデルディレクトリが見つかりません: {self.models_dir}")
                return models
            
            for model_dir in self.models_dir.iterdir():
                if model_dir.is_dir():
                    xml_files = list(model_dir.glob("*.xml"))
                    if xml_files:
                        model_info = self._extract_model_info(model_dir, xml_files[0])
                        if model_info:
                            models.append(model_info)
            
            models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            self.logger.info(f"利用可能なモデル: {len(models)}個")
            return models
            
        except Exception as e:
            self.logger.error(f"モデル一覧取得エラー: {e}")
            return []
    
    def _extract_model_info(self, model_dir: Path, xml_file: Path) -> Optional[Dict[str, Any]]:
        """モデル情報抽出"""
        try:
            metadata_file = model_dir / "metadata.json"
            
            model_info = {
                'name': model_dir.name,
                'path': str(xml_file),
                'directory': str(model_dir),
                'category': 'unknown',
                'algorithm': 'unknown',
                'created_at': '',
                'image_size': [256, 256]
            }
            
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    
                model_info.update({
                    'category': metadata.get('category', 'unknown'),
                    'algorithm': metadata.get('algorithm', 'unknown'),
                    'created_at': metadata.get('created_at', ''),
                    'image_size': metadata.get('image_size', [256, 256])
                })
            else:
                parts = model_dir.name.split('_')
                if len(parts) >= 2:
                    model_info['category'] = parts[0]
                    model_info['created_at'] = '_'.join(parts[1:])
            
            return model_info
            
        except Exception as e:
            self.logger.error(f"モデル情報抽出エラー: {model_dir}, {e}")
            return None
    
    def get_model_categories(self) -> List[str]:
        """モデルカテゴリ一覧取得"""
        models = self.get_available_models()
        categories = list(set(model['category'] for model in models if model['category'] != 'unknown'))
        categories.sort()
        return categories