"""設定管理モジュール - YAML設定の統一管理"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class ConfigManager:
    """設定管理クラス"""
    
    def __init__(self, environment: str = "development"):
        """
        設定管理初期化
        
        Args:
            environment: 環境名 ("development" or "production")
        """
        self.environment = environment
        self.config_dir = Path(__file__).parent.parent.parent / "config"
        self.config = self._load_config()
        
    def get_config(self) -> Dict[str, Any]:
        """設定取得"""
        return self.config.copy()
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        階層的設定値取得
        
        Args:
            key_path: "datasets.mvtec.base_path" のような階層指定
            default: デフォルト値
            
        Returns:
            設定値
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        設定更新
        
        Args:
            updates: 更新する設定
            
        Returns:
            更新成功可否
        """
        try:
            self.config.update(updates)
            return True
        except Exception as e:
            logging.error(f"設定更新エラー: {e}")
            return False
    
    def validate_config(self) -> bool:
        """
        設定検証
        
        Returns:
            検証結果
        """
        required_keys = [
            "datasets.mvtec.base_path",
            "models.save_path",
            "logging.level"
        ]
        
        for key_path in required_keys:
            if self.get(key_path) is None:
                logging.error(f"必須設定が見つかりません: {key_path}")
                return False
                
        return True
    
    def _load_config(self) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        try:
            # 共通設定読み込み
            app_config_path = self.config_dir / "app_config.yaml"
            with open(app_config_path, 'r', encoding='utf-8') as f:
                app_config = yaml.safe_load(f)
            
            # 環境別設定読み込み
            env_config_path = self.config_dir / f"{self.environment}_config.yaml"
            with open(env_config_path, 'r', encoding='utf-8') as f:
                env_config = yaml.safe_load(f)
            
            # 設定マージ
            merged_config = {**app_config, **env_config}
            
            # パス設定の絶対パス変換
            merged_config = self._resolve_paths(merged_config)
            
            return merged_config
            
        except Exception as e:
            logging.error(f"設定ファイル読み込みエラー: {e}")
            # 最小設定で継続
            return self._get_default_config()
    
    def _resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """相対パスを絶対パスに変換"""
        base_dir = Path(__file__).parent.parent.parent
        
        # datasets path
        if 'datasets' in config and 'mvtec' in config['datasets']:
            mvtec_config = config['datasets']['mvtec']
            if 'base_path' in mvtec_config:
                mvtec_config['base_path'] = str(base_dir / mvtec_config['base_path'])
        
        # models path
        if 'models' in config:
            models_config = config['models']
            for key in ['save_path', 'openvino_path', 'export_path']:
                if key in models_config:
                    models_config[key] = str(base_dir / models_config[key])
        
        # logging path
        if 'logging' in config and 'log_dir' in config['logging']:
            config['logging']['log_dir'] = str(base_dir / config['logging']['log_dir'])
        
        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        base_dir = Path(__file__).parent.parent.parent
        
        return {
            "app": {
                "name": "AI Anomaly Detection System",
                "version": "1.0.0",
                "phase": 1
            },
            "datasets": {
                "mvtec": {
                    "base_path": str(base_dir / "datasets" / "development" / "mvtec"),
                    "category": "bottle",
                    "image_size": [256, 256]
                }
            },
            "models": {
                "save_path": str(base_dir / "models" / "development"),
                "openvino_path": str(base_dir / "models" / "openvino")
            },
            "logging": {
                "level": "INFO",
                "log_dir": str(base_dir / "logs")
            },
            "training": {
                "model_name": "patchcore",
                "batch_size": 16,
                "max_epochs": 50
            },
            "inference": {
                "model_format": "openvino",
                "processing_device": "CPU",
                "confidence_threshold": 0.5
            }
        }


# シングルトンインスタンス用
_config_instances = {}


def get_config_manager(environment: str = "development") -> ConfigManager:
    """設定管理インスタンス取得（シングルトン）"""
    if environment not in _config_instances:
        _config_instances[environment] = ConfigManager(environment)
    return _config_instances[environment]