"""画像入力・前処理管理"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from shared.config import get_config_manager
from shared.utils import (
    setup_logger, log_function_call, validate_image_format,
    get_image_info, load_image_rgb, preprocess_for_anomalib, create_thumbnail
)


class ImageManager:
    """画像管理（読み込み・前処理・検証）"""
    
    def __init__(self, config_manager=None):
        """
        初期化
        
        Args:
            config_manager: 設定管理インスタンス
        """
        self.config_manager = config_manager or get_config_manager()
        self.config = self.config_manager.get_config()
        self.logger = setup_logger("image_manager")
        
        # 設定取得
        self.image_size = tuple(self.config.get('datasets.mvtec.image_size', [256, 256]))
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 処理済み画像キャッシュ
        self.current_image = None
        self.current_image_path = None
        self.current_preprocessed = None
        
        self.logger.info(f"ImageManager初期化: 画像サイズ={self.image_size}")
    
    @log_function_call
    def load_image(self, file_path: str) -> Optional[np.ndarray]:
        """
        画像読み込み
        
        Args:
            file_path: 画像ファイルパス
            
        Returns:
            読み込み済み画像配列 (RGB形式)
        """
        try:
            file_path = Path(file_path)
            
            # ファイル存在確認
            if not file_path.exists():
                self.logger.error(f"ファイルが見つかりません: {file_path}")
                return None
            
            # 形式検証
            if not self._is_supported_format(file_path):
                self.logger.error(f"サポートされていない画像形式: {file_path}")
                return None
            
            # 画像読み込み
            image = load_image_rgb(file_path)
            if image is None:
                self.logger.error(f"画像読み込み失敗: {file_path}")
                return None
            
            # 基本検証
            if not self._validate_image_basic(image, file_path):
                return None
            
            # キャッシュ更新
            self.current_image = image
            self.current_image_path = str(file_path)
            self.current_preprocessed = None  # 前処理キャッシュクリア
            
            self.logger.info(f"画像読み込み成功: {file_path}, サイズ={image.shape}")
            return image
            
        except Exception as e:
            self.logger.error(f"画像読み込みエラー: {file_path}, エラー: {e}")
            return None
    
    @log_function_call
    def preprocess_image(self, image: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        前処理実行
        
        Args:
            image: 前処理対象画像（Noneの場合は最後に読み込んだ画像を使用）
            
        Returns:
            前処理済み画像
        """
        try:
            # 画像取得
            if image is None:
                if self.current_image is None:
                    self.logger.error("前処理対象の画像がありません")
                    return None
                image = self.current_image
            
            # キャッシュ確認
            if (self.current_preprocessed is not None and 
                np.array_equal(image, self.current_image)):
                self.logger.debug("前処理済み画像のキャッシュを使用")
                return self.current_preprocessed
            
            # 前処理実行
            preprocessed = preprocess_for_anomalib(image, self.image_size)
            
            # キャッシュ更新
            self.current_preprocessed = preprocessed
            
            self.logger.debug(f"前処理完了: {image.shape} -> {preprocessed.shape}")
            return preprocessed
            
        except Exception as e:
            self.logger.error(f"前処理エラー: {e}")
            return None
    
    def validate_image(self, image: np.ndarray) -> bool:
        """
        画像検証
        
        Args:
            image: 検証対象画像
            
        Returns:
            検証結果
        """
        return self._validate_image_basic(image)
    
    def get_image_info(self, file_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        画像情報取得
        
        Args:
            file_path: 画像ファイルパス（Noneの場合は現在の画像）
            
        Returns:
            画像情報辞書
        """
        try:
            if file_path:
                # 指定されたファイルの情報取得
                return get_image_info(Path(file_path))
            elif self.current_image_path:
                # 現在の画像の情報取得
                info = get_image_info(Path(self.current_image_path))
                if info and self.current_image is not None:
                    # 現在のメモリ内画像情報を追加
                    info['current_shape'] = self.current_image.shape
                    info['current_dtype'] = str(self.current_image.dtype)
                return info
            else:
                self.logger.warning("画像情報取得対象がありません")
                return None
                
        except Exception as e:
            self.logger.error(f"画像情報取得エラー: {e}")
            return None
    
    def create_thumbnail(self, image: Optional[np.ndarray] = None, max_size: int = 200) -> Optional[np.ndarray]:
        """
        サムネイル作成
        
        Args:
            image: サムネイル作成対象画像
            max_size: 最大サイズ
            
        Returns:
            サムネイル画像
        """
        try:
            if image is None:
                if self.current_image is None:
                    return None
                image = self.current_image
            
            return create_thumbnail(image, max_size)
            
        except Exception as e:
            self.logger.error(f"サムネイル作成エラー: {e}")
            return None
    
    def clear_cache(self):
        """キャッシュクリア"""
        self.current_image = None
        self.current_image_path = None
        self.current_preprocessed = None
        self.logger.debug("画像キャッシュクリア")
    
    def get_current_image(self) -> Optional[np.ndarray]:
        """現在の画像取得"""
        return self.current_image
    
    def get_current_image_path(self) -> Optional[str]:
        """現在の画像パス取得"""
        return self.current_image_path
    
    def is_image_loaded(self) -> bool:
        """画像読み込み状態確認"""
        return self.current_image is not None
    
    def _is_supported_format(self, file_path: Path) -> bool:
        """サポート形式確認"""
        return file_path.suffix.lower() in self.supported_formats
    
    def _validate_image_basic(self, image: np.ndarray, file_path: Optional[Path] = None) -> bool:
        """基本画像検証"""
        try:
            # 形状確認
            if len(image.shape) not in [2, 3]:
                self.logger.error(f"不正な画像形状: {image.shape}")
                return False
            
            # サイズ確認
            height, width = image.shape[:2]
            if height < 32 or width < 32:
                self.logger.error(f"画像サイズが小さすぎます: {width}x{height}")
                return False
            
            if height > 4096 or width > 4096:
                self.logger.warning(f"画像サイズが大きいです: {width}x{height}")
            
            # データ型確認
            if image.dtype not in [np.uint8, np.float32, np.float64]:
                self.logger.warning(f"推奨されないデータ型: {image.dtype}")
            
            # 値範囲確認
            if image.dtype == np.uint8:
                if image.min() < 0 or image.max() > 255:
                    self.logger.error("uint8画像の値が範囲外です")
                    return False
            elif image.dtype in [np.float32, np.float64]:
                if image.min() < 0 or image.max() > 1:
                    # 0-255範囲の可能性もあるので警告のみ
                    self.logger.warning(f"float画像の値範囲: {image.min():.3f} - {image.max():.3f}")
            
            # チャンネル数確認
            if len(image.shape) == 3:
                channels = image.shape[2]
                if channels not in [1, 3, 4]:
                    self.logger.error(f"不正なチャンネル数: {channels}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"画像検証エラー: {e}")
            return False
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """処理統計取得"""
        stats = {
            "image_loaded": self.is_image_loaded(),
            "current_image_path": self.current_image_path,
            "target_size": self.image_size,
            "supported_formats": list(self.supported_formats),
        }
        
        if self.current_image is not None:
            stats.update({
                "current_shape": self.current_image.shape,
                "current_dtype": str(self.current_image.dtype),
                "current_size_mb": self.current_image.nbytes / (1024 * 1024),
                "preprocessed_available": self.current_preprocessed is not None
            })
        
        return stats