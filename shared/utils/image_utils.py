"""画像処理ユーティリティ"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    画像リサイズ
    
    Args:
        image: 入力画像
        target_size: (width, height)
        
    Returns:
        リサイズ後画像
    """
    try:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    except Exception as e:
        logger.error(f"画像リサイズエラー: {e}")
        raise


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    画像正規化（0-1範囲）
    
    Args:
        image: 入力画像
        
    Returns:
        正規化画像
    """
    try:
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        return image / 255.0
    except Exception as e:
        logger.error(f"画像正規化エラー: {e}")
        raise


def validate_image_format(file_path: Path) -> bool:
    """
    画像形式検証
    
    Args:
        file_path: 画像ファイルパス
        
    Returns:
        有効な画像形式かどうか
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    try:
        # 拡張子チェック
        if file_path.suffix.lower() not in valid_extensions:
            return False
        
        # 実際に画像として読み込めるかチェック
        with Image.open(file_path) as img:
            img.verify()
        
        return True
        
    except Exception as e:
        logger.warning(f"画像形式検証失敗: {file_path}, エラー: {e}")
        return False


def get_image_info(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    画像情報取得
    
    Args:
        file_path: 画像ファイルパス
        
    Returns:
        画像情報辞書
    """
    try:
        with Image.open(file_path) as img:
            return {
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "format": img.format,
                "file_size": file_path.stat().st_size,
                "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2)
            }
    except Exception as e:
        logger.error(f"画像情報取得エラー: {file_path}, エラー: {e}")
        return None


def load_image_cv2(file_path: Path) -> Optional[np.ndarray]:
    """
    OpenCV形式で画像読み込み
    
    Args:
        file_path: 画像ファイルパス
        
    Returns:
        画像配列 (BGR形式)
    """
    try:
        image = cv2.imread(str(file_path))
        if image is None:
            logger.error(f"画像読み込み失敗: {file_path}")
            return None
        return image
    except Exception as e:
        logger.error(f"画像読み込みエラー: {file_path}, エラー: {e}")
        return None


def load_image_rgb(file_path: Path) -> Optional[np.ndarray]:
    """
    RGB形式で画像読み込み
    
    Args:
        file_path: 画像ファイルパス
        
    Returns:
        画像配列 (RGB形式)
    """
    try:
        with Image.open(file_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img)
    except Exception as e:
        logger.error(f"画像読み込みエラー: {file_path}, エラー: {e}")
        return None


def save_image(image: np.ndarray, file_path: Path, format: str = 'PNG') -> bool:
    """
    画像保存
    
    Args:
        image: 保存する画像配列
        file_path: 保存先パス
        format: 保存形式
        
    Returns:
        保存成功可否
    """
    try:
        # ディレクトリ作成
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 画像形式変換（必要に応じて）
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        
        # PIL形式で保存
        if len(image.shape) == 3 and image.shape[2] == 3:
            # RGB画像
            pil_image = Image.fromarray(image, 'RGB')
        else:
            # グレースケール画像
            pil_image = Image.fromarray(image, 'L')
        
        pil_image.save(file_path, format=format)
        logger.info(f"画像保存完了: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"画像保存エラー: {file_path}, エラー: {e}")
        return False


def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    BGR形式からRGB形式に変換
    
    Args:
        image: BGR画像
        
    Returns:
        RGB画像
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    RGB形式からBGR形式に変換
    
    Args:
        image: RGB画像
        
    Returns:
        BGR画像
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def preprocess_for_anomalib(image: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    anomalib用前処理
    
    Args:
        image: 入力画像
        target_size: 目標サイズ
        
    Returns:
        前処理済み画像
    """
    try:
        # リサイズ
        processed = resize_image(image, target_size)
        
        # 正規化
        processed = normalize_image(processed)
        
        # RGB確認・変換
        if len(processed.shape) == 3 and processed.shape[2] == 3:
            # BGRの場合はRGBに変換
            if processed.max() <= 1.0:  # 正規化済みの場合
                processed = convert_bgr_to_rgb(processed)
        
        return processed
        
    except Exception as e:
        logger.error(f"anomalib前処理エラー: {e}")
        raise


def create_thumbnail(image: np.ndarray, max_size: int = 200) -> np.ndarray:
    """
    サムネイル作成
    
    Args:
        image: 入力画像
        max_size: 最大サイズ
        
    Returns:
        サムネイル画像
    """
    try:
        height, width = image.shape[:2]
        
        # アスペクト比維持してリサイズ
        if height > width:
            new_height = max_size
            new_width = int(width * max_size / height)
        else:
            new_width = max_size
            new_height = int(height * max_size / width)
        
        return resize_image(image, (new_width, new_height))
        
    except Exception as e:
        logger.error(f"サムネイル作成エラー: {e}")
        return image