"""ファイル操作ユーティリティ"""

import os
import shutil
import hashlib
import zipfile
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


def safe_copy_file(src: Path, dst: Path) -> bool:
    """
    安全なファイルコピー
    
    Args:
        src: コピー元パス
        dst: コピー先パス
        
    Returns:
        コピー成功可否
    """
    try:
        # コピー先ディレクトリ作成
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        # ファイルコピー
        shutil.copy2(src, dst)
        
        # コピー後検証
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            logger.info(f"ファイルコピー成功: {src} -> {dst}")
            return True
        else:
            logger.error(f"ファイルコピー検証失敗: {src} -> {dst}")
            return False
            
    except Exception as e:
        logger.error(f"ファイルコピーエラー: {src} -> {dst}, エラー: {e}")
        return False


def create_directory_if_not_exists(path: Path) -> bool:
    """
    ディレクトリ作成（存在しない場合のみ）
    
    Args:
        path: 作成するディレクトリパス
        
    Returns:
        作成成功可否
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"ディレクトリ作成: {path}")
        return True
    except Exception as e:
        logger.error(f"ディレクトリ作成エラー: {path}, エラー: {e}")
        return False


def get_file_hash(file_path: Path, algorithm: str = "md5") -> Optional[str]:
    """
    ファイルハッシュ取得
    
    Args:
        file_path: ファイルパス
        algorithm: ハッシュアルゴリズム
        
    Returns:
        ハッシュ値
    """
    try:
        hash_func = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except Exception as e:
        logger.error(f"ハッシュ計算エラー: {file_path}, エラー: {e}")
        return None


def compress_directory(dir_path: Path, output_path: Path) -> bool:
    """
    ディレクトリ圧縮
    
    Args:
        dir_path: 圧縮対象ディレクトリ
        output_path: 出力ZIPファイルパス
        
    Returns:
        圧縮成功可否
    """
    try:
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in dir_path.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(dir_path)
                    zipf.write(file_path, arcname)
        
        logger.info(f"ディレクトリ圧縮完了: {dir_path} -> {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"ディレクトリ圧縮エラー: {dir_path}, エラー: {e}")
        return False


def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """
    アーカイブ展開
    
    Args:
        archive_path: アーカイブファイルパス
        extract_to: 展開先ディレクトリ
        
    Returns:
        展開成功可否
    """
    try:
        create_directory_if_not_exists(extract_to)
        
        with zipfile.ZipFile(archive_path, 'r') as zipf:
            zipf.extractall(extract_to)
        
        logger.info(f"アーカイブ展開完了: {archive_path} -> {extract_to}")
        return True
        
    except Exception as e:
        logger.error(f"アーカイブ展開エラー: {archive_path}, エラー: {e}")
        return False


def get_directory_size(dir_path: Path) -> int:
    """
    ディレクトリサイズ取得
    
    Args:
        dir_path: ディレクトリパス
        
    Returns:
        サイズ（バイト）
    """
    total_size = 0
    try:
        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
    except Exception as e:
        logger.error(f"ディレクトリサイズ取得エラー: {dir_path}, エラー: {e}")
    
    return total_size


def find_files_by_pattern(dir_path: Path, pattern: str) -> List[Path]:
    """
    パターンマッチングによるファイル検索
    
    Args:
        dir_path: 検索対象ディレクトリ
        pattern: 検索パターン（glob形式）
        
    Returns:
        見つかったファイルパスのリスト
    """
    try:
        return list(dir_path.glob(pattern))
    except Exception as e:
        logger.error(f"ファイル検索エラー: {dir_path}, パターン: {pattern}, エラー: {e}")
        return []


def safe_remove_file(file_path: Path) -> bool:
    """
    安全なファイル削除
    
    Args:
        file_path: 削除対象ファイルパス
        
    Returns:
        削除成功可否
    """
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"ファイル削除: {file_path}")
        return True
    except Exception as e:
        logger.error(f"ファイル削除エラー: {file_path}, エラー: {e}")
        return False


def safe_remove_directory(dir_path: Path) -> bool:
    """
    安全なディレクトリ削除
    
    Args:
        dir_path: 削除対象ディレクトリパス
        
    Returns:
        削除成功可否
    """
    try:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            logger.info(f"ディレクトリ削除: {dir_path}")
        return True
    except Exception as e:
        logger.error(f"ディレクトリ削除エラー: {dir_path}, エラー: {e}")
        return False


def cleanup_openvino_cache_directories(project_root: Path = None) -> int:
    """
    OpenVINOが作成した16進数キャッシュディレクトリをクリーンアップ
    
    Args:
        project_root: プロジェクトルートディレクトリ（Noneの場合は現在のディレクトリ）
        
    Returns:
        削除したディレクトリ数
    """
    try:
        import re
        
        if project_root is None:
            project_root = Path(".")
            
        # 16桁の16進数パターン（OpenVINOキャッシュフォルダの命名規則）
        hex_pattern = re.compile(r'^[0-9A-F]{16}$', re.IGNORECASE)
        deleted_count = 0
        
        logger.info("OpenVINOキャッシュディレクトリのクリーンアップを開始")
        
        for item in project_root.iterdir():
            if item.is_dir() and hex_pattern.match(item.name):
                # .blobファイルが含まれているかチェック（OpenVINOキャッシュの特徴）
                blob_files = list(item.glob("*.blob"))
                if blob_files:
                    try:
                        shutil.rmtree(item)
                        logger.info(f"OpenVINOキャッシュディレクトリを削除: {item.name}")
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"キャッシュディレクトリ削除失敗: {item.name}, エラー: {e}")
        
        logger.info(f"OpenVINOキャッシュクリーンアップ完了: {deleted_count}個のディレクトリを削除")
        return deleted_count
        
    except Exception as e:
        logger.error(f"OpenVINOキャッシュクリーンアップエラー: {e}")
        return 0