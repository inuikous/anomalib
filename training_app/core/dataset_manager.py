"""汎用データセット管理 - MVTec + カスタムデータセット対応"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

from shared.config import get_config_manager
from shared.domain import DatasetInfo, ImageInfo
from shared.utils import (
    setup_logger, log_function_call, validate_image_format, 
    get_image_info, find_files_by_pattern, create_directory_if_not_exists
)


class DatasetManager:
    """汎用データセット管理（MVTec + カスタム対応）"""
    
    def __init__(self, config_manager=None, dataset_type: str = None, category: str = None):
        """
        初期化
        
        Args:
            config_manager: 設定管理インスタンス
            dataset_type: データセットタイプ ("mvtec" or "custom")
            category: カテゴリ名
        """
        self.config_manager = config_manager or get_config_manager()
        self.config = self.config_manager.get_config()
        self.logger = setup_logger("dataset_manager")
        
        # データセットタイプとカテゴリ設定
        datasets_config = self.config.get('datasets', {})
        self.dataset_type = dataset_type or datasets_config.get('active_type', 'mvtec')
        self.category = category or datasets_config.get('active_category', 'bottle')
        
        # 設定からパス取得
        try:
            type_config = datasets_config.get(self.dataset_type, {})
            base_path_config = type_config.get('base_path')
            if base_path_config is None:
                raise ValueError(f"{self.dataset_type}データセットのbase_pathが設定されていません")
            self.base_path = Path(base_path_config)
        except (KeyError, TypeError) as e:
            self.logger.error(f"設定エラー: {e}")
            # デフォルトパス設定
            self.base_path = Path(f"./datasets/development/{self.dataset_type}")
            self.logger.warning(f"デフォルトパスを使用: {self.base_path}")
        
        self.category_path = self.base_path / self.category
        
        self.logger.info(f"DatasetManager初期化: タイプ={self.dataset_type}, カテゴリ={self.category}, パス={self.category_path}")
    
    def get_available_categories(self) -> List[str]:
        """
        利用可能なカテゴリ一覧取得
        
        Returns:
            カテゴリ一覧
        """
        try:
            if self.dataset_type == "mvtec":
                # MVTecの場合は設定から取得
                datasets_config = self.config.get('datasets', {})
                mvtec_config = datasets_config.get('mvtec', {})
                return mvtec_config.get('categories', [])
            else:
                # カスタムの場合はディレクトリから動的に検出
                if not self.base_path.exists():
                    return []
                
                categories = []
                for item in self.base_path.iterdir():
                    if item.is_dir() and self._is_valid_category_structure(item):
                        categories.append(item.name)
                
                return sorted(categories)
                
        except Exception as e:
            self.logger.error(f"カテゴリ一覧取得エラー: {e}")
            return []
    
    def _is_valid_category_structure(self, category_path: Path) -> bool:
        """
        カテゴリディレクトリ構造の妥当性チェック
        
        Args:
            category_path: カテゴリパス
            
        Returns:
            妥当性
        """
        required_dirs = ["train", "test"]
        return all((category_path / dir_name).exists() for dir_name in required_dirs)
    
    def set_category(self, category: str):
        """
        カテゴリ設定
        
        Args:
            category: カテゴリ名
        """
        self.category = category
        self.category_path = self.base_path / self.category
        self.logger.info(f"カテゴリ変更: {category}")
    
    def set_dataset_type(self, dataset_type: str):
        """
        データセットタイプ設定
        
        Args:
            dataset_type: データセットタイプ ("mvtec" or "custom")
        """
        self.dataset_type = dataset_type
        
        # パス再設定
        datasets_config = self.config.get('datasets', {})
        type_config = datasets_config.get(dataset_type, {})
        base_path_config = type_config.get('base_path')
        
        if base_path_config:
            self.base_path = Path(base_path_config)
        else:
            self.base_path = Path(f"./datasets/development/{dataset_type}")
            
        self.category_path = self.base_path / self.category
        self.logger.info(f"データセットタイプ変更: {dataset_type}, パス={self.base_path}")
    
    @log_function_call
    def validate_dataset(self, dataset_path: Optional[str] = None) -> bool:
        """
        データセット検証（MVTec/カスタム対応）
        
        Args:
            dataset_path: データセットパス（指定されない場合は設定から取得）
            
        Returns:
            検証結果
        """
        try:
            # パス決定
            if dataset_path:
                category_path = Path(dataset_path)
            else:
                category_path = self.category_path
            
            # 基本ディレクトリ構造確認
            required_dirs = [
                category_path / "train",
                category_path / "test"
            ]
            
            # MVTecの場合は特定の構造をチェック
            if self.dataset_type == "mvtec":
                required_dirs.extend([
                    category_path / "train" / "good",
                    category_path / "test" / "good"
                ])
            else:
                # カスタムの場合は柔軟な構造を許可
                # train/test フォルダ内にサブフォルダがあることを確認
                train_path = category_path / "train"
                test_path = category_path / "test"
                
                if train_path.exists():
                    train_subdirs = [d for d in train_path.iterdir() if d.is_dir()]
                    if not train_subdirs:
                        self.logger.error(f"trainフォルダにサブフォルダがありません: {train_path}")
                        return False
                
                if test_path.exists():
                    test_subdirs = [d for d in test_path.iterdir() if d.is_dir()]
                    if not test_subdirs:
                        self.logger.error(f"testフォルダにサブフォルダがありません: {test_path}")
                        return False
            
            for dir_path in required_dirs:
                if not dir_path.exists():
                    self.logger.error(f"必須ディレクトリが見つかりません: {dir_path}")
                    return False
            
            # 最小限の画像数確認
            train_images = self._get_train_images()
            test_good_images = self._get_test_good_images()
            
            if len(train_images) < 5:
                self.logger.error(f"学習用画像が不足: {len(train_images)}枚 (最低5枚必要)")
                return False
            
            if len(test_good_images) < 2:
                self.logger.error(f"テスト用正常画像が不足: {len(test_good_images)}枚 (最低2枚必要)")
                return False
            
            self.logger.info(f"データセット検証成功: 学習={len(train_images)}, テスト正常={len(test_good_images)}")
            return True
            
        except Exception as e:
            self.logger.error(f"データセット検証エラー: {e}")
            return False
    
    def get_dataset_info(self, dataset_path: Optional[str] = None) -> DatasetInfo:
        """
        データセット情報取得
        
        Args:
            dataset_path: データセットパス（指定されない場合は設定から取得）
        
        Returns:
            データセット情報
        """
        try:
            # パス決定
            if dataset_path:
                self.category_path = Path(dataset_path)
            train_images = self._get_train_images()
            test_good_images = self._get_test_good_images()
            test_defect_images, defect_types = self._get_test_defect_images()
            
            validation_errors = []
            
            # 検証エラー収集
            if len(train_images) < 5:
                validation_errors.append(f"学習用画像不足: {len(train_images)}枚")
            
            if len(test_good_images) < 2:
                validation_errors.append(f"テスト用正常画像不足: {len(test_good_images)}枚")
            
            # 不正画像チェック（高速化のため一時的に無効化）
            # invalid_images = self._check_invalid_images()
            # if invalid_images:
            #     validation_errors.extend(invalid_images)
            invalid_images = []  # 高速化のため
            
            is_valid = len(validation_errors) == 0
            
            dataset_info = DatasetInfo(
                category=self.category,
                train_count=len(train_images),
                test_normal_count=len(test_good_images),
                test_defect_count=len(test_defect_images),
                defect_types=defect_types,
                is_valid=is_valid,
                validation_errors=validation_errors,
                base_path=str(self.base_path)
            )
            
            self.logger.info(f"データセット情報取得完了: {dataset_info.get_summary()}")
            return dataset_info
            
        except Exception as e:
            self.logger.error(f"データセット情報取得エラー: {e}")
            return DatasetInfo(
                category=self.category,
                train_count=0,
                test_normal_count=0,
                test_defect_count=0,
                defect_types=[],
                is_valid=False,
                validation_errors=[f"情報取得エラー: {str(e)}"],
                base_path=str(self.base_path)
            )
    
    def get_train_images(self) -> List[Path]:
        """
        学習用画像パス一覧取得
        
        Returns:
            学習用画像パスリスト
        """
        return self._get_train_images()
    
    def get_test_images(self) -> Tuple[List[Path], List[Path]]:
        """
        テスト用画像パス一覧取得
        
        Returns:
            (正常画像パス, 異常画像パス)
        """
        test_good = self._get_test_good_images()
        test_defect, _ = self._get_test_defect_images()
        return test_good, test_defect
    
    def setup_directory_structure(self) -> bool:
        """
        ディレクトリ構造作成
        
        Returns:
            作成成功可否
        """
        try:
            required_dirs = [
                self.category_path / "train" / "good",
                self.category_path / "test" / "good"
            ]
            
            for dir_path in required_dirs:
                create_directory_if_not_exists(dir_path)
            
            # README作成
            readme_path = self.category_path / "README.md"
            if not readme_path.exists():
                readme_content = f"""# {self.category.upper()} Dataset

## 構造
```
{self.category}/
├── train/
│   └── good/          # 学習用正常画像
└── test/
    ├── good/          # テスト用正常画像
    └── [defect_type]/ # テスト用異常画像（異常タイプ別）
```

## 注意事項
- 画像形式: PNG, JPEG, BMP対応
- 推奨サイズ: 256x256以上
- Phase1では{self.category}カテゴリのみ対応

作成日時: {datetime.now().isoformat()}
"""
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(readme_content)
            
            self.logger.info(f"ディレクトリ構造作成完了: {self.category_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"ディレクトリ構造作成エラー: {e}")
            return False
    
    def get_image_details(self, image_path: Path) -> Optional[ImageInfo]:
        """
        画像詳細情報取得
        
        Args:
            image_path: 画像パス
            
        Returns:
            画像情報
        """
        try:
            if not validate_image_format(image_path):
                return None
            
            # パスから情報推定
            path_parts = image_path.parts
            is_defect = "good" not in str(image_path)
            defect_type = None
            
            # 異常タイプ推定
            if is_defect:
                for part in reversed(path_parts):
                    if part not in ["test", self.category] and part.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        continue
                    if part not in ["test", self.category]:
                        defect_type = part
                        break
            
            # 画像メタデータ取得
            img_info = get_image_info(image_path)
            
            image_info = ImageInfo(
                file_path=image_path,
                category=self.category,
                is_defect=is_defect,
                defect_type=defect_type,
                width=img_info['width'] if img_info else None,
                height=img_info['height'] if img_info else None,
                file_size=img_info['file_size'] if img_info else None
            )
            
            return image_info
            
        except Exception as e:
            self.logger.error(f"画像詳細情報取得エラー: {image_path}, エラー: {e}")
            return None
    
    def _get_train_images(self) -> List[Path]:
        """学習用画像取得"""
        train_dir = self.category_path / "train" / "good"
        if not train_dir.exists():
            return []
        
        image_patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
        images = []
        
        for pattern in image_patterns:
            images.extend(find_files_by_pattern(train_dir, pattern))
        
        # 有効な画像のみフィルタ
        valid_images = [img for img in images if validate_image_format(img)]
        
        self.logger.debug(f"学習用画像: {len(valid_images)}枚")
        return valid_images
    
    def _get_test_good_images(self) -> List[Path]:
        """テスト用正常画像取得"""
        test_good_dir = self.category_path / "test" / "good"
        if not test_good_dir.exists():
            return []
        
        image_patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
        images = []
        
        for pattern in image_patterns:
            images.extend(find_files_by_pattern(test_good_dir, pattern))
        
        valid_images = [img for img in images if validate_image_format(img)]
        
        self.logger.debug(f"テスト用正常画像: {len(valid_images)}枚")
        return valid_images
    
    def _get_test_defect_images(self) -> Tuple[List[Path], List[str]]:
        """テスト用異常画像取得"""
        test_dir = self.category_path / "test"
        if not test_dir.exists():
            return [], []
        
        defect_images = []
        defect_types = []
        
        # goodディレクトリ以外をスキャン
        for subdir in test_dir.iterdir():
            if subdir.is_dir() and subdir.name != "good":
                defect_types.append(subdir.name)
                
                image_patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
                for pattern in image_patterns:
                    images = find_files_by_pattern(subdir, pattern)
                    valid_images = [img for img in images if validate_image_format(img)]
                    defect_images.extend(valid_images)
        
        self.logger.debug(f"テスト用異常画像: {len(defect_images)}枚, 異常タイプ: {defect_types}")
        return defect_images, defect_types
    
    def _check_invalid_images(self) -> List[str]:
        """不正画像チェック（簡略版）"""
        errors = []
        
        try:
            # 基本的な存在チェックのみ（高速化）
            train_images = self._get_train_images()
            test_good_images = self._get_test_good_images()
            defect_images, _ = self._get_test_defect_images()
            
            # 画像ファイルの存在確認のみ
            all_images = []
            all_images.extend(train_images[:5])  # 最初の5枚のみチェック
            all_images.extend(test_good_images[:5])
            all_images.extend(defect_images[:5])
            
            for image_path in all_images:
                if not image_path.exists():
                    errors.append(f"画像ファイルが存在しません: {image_path.name}")
                elif image_path.stat().st_size == 0:
                    errors.append(f"空のファイル: {image_path.name}")
            
            self.logger.debug(f"簡易画像検証完了: {len(all_images)}枚をチェック")
                
        except Exception as e:
            errors.append(f"画像チェックエラー: {str(e)}")
        
        return errors