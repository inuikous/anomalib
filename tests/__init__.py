"""テストパッケージ初期化"""

# テスト用の共通設定やユーティリティ関数をここに配置

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "shared"))

__all__ = []