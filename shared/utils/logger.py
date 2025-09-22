"""構造化ログ管理モジュール"""

import os
import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from functools import wraps


class StructuredFormatter(logging.Formatter):
    """構造化ログフォーマッター"""
    
    def format(self, record):
        """ログレコードを構造化JSON形式にフォーマット"""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
            "function": record.funcName,
            "line": record.lineno
        }
        
        # 追加情報があれば含める
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        return json.dumps(log_data, ensure_ascii=False)


def setup_logger(name: str, level: str = "INFO", log_dir: Optional[str] = None) -> logging.Logger:
    """
    ロガー設定
    
    Args:
        name: ロガー名
        level: ログレベル
        log_dir: ログディレクトリ
        
    Returns:
        設定されたロガー
    """
    logger = logging.getLogger(name)
    
    # 既に設定済みの場合はそのまま返す
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # ログディレクトリ設定
    if log_dir is None:
        log_dir = Path(__file__).parent.parent.parent / "logs"
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(exist_ok=True)
    
    # ファイルハンドラー（日次ローテーション）
    log_file = log_dir / f"{name}.log"
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_file,
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    file_handler.setFormatter(StructuredFormatter())
    logger.addHandler(file_handler)
    
    # コンソールハンドラー（開発時用）
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def log_function_call(func):
    """関数呼び出しログデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        
        # 引数情報（sensitive dataは除外）
        safe_args = []
        for arg in args[1:]:  # selfは除外
            if isinstance(arg, (str, int, float, bool)):
                safe_args.append(arg)
            else:
                safe_args.append(type(arg).__name__)
        
        safe_kwargs = {k: v for k, v in kwargs.items() 
                      if not k.lower().startswith('password')}
        
        logger.info(
            f"関数呼び出し: {func.__name__}",
            extra={"extra_data": {
                "function": func.__name__,
                "args": safe_args,
                "kwargs": safe_kwargs
            }}
        )
        
        try:
            result = func(*args, **kwargs)
            logger.info(f"関数完了: {func.__name__}")
            return result
        except Exception as e:
            logger.error(
                f"関数エラー: {func.__name__}",
                extra={"extra_data": {
                    "function": func.__name__,
                    "error": str(e),
                    "error_type": type(e).__name__
                }}
            )
            raise
    
    return wrapper


def log_error(logger: logging.Logger, error: Exception, context: str = "", **extra_data):
    """エラーログ記録"""
    logger.error(
        f"エラー発生: {context}",
        extra={"extra_data": {
            "context": context,
            "error_message": str(error),
            "error_type": type(error).__name__,
            **extra_data
        }}
    )


def log_performance(logger: logging.Logger, operation: str, duration: float, **extra_data):
    """パフォーマンスログ記録"""
    logger.info(
        f"パフォーマンス: {operation}",
        extra={"extra_data": {
            "operation": operation,
            "duration_ms": round(duration * 1000, 2),
            **extra_data
        }}
    )


def log_user_action(logger: logging.Logger, action: str, user_id: str = "unknown", **extra_data):
    """ユーザーアクションログ記録"""
    logger.info(
        f"ユーザーアクション: {action}",
        extra={"extra_data": {
            "action": action,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            **extra_data
        }}
    )


class PerformanceTimer:
    """パフォーマンス計測用コンテキストマネージャー"""
    
    def __init__(self, logger: logging.Logger, operation: str, **extra_data):
        self.logger = logger
        self.operation = operation
        self.extra_data = extra_data
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            log_performance(self.logger, self.operation, duration, **self.extra_data)