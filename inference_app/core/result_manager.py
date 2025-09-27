"""結果管理・ログ記録"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from collections import defaultdict

from shared.config import get_config_manager
from shared.domain import DetectionResult
from shared.utils import setup_logger, log_function_call, create_directory_if_not_exists


class ResultManager:
    """結果管理・ログ記録"""
    
    def __init__(self, config_manager=None):
        """
        初期化
        
        Args:
            config_manager: 設定管理インスタンス
        """
        self.config_manager = config_manager or get_config_manager()
        self.config = self.config_manager.get_config()
        self.logger = setup_logger("result_manager")
        
        # パス設定
        self.log_dir = Path(self.config.get('logging.log_dir', 'logs'))
        self.results_dir = self.log_dir / "results"
        self.reports_dir = self.log_dir / "reports"
        
        # ディレクトリ作成
        create_directory_if_not_exists(self.results_dir)
        create_directory_if_not_exists(self.reports_dir)
        
        # メモリ内履歴（セッション中のみ）
        self.session_results = []
        self.max_session_results = 1000  # メモリ使用量制限
        
        # 自動保存設定
        self.auto_save = self.config.get('inference.auto_save_results', True)
        
        self.logger.info(f"ResultManager初期化: 結果保存={self.auto_save}")
    
    @log_function_call
    def save_result(self, result: DetectionResult, image_path: str) -> None:
        """
        結果保存
        
        Args:
            result: 検知結果
            image_path: 画像パス
        """
        try:
            # 画像パス設定
            if not result.image_path:
                result.image_path = image_path
            
            # セッション履歴に追加
            self.session_results.append(result)
            
            # メモリ使用量制限
            if len(self.session_results) > self.max_session_results:
                self.session_results = self.session_results[-self.max_session_results:]
            
            # 自動保存
            if self.auto_save:
                self._save_result_to_file(result)
            
            self.logger.debug(f"結果保存: {result.filename}, {result.status_text}")
            
        except Exception as e:
            self.logger.error(f"結果保存エラー: {e}")
    
    def get_result_history(self, limit: Optional[int] = None) -> List[DetectionResult]:
        """
        結果履歴取得
        
        Args:
            limit: 取得件数制限
            
        Returns:
            結果履歴リスト
        """
        try:
            results = self.session_results.copy()
            
            # 時系列でソート（新しい順）
            results.sort(key=lambda x: x.timestamp, reverse=True)
            
            if limit:
                results = results[:limit]
            
            return results
            
        except Exception as e:
            self.logger.error(f"結果履歴取得エラー: {e}")
            return []
    
    @log_function_call
    def export_results_csv(self, export_path: Optional[str] = None, 
                          date_filter: Optional[date] = None) -> bool:
        """
        CSV形式エクスポート
        
        Args:
            export_path: エクスポート先パス
            date_filter: 日付フィルタ
            
        Returns:
            エクスポート成功可否
        """
        try:
            # エクスポート先設定
            if export_path:
                csv_path = Path(export_path)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = self.reports_dir / f"detection_results_{timestamp}.csv"
            
            # データ準備
            results = self.session_results
            if date_filter:
                results = [r for r in results 
                          if r.timestamp.date() == date_filter]
            
            if not results:
                self.logger.warning("エクスポート対象データがありません")
                return False
            
            # CSV書き込み
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # ヘッダー
                headers = [
                    'timestamp', 'filename', 'image_path', 'status', 
                    'is_anomaly', 'confidence_score', 'confidence_percentage',
                    'processing_time_ms', 'model_version'
                ]
                writer.writerow(headers)
                
                # データ行
                for result in sorted(results, key=lambda x: x.timestamp):
                    row = [
                        result.timestamp.isoformat(),
                        result.filename,
                        result.image_path,
                        result.status_text,
                        result.is_anomaly,
                        f"{result.confidence_score:.6f}",
                        result.confidence_percentage,
                        f"{result.processing_time_ms:.2f}",
                        result.model_version
                    ]
                    writer.writerow(row)
            
            self.logger.info(f"CSV エクスポート完了: {csv_path} ({len(results)}件)")
            return True
            
        except Exception as e:
            self.logger.error(f"CSV エクスポートエラー: {e}")
            return False
    
    def generate_simple_report(self) -> Dict[str, Any]:
        """
        簡易レポート生成
        
        Returns:
            レポートデータ
        """
        try:
            results = self.session_results
            
            if not results:
                return {
                    "summary": "データなし",
                    "total_count": 0,
                    "normal_count": 0,
                    "anomaly_count": 0,
                    "accuracy_rate": 0.0,
                    "average_processing_time": 0.0,
                    "generated_at": datetime.now().isoformat()
                }
            
            # 基本統計
            total_count = len(results)
            anomaly_count = sum(1 for r in results if r.is_anomaly)
            normal_count = total_count - anomaly_count
            
            # 処理時間統計
            processing_times = [r.processing_time_ms for r in results]
            avg_processing_time = sum(processing_times) / len(processing_times)
            
            # 信頼度統計
            confidence_scores = [r.confidence_score for r in results]
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            
            # 時間範囲
            timestamps = [r.timestamp for r in results]
            time_range = {
                "start": min(timestamps).isoformat(),
                "end": max(timestamps).isoformat()
            }
            
            # モデル情報
            model_versions = list(set(r.model_version for r in results))
            
            # 日別統計
            daily_stats = self._calculate_daily_stats(results)
            
            return {
                "summary": f"総数: {total_count}件, 正常: {normal_count}件, 異常: {anomaly_count}件",
                "total_count": total_count,
                "normal_count": normal_count,
                "anomaly_count": anomaly_count,
                "anomaly_rate": (anomaly_count / total_count * 100) if total_count > 0 else 0.0,
                "average_processing_time_ms": round(avg_processing_time, 2),
                "average_confidence": round(avg_confidence, 3),
                "time_range": time_range,
                "model_versions": model_versions,
                "daily_stats": daily_stats,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"簡易レポート生成エラー: {e}")
            return {"error": str(e)}
    
    @log_function_call
    def save_report(self, report_data: Dict[str, Any], 
                   report_name: Optional[str] = None) -> bool:
        """
        レポート保存
        
        Args:
            report_data: レポートデータ
            report_name: レポート名
            
        Returns:
            保存成功可否
        """
        try:
            if report_name:
                filename = f"{report_name}.json"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"report_{timestamp}.json"
            
            report_path = self.reports_dir / filename
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"レポート保存完了: {report_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"レポート保存エラー: {e}")
            return False
    
    def clear_session_results(self):
        """セッション結果クリア"""
        self.session_results.clear()
        self.logger.info("セッション結果クリア")
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報取得"""
        try:
            results = self.session_results
            
            if not results:
                return {"message": "データなし"}
            
            # 基本統計
            stats = {
                "total_detections": len(results),
                "anomaly_detections": sum(1 for r in results if r.is_anomaly),
                "normal_detections": sum(1 for r in results if not r.is_anomaly),
            }
            
            stats["anomaly_rate"] = (stats["anomaly_detections"] / stats["total_detections"] * 100
                                   if stats["total_detections"] > 0 else 0.0)
            
            # 処理時間統計
            processing_times = [r.processing_time_ms for r in results]
            stats.update({
                "avg_processing_time_ms": sum(processing_times) / len(processing_times),
                "min_processing_time_ms": min(processing_times),
                "max_processing_time_ms": max(processing_times)
            })
            
            # 信頼度統計
            confidence_scores = [r.confidence_score for r in results]
            stats.update({
                "avg_confidence": sum(confidence_scores) / len(confidence_scores),
                "min_confidence": min(confidence_scores),
                "max_confidence": max(confidence_scores)
            })
            
            return stats
            
        except Exception as e:
            self.logger.error(f"統計情報取得エラー: {e}")
            return {"error": str(e)}
    
    def _save_result_to_file(self, result: DetectionResult):
        """結果ファイル保存"""
        try:
            # 日別ファイル
            date_str = result.timestamp.strftime("%Y%m%d")
            file_path = self.results_dir / f"results_{date_str}.jsonl"
            
            # JSONL形式で追記
            with open(file_path, 'a', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, ensure_ascii=False)
                f.write('\n')
                
        except Exception as e:
            self.logger.error(f"結果ファイル保存エラー: {e}")
    
    def _calculate_daily_stats(self, results: List[DetectionResult]) -> Dict[str, Dict[str, int]]:
        """日別統計計算"""
        daily_stats = defaultdict(lambda: {"total": 0, "normal": 0, "anomaly": 0})
        
        for result in results:
            date_key = result.timestamp.strftime("%Y-%m-%d")
            daily_stats[date_key]["total"] += 1
            if result.is_anomaly:
                daily_stats[date_key]["anomaly"] += 1
            else:
                daily_stats[date_key]["normal"] += 1
        
        return dict(daily_stats)
    
    def load_historical_results(self, date_filter: Optional[date] = None) -> List[DetectionResult]:
        """履歴結果読み込み"""
        try:
            results = []
            
            if date_filter:
                # 特定日のファイル読み込み
                date_str = date_filter.strftime("%Y%m%d")
                file_path = self.results_dir / f"results_{date_str}.jsonl"
                if file_path.exists():
                    results.extend(self._load_results_from_file(file_path))
            else:
                # 全ファイル読み込み
                for file_path in self.results_dir.glob("results_*.jsonl"):
                    results.extend(self._load_results_from_file(file_path))
            
            return results
            
        except Exception as e:
            self.logger.error(f"履歴結果読み込みエラー: {e}")
            return []
    
    def _load_results_from_file(self, file_path: Path) -> List[DetectionResult]:
        """ファイルから結果読み込み"""
        results = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        result = DetectionResult.from_dict(data)
                        results.append(result)
                        
        except Exception as e:
            self.logger.error(f"結果ファイル読み込みエラー: {file_path}, {e}")
        
        return results
    
    @log_function_call
    def save_batch_results(self, results: List[DetectionResult]) -> bool:
        """
        一括結果保存
        
        Args:
            results: DetectionResultのリスト
            
        Returns:
            保存成功可否
        """
        try:
            if not results:
                self.logger.warning("保存する結果がありません")
                return True
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_filename = f"batch_results_{timestamp}.jsonl"
            batch_file = self.results_dir / batch_filename
            
            with open(batch_file, 'w', encoding='utf-8') as f:
                for result in results:
                    result_data = result.to_dict()
                    f.write(json.dumps(result_data, ensure_ascii=False) + '\n')
            
            self.session_results.extend(results)
            
            if len(self.session_results) > self.max_session_results:
                excess = len(self.session_results) - self.max_session_results
                self.session_results = self.session_results[excess:]
                self.logger.debug(f"セッション履歴制限: {excess}件削除")
            
            self.logger.info(f"一括結果保存完了: {len(results)}件 -> {batch_filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"一括結果保存エラー: {e}")
            return False
    
    @log_function_call
    def export_batch_results_csv(self, results: List[DetectionResult], file_path: str) -> bool:
        """
        一括結果のCSVエクスポート
        
        Args:
            results: DetectionResultのリスト
            file_path: エクスポート先ファイルパス
            
        Returns:
            エクスポート成功可否
        """
        try:
            if not results:
                self.logger.warning("エクスポートする結果がありません")
                return False
            
            file_path_obj = Path(file_path)
            
            fieldnames = [
                'ファイル名', '画像パス', '判定結果', '信頼度', '信頼度(%)',
                '処理時間(ms)', '検知日時', 'モデルバージョン', '閾値', 'デバイス'
            ]
            
            with open(file_path_obj, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    metadata = result.metadata
                    row = {
                        'ファイル名': result.filename,
                        '画像パス': result.image_path,
                        '判定結果': result.status_text,
                        '信頼度': f"{result.confidence_score:.4f}",
                        '信頼度(%)': f"{result.confidence_percentage}%",
                        '処理時間(ms)': f"{result.processing_time_ms:.2f}",
                        '検知日時': result.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        'モデルバージョン': result.model_version,
                        '閾値': metadata.get('threshold', 'unknown'),
                        'デバイス': metadata.get('device', 'unknown')
                    }
                    writer.writerow(row)
            
            self.logger.info(f"一括CSV エクスポート完了: {len(results)}件 -> {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"一括CSV エクスポートエラー: {e}")
            return False