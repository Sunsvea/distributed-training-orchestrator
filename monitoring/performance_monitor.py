import asyncio
import time
import json
import csv
import io
import logging
from typing import Dict, List, Any, Optional
import statistics
import threading

from .metrics_collector import MetricsCollector, SystemMetrics, TrainingMetrics, NetworkMetrics
from .alert_manager import AlertManager, AlertRule, AlertSeverity, MetricType

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Main performance monitoring system"""
    
    def __init__(self, collection_interval: float = 5.0, 
                 alert_evaluation_interval: float = 10.0):
        self.collection_interval = collection_interval
        self.alert_evaluation_interval = alert_evaluation_interval
        
        # Components
        self.metrics_collector = MetricsCollector(collection_interval)
        self.alert_manager = AlertManager()
        
        # State
        self.current_training_state: Optional[Dict[str, Any]] = None
        self.current_network_connections: Optional[List[Dict[str, Any]]] = None
        self.custom_metrics: Dict[str, Any] = {}
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.alert_evaluation_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Thread safety
        self.state_lock = threading.Lock()
    
    def register_training_state(self, training_state: Dict[str, Any]):
        """Register current training state"""
        with self.state_lock:
            self.current_training_state = training_state
            self.metrics_collector.update_training_state(training_state)
    
    def register_network_connections(self, connections: List[Dict[str, Any]]):
        """Register current network connections"""
        with self.state_lock:
            self.current_network_connections = connections
            self.metrics_collector.update_network_connections(connections)
    
    def add_custom_metric(self, name: str, value: Any):
        """Add a custom metric"""
        with self.state_lock:
            self.custom_metrics[name] = value
    
    def get_current_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        current_metrics = self.metrics_collector.get_current_metrics()
        
        return {
            "timestamp": time.time(),
            "system": current_metrics.get("system"),
            "training": current_metrics.get("training"),
            "network": current_metrics.get("network"),
            "custom_metrics": self.custom_metrics.copy(),
            "alerts": {
                "active_count": len(self.alert_manager.get_active_alerts()),
                "active_alerts": self.alert_manager.get_active_alerts()
            }
        }
    
    def get_performance_trends(self, window_size: int = 100) -> Dict[str, Any]:
        """Get performance trends over time"""
        history = self.metrics_collector.get_metrics_history(limit=window_size)
        
        if len(history) < 2:
            return {"insufficient_data": True}
        
        trends = {}
        
        # System metrics trends
        system_values = {}
        for record in history:
            if record["system"]:
                for field in ["cpu_percent", "memory_percent", "memory_used_gb"]:
                    if field not in system_values:
                        system_values[field] = []
                    system_values[field].append(getattr(record["system"], field))
        
        system_trends = {}
        for field, values in system_values.items():
            if len(values) >= 2:
                # Calculate trend (simple linear regression slope)
                n = len(values)
                x_values = list(range(n))
                slope = self._calculate_trend_slope(x_values, values)
                
                system_trends[field] = {
                    "current": values[-1],
                    "trend": "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable",
                    "slope": slope,
                    "change_rate": (values[-1] - values[0]) / values[0] if values[0] != 0 else 0
                }
        
        trends["system"] = system_trends
        
        # Training metrics trends
        training_values = {}
        for record in history:
            if record["training"]:
                for field in ["loss", "accuracy", "throughput"]:
                    if field not in training_values:
                        training_values[field] = []
                    training_values[field].append(getattr(record["training"], field))
        
        training_trends = {}
        for field, values in training_values.items():
            if len(values) >= 2:
                slope = self._calculate_trend_slope(list(range(len(values))), values)
                
                training_trends[field] = {
                    "current": values[-1],
                    "trend": "increasing" if slope > 0.001 else "decreasing" if slope < -0.001 else "stable",
                    "slope": slope,
                    "change_rate": (values[-1] - values[0]) / values[0] if values[0] != 0 else 0
                }
        
        trends["training"] = training_trends
        
        return trends
    
    def _calculate_trend_slope(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate linear regression slope"""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        n = len(x_values)
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(y_values)
        
        numerator = sum((x_values[i] - x_mean) * (y_values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def configure_alerting(self, rules_config: List[Dict[str, Any]]):
        """Configure alerting rules"""
        for rule_config in rules_config:
            rule = AlertRule(
                name=rule_config["name"],
                metric_type=MetricType(rule_config["metric_type"]),
                metric_name=rule_config["metric_name"],
                condition=rule_config["condition"],
                threshold=rule_config["threshold"],
                severity=AlertSeverity(rule_config["severity"]),
                duration=rule_config.get("duration", 60.0),
                message_template=rule_config.get("message_template", "")
            )
            
            self.alert_manager.add_alert_rule(rule)
        
        logger.info(f"Configured {len(rules_config)} alert rules")
    
    async def start_monitoring(self):
        """Start performance monitoring"""
        self._shutdown = False
        
        # Start metrics collection
        await self.metrics_collector.start_collection(
            self.current_training_state,
            self.current_network_connections
        )
        
        # Start monitoring tasks
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        if self.alert_evaluation_task is None:
            self.alert_evaluation_task = asyncio.create_task(self._alert_evaluation_loop())
        
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self._shutdown = True
        
        # Stop metrics collection
        await self.metrics_collector.stop_collection()
        
        # Stop monitoring tasks
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
        
        if self.alert_evaluation_task:
            self.alert_evaluation_task.cancel()
            try:
                await self.alert_evaluation_task
            except asyncio.CancelledError:
                pass
            self.alert_evaluation_task = None
        
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self._shutdown:
            try:
                # Update custom metrics or perform other monitoring tasks
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                logger.info("Monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _alert_evaluation_loop(self):
        """Alert evaluation loop"""
        while not self._shutdown:
            try:
                # Get current metrics
                current_metrics = self.metrics_collector.get_current_metrics()
                
                # Add custom metrics
                current_metrics["custom"] = self.custom_metrics.copy()
                
                # Evaluate alerts
                triggered_alerts = self.alert_manager.evaluate_alert_rules(current_metrics)
                
                # Log triggered alerts
                for alert in triggered_alerts:
                    logger.warning(f"Alert triggered: {alert.name} - {alert.message}")
                
                await asyncio.sleep(self.alert_evaluation_interval)
                
            except asyncio.CancelledError:
                logger.info("Alert evaluation loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in alert evaluation loop: {e}")
                await asyncio.sleep(1.0)
    
    def export_metrics(self, format: str = "json", limit: Optional[int] = None) -> str:
        """Export metrics in specified format"""
        history = self.metrics_collector.get_metrics_history(limit=limit)
        
        if format.lower() == "json":
            return self._export_as_json(history)
        elif format.lower() == "csv":
            return self._export_as_csv(history)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_as_json(self, history: List[Dict[str, Any]]) -> str:
        """Export metrics as JSON"""
        # Convert dataclass objects to dictionaries
        json_data = []
        
        for record in history:
            json_record = {
                "timestamp": record["timestamp"]
            }
            
            if record["system"]:
                json_record["system"] = {
                    "timestamp": record["system"].timestamp,
                    "cpu_percent": record["system"].cpu_percent,
                    "memory_percent": record["system"].memory_percent,
                    "memory_used_gb": record["system"].memory_used_gb,
                    "memory_total_gb": record["system"].memory_total_gb,
                    "disk_used_gb": record["system"].disk_used_gb,
                    "disk_total_gb": record["system"].disk_total_gb
                }
            
            if record["training"]:
                json_record["training"] = {
                    "timestamp": record["training"].timestamp,
                    "loss": record["training"].loss,
                    "accuracy": record["training"].accuracy,
                    "learning_rate": record["training"].learning_rate,
                    "throughput": record["training"].throughput
                }
            
            json_data.append(json_record)
        
        return json.dumps(json_data, indent=2)
    
    def _export_as_csv(self, history: List[Dict[str, Any]]) -> str:
        """Export metrics as CSV"""
        if not history:
            return ""
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        header = ["timestamp", "cpu_percent", "memory_percent", "memory_used_gb", 
                 "disk_used_gb", "loss", "accuracy", "learning_rate", "throughput"]
        writer.writerow(header)
        
        # Write data
        for record in history:
            row = [record["timestamp"]]
            
            if record["system"]:
                row.extend([
                    record["system"].cpu_percent,
                    record["system"].memory_percent,
                    record["system"].memory_used_gb,
                    record["system"].disk_used_gb
                ])
            else:
                row.extend([None, None, None, None])
            
            if record["training"]:
                row.extend([
                    record["training"].loss,
                    record["training"].accuracy,
                    record["training"].learning_rate,
                    record["training"].throughput
                ])
            else:
                row.extend([None, None, None, None])
            
            writer.writerow(row)
        
        return output.getvalue()
    
    def get_health_score(self) -> float:
        """Calculate overall system health score (0-100)"""
        current_metrics = self.metrics_collector.get_current_metrics()
        
        if not current_metrics.get("system"):
            return 0.0
        
        system_metrics = current_metrics["system"]
        
        # Score components (0-100 each)
        cpu_score = max(0, 100 - system_metrics.cpu_percent)
        memory_score = max(0, 100 - system_metrics.memory_percent)
        disk_score = max(0, 100 - (system_metrics.disk_used_gb / system_metrics.disk_total_gb * 100))
        
        # Alert penalty
        active_alerts = len(self.alert_manager.get_active_alerts())
        alert_penalty = min(active_alerts * 10, 50)  # Max 50 point penalty
        
        # Calculate weighted average
        weights = [0.3, 0.3, 0.2, 0.2]  # CPU, Memory, Disk, Alert penalty
        scores = [cpu_score, memory_score, disk_score, max(0, 100 - alert_penalty)]
        
        health_score = sum(w * s for w, s in zip(weights, scores))
        
        return max(0, min(100, health_score))
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            "monitoring_active": self.monitoring_task is not None,
            "collection_interval": self.collection_interval,
            "alert_evaluation_interval": self.alert_evaluation_interval,
            "metrics_history_size": len(self.metrics_collector.metrics_history),
            "active_alerts": len(self.alert_manager.get_active_alerts()),
            "alert_rules": len(self.alert_manager.alert_rules),
            "health_score": self.get_health_score(),
            "uptime": time.time() - getattr(self, '_start_time', time.time())
        }
    
    def reset_metrics(self):
        """Reset all collected metrics"""
        self.metrics_collector.clear_history()
        self.alert_manager.clear_alert_history()
        self.custom_metrics.clear()
        
        logger.info("All metrics reset")
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights and recommendations"""
        current_metrics = self.metrics_collector.get_current_metrics()
        trends = self.get_performance_trends()
        
        insights = {
            "recommendations": [],
            "warnings": [],
            "optimizations": []
        }
        
        # System insights
        if current_metrics.get("system"):
            sys_metrics = current_metrics["system"]
            
            if sys_metrics.cpu_percent > 80:
                insights["warnings"].append("High CPU usage detected")
                insights["recommendations"].append("Consider reducing batch size or adding more workers")
            
            if sys_metrics.memory_percent > 85:
                insights["warnings"].append("High memory usage detected")
                insights["recommendations"].append("Consider optimizing model size or reducing batch size")
            
            disk_usage = (sys_metrics.disk_used_gb / sys_metrics.disk_total_gb) * 100
            if disk_usage > 90:
                insights["warnings"].append("Low disk space")
                insights["recommendations"].append("Clean up old checkpoints or add more storage")
        
        # Training insights
        if current_metrics.get("training") and trends.get("training"):
            if "loss" in trends["training"]:
                loss_trend = trends["training"]["loss"]["trend"]
                if loss_trend == "increasing":
                    insights["warnings"].append("Loss is increasing")
                    insights["recommendations"].append("Consider reducing learning rate or checking data quality")
                elif loss_trend == "stable":
                    insights["optimizations"].append("Loss has plateaued, consider learning rate scheduling")
        
        return insights