import pytest
import asyncio
import time
import torch
import psutil
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from monitoring.performance_monitor import PerformanceMonitor, MetricType, AlertRule, AlertSeverity
from monitoring.metrics_collector import MetricsCollector, SystemMetrics, TrainingMetrics, NetworkMetrics
from monitoring.alert_manager import AlertManager, Alert, AlertRule as AlertRuleClass
from monitoring.dashboard_server import DashboardServer

class TestMetricsCollector:
    
    @pytest.fixture
    def metrics_collector(self):
        return MetricsCollector(
            collection_interval=1.0,
            max_history_size=100
        )
    
    def test_metrics_collector_initialization(self, metrics_collector):
        """Test metrics collector initialization"""
        assert metrics_collector.collection_interval == 1.0
        assert metrics_collector.max_history_size == 100
        assert len(metrics_collector.metrics_history) == 0
        assert metrics_collector.system_metrics is None
    
    def test_collect_system_metrics(self, metrics_collector):
        """Test system metrics collection"""
        metrics = metrics_collector.collect_system_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert 0 <= metrics.cpu_percent <= 100
        assert 0 <= metrics.memory_percent <= 100
        assert metrics.memory_used_gb > 0
        assert metrics.memory_total_gb > 0
        assert metrics.disk_used_gb >= 0
        assert metrics.disk_total_gb > 0
        assert len(metrics.gpu_metrics) >= 0
        assert metrics.network_io_bytes_sent >= 0
        assert metrics.network_io_bytes_recv >= 0
    
    def test_collect_training_metrics(self, metrics_collector):
        """Test training metrics collection"""
        mock_training_state = {
            "loss": 0.5,
            "accuracy": 0.85,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epoch": 10,
            "iteration": 1000,
            "throughput": 150.0,
            "gradient_norm": 0.1
        }
        
        metrics = metrics_collector.collect_training_metrics(mock_training_state)
        
        assert isinstance(metrics, TrainingMetrics)
        assert metrics.loss == 0.5
        assert metrics.accuracy == 0.85
        assert metrics.learning_rate == 0.001
        assert metrics.batch_size == 32
        assert metrics.epoch == 10
        assert metrics.iteration == 1000
        assert metrics.throughput == 150.0
        assert metrics.gradient_norm == 0.1
    
    def test_collect_network_metrics(self, metrics_collector):
        """Test network metrics collection"""
        mock_connections = [
            {"worker_id": "worker-1", "latency": 10.0, "bandwidth": 1000.0, "packet_loss": 0.0},
            {"worker_id": "worker-2", "latency": 15.0, "bandwidth": 800.0, "packet_loss": 0.1}
        ]
        
        metrics = metrics_collector.collect_network_metrics(mock_connections)
        
        assert isinstance(metrics, NetworkMetrics)
        assert len(metrics.worker_connections) == 2
        assert metrics.worker_connections["worker-1"]["latency"] == 10.0
        assert metrics.worker_connections["worker-2"]["bandwidth"] == 800.0
        assert metrics.average_latency == 12.5
        assert metrics.total_bandwidth == 1800.0
        assert metrics.packet_loss_rate == 0.05
    
    def test_record_metrics(self, metrics_collector):
        """Test metrics recording"""
        system_metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_gb=8.0,
            memory_total_gb=16.0,
            disk_used_gb=100.0,
            disk_total_gb=500.0,
            gpu_metrics=[],
            network_io_bytes_sent=1000,
            network_io_bytes_recv=2000
        )
        
        training_metrics = TrainingMetrics(
            timestamp=time.time(),
            loss=0.3,
            accuracy=0.9,
            learning_rate=0.0005,
            batch_size=64,
            epoch=5,
            iteration=500,
            throughput=200.0,
            gradient_norm=0.05
        )
        
        metrics_collector.record_metrics(system_metrics, training_metrics, None)
        
        assert len(metrics_collector.metrics_history) == 1
        assert metrics_collector.metrics_history[0]["system"] == system_metrics
        assert metrics_collector.metrics_history[0]["training"] == training_metrics
    
    def test_get_metrics_history(self, metrics_collector):
        """Test getting metrics history"""
        base_time = time.time()
        
        # Add some test metrics
        for i in range(5):
            system_metrics = SystemMetrics(
                timestamp=base_time + i,
                cpu_percent=50.0 + i,
                memory_percent=60.0,
                memory_used_gb=8.0,
                memory_total_gb=16.0,
                disk_used_gb=100.0,
                disk_total_gb=500.0,
                gpu_metrics=[],
                network_io_bytes_sent=1000,
                network_io_bytes_recv=2000
            )
            
            metrics_collector.record_metrics(system_metrics, None, None)
        
        # Get all history
        history = metrics_collector.get_metrics_history()
        assert len(history) == 5
        
        # Get limited history
        history = metrics_collector.get_metrics_history(limit=3)
        assert len(history) == 3
        
        # Get time-filtered history (filter uses record timestamp, not system timestamp)
        # We need to use the record timestamp for filtering
        start_time = base_time
        history = metrics_collector.get_metrics_history(start_time=start_time)
        assert len(history) == 5  # All records should be included
    
    def test_get_aggregated_metrics(self, metrics_collector):
        """Test getting aggregated metrics"""
        # Add test metrics
        for i in range(10):
            system_metrics = SystemMetrics(
                timestamp=time.time() + i,
                cpu_percent=50.0 + i,
                memory_percent=60.0 + i,
                memory_used_gb=8.0,
                memory_total_gb=16.0,
                disk_used_gb=100.0,
                disk_total_gb=500.0,
                gpu_metrics=[],
                network_io_bytes_sent=1000,
                network_io_bytes_recv=2000
            )
            
            metrics_collector.record_metrics(system_metrics, None, None)
        
        aggregated = metrics_collector.get_aggregated_metrics()
        
        assert "system" in aggregated
        assert "cpu_percent" in aggregated["system"]
        assert "avg" in aggregated["system"]["cpu_percent"]
        assert "min" in aggregated["system"]["cpu_percent"]
        assert "max" in aggregated["system"]["cpu_percent"]
        assert aggregated["system"]["cpu_percent"]["avg"] == 54.5  # (50+59)/2
        assert aggregated["system"]["cpu_percent"]["min"] == 50.0
        assert aggregated["system"]["cpu_percent"]["max"] == 59.0
    
    @pytest.mark.asyncio
    async def test_metrics_collection_lifecycle(self, metrics_collector):
        """Test metrics collection start/stop"""
        # Mock system state
        mock_training_state = {"loss": 0.5, "accuracy": 0.8}
        mock_network_connections = []
        
        # Start collection
        await metrics_collector.start_collection(mock_training_state, mock_network_connections)
        assert metrics_collector.collection_task is not None
        
        # Let it collect a few metrics
        await asyncio.sleep(0.1)
        
        # Stop collection
        await metrics_collector.stop_collection()
        assert metrics_collector.collection_task is None
        assert metrics_collector._shutdown == True

class TestAlertManager:
    
    @pytest.fixture
    def alert_manager(self):
        return AlertManager()
    
    def test_alert_manager_initialization(self, alert_manager):
        """Test alert manager initialization"""
        assert len(alert_manager.alert_rules) == 0
        assert len(alert_manager.active_alerts) == 0
        assert len(alert_manager.alert_history) == 0
    
    def test_add_alert_rule(self, alert_manager):
        """Test adding alert rules"""
        rule = AlertRuleClass(
            name="high_cpu",
            metric_type=MetricType.SYSTEM,
            metric_name="cpu_percent",
            condition="greater_than",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            duration=60.0
        )
        
        alert_manager.add_alert_rule(rule)
        
        assert len(alert_manager.alert_rules) == 1
        assert alert_manager.alert_rules[0] == rule
    
    def test_evaluate_alert_rules(self, alert_manager):
        """Test evaluating alert rules against metrics"""
        # Add alert rule
        rule = AlertRuleClass(
            name="high_memory",
            metric_type=MetricType.SYSTEM,
            metric_name="memory_percent",
            condition="greater_than",
            threshold=90.0,
            severity=AlertSeverity.CRITICAL,
            duration=0.0  # No duration for test
        )
        alert_manager.add_alert_rule(rule)
        
        # Create metrics that should trigger alert
        metrics = {
            "system": SystemMetrics(
                timestamp=time.time(),
                cpu_percent=50.0,
                memory_percent=95.0,  # Above threshold
                memory_used_gb=15.0,
                memory_total_gb=16.0,
                disk_used_gb=100.0,
                disk_total_gb=500.0,
                gpu_metrics=[],
                network_io_bytes_sent=1000,
                network_io_bytes_recv=2000
            )
        }
        
        triggered_alerts = alert_manager.evaluate_alert_rules(metrics)
        
        assert len(triggered_alerts) == 1
        assert triggered_alerts[0].name == "high_memory"
        assert triggered_alerts[0].severity == AlertSeverity.CRITICAL
    
    def test_alert_deduplication(self, alert_manager):
        """Test alert deduplication"""
        rule = AlertRuleClass(
            name="test_alert",
            metric_type=MetricType.SYSTEM,
            metric_name="cpu_percent",
            condition="greater_than",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            duration=0.0  # No duration for test
        )
        alert_manager.add_alert_rule(rule)
        
        metrics = {
            "system": SystemMetrics(
                timestamp=time.time(),
                cpu_percent=85.0,
                memory_percent=60.0,
                memory_used_gb=8.0,
                memory_total_gb=16.0,
                disk_used_gb=100.0,
                disk_total_gb=500.0,
                gpu_metrics=[],
                network_io_bytes_sent=1000,
                network_io_bytes_recv=2000
            )
        }
        
        # First evaluation should trigger alert
        alerts1 = alert_manager.evaluate_alert_rules(metrics)
        assert len(alerts1) == 1
        
        # Second evaluation should not trigger duplicate alert
        alerts2 = alert_manager.evaluate_alert_rules(metrics)
        assert len(alerts2) == 0
        
        # Active alerts should contain one alert
        assert len(alert_manager.active_alerts) == 1
    
    def test_alert_resolution(self, alert_manager):
        """Test alert resolution when conditions are no longer met"""
        rule = AlertRuleClass(
            name="cpu_alert",
            metric_type=MetricType.SYSTEM,
            metric_name="cpu_percent",
            condition="greater_than",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            duration=0.0  # No duration for test
        )
        alert_manager.add_alert_rule(rule)
        
        # High CPU metrics - should trigger alert
        high_cpu_metrics = {
            "system": SystemMetrics(
                timestamp=time.time(),
                cpu_percent=85.0,
                memory_percent=60.0,
                memory_used_gb=8.0,
                memory_total_gb=16.0,
                disk_used_gb=100.0,
                disk_total_gb=500.0,
                gpu_metrics=[],
                network_io_bytes_sent=1000,
                network_io_bytes_recv=2000
            )
        }
        
        alerts = alert_manager.evaluate_alert_rules(high_cpu_metrics)
        assert len(alerts) == 1
        assert len(alert_manager.active_alerts) == 1
        
        # Normal CPU metrics - should resolve alert
        normal_cpu_metrics = {
            "system": SystemMetrics(
                timestamp=time.time(),
                cpu_percent=50.0,
                memory_percent=60.0,
                memory_used_gb=8.0,
                memory_total_gb=16.0,
                disk_used_gb=100.0,
                disk_total_gb=500.0,
                gpu_metrics=[],
                network_io_bytes_sent=1000,
                network_io_bytes_recv=2000
            )
        }
        
        alert_manager.evaluate_alert_rules(normal_cpu_metrics)
        assert len(alert_manager.active_alerts) == 0
    
    def test_get_alert_history(self, alert_manager):
        """Test getting alert history"""
        # Add some alerts to history
        for i in range(5):
            alert = Alert(
                name=f"test_alert_{i}",
                severity=AlertSeverity.WARNING,
                message=f"Test alert {i}",
                timestamp=time.time() + i,
                resolved=i % 2 == 0,
                resolved_timestamp=time.time() + i + 1 if i % 2 == 0 else None
            )
            alert_manager.alert_history.append(alert)
        
        # Get all history
        history = alert_manager.get_alert_history()
        assert len(history) == 5
        
        # Get limited history
        history = alert_manager.get_alert_history(limit=3)
        assert len(history) == 3
        
        # Get only active alerts
        active_history = alert_manager.get_alert_history(active_only=True)
        assert len(active_history) == 2  # Only unresolved alerts

class TestPerformanceMonitor:
    
    @pytest.fixture
    def performance_monitor(self):
        return PerformanceMonitor(
            collection_interval=1.0,
            alert_evaluation_interval=5.0
        )
    
    def test_performance_monitor_initialization(self, performance_monitor):
        """Test performance monitor initialization"""
        assert performance_monitor.collection_interval == 1.0
        assert performance_monitor.alert_evaluation_interval == 5.0
        assert performance_monitor.metrics_collector is not None
        assert performance_monitor.alert_manager is not None
    
    def test_register_training_state(self, performance_monitor):
        """Test registering training state"""
        training_state = {
            "loss": 0.4,
            "accuracy": 0.88,
            "learning_rate": 0.0008,
            "batch_size": 48,
            "epoch": 8,
            "iteration": 800
        }
        
        performance_monitor.register_training_state(training_state)
        assert performance_monitor.current_training_state == training_state
    
    def test_register_network_connections(self, performance_monitor):
        """Test registering network connections"""
        connections = [
            {"worker_id": "worker-1", "latency": 12.0, "bandwidth": 900.0, "packet_loss": 0.0},
            {"worker_id": "worker-2", "latency": 18.0, "bandwidth": 750.0, "packet_loss": 0.2}
        ]
        
        performance_monitor.register_network_connections(connections)
        assert performance_monitor.current_network_connections == connections
    
    def test_add_custom_metric(self, performance_monitor):
        """Test adding custom metrics"""
        performance_monitor.add_custom_metric("custom_throughput", 250.0)
        performance_monitor.add_custom_metric("custom_latency", 5.0)
        
        assert "custom_throughput" in performance_monitor.custom_metrics
        assert "custom_latency" in performance_monitor.custom_metrics
        assert performance_monitor.custom_metrics["custom_throughput"] == 250.0
        assert performance_monitor.custom_metrics["custom_latency"] == 5.0
    
    def test_get_current_performance_summary(self, performance_monitor):
        """Test getting current performance summary"""
        # Set up some state
        training_state = {"loss": 0.3, "accuracy": 0.92}
        performance_monitor.register_training_state(training_state)
        performance_monitor.add_custom_metric("test_metric", 100.0)
        
        summary = performance_monitor.get_current_performance_summary()
        
        assert "system" in summary
        assert "training" in summary
        assert "custom_metrics" in summary
        assert "alerts" in summary
        assert summary["custom_metrics"]["test_metric"] == 100.0
    
    def test_get_performance_trends(self, performance_monitor):
        """Test getting performance trends"""
        # Add some historical metrics
        for i in range(10):
            system_metrics = SystemMetrics(
                timestamp=time.time() + i,
                cpu_percent=50.0 + i,
                memory_percent=60.0,
                memory_used_gb=8.0,
                memory_total_gb=16.0,
                disk_used_gb=100.0,
                disk_total_gb=500.0,
                gpu_metrics=[],
                network_io_bytes_sent=1000,
                network_io_bytes_recv=2000
            )
            
            training_metrics = TrainingMetrics(
                timestamp=time.time() + i,
                loss=0.5 - i * 0.01,  # Decreasing loss
                accuracy=0.8 + i * 0.01,  # Increasing accuracy
                learning_rate=0.001,
                batch_size=32,
                epoch=i // 2,
                iteration=i * 100,
                throughput=150.0 + i * 5,  # Increasing throughput
                gradient_norm=0.1
            )
            
            performance_monitor.metrics_collector.record_metrics(system_metrics, training_metrics, None)
        
        trends = performance_monitor.get_performance_trends()
        
        assert "system" in trends
        assert "training" in trends
        assert "cpu_percent" in trends["system"]
        assert "loss" in trends["training"]
        assert "accuracy" in trends["training"]
        assert trends["training"]["loss"]["trend"] == "decreasing"
        assert trends["training"]["accuracy"]["trend"] == "increasing"
    
    def test_configure_alerting(self, performance_monitor):
        """Test configuring alerting rules"""
        rules = [
            {
                "name": "high_cpu",
                "metric_type": "system",
                "metric_name": "cpu_percent",
                "condition": "greater_than",
                "threshold": 80.0,
                "severity": "warning",
                "duration": 60.0
            },
            {
                "name": "low_accuracy",
                "metric_type": "training",
                "metric_name": "accuracy",
                "condition": "less_than",
                "threshold": 0.8,
                "severity": "critical",
                "duration": 30.0
            }
        ]
        
        performance_monitor.configure_alerting(rules)
        
        assert len(performance_monitor.alert_manager.alert_rules) == 2
        assert performance_monitor.alert_manager.alert_rules[0].name == "high_cpu"
        assert performance_monitor.alert_manager.alert_rules[1].name == "low_accuracy"
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, performance_monitor):
        """Test monitoring start/stop lifecycle"""
        # Start monitoring
        await performance_monitor.start_monitoring()
        assert performance_monitor.monitoring_task is not None
        assert performance_monitor.metrics_collector.collection_task is not None
        
        # Stop monitoring
        await performance_monitor.stop_monitoring()
        assert performance_monitor.monitoring_task is None
        assert performance_monitor.metrics_collector.collection_task is None
        assert performance_monitor._shutdown == True
    
    def test_export_metrics(self, performance_monitor):
        """Test exporting metrics in different formats"""
        # Add some test metrics
        system_metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=75.0,
            memory_percent=65.0,
            memory_used_gb=10.0,
            memory_total_gb=16.0,
            disk_used_gb=150.0,
            disk_total_gb=500.0,
            gpu_metrics=[],
            network_io_bytes_sent=5000,
            network_io_bytes_recv=10000
        )
        
        training_metrics = TrainingMetrics(
            timestamp=time.time(),
            loss=0.25,
            accuracy=0.94,
            learning_rate=0.0003,
            batch_size=64,
            epoch=15,
            iteration=1500,
            throughput=300.0,
            gradient_norm=0.03
        )
        
        performance_monitor.metrics_collector.record_metrics(system_metrics, training_metrics, None)
        
        # Export as JSON
        json_export = performance_monitor.export_metrics("json")
        assert isinstance(json_export, str)
        
        # Export as CSV
        csv_export = performance_monitor.export_metrics("csv")
        assert isinstance(csv_export, str)
        assert "timestamp" in csv_export
        assert "cpu_percent" in csv_export
        assert "loss" in csv_export
    
    def test_get_health_score(self, performance_monitor):
        """Test calculating system health score"""
        # Add metrics that should result in a specific health score
        system_metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=30.0,  # Good
            memory_percent=40.0,  # Good
            memory_used_gb=6.0,
            memory_total_gb=16.0,
            disk_used_gb=100.0,  # Low usage
            disk_total_gb=500.0,
            gpu_metrics=[],
            network_io_bytes_sent=2000,
            network_io_bytes_recv=4000
        )
        
        performance_monitor.metrics_collector.record_metrics(system_metrics, None, None)
        
        health_score = performance_monitor.get_health_score()
        
        assert 0 <= health_score <= 100
        assert isinstance(health_score, (int, float))
        
        # Health score should be reasonably high for good metrics
        assert health_score > 60