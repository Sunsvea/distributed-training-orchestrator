"""
Tests for the real-time dashboard functionality
"""
import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient

from dashboard.server import DashboardServer
from monitoring.performance_monitor import PerformanceMonitor
from monitoring.metrics_collector import SystemMetrics, TrainingMetrics

class TestDashboardServer:
    """Test dashboard server functionality"""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create a mock performance monitor"""
        monitor = Mock(spec=PerformanceMonitor)
        
        # Mock methods that return data
        monitor.get_current_performance_summary.return_value = {
            "timestamp": 1234567890,
            "system": SystemMetrics(
                timestamp=1234567890,
                cpu_percent=45.0,
                memory_percent=65.0,
                memory_used_gb=8.0,
                memory_total_gb=16.0,
                disk_used_gb=100.0,
                disk_total_gb=500.0,
                gpu_metrics=[],
                network_io_bytes_sent=1000,
                network_io_bytes_recv=2000
            ),
            "training": TrainingMetrics(
                timestamp=1234567890,
                loss=0.35,
                accuracy=0.89,
                learning_rate=0.001,
                batch_size=32,
                epoch=10,
                iteration=1000,
                throughput=200.0,
                gradient_norm=0.05
            ),
            "network": None,
            "custom_metrics": {"test_metric": 42},
            "alerts": {"active_count": 1, "active_alerts": []}
        }
        
        monitor.get_health_score.return_value = 85.0
        monitor.get_monitoring_status.return_value = {
            "monitoring_active": True,
            "metrics_history_size": 50,
            "active_alerts": 1,
            "uptime": 3600
        }
        
        monitor.get_performance_insights.return_value = {
            "recommendations": ["Consider increasing batch size"],
            "warnings": ["High memory usage detected"],
            "optimizations": ["Enable gradient compression"]
        }
        
        # Mock metrics collector
        monitor.metrics_collector = Mock()
        monitor.metrics_collector.get_metrics_history.return_value = []
        
        # Mock alert manager
        monitor.alert_manager = Mock()
        monitor.alert_manager.get_alert_history.return_value = []
        monitor.alert_manager.add_alert_rule = Mock()
        
        # Mock export_metrics method
        monitor.export_metrics = Mock()
        monitor.export_metrics.return_value = '{"sample": "data"}'
        
        return monitor
    
    @pytest.fixture
    def dashboard_server(self, performance_monitor):
        """Create dashboard server instance"""
        return DashboardServer(performance_monitor, host="127.0.0.1", port=8081)
    
    def test_dashboard_initialization(self, dashboard_server, performance_monitor):
        """Test dashboard server initialization"""
        assert dashboard_server.performance_monitor == performance_monitor
        assert dashboard_server.host == "127.0.0.1"
        assert dashboard_server.port == 8081
        assert dashboard_server.active_connections == []
        assert dashboard_server.app is not None
    
    def test_api_status_endpoint(self, dashboard_server):
        """Test the /api/status endpoint"""
        client = TestClient(dashboard_server.app)
        response = client.get("/api/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "health_score" in data
        assert "timestamp" in data
        assert "monitoring_active" in data
        assert data["health_score"] == 85.0
        assert data["status"] == "healthy"
    
    def test_api_current_metrics_endpoint(self, dashboard_server):
        """Test the /api/metrics/current endpoint"""
        client = TestClient(dashboard_server.app)
        response = client.get("/api/metrics/current")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "system" in data
        assert "training" in data
        assert "custom_metrics" in data
        assert data["custom_metrics"]["test_metric"] == 42
    
    def test_api_insights_endpoint(self, dashboard_server):
        """Test the /api/insights endpoint"""
        client = TestClient(dashboard_server.app)
        response = client.get("/api/insights")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "recommendations" in data
        assert "warnings" in data
        assert "optimizations" in data
        assert len(data["recommendations"]) == 1
        assert "Consider increasing batch size" in data["recommendations"]
    
    def test_api_alerts_endpoint(self, dashboard_server):
        """Test the /api/alerts endpoint"""
        client = TestClient(dashboard_server.app)
        response = client.get("/api/alerts")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_dashboard_html_page(self, dashboard_server):
        """Test the main dashboard HTML page"""
        client = TestClient(dashboard_server.app)
        response = client.get("/")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Distributed Training Orchestrator" in response.text
        assert "WebSocket" in response.text
        assert "Performance Trends" in response.text
    
    def test_add_alert_rule_endpoint(self, dashboard_server):
        """Test adding alert rule via API"""
        client = TestClient(dashboard_server.app)
        
        rule_data = {
            "name": "test_rule",
            "metric_type": "system",
            "metric_name": "cpu_percent",
            "condition": "greater_than",
            "threshold": 80.0,
            "severity": "warning",
            "duration": 60.0,
            "message_template": "High CPU usage: {metric_value}%"
        }
        
        response = client.post("/api/alerts/rules", json=rule_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "Alert rule added successfully" in data["message"]
    
    def test_export_metrics_json(self, dashboard_server):
        """Test metrics export in JSON format"""
        client = TestClient(dashboard_server.app)
        response = client.post("/api/metrics/export", params={"format": "json"})
        
        assert response.status_code == 200
        # The response should contain exported metrics
    
    def test_export_metrics_csv(self, dashboard_server):
        """Test metrics export in CSV format"""
        # Mock CSV export specifically
        dashboard_server.performance_monitor.export_metrics = Mock()
        dashboard_server.performance_monitor.export_metrics.return_value = "timestamp,cpu_percent,memory_percent\n1234567890,45.0,65.0"
        
        client = TestClient(dashboard_server.app)
        response = client.post("/api/metrics/export", params={"format": "csv"})
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, dashboard_server):
        """Test WebSocket connection handling"""
        # Mock WebSocket
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        
        # Test connection
        await dashboard_server.connect_websocket(mock_websocket)
        
        assert mock_websocket in dashboard_server.active_connections
        assert len(dashboard_server.active_connections) == 1
        
        # Test disconnection
        dashboard_server.disconnect_websocket(mock_websocket)
        assert len(dashboard_server.active_connections) == 0
    
    @pytest.mark.asyncio
    async def test_websocket_broadcast(self, dashboard_server):
        """Test WebSocket broadcast functionality"""
        # Mock WebSocket connections
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        
        await dashboard_server.connect_websocket(mock_ws1)
        await dashboard_server.connect_websocket(mock_ws2)
        
        # Test broadcast
        message = {"type": "test", "data": "hello"}
        await dashboard_server.broadcast_to_websockets(message)
        
        # Verify both connections received the message
        mock_ws1.send_text.assert_called_once()
        mock_ws2.send_text.assert_called_once()
        
        # Verify message content
        sent_message = mock_ws1.send_text.call_args[0][0]
        assert json.loads(sent_message) == message


class TestDashboardIntegration:
    """Integration tests for dashboard with monitoring system"""
    
    @pytest.mark.asyncio
    async def test_dashboard_with_real_monitor(self):
        """Test dashboard with real performance monitor"""
        # Create real performance monitor
        monitor = PerformanceMonitor(collection_interval=0.1)
        
        # Add sample training state
        monitor.register_training_state({
            "loss": 0.4,
            "accuracy": 0.85,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epoch": 5,
            "iteration": 500,
            "throughput": 150.0,
            "gradient_norm": 0.1
        })
        
        # Create dashboard
        dashboard = DashboardServer(monitor, host="127.0.0.1", port=8082)
        
        # Test that dashboard can access monitor data
        client = TestClient(dashboard.app)
        response = client.get("/api/metrics/current")
        
        assert response.status_code == 200
        data = response.json()
        assert "system" in data
        assert "training" in data
        
        # Cleanup
        await monitor.stop_monitoring()
    
    def test_dashboard_error_handling(self):
        """Test dashboard error handling"""
        # Create monitor that throws errors
        monitor = Mock(spec=PerformanceMonitor)
        monitor.get_current_performance_summary.side_effect = Exception("Test error")
        
        dashboard = DashboardServer(monitor)
        client = TestClient(dashboard.app)
        
        response = client.get("/api/metrics/current")
        assert response.status_code == 500
    
    def test_dashboard_metrics_history(self):
        """Test dashboard metrics history endpoint"""
        monitor = Mock(spec=PerformanceMonitor)
        monitor.metrics_collector = Mock()
        monitor.metrics_collector.get_metrics_history.return_value = [
            {"timestamp": 1234567890, "system": None, "training": None},
            {"timestamp": 1234567891, "system": None, "training": None}
        ]
        
        dashboard = DashboardServer(monitor)
        client = TestClient(dashboard.app)
        
        response = client.get("/api/metrics/history?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
    
    def test_dashboard_performance_trends(self):
        """Test dashboard performance trends endpoint"""
        monitor = Mock(spec=PerformanceMonitor)
        monitor.get_performance_trends.return_value = {
            "system": {
                "cpu_percent": {"trend": "stable", "current": 45.0},
                "memory_percent": {"trend": "increasing", "current": 65.0}
            },
            "training": {
                "loss": {"trend": "decreasing", "current": 0.35},
                "accuracy": {"trend": "increasing", "current": 0.89}
            }
        }
        
        dashboard = DashboardServer(monitor)
        client = TestClient(dashboard.app)
        
        response = client.get("/api/metrics/trends")
        assert response.status_code == 200
        data = response.json()
        assert "system" in data
        assert "training" in data
        assert data["system"]["cpu_percent"]["trend"] == "stable"
        assert data["training"]["loss"]["trend"] == "decreasing"