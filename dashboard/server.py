"""
Real-time dashboard server for distributed training orchestrator
Provides WebSocket-based real-time monitoring and REST API endpoints
"""
import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.performance_monitor import PerformanceMonitor, MetricType, AlertSeverity
from monitoring.metrics_collector import MetricsCollector
from monitoring.alert_manager import AlertManager, AlertRule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardServer:
    """Real-time dashboard server with WebSocket support"""
    
    def __init__(self, performance_monitor: PerformanceMonitor, host: str = "0.0.0.0", port: int = 8080):
        self.performance_monitor = performance_monitor
        self.host = host
        self.port = port
        
        # WebSocket connection manager
        self.active_connections: List[WebSocket] = []
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Distributed Training Orchestrator Dashboard",
            description="Real-time monitoring dashboard for distributed ML training",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        # Background task for broadcasting metrics
        self.broadcast_task: Optional[asyncio.Task] = None
        self._shutdown = False
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            """Serve the main dashboard page"""
            return await self._get_dashboard_html()
        
        @self.app.get("/api/status")
        async def get_system_status():
            """Get current system status"""
            try:
                summary = self.performance_monitor.get_current_performance_summary()
                health_score = self.performance_monitor.get_health_score()
                monitoring_status = self.performance_monitor.get_monitoring_status()
                
                return {
                    "status": "healthy" if health_score > 70 else "warning" if health_score > 40 else "critical",
                    "health_score": health_score,
                    "timestamp": time.time(),
                    "monitoring_active": monitoring_status["monitoring_active"],
                    "metrics_count": monitoring_status["metrics_history_size"],
                    "active_alerts": monitoring_status["active_alerts"],
                    "uptime": monitoring_status["uptime"]
                }
            except Exception as e:
                logger.error(f"Error getting system status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/metrics/current")
        async def get_current_metrics():
            """Get current performance metrics"""
            try:
                return self.performance_monitor.get_current_performance_summary()
            except Exception as e:
                logger.error(f"Error getting current metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/metrics/history")
        async def get_metrics_history(limit: int = 100, start_time: Optional[float] = None):
            """Get historical metrics"""
            try:
                return self.performance_monitor.metrics_collector.get_metrics_history(
                    limit=limit, start_time=start_time
                )
            except Exception as e:
                logger.error(f"Error getting metrics history: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/metrics/trends")
        async def get_performance_trends():
            """Get performance trends"""
            try:
                return self.performance_monitor.get_performance_trends()
            except Exception as e:
                logger.error(f"Error getting performance trends: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/alerts")
        async def get_alerts(active_only: bool = False, limit: int = 50):
            """Get alerts"""
            try:
                history = self.performance_monitor.alert_manager.get_alert_history(
                    limit=limit, active_only=active_only
                )
                return [
                    {
                        "name": alert.name,
                        "severity": alert.severity.value,
                        "message": alert.message,
                        "timestamp": alert.timestamp,
                        "resolved": alert.resolved,
                        "resolved_timestamp": alert.resolved_timestamp,
                        "metadata": alert.metadata
                    }
                    for alert in history
                ]
            except Exception as e:
                logger.error(f"Error getting alerts: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/insights")
        async def get_performance_insights():
            """Get performance insights and recommendations"""
            try:
                return self.performance_monitor.get_performance_insights()
            except Exception as e:
                logger.error(f"Error getting performance insights: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/alerts/rules")
        async def add_alert_rule(rule_data: dict):
            """Add a new alert rule"""
            try:
                rule = AlertRule(
                    name=rule_data["name"],
                    metric_type=MetricType(rule_data["metric_type"]),
                    metric_name=rule_data["metric_name"],
                    condition=rule_data["condition"],
                    threshold=float(rule_data["threshold"]),
                    severity=AlertSeverity(rule_data["severity"]),
                    duration=float(rule_data.get("duration", 60.0)),
                    message_template=rule_data.get("message_template", "")
                )
                self.performance_monitor.alert_manager.add_alert_rule(rule)
                return {"success": True, "message": "Alert rule added successfully"}
            except Exception as e:
                logger.error(f"Error adding alert rule: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/api/metrics/export")
        async def export_metrics(format: str = "json", limit: Optional[int] = None):
            """Export metrics in specified format"""
            try:
                exported_data = self.performance_monitor.export_metrics(format, limit)
                
                if format.lower() == "json":
                    return JSONResponse(
                        content=json.loads(exported_data),
                        headers={"Content-Disposition": "attachment; filename=metrics.json"}
                    )
                elif format.lower() == "csv":
                    return JSONResponse(
                        content={"data": exported_data},
                        headers={"Content-Disposition": "attachment; filename=metrics.csv"}
                    )
                else:
                    raise HTTPException(status_code=400, detail="Unsupported format")
            except Exception as e:
                logger.error(f"Error exporting metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await self.connect_websocket(websocket)
            try:
                while True:
                    # Keep connection alive and handle incoming messages
                    message = await websocket.receive_text()
                    data = json.loads(message)
                    
                    # Handle different message types
                    if data.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                    elif data.get("type") == "subscribe":
                        # Client is subscribing to updates
                        await websocket.send_text(json.dumps({
                            "type": "subscription_confirmed",
                            "timestamp": time.time()
                        }))
                    
            except WebSocketDisconnect:
                self.disconnect_websocket(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.disconnect_websocket(websocket)
    
    async def connect_websocket(self, websocket: WebSocket):
        """Add new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Active connections: {len(self.active_connections)}")
    
    def disconnect_websocket(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Active connections: {len(self.active_connections)}")
    
    async def broadcast_to_websockets(self, message: dict):
        """Broadcast message to all connected WebSocket clients"""
        if not self.active_connections:
            return
        
        message_str = json.dumps(message)
        disconnected_connections = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected_connections.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected_connections:
            self.disconnect_websocket(connection)
    
    async def _broadcast_metrics_loop(self):
        """Background task to broadcast metrics to WebSocket clients"""
        while not self._shutdown:
            try:
                if self.active_connections:
                    # Get current metrics and status
                    current_metrics = self.performance_monitor.get_current_performance_summary()
                    health_score = self.performance_monitor.get_health_score()
                    
                    # Get recent alerts
                    recent_alerts = self.performance_monitor.alert_manager.get_alert_history(limit=5)
                    alerts_data = [
                        {
                            "name": alert.name,
                            "severity": alert.severity.value,
                            "message": alert.message,
                            "timestamp": alert.timestamp,
                            "resolved": alert.resolved
                        }
                        for alert in recent_alerts
                    ]
                    
                    # Broadcast update
                    update_message = {
                        "type": "metrics_update",
                        "timestamp": time.time(),
                        "health_score": health_score,
                        "metrics": current_metrics,
                        "alerts": alerts_data
                    }
                    
                    await self.broadcast_to_websockets(update_message)
                
                # Wait before next broadcast
                await asyncio.sleep(2.0)  # Broadcast every 2 seconds
                
            except asyncio.CancelledError:
                logger.info("Metrics broadcast loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in metrics broadcast loop: {e}")
                await asyncio.sleep(5.0)  # Wait longer on error
    
    async def _get_dashboard_html(self) -> str:
        """Generate the dashboard HTML page"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Distributed Training Orchestrator Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            color: #333;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            font-size: 1.8rem;
            font-weight: 300;
        }
        
        .header .status {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-top: 0.5rem;
        }
        
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        .status-healthy { background-color: #4CAF50; }
        .status-warning { background-color: #FF9800; }
        .status-critical { background-color: #f44336; }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .dashboard {
            padding: 2rem;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .card {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }
        
        .card:hover {
            transform: translateY(-2px);
        }
        
        .card h3 {
            color: #333;
            margin-bottom: 1rem;
            font-size: 1.2rem;
            font-weight: 500;
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 1rem;
        }
        
        .metric {
            text-align: center;
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 6px;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: #667eea;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #666;
            margin-top: 0.25rem;
        }
        
        .chart-container {
            position: relative;
            height: 300px;
        }
        
        .alerts-list {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .alert-item {
            padding: 0.75rem;
            margin: 0.5rem 0;
            border-radius: 4px;
            border-left: 4px solid;
        }
        
        .alert-critical {
            background-color: #ffebee;
            border-left-color: #f44336;
        }
        
        .alert-warning {
            background-color: #fff3e0;
            border-left-color: #ff9800;
        }
        
        .alert-info {
            background-color: #e3f2fd;
            border-left-color: #2196f3;
        }
        
        .alert-severity {
            font-weight: bold;
            text-transform: uppercase;
            font-size: 0.8rem;
        }
        
        .alert-message {
            margin-top: 0.25rem;
            font-size: 0.9rem;
        }
        
        .alert-time {
            font-size: 0.8rem;
            color: #666;
            margin-top: 0.25rem;
        }
        
        .connection-status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.9rem;
        }
        
        .insights-list {
            max-height: 200px;
            overflow-y: auto;
        }
        
        .insight-item {
            padding: 0.5rem;
            margin: 0.25rem 0;
            background: #f8f9fa;
            border-radius: 4px;
            font-size: 0.9rem;
        }
        
        .insight-recommendation {
            color: #4CAF50;
        }
        
        .insight-warning {
            color: #FF9800;
        }
        
        .full-width {
            grid-column: 1 / -1;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Distributed Training Orchestrator</h1>
        <div class="status">
            <div class="status-indicator" id="statusIndicator"></div>
            <span id="statusText">Connecting...</span>
            <div class="connection-status">
                <span id="connectionStatus">●</span>
                <span>WebSocket</span>
            </div>
        </div>
    </div>
    
    <div class="dashboard">
        <!-- System Overview -->
        <div class="card">
            <h3>System Overview</h3>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-value" id="healthScore">--</div>
                    <div class="metric-label">Health Score</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="activeAlerts">--</div>
                    <div class="metric-label">Active Alerts</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="uptime">--</div>
                    <div class="metric-label">Uptime</div>
                </div>
            </div>
        </div>
        
        <!-- System Metrics -->
        <div class="card">
            <h3>System Resources</h3>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-value" id="cpuUsage">--</div>
                    <div class="metric-label">CPU %</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="memoryUsage">--</div>
                    <div class="metric-label">Memory %</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="diskUsage">--</div>
                    <div class="metric-label">Disk %</div>
                </div>
            </div>
        </div>
        
        <!-- Training Metrics -->
        <div class="card">
            <h3>Training Progress</h3>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-value" id="currentLoss">--</div>
                    <div class="metric-label">Loss</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="currentAccuracy">--</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="throughput">--</div>
                    <div class="metric-label">Throughput</div>
                </div>
            </div>
        </div>
        
        <!-- Performance Charts -->
        <div class="card full-width">
            <h3>Performance Trends</h3>
            <div class="chart-container">
                <canvas id="performanceChart"></canvas>
            </div>
        </div>
        
        <!-- Alerts -->
        <div class="card">
            <h3>Recent Alerts</h3>
            <div class="alerts-list" id="alertsList">
                <div style="text-align: center; color: #666; padding: 2rem;">
                    No alerts to display
                </div>
            </div>
        </div>
        
        <!-- Insights -->
        <div class="card">
            <h3>Performance Insights</h3>
            <div class="insights-list" id="insightsList">
                <div style="text-align: center; color: #666; padding: 2rem;">
                    Loading insights...
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Dashboard JavaScript
        let ws = null;
        let chart = null;
        let chartData = {
            labels: [],
            datasets: [
                {
                    label: 'CPU %',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.1
                },
                {
                    label: 'Memory %',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    tension: 0.1
                }
            ]
        };
        
        // Initialize dashboard
        function initDashboard() {
            initChart();
            connectWebSocket();
            loadInitialData();
        }
        
        // Initialize performance chart
        function initChart() {
            const ctx = document.getElementById('performanceChart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'line',
                data: chartData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    },
                    animation: {
                        duration: 0 // Disable animations for real-time updates
                    }
                }
            });
        }
        
        // Connect to WebSocket
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                console.log('WebSocket connected');
                updateConnectionStatus(true);
                
                // Subscribe to updates
                ws.send(JSON.stringify({type: 'subscribe'}));
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            ws.onclose = function() {
                console.log('WebSocket disconnected');
                updateConnectionStatus(false);
                
                // Attempt to reconnect after 5 seconds
                setTimeout(connectWebSocket, 5000);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
                updateConnectionStatus(false);
            };
        }
        
        // Handle WebSocket messages
        function handleWebSocketMessage(data) {
            if (data.type === 'metrics_update') {
                updateMetrics(data.metrics);
                updateHealthScore(data.health_score);
                updateAlerts(data.alerts);
                updateChart(data.metrics);
            }
        }
        
        // Update connection status
        function updateConnectionStatus(connected) {
            const statusElement = document.getElementById('connectionStatus');
            statusElement.style.color = connected ? '#4CAF50' : '#f44336';
        }
        
        // Update metrics display
        function updateMetrics(metrics) {
            // System metrics
            if (metrics.system) {
                document.getElementById('cpuUsage').textContent = 
                    metrics.system.cpu_percent?.toFixed(1) || '--';
                document.getElementById('memoryUsage').textContent = 
                    metrics.system.memory_percent?.toFixed(1) || '--';
                    
                const diskPercent = metrics.system.disk_used_gb && metrics.system.disk_total_gb ?
                    (metrics.system.disk_used_gb / metrics.system.disk_total_gb * 100).toFixed(1) : '--';
                document.getElementById('diskUsage').textContent = diskPercent;
            }
            
            // Training metrics
            if (metrics.training) {
                document.getElementById('currentLoss').textContent = 
                    metrics.training.loss?.toFixed(4) || '--';
                document.getElementById('currentAccuracy').textContent = 
                    (metrics.training.accuracy * 100)?.toFixed(2) + '%' || '--';
                document.getElementById('throughput').textContent = 
                    metrics.training.throughput?.toFixed(1) || '--';
            }
            
            // Alert count
            if (metrics.alerts) {
                document.getElementById('activeAlerts').textContent = metrics.alerts.active_count || 0;
            }
        }
        
        // Update health score
        function updateHealthScore(healthScore) {
            const healthElement = document.getElementById('healthScore');
            const statusIndicator = document.getElementById('statusIndicator');
            const statusText = document.getElementById('statusText');
            
            healthElement.textContent = healthScore?.toFixed(0) || '--';
            
            // Update status
            let status, statusClass;
            if (healthScore >= 70) {
                status = 'Healthy';
                statusClass = 'status-healthy';
            } else if (healthScore >= 40) {
                status = 'Warning';
                statusClass = 'status-warning';
            } else {
                status = 'Critical';
                statusClass = 'status-critical';
            }
            
            statusText.textContent = status;
            statusIndicator.className = `status-indicator ${statusClass}`;
        }
        
        // Update alerts display
        function updateAlerts(alerts) {
            const alertsList = document.getElementById('alertsList');
            
            if (!alerts || alerts.length === 0) {
                alertsList.innerHTML = '<div style="text-align: center; color: #666; padding: 2rem;">No recent alerts</div>';
                return;
            }
            
            const alertsHtml = alerts.map(alert => {
                const severity = alert.severity || 'info';
                const time = new Date(alert.timestamp * 1000).toLocaleTimeString();
                
                return `
                    <div class="alert-item alert-${severity}">
                        <div class="alert-severity">${severity}</div>
                        <div class="alert-message">${alert.message}</div>
                        <div class="alert-time">${time}</div>
                    </div>
                `;
            }).join('');
            
            alertsList.innerHTML = alertsHtml;
        }
        
        // Update performance chart
        function updateChart(metrics) {
            if (!metrics.system) return;
            
            const now = new Date().toLocaleTimeString();
            
            // Add new data point
            chartData.labels.push(now);
            chartData.datasets[0].data.push(metrics.system.cpu_percent || 0);
            chartData.datasets[1].data.push(metrics.system.memory_percent || 0);
            
            // Keep only last 20 data points
            if (chartData.labels.length > 20) {
                chartData.labels.shift();
                chartData.datasets[0].data.shift();
                chartData.datasets[1].data.shift();
            }
            
            chart.update('none'); // Update without animation
        }
        
        // Load initial data
        async function loadInitialData() {
            try {
                // Load performance insights
                const insightsResponse = await fetch('/api/insights');
                const insights = await insightsResponse.json();
                updateInsights(insights);
                
                // Load current status
                const statusResponse = await fetch('/api/status');
                const status = await statusResponse.json();
                
                document.getElementById('uptime').textContent = 
                    formatUptime(status.uptime);
                
            } catch (error) {
                console.error('Error loading initial data:', error);
            }
        }
        
        // Update insights display
        function updateInsights(insights) {
            const insightsList = document.getElementById('insightsList');
            
            const allInsights = [
                ...(insights.recommendations || []).map(item => ({text: item, type: 'recommendation'})),
                ...(insights.warnings || []).map(item => ({text: item, type: 'warning'})),
                ...(insights.optimizations || []).map(item => ({text: item, type: 'recommendation'}))
            ];
            
            if (allInsights.length === 0) {
                insightsList.innerHTML = '<div style="text-align: center; color: #666; padding: 2rem;">No insights available</div>';
                return;
            }
            
            const insightsHtml = allInsights.map(insight => `
                <div class="insight-item insight-${insight.type}">
                    ${insight.text}
                </div>
            `).join('');
            
            insightsList.innerHTML = insightsHtml;
        }
        
        // Format uptime
        function formatUptime(seconds) {
            if (!seconds) return '--';
            
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            
            if (hours > 0) {
                return `${hours}h ${minutes}m`;
            } else {
                return `${minutes}m`;
            }
        }
        
        // Initialize dashboard when page loads
        document.addEventListener('DOMContentLoaded', initDashboard);
    </script>
</body>
</html>
        '''
    
    async def start(self):
        """Start the dashboard server"""
        self._shutdown = False
        
        # Start background metrics broadcasting
        self.broadcast_task = asyncio.create_task(self._broadcast_metrics_loop())
        
        logger.info(f"Starting dashboard server on {self.host}:{self.port}")
        
        # Start the FastAPI server
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def stop(self):
        """Stop the dashboard server"""
        self._shutdown = True
        
        if self.broadcast_task:
            self.broadcast_task.cancel()
            try:
                await self.broadcast_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Dashboard server stopped")


# Standalone server runner
async def run_dashboard(performance_monitor: PerformanceMonitor, host: str = "0.0.0.0", port: int = 8080):
    """Run the dashboard server"""
    dashboard = DashboardServer(performance_monitor, host, port)
    await dashboard.start()


if __name__ == "__main__":
    # Example usage - would normally get performance_monitor from main application
    from monitoring.performance_monitor import PerformanceMonitor
    
    monitor = PerformanceMonitor()
    
    # Start monitoring
    asyncio.create_task(monitor.start_monitoring())
    
    # Run dashboard
    asyncio.run(run_dashboard(monitor))