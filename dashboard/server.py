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
                current_metrics = self.performance_monitor.get_current_performance_summary()
                return self._convert_metrics_to_json(current_metrics)
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
        
        # Demo control endpoints
        @self.app.post("/api/demo/add-worker")
        async def add_demo_worker():
            """Add a worker to the demo cluster"""
            try:
                # Signal to demo orchestrator to add worker
                if hasattr(self, 'demo_orchestrator') and self.demo_orchestrator:
                    worker_id = f"worker-{len(self.demo_orchestrator.demo_state['active_workers']) + 1}"
                    await self.demo_orchestrator._start_worker(worker_id)
                    return {"success": True, "worker_id": worker_id, "message": "Worker added successfully"}
                else:
                    return {"success": False, "message": "Demo orchestrator not available"}
            except Exception as e:
                logger.error(f"Error adding demo worker: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/demo/remove-worker")
        async def remove_demo_worker():
            """Remove a worker from the demo cluster"""
            try:
                # Signal to demo orchestrator to remove worker
                if hasattr(self, 'demo_orchestrator') and self.demo_orchestrator:
                    if len(self.demo_orchestrator.demo_state['active_workers']) > 1:
                        worker_id = self.demo_orchestrator.demo_state['active_workers'][-1]['id']
                        await self.demo_orchestrator._stop_worker(worker_id)
                        return {"success": True, "worker_id": worker_id, "message": "Worker removed successfully"}
                    else:
                        return {"success": False, "message": "Cannot remove last worker"}
                else:
                    return {"success": False, "message": "Demo orchestrator not available"}
            except Exception as e:
                logger.error(f"Error removing demo worker: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/demo/inject-failure")
        async def inject_demo_failure():
            """Inject a failure into the demo cluster"""
            try:
                # Signal to demo orchestrator to inject failure
                if hasattr(self, 'demo_orchestrator') and self.demo_orchestrator:
                    # Get active workers (not already failed)
                    active_workers = [w for w in self.demo_orchestrator.demo_state['active_workers'] if w['status'] == 'active']
                    
                    if len(active_workers) >= 1:
                        # Randomly select an active worker to fail
                        import random
                        worker = random.choice(active_workers)
                        worker_id = worker['id']
                        
                        # Fail the worker immediately
                        worker['status'] = 'failed'
                        
                        # Record fault injection
                        self.demo_orchestrator.demo_state["fault_injections"].append({
                            "type": "worker_failure",
                            "target": worker_id,
                            "timestamp": time.time()
                        })
                        
                        # Simulate impact on training - temporary loss increase
                        current_loss = self.demo_orchestrator.demo_state["training_progress"]["loss"]
                        self.demo_orchestrator.demo_state["training_progress"]["loss"] = min(1.0, current_loss + random.uniform(0.05, 0.15))
                        
                        # Schedule recovery after 8-12 seconds
                        recovery_delay = random.uniform(8, 12)
                        asyncio.create_task(self._schedule_worker_recovery(worker_id, recovery_delay))
                        
                        return {"success": True, "worker_id": worker_id, "message": f"Worker {worker_id} failed - recovery in {recovery_delay:.1f}s"}
                    else:
                        return {"success": False, "message": "No active workers available to fail"}
                else:
                    return {"success": False, "message": "Demo orchestrator not available"}
            except Exception as e:
                logger.error(f"Error injecting demo failure: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/demo/switch-strategy")
        async def switch_demo_strategy():
            """Switch gradient synchronization strategy"""
            try:
                # Signal to demo orchestrator to switch strategy
                if hasattr(self, 'demo_orchestrator') and self.demo_orchestrator:
                    current_strategy = getattr(self.demo_orchestrator, 'current_strategy', 'AllReduce')
                    new_strategy = 'ParameterServer' if current_strategy == 'AllReduce' else 'AllReduce'
                    
                    # Switch strategy logic would go here
                    self.demo_orchestrator.current_strategy = new_strategy
                    
                    return {"success": True, "new_strategy": new_strategy, "message": f"Switched to {new_strategy} strategy"}
                else:
                    return {"success": False, "message": "Demo orchestrator not available"}
            except Exception as e:
                logger.error(f"Error switching demo strategy: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/demo/reset-training")
        async def reset_demo_training():
            """Reset training progress to fresh state"""
            try:
                # Signal to demo orchestrator to reset training
                if hasattr(self, 'demo_orchestrator') and self.demo_orchestrator:
                    # Reset training progress to initial state
                    self.demo_orchestrator.demo_state["training_progress"] = {
                        "epoch": 0,
                        "loss": 1.0,
                        "accuracy": 0.0,
                        "throughput": 200.0
                    }
                    
                    # Clear fault injections
                    self.demo_orchestrator.demo_state["fault_injections"] = []
                    
                    # Reset cluster health
                    self.demo_orchestrator.demo_state["cluster_health"] = 100.0
                    
                    # Reset all workers to active state
                    for worker in self.demo_orchestrator.demo_state["active_workers"]:
                        worker["status"] = "active"
                    
                    return {"success": True, "message": "Training progress reset to fresh state"}
                else:
                    return {"success": False, "message": "Demo orchestrator not available"}
            except Exception as e:
                logger.error(f"Error resetting demo training: {e}")
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
    
    async def _schedule_worker_recovery(self, worker_id: str, delay: float):
        """Schedule automatic recovery of a failed worker"""
        await asyncio.sleep(delay)
        
        # Find and recover the worker
        if hasattr(self, 'demo_orchestrator') and self.demo_orchestrator:
            for worker in self.demo_orchestrator.demo_state['active_workers']:
                if worker['id'] == worker_id and worker['status'] == 'failed':
                    worker['status'] = 'active'
                    logger.info(f"🔧 Worker {worker_id} automatically recovered")
                    break
    
    def _convert_metrics_to_json(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metrics with dataclass objects to JSON-serializable format"""
        json_metrics = {}
        
        for key, value in metrics.items():
            if key == "system" and value is not None:
                json_metrics[key] = {
                    "timestamp": value.timestamp,
                    "cpu_percent": value.cpu_percent,
                    "memory_percent": value.memory_percent,
                    "memory_used_gb": value.memory_used_gb,
                    "memory_total_gb": value.memory_total_gb,
                    "disk_used_gb": value.disk_used_gb,
                    "disk_total_gb": value.disk_total_gb,
                    "gpu_metrics": value.gpu_metrics,
                    "network_io_bytes_sent": value.network_io_bytes_sent,
                    "network_io_bytes_recv": value.network_io_bytes_recv
                }
            elif key == "training" and value is not None:
                json_metrics[key] = {
                    "timestamp": value.timestamp,
                    "loss": value.loss,
                    "accuracy": value.accuracy,
                    "learning_rate": value.learning_rate,
                    "batch_size": value.batch_size,
                    "epoch": value.epoch,
                    "iteration": value.iteration,
                    "throughput": value.throughput,
                    "gradient_norm": value.gradient_norm
                }
            elif key == "network" and value is not None:
                json_metrics[key] = {
                    "timestamp": value.timestamp,
                    "worker_connections": value.worker_connections,
                    "average_latency": value.average_latency,
                    "total_bandwidth": value.total_bandwidth,
                    "packet_loss_rate": value.packet_loss_rate
                }
            elif key == "alerts" and value is not None:
                # Handle alerts with proper serialization
                if isinstance(value, dict):
                    json_metrics[key] = {
                        "active_count": value.get("active_count", 0),
                        "active_alerts": self._serialize_alerts(value.get("active_alerts", []))
                    }
                else:
                    json_metrics[key] = value
            else:
                # For other keys (custom_metrics, etc.)
                json_metrics[key] = value
        
        return json_metrics
    
    def _serialize_alerts(self, alerts: List[Any]) -> List[Dict[str, Any]]:
        """Serialize Alert objects to JSON-compatible format"""
        serialized_alerts = []
        
        for alert in alerts:
            try:
                # Handle Alert dataclass objects
                if hasattr(alert, 'name') and hasattr(alert, 'severity'):
                    alert_dict = {
                        "name": str(alert.name),
                        "severity": str(alert.severity.value) if hasattr(alert.severity, 'value') else str(alert.severity),
                        "message": str(alert.message),
                        "timestamp": float(alert.timestamp),
                        "resolved": bool(alert.resolved)
                    }
                    
                    # Include optional fields if they exist
                    if hasattr(alert, 'resolved_timestamp') and alert.resolved_timestamp is not None:
                        alert_dict["resolved_timestamp"] = float(alert.resolved_timestamp)
                    if hasattr(alert, 'metadata') and alert.metadata:
                        alert_dict["metadata"] = alert.metadata
                    
                    serialized_alerts.append(alert_dict)
                else:
                    # Handle already serialized alerts or dictionaries
                    serialized_alerts.append(alert)
                    
            except Exception as e:
                logger.warning(f"Error serializing alert: {e}")
                # Add a fallback alert representation
                serialized_alerts.append({
                    "name": "serialization_error",
                    "severity": "warning",
                    "message": f"Error serializing alert: {str(e)}",
                    "timestamp": time.time(),
                    "resolved": False
                })
        
        return serialized_alerts
    
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
                    
                    # Get recent alerts and add to metrics before conversion
                    recent_alerts = self.performance_monitor.alert_manager.get_alert_history(limit=5)
                    current_metrics["alerts"] = {
                        "active_count": len([a for a in recent_alerts if not a.resolved]),
                        "active_alerts": recent_alerts
                    }
                    
                    # Convert metrics to JSON-serializable format
                    serializable_metrics = self._convert_metrics_to_json(current_metrics)
                    
                    # Add cluster and scenario data if available
                    cluster_data = None
                    scenario_data = None
                    
                    if hasattr(self, 'demo_orchestrator') and self.demo_orchestrator:
                        cluster_data = {
                            "coordinator": {"status": "active"},
                            "workers": self.demo_orchestrator.demo_state.get("active_workers", [])
                        }
                        scenario_data = self.demo_orchestrator.demo_state.get("current_scenario", "baseline_training")
                    
                    # Broadcast update
                    update_message = {
                        "type": "metrics_update",
                        "timestamp": time.time(),
                        "health_score": health_score,
                        "metrics": serializable_metrics,
                        "cluster": cluster_data,
                        "scenario": scenario_data
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
        
        /* Cluster Visualization */
        .cluster-viz-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .cluster-node {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 1rem;
            margin: 0.5rem;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }
        
        .cluster-node:hover {
            transform: translateY(-2px);
        }
        
        .cluster-node.coordinator {
            border: 2px solid #667eea;
        }
        
        .cluster-node.worker {
            border: 2px solid #4CAF50;
        }
        
        .cluster-node.failed {
            border: 2px solid #f44336;
            opacity: 0.7;
        }
        
        .node-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        
        .node-label {
            font-weight: bold;
            margin-bottom: 0.25rem;
        }
        
        .node-status {
            font-size: 0.8rem;
            color: #666;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            background: #e9ecef;
        }
        
        .node-status.active {
            background: #d4edda;
            color: #155724;
        }
        
        .node-status.failed {
            background: #f8d7da;
            color: #721c24;
        }
        
        .workers-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            margin-top: 1rem;
        }
        
        
        /* Training Progress */
        .training-progress {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .progress-item {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .progress-item label {
            font-weight: bold;
            font-size: 0.9rem;
        }
        
        .mini-chart {
            height: 100px;
            background: #f8f9fa;
            border-radius: 4px;
            padding: 0.5rem;
        }
        
        /* Demo Controls - standalone card */
        .demo-controls {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .control-group label {
            font-weight: bold;
            font-size: 0.9rem;
            color: #333;
        }
        
        .scenario-display, .strategy-display {
            padding: 0.5rem;
            background: #e9ecef;
            border-radius: 4px;
            font-weight: bold;
            color: #667eea;
        }
        
        .demo-btn {
            padding: 0.5rem 1rem;
            margin: 0.25rem;
            border: none;
            border-radius: 4px;
            background: #667eea;
            color: white;
            cursor: pointer;
            font-size: 0.9rem;
            transition: background 0.2s ease;
        }
        
        .demo-btn:hover {
            background: #5a6fd8;
        }
        
        .demo-btn:active {
            transform: translateY(1px);
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
        
        <!-- Cluster Visualization -->
        <div class="card">
            <h3>Cluster Topology</h3>
            <div class="cluster-viz-container">
                <div class="cluster-node coordinator" id="coordinatorNode">
                    <div class="node-icon">🎯</div>
                    <div class="node-label">Coordinator</div>
                    <div class="node-status">Active</div>
                </div>
                <div class="workers-container" id="workersContainer">
                    <!-- Worker nodes will be added dynamically -->
                </div>
            </div>
        </div>
        
        <!-- Demo Controls -->
        <div class="card">
            <h3>Demo Controls</h3>
            <div class="demo-controls">
                <div class="control-group">
                    <label>Current Scenario:</label>
                    <div class="scenario-display" id="currentScenario">Baseline Training</div>
                </div>
                <div class="control-group">
                    <label>Interactive Actions:</label>
                    <button class="demo-btn" onclick="addWorker()">➕ Add Worker</button>
                    <button class="demo-btn" onclick="removeWorker()">➖ Remove Worker</button>
                    <button class="demo-btn" onclick="injectFailure()">💥 Inject Failure</button>
                    <button class="demo-btn" onclick="switchStrategy()">🔄 Switch Strategy</button>
                    <button class="demo-btn" onclick="resetTraining()">🔄 Reset Training</button>
                </div>
                <div class="control-group">
                    <label>Gradient Strategy:</label>
                    <div class="strategy-display" id="gradientStrategy">AllReduce</div>
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
        
        <!-- Training Progress Charts -->
        <div class="card">
            <h3>Training Progress</h3>
            <div class="training-progress">
                <div class="progress-item">
                    <label>Loss Curve:</label>
                    <div class="mini-chart">
                        <canvas id="lossChart"></canvas>
                    </div>
                </div>
                <div class="progress-item">
                    <label>Accuracy Curve:</label>
                    <div class="mini-chart">
                        <canvas id="accuracyChart"></canvas>
                    </div>
                </div>
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
        
        // Demo control functions
        function addWorker() {
            console.log('Adding worker...');
            // Send request to add worker
            fetch('/api/demo/add-worker', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    console.log('Worker added:', data);
                    updateStatusMessage('Worker added successfully');
                })
                .catch(error => {
                    console.error('Error adding worker:', error);
                    updateStatusMessage('Error adding worker', 'error');
                });
        }
        
        function removeWorker() {
            console.log('Removing worker...');
            // Send request to remove worker
            fetch('/api/demo/remove-worker', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    console.log('Worker removed:', data);
                    updateStatusMessage('Worker removed successfully');
                })
                .catch(error => {
                    console.error('Error removing worker:', error);
                    updateStatusMessage('Error removing worker', 'error');
                });
        }
        
        function injectFailure() {
            console.log('Injecting failure...');
            // Send request to inject failure
            fetch('/api/demo/inject-failure', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    console.log('Failure injected:', data);
                    updateStatusMessage('Failure injected - observe recovery');
                })
                .catch(error => {
                    console.error('Error injecting failure:', error);
                    updateStatusMessage('Error injecting failure', 'error');
                });
        }
        
        function switchStrategy() {
            console.log('Switching strategy...');
            // Send request to switch gradient strategy
            fetch('/api/demo/switch-strategy', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    console.log('Strategy switched:', data);
                    updateStatusMessage('Gradient strategy switched');
                    document.getElementById('gradientStrategy').textContent = data.new_strategy;
                })
                .catch(error => {
                    console.error('Error switching strategy:', error);
                    updateStatusMessage('Error switching strategy', 'error');
                });
        }
        
        function resetTraining() {
            console.log('Resetting training...');
            // Send request to reset training progress
            fetch('/api/demo/reset-training', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    console.log('Training reset:', data);
                    updateStatusMessage('Training progress reset to fresh state');
                    
                    // Clear the charts
                    if (lossChart) {
                        lossChart.data.labels = [];
                        lossChart.data.datasets[0].data = [];
                        lossChart.update();
                    }
                    if (accuracyChart) {
                        accuracyChart.data.labels = [];
                        accuracyChart.data.datasets[0].data = [];
                        accuracyChart.update();
                    }
                })
                .catch(error => {
                    console.error('Error resetting training:', error);
                    updateStatusMessage('Error resetting training', 'error');
                });
        }
        
        function updateStatusMessage(message, type = 'info') {
            // Create status message element
            const statusMsg = document.createElement('div');
            statusMsg.className = `status-message ${type}`;
            statusMsg.textContent = message;
            statusMsg.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 10px 20px;
                border-radius: 4px;
                color: white;
                background: ${type === 'error' ? '#f44336' : '#4CAF50'};
                z-index: 1000;
                animation: slideIn 0.3s ease;
            `;
            
            document.body.appendChild(statusMsg);
            
            // Remove after 3 seconds
            setTimeout(() => {
                statusMsg.remove();
            }, 3000);
        }
        
        // Update cluster visualization
        function updateClusterViz(clusterData) {
            const workersContainer = document.getElementById('workersContainer');
            workersContainer.innerHTML = '';
            
            if (clusterData && clusterData.workers) {
                clusterData.workers.forEach(worker => {
                    const workerNode = document.createElement('div');
                    workerNode.className = `cluster-node worker ${worker.status === 'failed' ? 'failed' : ''}`;
                    workerNode.innerHTML = `
                        <div class="node-icon">${worker.status === 'failed' ? '❌' : '⚙️'}</div>
                        <div class="node-label">${worker.id}</div>
                        <div class="node-status ${worker.status === 'active' ? 'active' : 'failed'}">${worker.status}</div>
                    `;
                    workersContainer.appendChild(workerNode);
                });
            }
        }
        
        // Update scenario display
        function updateScenario(scenario) {
            const scenarioDisplay = document.getElementById('currentScenario');
            if (scenarioDisplay) {
                scenarioDisplay.textContent = scenario.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
            }
        }
        
        // Initialize mini charts for training progress
        let lossChart = null;
        let accuracyChart = null;
        
        function initMiniCharts() {
            // Loss chart
            const lossCtx = document.getElementById('lossChart').getContext('2d');
            lossChart = new Chart(lossCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Loss',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { beginAtZero: true }
                    },
                    plugins: {
                        legend: { display: false }
                    },
                    animation: { duration: 0 }
                }
            });
            
            // Accuracy chart
            const accuracyCtx = document.getElementById('accuracyChart').getContext('2d');
            accuracyChart = new Chart(accuracyCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Accuracy',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { beginAtZero: true, max: 100 }
                    },
                    plugins: {
                        legend: { display: false }
                    },
                    animation: { duration: 0 }
                }
            });
        }
        
        function updateMiniCharts(metrics) {
            if (!metrics.training) return;
            
            const now = new Date().toLocaleTimeString();
            
            // Update loss chart
            if (lossChart) {
                lossChart.data.labels.push(now);
                lossChart.data.datasets[0].data.push(metrics.training.loss || 0);
                
                if (lossChart.data.labels.length > 10) {
                    lossChart.data.labels.shift();
                    lossChart.data.datasets[0].data.shift();
                }
                
                lossChart.update('none');
            }
            
            // Update accuracy chart
            if (accuracyChart) {
                accuracyChart.data.labels.push(now);
                accuracyChart.data.datasets[0].data.push((metrics.training.accuracy * 100) || 0);
                
                if (accuracyChart.data.labels.length > 10) {
                    accuracyChart.data.labels.shift();
                    accuracyChart.data.datasets[0].data.shift();
                }
                
                accuracyChart.update('none');
            }
        }
        
        // Override the initDashboard function to include new features
        function initDashboard() {
            initChart();
            initMiniCharts();
            connectWebSocket();
            loadInitialData();
        }
        
        // Override handleWebSocketMessage to handle new demo data
        function handleWebSocketMessage(data) {
            if (data.type === 'metrics_update') {
                updateMetrics(data.metrics);
                updateHealthScore(data.health_score);
                updateAlerts(data.alerts);
                updateChart(data.metrics);
                updateMiniCharts(data.metrics);
                
                // Update cluster visualization if available
                if (data.cluster) {
                    updateClusterViz(data.cluster);
                }
                
                // Update scenario if available
                if (data.scenario) {
                    updateScenario(data.scenario);
                }
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