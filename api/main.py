#!/usr/bin/env python3
"""
Vercel-compatible version of the Distributed Training Orchestrator Dashboard
This version uses polling instead of WebSockets for Vercel compatibility
"""
import asyncio
import json
import logging
import time
import random
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for demo
class DemoState:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.active_workers = [
            {"id": "worker-1", "status": "active", "connected_at": time.time()},
            {"id": "worker-2", "status": "active", "connected_at": time.time()},
            {"id": "worker-3", "status": "active", "connected_at": time.time()},
        ]
        self.coordinator_status = "active"
        self.training_progress = {
            "epoch": 0,
            "loss": 1.0,
            "accuracy": 0.0,
            "throughput": 200.0
        }
        self.fault_injections = []
        self.cluster_health = 100.0
        self.current_strategy = "AllReduce"
        self.current_scenario = "baseline_training"
        self.start_time = time.time()

# Global demo state
demo_state = DemoState()

# FastAPI app
app = FastAPI(
    title="Distributed Training Orchestrator Dashboard",
    description="Real-time monitoring dashboard for distributed ML training",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def simulate_training_progress():
    """Simulate realistic training progress"""
    current_loss = demo_state.training_progress["loss"]
    current_accuracy = demo_state.training_progress["accuracy"]
    
    # Simulate loss improvement
    if current_loss > 0.5:
        loss_reduction = random.uniform(0.03, 0.08)
    elif current_loss > 0.2:
        loss_reduction = random.uniform(0.01, 0.04)
    else:
        loss_reduction = random.uniform(0.002, 0.01)
    
    new_loss = max(0.05, current_loss - loss_reduction)
    demo_state.training_progress["loss"] = new_loss
    
    # Simulate accuracy improvement
    if current_accuracy < 0.5:
        accuracy_gain = random.uniform(0.02, 0.06)
    elif current_accuracy < 0.8:
        accuracy_gain = random.uniform(0.01, 0.03)
    else:
        accuracy_gain = random.uniform(0.001, 0.01)
    
    new_accuracy = min(0.98, current_accuracy + accuracy_gain)
    demo_state.training_progress["accuracy"] = new_accuracy
    
    # Update epoch
    demo_state.training_progress["epoch"] += 1
    
    # Update throughput with some variation
    demo_state.training_progress["throughput"] = 200.0 + random.uniform(-20, 20)

def simulate_system_metrics():
    """Simulate system metrics"""
    return {
        "cpu_percent": 45.0 + random.uniform(-15, 25),
        "memory_percent": 60.0 + random.uniform(-10, 20),
        "disk_used_gb": 85.0,
        "disk_total_gb": 500.0,
        "network_io_bytes": 1024 * 1024 * random.uniform(50, 200),
        "timestamp": time.time()
    }

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard page"""
    return await get_dashboard_html()

@app.get("/api/status")
async def get_system_status():
    """Get current system status"""
    uptime = time.time() - demo_state.start_time
    health_score = 75.0 + random.uniform(-10, 15)
    
    return {
        "status": "healthy" if health_score > 70 else "warning" if health_score > 40 else "critical",
        "health_score": health_score,
        "timestamp": time.time(),
        "monitoring_active": True,
        "metrics_count": 5,
        "active_alerts": 0,
        "uptime": uptime
    }

@app.get("/api/metrics/current")
async def get_current_metrics():
    """Get current performance metrics"""
    # Simulate training progress
    simulate_training_progress()
    
    # Get system metrics
    system_metrics = simulate_system_metrics()
    
    return {
        "system": system_metrics,
        "training": demo_state.training_progress,
        "cluster": {
            "coordinator": {"status": demo_state.coordinator_status},
            "workers": demo_state.active_workers
        },
        "scenario": demo_state.current_scenario,
        "alerts": {
            "active_count": 0,
            "active_alerts": []
        }
    }

@app.get("/api/insights")
async def get_performance_insights():
    """Get performance insights"""
    insights = [
        "Training loss is decreasing steadily",
        "Cluster performance is optimal",
        "All workers are healthy and responding",
        "Gradient synchronization is efficient"
    ]
    return {"recommendations": insights, "warnings": [], "optimizations": []}

@app.post("/api/demo/add-worker")
async def add_demo_worker():
    """Add a worker to the demo cluster"""
    try:
        worker_id = f"worker-{len(demo_state.active_workers) + 1}"
        demo_state.active_workers.append({
            "id": worker_id,
            "status": "active",
            "connected_at": time.time()
        })
        return {"success": True, "worker_id": worker_id, "message": "Worker added successfully"}
    except Exception as e:
        logger.error(f"Error adding demo worker: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/demo/remove-worker")
async def remove_demo_worker():
    """Remove a worker from the demo cluster"""
    try:
        if len(demo_state.active_workers) > 1:
            removed_worker = demo_state.active_workers.pop()
            return {"success": True, "worker_id": removed_worker["id"], "message": "Worker removed successfully"}
        else:
            return {"success": False, "message": "Cannot remove last worker"}
    except Exception as e:
        logger.error(f"Error removing demo worker: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/demo/inject-failure")
async def inject_demo_failure():
    """Inject a failure into the demo cluster"""
    try:
        active_workers = [w for w in demo_state.active_workers if w["status"] == "active"]
        
        if len(active_workers) >= 1:
            worker = random.choice(active_workers)
            worker_id = worker["id"]
            worker["status"] = "failed"
            
            # Record fault injection
            demo_state.fault_injections.append({
                "type": "worker_failure",
                "target": worker_id,
                "timestamp": time.time()
            })
            
            # Simulate impact on training
            current_loss = demo_state.training_progress["loss"]
            demo_state.training_progress["loss"] = min(1.0, current_loss + random.uniform(0.05, 0.15))
            
            return {"success": True, "worker_id": worker_id, "message": f"Worker {worker_id} failed"}
        else:
            return {"success": False, "message": "No active workers available to fail"}
    except Exception as e:
        logger.error(f"Error injecting demo failure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/demo/switch-strategy")
async def switch_demo_strategy():
    """Switch gradient synchronization strategy"""
    try:
        new_strategy = 'ParameterServer' if demo_state.current_strategy == 'AllReduce' else 'AllReduce'
        demo_state.current_strategy = new_strategy
        return {"success": True, "new_strategy": new_strategy, "message": f"Switched to {new_strategy} strategy"}
    except Exception as e:
        logger.error(f"Error switching demo strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/demo/reset-training")
async def reset_demo_training():
    """Reset training progress to fresh state"""
    try:
        demo_state.training_progress = {
            "epoch": 0,
            "loss": 1.0,
            "accuracy": 0.0,
            "throughput": 200.0
        }
        return {"success": True, "message": "Training progress reset to fresh state"}
    except Exception as e:
        logger.error(f"Error resetting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_dashboard_html() -> str:
    """Generate the dashboard HTML page with polling instead of WebSockets"""
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
            grid-template-columns: repeat(4, 1fr);
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
        
        .cluster-with-controls {
            grid-column: span 3;
        }
        
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
        
        .console-container {
            height: 300px;
            background: #1a1a1a;
            border-radius: 4px;
            padding: 1rem;
            overflow: hidden;
            position: relative;
        }
        
        .console-output {
            height: 100%;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
            line-height: 1.4;
            color: #e0e0e0;
        }
        
        .console-line {
            margin-bottom: 2px;
            word-wrap: break-word;
        }
        
        .console-timestamp {
            color: #888;
            font-weight: bold;
        }
        
        .console-level {
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 0.75rem;
            margin: 0 8px;
        }
        
        .console-level.info {
            background: #2196f3;
            color: white;
        }
        
        .console-level.warning {
            background: #ff9800;
            color: white;
        }
        
        .console-level.error {
            background: #f44336;
            color: white;
        }
        
        .console-level.success {
            background: #4caf50;
            color: white;
        }
        
        .console-message {
            color: #e0e0e0;
        }
        
        .console-output::-webkit-scrollbar {
            width: 8px;
        }
        
        .console-output::-webkit-scrollbar-track {
            background: #333;
            border-radius: 4px;
        }
        
        .console-output::-webkit-scrollbar-thumb {
            background: #666;
            border-radius: 4px;
        }
        
        .console-output::-webkit-scrollbar-thumb:hover {
            background: #888;
        }
        
        .full-width {
            grid-column: 1 / -1;
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
        
        .connection-status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Distributed Training Orchestrator</h1>
        <div class="status">
            <div class="status-indicator" id="statusIndicator"></div>
            <span id="statusText">Loading...</span>
            <div class="connection-status">
                <span id="connectionStatus">●</span>
                <span>Polling</span>
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
        
        <!-- Cluster Visualization -->
        <div class="card cluster-with-controls">
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
        
        <!-- System Console -->
        <div class="card full-width">
            <h3>System Console</h3>
            <div class="console-container">
                <div class="console-output" id="consoleOutput">
                    <div class="console-line">
                        <span class="console-timestamp">[00:00:00]</span>
                        <span class="console-level info">INFO</span>
                        <span class="console-message">System initialized - polling for updates...</span>
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
        // Dashboard JavaScript - Polling version for Vercel
        let consoleLines = [];
        let maxConsoleLines = 50;
        let previousMetrics = {};
        let lossChart = null;
        let accuracyChart = null;
        let pollingInterval = null;
        
        // Initialize dashboard
        function initDashboard() {
            initMiniCharts();
            loadInitialData();
            addConsoleMessage('System starting up...', 'info');
            startPolling();
        }
        
        // Start polling for updates
        function startPolling() {
            updateConnectionStatus(true);
            addConsoleMessage('Polling started - receiving updates every 2 seconds', 'success');
            
            // Poll every 2 seconds
            pollingInterval = setInterval(async () => {
                try {
                    const response = await fetch('/api/metrics/current');
                    const data = await response.json();
                    handleMetricsUpdate(data);
                } catch (error) {
                    console.error('Polling error:', error);
                    addConsoleMessage('Polling error - retrying...', 'error');
                }
            }, 2000);
        }
        
        // Handle metrics update
        function handleMetricsUpdate(data) {
            updateMetrics(data);
            updateHealthScore(75 + Math.random() * 10);
            updateMiniCharts(data);
            updateConsoleFromMetrics(data, data.cluster, data.scenario);
            
            // Update cluster visualization
            if (data.cluster) {
                updateClusterViz(data.cluster);
            }
            
            // Update scenario
            if (data.scenario) {
                updateScenario(data.scenario);
            }
        }
        
        // Console functionality
        function addConsoleMessage(message, level = 'info') {
            const now = new Date();
            const timestamp = now.toLocaleTimeString();
            
            const consoleLine = {
                timestamp: `[${timestamp}]`,
                level: level,
                message: message
            };
            
            consoleLines.push(consoleLine);
            
            if (consoleLines.length > maxConsoleLines) {
                consoleLines = consoleLines.slice(-maxConsoleLines);
            }
            
            updateConsoleDisplay();
        }
        
        function updateConsoleDisplay() {
            const consoleOutput = document.getElementById('consoleOutput');
            consoleOutput.innerHTML = consoleLines.map(line => 
                `<div class="console-line">
                    <span class="console-timestamp">${line.timestamp}</span>
                    <span class="console-level ${line.level}">${line.level.toUpperCase()}</span>
                    <span class="console-message">${line.message}</span>
                </div>`
            ).join('');
            
            consoleOutput.scrollTop = consoleOutput.scrollHeight;
        }
        
        // Update console based on metrics changes
        function updateConsoleFromMetrics(metrics, cluster, scenario) {
            // Track worker status changes
            if (cluster && cluster.workers) {
                cluster.workers.forEach(worker => {
                    const prevWorker = previousMetrics.cluster?.workers?.find(w => w.id === worker.id);
                    
                    if (!prevWorker) {
                        addConsoleMessage(`Worker ${worker.id} joined the cluster`, 'success');
                    } else if (prevWorker.status !== worker.status) {
                        if (worker.status === 'failed') {
                            addConsoleMessage(`Worker ${worker.id} failed - recovery needed`, 'error');
                        } else if (worker.status === 'active' && prevWorker.status === 'failed') {
                            addConsoleMessage(`Worker ${worker.id} recovered successfully`, 'success');
                        }
                    }
                });
            }
            
            // Track training progress
            if (metrics.training) {
                const prevTraining = previousMetrics.training;
                if (prevTraining) {
                    const lossDiff = prevTraining.loss - metrics.training.loss;
                    if (lossDiff > 0.01) {
                        addConsoleMessage(`Training loss improved: ${metrics.training.loss.toFixed(4)} (↓${lossDiff.toFixed(4)})`, 'info');
                    }
                    
                    const accDiff = metrics.training.accuracy - prevTraining.accuracy;
                    if (accDiff > 0.02) {
                        addConsoleMessage(`Training accuracy improved: ${(metrics.training.accuracy * 100).toFixed(2)}% (↑${(accDiff * 100).toFixed(2)}%)`, 'info');
                    }
                }
            }
            
            // Track scenario changes
            if (scenario && scenario !== previousMetrics.scenario) {
                addConsoleMessage(`Scenario changed to: ${scenario.replace('_', ' ')}`, 'info');
            }
            
            previousMetrics = {
                cluster: cluster,
                training: metrics.training,
                scenario: scenario
            };
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
        
        // Update scenario display
        function updateScenario(scenario) {
            const scenarioDisplay = document.getElementById('currentScenario');
            if (scenarioDisplay) {
                scenarioDisplay.textContent = scenario.replace('_', ' ').replace(/\\b\\w/g, l => l.toUpperCase());
            }
        }
        
        // Initialize mini charts
        function initMiniCharts() {
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
                        x: { 
                            title: { display: true, text: 'Epoch' },
                            beginAtZero: true
                        },
                        y: { 
                            title: { display: true, text: 'Loss' },
                            beginAtZero: true 
                        }
                    },
                    plugins: {
                        legend: { display: false }
                    },
                    animation: { duration: 0 }
                }
            });
            
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
                        x: { 
                            title: { display: true, text: 'Epoch' },
                            beginAtZero: true
                        },
                        y: { 
                            title: { display: true, text: 'Accuracy (%)' },
                            beginAtZero: true, 
                            max: 100 
                        }
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
            
            const epoch = metrics.training.epoch || 0;
            const loss = metrics.training.loss || 0;
            const accuracy = (metrics.training.accuracy * 100) || 0;
            
            // Update loss chart
            if (lossChart) {
                const lastEpoch = lossChart.data.labels[lossChart.data.labels.length - 1];
                if (epoch !== lastEpoch || Math.abs(loss - (lossChart.data.datasets[0].data[lossChart.data.datasets[0].data.length - 1] || 0)) > 0.001) {
                    lossChart.data.labels.push(epoch);
                    lossChart.data.datasets[0].data.push(loss);
                    
                    if (lossChart.data.labels.length > 200) {
                        lossChart.data.labels.shift();
                        lossChart.data.datasets[0].data.shift();
                    }
                    
                    lossChart.update('none');
                }
            }
            
            // Update accuracy chart
            if (accuracyChart) {
                const lastEpoch = accuracyChart.data.labels[accuracyChart.data.labels.length - 1];
                if (epoch !== lastEpoch || Math.abs(accuracy - (accuracyChart.data.datasets[0].data[accuracyChart.data.datasets[0].data.length - 1] || 0)) > 0.5) {
                    accuracyChart.data.labels.push(epoch);
                    accuracyChart.data.datasets[0].data.push(accuracy);
                    
                    if (accuracyChart.data.labels.length > 200) {
                        accuracyChart.data.labels.shift();
                        accuracyChart.data.datasets[0].data.shift();
                    }
                    
                    accuracyChart.update('none');
                }
            }
        }
        
        // Update cluster visualization
        function updateClusterViz(clusterData) {
            const workersContainer = document.getElementById('workersContainer');
            workersContainer.innerHTML = '';
            
            if (clusterData.workers) {
                clusterData.workers.forEach(worker => {
                    const workerNode = document.createElement('div');
                    workerNode.className = `cluster-node worker ${worker.status}`;
                    workerNode.innerHTML = `
                        <div class="node-icon">⚙️</div>
                        <div class="node-label">${worker.id}</div>
                        <div class="node-status ${worker.status}">${worker.status}</div>
                    `;
                    workersContainer.appendChild(workerNode);
                });
            }
        }
        
        // Load initial data
        async function loadInitialData() {
            try {
                const insightsResponse = await fetch('/api/insights');
                const insights = await insightsResponse.json();
                updateInsights(insights);
                
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
            addConsoleMessage('Requesting to add new worker...', 'info');
            fetch('/api/demo/add-worker', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    console.log('Worker added:', data);
                    updateStatusMessage('Worker added successfully');
                })
                .catch(error => {
                    console.error('Error adding worker:', error);
                    addConsoleMessage('Failed to add worker', 'error');
                    updateStatusMessage('Error adding worker', 'error');
                });
        }
        
        function removeWorker() {
            addConsoleMessage('Requesting to remove worker...', 'info');
            fetch('/api/demo/remove-worker', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    console.log('Worker removed:', data);
                    updateStatusMessage('Worker removed successfully');
                })
                .catch(error => {
                    console.error('Error removing worker:', error);
                    addConsoleMessage('Failed to remove worker', 'error');
                    updateStatusMessage('Error removing worker', 'error');
                });
        }
        
        function injectFailure() {
            addConsoleMessage('Injecting failure into random worker...', 'warning');
            fetch('/api/demo/inject-failure', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    console.log('Failure injected:', data);
                    updateStatusMessage('Failure injected - observe recovery');
                })
                .catch(error => {
                    console.error('Error injecting failure:', error);
                    addConsoleMessage('Failed to inject failure', 'error');
                    updateStatusMessage('Error injecting failure', 'error');
                });
        }
        
        function switchStrategy() {
            addConsoleMessage('Switching gradient synchronization strategy...', 'info');
            fetch('/api/demo/switch-strategy', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    console.log('Strategy switched:', data);
                    updateStatusMessage('Gradient strategy switched');
                    document.getElementById('gradientStrategy').textContent = data.new_strategy;
                })
                .catch(error => {
                    console.error('Error switching strategy:', error);
                    addConsoleMessage('Failed to switch strategy', 'error');
                    updateStatusMessage('Error switching strategy', 'error');
                });
        }
        
        function resetTraining() {
            addConsoleMessage('Resetting training progress to initial state...', 'info');
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
                    addConsoleMessage('Failed to reset training', 'error');
                    updateStatusMessage('Error resetting training', 'error');
                });
        }
        
        function updateStatusMessage(message, type = 'info') {
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
            
            setTimeout(() => {
                statusMsg.remove();
            }, 3000);
        }
        
        // Initialize dashboard when page loads
        document.addEventListener('DOMContentLoaded', initDashboard);
        
        // Clean up on page unload
        window.addEventListener('beforeunload', () => {
            if (pollingInterval) {
                clearInterval(pollingInterval);
            }
        });
    </script>
</body>
</html>
    '''

# For Vercel, the app needs to be accessible as a module
handler = app