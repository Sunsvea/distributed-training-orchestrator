#!/usr/bin/env python3
"""
Simple Interactive Demo for Distributed Training Orchestrator
This version focuses on the dashboard with simulated data for quick demonstrations
"""
import asyncio
import logging
import sys
import signal
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import time
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from monitoring.performance_monitor import PerformanceMonitor
from dashboard.server import DashboardServer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DemoScenario(Enum):
    BASELINE_TRAINING = "baseline_training"
    DYNAMIC_SCALING = "dynamic_scaling"
    WORKER_FAILURE = "worker_failure"
    GRADIENT_STRATEGIES = "gradient_strategies"

@dataclass
class DemoConfig:
    initial_workers: int = 3
    dashboard_port: int = 8080
    training_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.01

class SimpleDemoOrchestrator:
    """
    Simple demo orchestrator for quick dashboard demonstrations
    """
    
    def __init__(self, config: DemoConfig):
        self.config = config
        self.running = False
        self.current_scenario = DemoScenario.BASELINE_TRAINING
        
        # Demo state
        self.demo_state = {
            "current_scenario": self.current_scenario.value,
            "active_workers": [],
            "coordinator_status": "active",
            "training_progress": {
                "epoch": 0,
                "loss": 1.0,
                "accuracy": 0.0,
                "throughput": 200.0
            },
            "fault_injections": [],
            "cluster_health": 100.0,
            "current_strategy": "AllReduce"
        }
        
        # Components
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.dashboard: Optional[DashboardServer] = None
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        # Initialize demo workers
        self._init_demo_workers()
    
    def _init_demo_workers(self):
        """Initialize demo workers in the state"""
        for i in range(self.config.initial_workers):
            worker_id = f"worker-{i+1}"
            self.demo_state["active_workers"].append({
                "id": worker_id,
                "status": "active",
                "connected_at": time.time()
            })
    
    async def start_demo(self):
        """Start the demo environment"""
        logger.info("🚀 Starting Simple Interactive Demo")
        self.running = True
        
        try:
            # Initialize components
            await self._initialize_components()
            
            # Start dashboard
            await self._start_dashboard()
            
            # Start demo scenarios
            await self._start_demo_scenarios()
            
            # Print instructions
            self._print_demo_instructions()
            
            # Keep demo running
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Demo error: {e}")
            await self.stop_demo()
    
    async def _initialize_components(self):
        """Initialize performance monitor"""
        logger.info("Initializing performance monitor...")
        
        self.performance_monitor = PerformanceMonitor(
            collection_interval=2.0,
            alert_evaluation_interval=5.0
        )
        
        # Setup demo alerts
        await self._setup_demo_alerts()
        
        # Start monitoring
        await self.performance_monitor.start_monitoring()
        
        logger.info("✅ Performance monitor initialized")
    
    async def _setup_demo_alerts(self):
        """Setup demo alerts"""
        from monitoring.alert_manager import AlertRule
        from monitoring.performance_monitor import MetricType, AlertSeverity
        
        demo_alerts = [
            AlertRule(
                name="demo_high_cpu",
                metric_type=MetricType.SYSTEM,
                metric_name="cpu_percent",
                condition="greater_than",
                threshold=70.0,
                severity=AlertSeverity.WARNING,
                duration=10.0,
                message_template="High CPU usage detected: {metric_value}%"
            ),
            AlertRule(
                name="demo_training_stall",
                metric_type=MetricType.TRAINING,
                metric_name="loss",
                condition="greater_than",
                threshold=0.8,
                severity=AlertSeverity.WARNING,
                duration=30.0,
                message_template="Training loss not improving: {metric_value}"
            )
        ]
        
        for alert in demo_alerts:
            self.performance_monitor.alert_manager.add_alert_rule(alert)
    
    async def _start_dashboard(self):
        """Start the dashboard"""
        logger.info("Starting dashboard...")
        
        self.dashboard = DashboardServer(
            performance_monitor=self.performance_monitor,
            host="0.0.0.0",
            port=self.config.dashboard_port
        )
        
        # Connect dashboard to demo orchestrator
        self.dashboard.demo_orchestrator = self
        self.dashboard.current_strategy = "AllReduce"
        
        # Start dashboard in background
        dashboard_task = asyncio.create_task(self.dashboard.start())
        self.background_tasks.append(dashboard_task)
        
        await asyncio.sleep(2)  # Wait for dashboard to start
        
        logger.info(f"✅ Dashboard started at http://localhost:{self.config.dashboard_port}")
    
    async def _start_demo_scenarios(self):
        """Start background demo scenarios"""
        logger.info("Starting demo scenarios...")
        
        # Start scenario rotation
        scenario_task = asyncio.create_task(self._run_scenario_rotation())
        self.background_tasks.append(scenario_task)
        
        # Start metrics simulation
        metrics_task = asyncio.create_task(self._simulate_realistic_metrics())
        self.background_tasks.append(metrics_task)
        
        logger.info("✅ Demo scenarios started")
    
    async def _run_scenario_rotation(self):
        """Run different demo scenarios in rotation"""
        scenarios = [
            (DemoScenario.BASELINE_TRAINING, 45),  # More time to show loss convergence
            (DemoScenario.DYNAMIC_SCALING, 25),
            (DemoScenario.WORKER_FAILURE, 20),
            (DemoScenario.GRADIENT_STRATEGIES, 20)
        ]
        
        while self.running:
            for scenario, duration in scenarios:
                if not self.running:
                    break
                    
                logger.info(f"🎬 Starting scenario: {scenario.value}")
                self.current_scenario = scenario
                self.demo_state["current_scenario"] = scenario.value
                
                # Execute scenario repeatedly during its duration
                start_time = time.time()
                while time.time() - start_time < duration and self.running:
                    await self._execute_scenario(scenario)
                    await asyncio.sleep(2)  # Update every 2 seconds
    
    async def _execute_scenario(self, scenario: DemoScenario):
        """Execute a specific demo scenario"""
        if scenario == DemoScenario.BASELINE_TRAINING:
            await self._baseline_training_scenario()
        elif scenario == DemoScenario.DYNAMIC_SCALING:
            await self._dynamic_scaling_scenario()
        elif scenario == DemoScenario.WORKER_FAILURE:
            await self._worker_failure_scenario()
        elif scenario == DemoScenario.GRADIENT_STRATEGIES:
            await self._gradient_strategies_scenario()
    
    async def _baseline_training_scenario(self):
        """Simulate normal training progress"""
        # More aggressive loss improvement for better demo visualization
        current_loss = self.demo_state["training_progress"]["loss"]
        
        # Start with faster initial convergence, then slow down
        if current_loss > 0.5:
            # Fast initial convergence
            loss_reduction = random.uniform(0.03, 0.08)
        elif current_loss > 0.2:
            # Moderate convergence
            loss_reduction = random.uniform(0.01, 0.04)
        else:
            # Slow final convergence
            loss_reduction = random.uniform(0.002, 0.01)
        
        new_loss = max(0.05, current_loss - loss_reduction)
        self.demo_state["training_progress"]["loss"] = new_loss
        
        # Corresponding accuracy improvement
        current_accuracy = self.demo_state["training_progress"]["accuracy"]
        
        # Accuracy follows inverse relationship with loss
        if current_accuracy < 0.5:
            # Fast initial accuracy gains
            accuracy_gain = random.uniform(0.02, 0.06)
        elif current_accuracy < 0.8:
            # Moderate accuracy gains
            accuracy_gain = random.uniform(0.01, 0.03)
        else:
            # Slow final accuracy gains
            accuracy_gain = random.uniform(0.001, 0.01)
        
        new_accuracy = min(0.98, current_accuracy + accuracy_gain)
        self.demo_state["training_progress"]["accuracy"] = new_accuracy
        
        # Update training state in performance monitor
        training_state = {
            "loss": new_loss,
            "accuracy": new_accuracy,
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "epoch": self.demo_state["training_progress"]["epoch"],
            "iteration": self.demo_state["training_progress"]["epoch"] * 100,
            "throughput": 200.0 + random.uniform(-20, 20),
            "gradient_norm": 0.1 + random.uniform(-0.05, 0.05)
        }
        
        self.performance_monitor.register_training_state(training_state)
        self.demo_state["training_progress"]["epoch"] += 1
    
    async def _dynamic_scaling_scenario(self):
        """Simulate adding/removing workers"""
        # Continue baseline training during scaling
        await self._baseline_training_scenario()
        
        # Add worker occasionally
        if random.random() < 0.1 and len(self.demo_state['active_workers']) < 8:
            new_worker_id = f"worker-{len(self.demo_state['active_workers']) + 1}"
            self.demo_state["active_workers"].append({
                "id": new_worker_id,
                "status": "active",
                "connected_at": time.time()
            })
            logger.info(f"🔄 Added worker: {new_worker_id}")
            
            # Simulate slight throughput improvement with more workers
            current_throughput = self.demo_state["training_progress"]["throughput"]
            self.demo_state["training_progress"]["throughput"] = current_throughput + random.uniform(20, 40)
        
        # Remove worker occasionally
        elif random.random() < 0.1 and len(self.demo_state["active_workers"]) > 3:
            removed_worker = self.demo_state["active_workers"].pop()
            logger.info(f"🔄 Removed worker: {removed_worker['id']}")
            
            # Throughput decreases when worker is removed
            current_throughput = self.demo_state["training_progress"]["throughput"]
            self.demo_state["training_progress"]["throughput"] = max(150, current_throughput - random.uniform(20, 40))
    
    async def _worker_failure_scenario(self):
        """Simulate worker failure and recovery"""
        # Continue training progress
        await self._baseline_training_scenario()
        
        # Occasionally inject failures
        if random.random() < 0.05 and len(self.demo_state["active_workers"]) > 1:
            # Find a worker to fail
            active_workers = [w for w in self.demo_state["active_workers"] if w["status"] == "active"]
            if active_workers:
                worker = random.choice(active_workers)
                worker["status"] = "failed"
                logger.info(f"💥 Worker {worker['id']} failed")
                
                # Record fault injection
                self.demo_state["fault_injections"].append({
                    "type": "worker_failure",
                    "target": worker["id"],
                    "timestamp": time.time()
                })
                
                # Simulate impact on training - temporary loss increase
                current_loss = self.demo_state["training_progress"]["loss"]
                self.demo_state["training_progress"]["loss"] = min(1.0, current_loss + random.uniform(0.05, 0.15))
        
        # Recover failed workers occasionally
        elif random.random() < 0.1:
            failed_workers = [w for w in self.demo_state["active_workers"] if w["status"] == "failed"]
            if failed_workers:
                worker = random.choice(failed_workers)
                worker["status"] = "active"
                logger.info(f"🔧 Worker {worker['id']} recovered")
    
    async def _gradient_strategies_scenario(self):
        """Simulate switching gradient strategies"""
        # Continue training progress
        await self._baseline_training_scenario()
        
        # Occasionally switch strategies
        if random.random() < 0.1:
            current_strategy = self.demo_state["current_strategy"]
            if current_strategy == "AllReduce":
                logger.info("🔄 Switching to Parameter Server strategy")
                self.demo_state["current_strategy"] = "ParameterServer"
            else:
                logger.info("🔄 Switching back to AllReduce strategy")
                self.demo_state["current_strategy"] = "AllReduce"
    
    async def _simulate_realistic_metrics(self):
        """Simulate realistic system metrics"""
        while self.running:
            # Network metrics
            active_workers = [w for w in self.demo_state["active_workers"] if w["status"] == "active"]
            worker_connections = [
                {
                    "worker_id": worker["id"],
                    "status": worker["status"],
                    "connection_time": worker["connected_at"],
                    "last_seen": time.time(),
                    "latency": 1.0 + random.uniform(-0.5, 2.0),
                    "bandwidth": 100.0 + random.uniform(-20, 50),
                    "packet_loss": random.uniform(0, 0.05)
                }
                for worker in active_workers
            ]
            self.performance_monitor.register_network_connections(worker_connections)
            
            await asyncio.sleep(2)
    
    async def _start_worker(self, worker_id: str):
        """Add a worker to the demo"""
        self.demo_state["active_workers"].append({
            "id": worker_id,
            "status": "active",
            "connected_at": time.time()
        })
        logger.info(f"✅ Worker {worker_id} added")
    
    async def _stop_worker(self, worker_id: str):
        """Remove a worker from the demo"""
        self.demo_state["active_workers"] = [
            w for w in self.demo_state["active_workers"] 
            if w["id"] != worker_id
        ]
        logger.info(f"✅ Worker {worker_id} removed")
    
    async def _simulate_worker_failure(self, worker_id: str):
        """Simulate worker failure"""
        for worker in self.demo_state["active_workers"]:
            if worker["id"] == worker_id:
                worker["status"] = "failed"
                break
        
        # Record fault injection
        self.demo_state["fault_injections"].append({
            "type": "worker_failure",
            "target": worker_id,
            "timestamp": time.time()
        })
        
        logger.info(f"💥 Worker {worker_id} failed")
        
        # Auto-recovery after 5 seconds
        await asyncio.sleep(5)
        for worker in self.demo_state["active_workers"]:
            if worker["id"] == worker_id:
                worker["status"] = "active"
                break
        
        logger.info(f"🔧 Worker {worker_id} recovered")
    
    def _print_demo_instructions(self):
        """Print demo instructions"""
        print("\n" + "="*80)
        print("🚀 SIMPLE DISTRIBUTED TRAINING DEMO")
        print("="*80)
        print(f"📊 Dashboard: http://localhost:{self.config.dashboard_port}")
        print(f"👥 Workers: {len(self.demo_state['active_workers'])} simulated workers")
        print("\n🎬 DEMO SCENARIOS (Auto-rotating):")
        print("• Baseline Training: Simulated training progress")
        print("• Dynamic Scaling: Adding/removing workers")
        print("• Worker Failure: Fault tolerance demonstration")
        print("• Gradient Strategies: Strategy switching")
        print("\n🔍 INTERACTIVE CONTROLS:")
        print("• Use the dashboard buttons to:")
        print("  - ➕ Add Worker: Add a new worker")
        print("  - ➖ Remove Worker: Remove a worker")
        print("  - 💥 Inject Failure: Simulate worker failure")
        print("  - 🔄 Switch Strategy: Change gradient strategy")
        print("\n⚡ FEATURES DEMONSTRATED:")
        print("• Real-time metrics and performance monitoring")
        print("• Interactive cluster visualization")
        print("• Fault tolerance and recovery")
        print("• Multiple gradient synchronization strategies")
        print("• WebSocket-based live updates")
        print("\n🛑 Press Ctrl+C to stop the demo")
        print("="*80)
    
    async def stop_demo(self):
        """Stop the demo"""
        logger.info("🛑 Stopping demo...")
        self.running = False
        
        # Stop background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Stop performance monitor
        if self.performance_monitor:
            await self.performance_monitor.stop_monitoring()
        
        logger.info("✅ Demo stopped")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal")
            asyncio.create_task(self.stop_demo())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main entry point"""
    config = DemoConfig(
        initial_workers=3,
        dashboard_port=8080,
        training_epochs=100,
        batch_size=32,
        learning_rate=0.001
    )
    
    demo = SimpleDemoOrchestrator(config)
    demo.setup_signal_handlers()
    
    try:
        await demo.start_demo()
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    finally:
        await demo.stop_demo()

if __name__ == "__main__":
    asyncio.run(main())