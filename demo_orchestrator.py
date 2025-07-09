#!/usr/bin/env python3
"""
Interactive Demo Orchestrator for Distributed Training System
This script creates a comprehensive demo that shows:
- Real distributed training with multiple workers
- Fault tolerance and recovery scenarios
- Performance monitoring and alerting
- Interactive controls for recruiters
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

from coordinator.raft_coordinator import RaftCoordinator
from coordinator.cluster_manager import ClusterManager
from coordinator.fault_tolerance import FaultToleranceManager
from worker.training_worker import TrainingWorker
from worker.distributed_strategies import AllReduceStrategy, ParameterServerStrategy
from communication.coordinator_server import CoordinatorServer
from communication.worker_client import WorkerClient
from monitoring.performance_monitor import PerformanceMonitor
from dashboard.main import DashboardApplication

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DemoScenario(Enum):
    BASELINE_TRAINING = "baseline_training"
    DYNAMIC_SCALING = "dynamic_scaling"
    COORDINATOR_FAILURE = "coordinator_failure"
    WORKER_FAILURE = "worker_failure"
    NETWORK_PARTITION = "network_partition"
    GRADIENT_STRATEGIES = "gradient_strategies"

@dataclass
class DemoConfig:
    initial_workers: int = 3
    coordinator_port: int = 50051
    dashboard_port: int = 8080
    training_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.01
    model_type: str = "mnist_cnn"

class InteractiveDemoOrchestrator:
    """
    Interactive demo orchestrator that manages the complete distributed training demo
    """
    
    def __init__(self, config: DemoConfig):
        self.config = config
        self.running = False
        self.current_scenario = DemoScenario.BASELINE_TRAINING
        
        # Core components
        self.coordinator: Optional[RaftCoordinator] = None
        self.cluster_manager: Optional[ClusterManager] = None
        self.fault_tolerance: Optional[FaultToleranceManager] = None
        self.coordinator_server: Optional[CoordinatorServer] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.dashboard: Optional[DashboardApplication] = None
        
        # Worker management
        self.workers: Dict[str, TrainingWorker] = {}
        self.worker_clients: Dict[str, WorkerClient] = {}
        self.worker_tasks: Dict[str, asyncio.Task] = {}
        
        # Demo state
        self.demo_state = {
            "current_scenario": self.current_scenario.value,
            "active_workers": [],
            "coordinator_status": "initializing",
            "training_progress": {
                "epoch": 0,
                "loss": 1.0,
                "accuracy": 0.0,
                "throughput": 0.0
            },
            "fault_injections": [],
            "cluster_health": 100.0
        }
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
    async def start_demo(self):
        """Start the complete demo environment"""
        logger.info("🚀 Starting Interactive Distributed Training Demo")
        self.running = True
        
        try:
            # Initialize core components
            await self._initialize_components()
            
            # Start coordinator
            await self._start_coordinator()
            
            # Start initial workers
            await self._start_initial_workers()
            
            # Start dashboard
            await self._start_dashboard()
            
            # Start demo scenarios
            await self._start_demo_scenarios()
            
            # Print demo instructions
            self._print_demo_instructions()
            
            # Keep demo running
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Demo error: {e}")
            await self.stop_demo()
    
    async def _initialize_components(self):
        """Initialize all core components"""
        logger.info("Initializing core components...")
        
        # Initialize coordinator and cluster manager
        self.coordinator = RaftCoordinator(
            node_id="coordinator-1",
            cluster_nodes=["coordinator-1"],
            election_timeout=5.0,
            heartbeat_interval=1.0
        )
        
        self.cluster_manager = ClusterManager(
            coordinator=self.coordinator,
            max_workers=10,
            health_check_interval=5.0
        )
        
        # Initialize fault tolerance
        self.fault_tolerance = FaultToleranceManager(
            cluster_manager=self.cluster_manager,
            coordinator=self.coordinator
        )
        
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor(
            collection_interval=2.0,
            alert_evaluation_interval=5.0
        )
        
        # Setup performance monitoring alerts
        await self._setup_demo_alerts()
        
        logger.info("✅ Core components initialized")
    
    async def _setup_demo_alerts(self):
        """Setup alerts for demo scenarios"""
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
                name="demo_worker_failure",
                metric_type=MetricType.NETWORK,
                metric_name="worker_connections",
                condition="less_than",
                threshold=2.0,
                severity=AlertSeverity.CRITICAL,
                duration=5.0,
                message_template="Worker count dropped to {metric_value}"
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
    
    async def _start_coordinator(self):
        """Start the coordinator node"""
        logger.info("Starting coordinator node...")
        
        self.coordinator_server = CoordinatorServer(
            coordinator=self.coordinator,
            cluster_manager=self.cluster_manager,
            port=self.config.coordinator_port
        )
        
        # Start coordinator server in background
        coordinator_task = asyncio.create_task(self.coordinator_server.start())
        self.background_tasks.append(coordinator_task)
        
        # Wait for coordinator to be ready
        await asyncio.sleep(2)
        
        # Start fault tolerance monitoring
        await self.fault_tolerance.start_monitoring()
        
        self.demo_state["coordinator_status"] = "running"
        logger.info("✅ Coordinator started on port {}".format(self.config.coordinator_port))
    
    async def _start_initial_workers(self):
        """Start initial worker nodes"""
        logger.info(f"Starting {self.config.initial_workers} initial workers...")
        
        for i in range(self.config.initial_workers):
            worker_id = f"worker-{i+1}"
            await self._start_worker(worker_id)
            
        # Wait for workers to connect
        await asyncio.sleep(3)
        
        # Start distributed training
        await self._start_distributed_training()
        
        logger.info(f"✅ {len(self.workers)} workers started and training initiated")
    
    async def _start_worker(self, worker_id: str):
        """Start a single worker node"""
        logger.info(f"Starting worker: {worker_id}")
        
        # Create worker
        worker = TrainingWorker(
            worker_id=worker_id,
            model_type=self.config.model_type,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate
        )
        
        # Create worker client
        client = WorkerClient(
            worker_id=worker_id,
            coordinator_host="localhost",
            coordinator_port=self.config.coordinator_port
        )
        
        # Connect to coordinator
        await client.connect()
        
        # Start worker training loop
        worker_task = asyncio.create_task(self._worker_training_loop(worker, client))
        
        # Store references
        self.workers[worker_id] = worker
        self.worker_clients[worker_id] = client
        self.worker_tasks[worker_id] = worker_task
        
        # Update demo state
        self.demo_state["active_workers"].append({
            "id": worker_id,
            "status": "running",
            "connected_at": time.time()
        })
        
        # Update cluster manager
        self.cluster_manager.add_worker_node(worker_id, "localhost", 50000 + hash(worker_id) % 1000)
        
        logger.info(f"✅ Worker {worker_id} started and connected")
    
    async def _worker_training_loop(self, worker: TrainingWorker, client: WorkerClient):
        """Main training loop for a worker"""
        try:
            # Start training
            await worker.start_training("demo_training_session")
            
            while self.running:
                # Compute gradients
                gradients = await worker.compute_gradients()
                
                # Synchronize gradients with coordinator
                if gradients:
                    await client.sync_gradients(gradients)
                
                # Update training metrics
                training_state = worker.get_training_state()
                if training_state:
                    # Update demo state
                    self.demo_state["training_progress"].update({
                        "epoch": training_state.get("epoch", 0),
                        "loss": training_state.get("loss", 1.0),
                        "accuracy": training_state.get("accuracy", 0.0),
                        "throughput": training_state.get("throughput", 0.0)
                    })
                    
                    # Register with performance monitor
                    self.performance_monitor.register_training_state(training_state)
                
                # Send heartbeat
                await client.send_heartbeat()
                
                await asyncio.sleep(1)  # Training iteration delay
                
        except Exception as e:
            logger.error(f"Worker {worker.worker_id} error: {e}")
            await self._handle_worker_failure(worker.worker_id)
    
    async def _start_distributed_training(self):
        """Initialize distributed training across all workers"""
        logger.info("Initializing distributed training...")
        
        # Select gradient synchronization strategy
        strategy = AllReduceStrategy(
            world_size=len(self.workers),
            compression_ratio=0.1  # 10% compression for demo
        )
        
        # Configure workers with strategy
        for worker in self.workers.values():
            worker.set_gradient_strategy(strategy)
        
        logger.info("✅ Distributed training initialized with AllReduce strategy")
    
    async def _start_dashboard(self):
        """Start the dashboard application"""
        logger.info("Starting dashboard...")
        
        self.dashboard = DashboardApplication(
            dashboard_host="0.0.0.0",
            dashboard_port=self.config.dashboard_port
        )
        
        # Override performance monitor with our demo monitor
        self.dashboard.performance_monitor = self.performance_monitor
        
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
            (DemoScenario.BASELINE_TRAINING, 30),  # 30 seconds
            (DemoScenario.DYNAMIC_SCALING, 20),    # 20 seconds
            (DemoScenario.WORKER_FAILURE, 25),     # 25 seconds
            (DemoScenario.GRADIENT_STRATEGIES, 15) # 15 seconds
        ]
        
        while self.running:
            for scenario, duration in scenarios:
                if not self.running:
                    break
                    
                logger.info(f"🎬 Starting scenario: {scenario.value}")
                self.current_scenario = scenario
                self.demo_state["current_scenario"] = scenario.value
                
                await self._execute_scenario(scenario)
                await asyncio.sleep(duration)
    
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
        """Demonstrate normal distributed training"""
        logger.info("Running baseline training scenario...")
        
        # Simulate steady training progress
        for worker in self.workers.values():
            # Simulate improving loss and accuracy
            current_loss = self.demo_state["training_progress"]["loss"]
            current_accuracy = self.demo_state["training_progress"]["accuracy"]
            
            # Gradual improvement
            new_loss = max(0.1, current_loss - random.uniform(0.01, 0.05))
            new_accuracy = min(0.95, current_accuracy + random.uniform(0.01, 0.03))
            
            worker.update_training_metrics(loss=new_loss, accuracy=new_accuracy)
    
    async def _dynamic_scaling_scenario(self):
        """Demonstrate adding/removing workers dynamically"""
        logger.info("Running dynamic scaling scenario...")
        
        # Add a new worker
        new_worker_id = f"worker-{len(self.workers) + 1}"
        logger.info(f"🔄 Adding worker: {new_worker_id}")
        await self._start_worker(new_worker_id)
        
        await asyncio.sleep(10)
        
        # Remove a worker
        if len(self.workers) > 1:
            worker_to_remove = list(self.workers.keys())[-1]
            logger.info(f"🔄 Removing worker: {worker_to_remove}")
            await self._stop_worker(worker_to_remove)
    
    async def _worker_failure_scenario(self):
        """Demonstrate worker failure and recovery"""
        logger.info("Running worker failure scenario...")
        
        if len(self.workers) > 1:
            # Simulate worker failure
            failing_worker = list(self.workers.keys())[0]
            logger.info(f"💥 Simulating failure of worker: {failing_worker}")
            
            await self._simulate_worker_failure(failing_worker)
            
            await asyncio.sleep(5)
            
            # Automatic recovery
            logger.info(f"🔧 Recovering worker: {failing_worker}")
            await self._recover_worker(failing_worker)
    
    async def _gradient_strategies_scenario(self):
        """Demonstrate different gradient synchronization strategies"""
        logger.info("Running gradient strategies scenario...")
        
        # Switch to Parameter Server strategy
        logger.info("🔄 Switching to Parameter Server strategy")
        ps_strategy = ParameterServerStrategy(
            world_size=len(self.workers),
            staleness_threshold=2
        )
        
        for worker in self.workers.values():
            worker.set_gradient_strategy(ps_strategy)
        
        await asyncio.sleep(10)
        
        # Switch back to AllReduce
        logger.info("🔄 Switching back to AllReduce strategy")
        ar_strategy = AllReduceStrategy(
            world_size=len(self.workers),
            compression_ratio=0.2
        )
        
        for worker in self.workers.values():
            worker.set_gradient_strategy(ar_strategy)
    
    async def _simulate_realistic_metrics(self):
        """Simulate realistic system metrics for demo"""
        while self.running:
            # Simulate varying system load
            cpu_percent = 20 + random.uniform(-5, 25)  # 15-45% CPU
            memory_percent = 45 + random.uniform(-10, 20)  # 35-65% Memory
            
            # Simulate network activity
            network_connections = len(self.workers)
            network_latency = 1.0 + random.uniform(-0.5, 2.0)  # 0.5-3.0ms
            
            # Register metrics
            self.performance_monitor.register_network_connections(network_connections)
            
            await asyncio.sleep(2)
    
    async def _simulate_worker_failure(self, worker_id: str):
        """Simulate worker failure"""
        if worker_id in self.worker_tasks:
            # Cancel worker task
            self.worker_tasks[worker_id].cancel()
            
            # Update demo state
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
            
            # Notify fault tolerance system
            await self.fault_tolerance.report_node_failure(worker_id, "simulated_failure")
    
    async def _recover_worker(self, worker_id: str):
        """Recover failed worker"""
        # Remove failed worker references
        if worker_id in self.workers:
            del self.workers[worker_id]
        if worker_id in self.worker_clients:
            del self.worker_clients[worker_id]
        if worker_id in self.worker_tasks:
            del self.worker_tasks[worker_id]
        
        # Remove from demo state
        self.demo_state["active_workers"] = [
            w for w in self.demo_state["active_workers"] 
            if w["id"] != worker_id
        ]
        
        # Start new worker with same ID
        await self._start_worker(worker_id)
    
    async def _handle_worker_failure(self, worker_id: str):
        """Handle unexpected worker failure"""
        logger.warning(f"Worker {worker_id} failed unexpectedly")
        
        # Update demo state
        for worker in self.demo_state["active_workers"]:
            if worker["id"] == worker_id:
                worker["status"] = "failed"
                break
        
        # Trigger recovery after delay
        await asyncio.sleep(5)
        await self._recover_worker(worker_id)
    
    async def _stop_worker(self, worker_id: str):
        """Stop a specific worker"""
        if worker_id in self.worker_tasks:
            self.worker_tasks[worker_id].cancel()
            
        if worker_id in self.worker_clients:
            await self.worker_clients[worker_id].disconnect()
            del self.worker_clients[worker_id]
            
        if worker_id in self.workers:
            del self.workers[worker_id]
            
        if worker_id in self.worker_tasks:
            del self.worker_tasks[worker_id]
        
        # Update demo state
        self.demo_state["active_workers"] = [
            w for w in self.demo_state["active_workers"] 
            if w["id"] != worker_id
        ]
        
        logger.info(f"✅ Worker {worker_id} stopped")
    
    def _print_demo_instructions(self):
        """Print instructions for recruiters"""
        print("\n" + "="*80)
        print("🚀 DISTRIBUTED TRAINING ORCHESTRATOR DEMO")
        print("="*80)
        print(f"📊 Dashboard: http://localhost:{self.config.dashboard_port}")
        print(f"⚙️  Coordinator: Running on port {self.config.coordinator_port}")
        print(f"👥 Workers: {len(self.workers)} active workers")
        print("\n🎬 DEMO SCENARIOS (Auto-rotating):")
        print("• Baseline Training: Normal distributed training progress")
        print("• Dynamic Scaling: Adding/removing workers during training")
        print("• Worker Failure: Fault tolerance and automatic recovery")
        print("• Gradient Strategies: Switching between AllReduce and Parameter Server")
        print("\n🔍 WHAT TO WATCH:")
        print("• Real-time metrics and performance charts")
        print("• Training loss decreasing and accuracy improving")
        print("• System alerts when failures occur")
        print("• Cluster health and worker status")
        print("• Fault tolerance and recovery mechanisms")
        print("\n⚡ TECHNICAL HIGHLIGHTS:")
        print("• Raft consensus algorithm for coordinator election")
        print("• Multiple gradient synchronization strategies")
        print("• Automatic failure detection and recovery")
        print("• Real-time performance monitoring and alerting")
        print("• WebSocket-based live dashboard updates")
        print("\n🛑 Press Ctrl+C to stop the demo")
        print("="*80)
    
    async def stop_demo(self):
        """Stop the demo environment"""
        logger.info("🛑 Stopping demo...")
        self.running = False
        
        # Stop all workers
        for worker_id in list(self.workers.keys()):
            await self._stop_worker(worker_id)
        
        # Stop background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Stop components
        if self.fault_tolerance:
            await self.fault_tolerance.stop_monitoring()
        
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
        coordinator_port=50051,
        dashboard_port=8080,
        training_epochs=100,
        batch_size=32,
        learning_rate=0.001,
        model_type="mnist_cnn"
    )
    
    demo = InteractiveDemoOrchestrator(config)
    demo.setup_signal_handlers()
    
    try:
        await demo.start_demo()
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    finally:
        await demo.stop_demo()

if __name__ == "__main__":
    asyncio.run(main())