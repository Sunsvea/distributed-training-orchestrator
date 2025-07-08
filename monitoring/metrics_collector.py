import asyncio
import time
import logging
import psutil
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import statistics
import threading

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_used_gb: float
    disk_total_gb: float
    gpu_metrics: List[Dict[str, Any]] = field(default_factory=list)
    network_io_bytes_sent: int = 0
    network_io_bytes_recv: int = 0

@dataclass
class TrainingMetrics:
    """Training performance metrics"""
    timestamp: float
    loss: float
    accuracy: float
    learning_rate: float
    batch_size: int
    epoch: int
    iteration: int
    throughput: float
    gradient_norm: float

@dataclass
class NetworkMetrics:
    """Network connectivity metrics"""
    timestamp: float
    worker_connections: Dict[str, Dict[str, float]]
    average_latency: float
    total_bandwidth: float
    packet_loss_rate: float

class MetricsCollector:
    """Collects and aggregates performance metrics"""
    
    def __init__(self, collection_interval: float = 5.0, max_history_size: int = 1000):
        self.collection_interval = collection_interval
        self.max_history_size = max_history_size
        
        # State
        self.metrics_history: List[Dict[str, Any]] = []
        self.system_metrics: Optional[SystemMetrics] = None
        self.training_metrics: Optional[TrainingMetrics] = None
        self.network_metrics: Optional[NetworkMetrics] = None
        
        # Background collection
        self.collection_task: Optional[asyncio.Task] = None
        self.current_training_state: Optional[Dict[str, Any]] = None
        self.current_network_connections: Optional[List[Dict[str, Any]]] = None
        self._shutdown = False
        
        # Thread safety
        self.metrics_lock = threading.Lock()
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_io_bytes_sent = network_io.bytes_sent
            network_io_bytes_recv = network_io.bytes_recv
            
            # GPU metrics (would need nvidia-ml-py for real GPU monitoring)
            gpu_metrics = []
            
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory_used_gb,
                memory_total_gb=memory_total_gb,
                disk_used_gb=disk_used_gb,
                disk_total_gb=disk_total_gb,
                gpu_metrics=gpu_metrics,
                network_io_bytes_sent=network_io_bytes_sent,
                network_io_bytes_recv=network_io_bytes_recv
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            # Return default metrics
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_gb=0.0,
                memory_total_gb=0.0,
                disk_used_gb=0.0,
                disk_total_gb=0.0,
                gpu_metrics=[],
                network_io_bytes_sent=0,
                network_io_bytes_recv=0
            )
    
    def collect_training_metrics(self, training_state: Dict[str, Any]) -> TrainingMetrics:
        """Collect training performance metrics"""
        return TrainingMetrics(
            timestamp=time.time(),
            loss=training_state.get("loss", 0.0),
            accuracy=training_state.get("accuracy", 0.0),
            learning_rate=training_state.get("learning_rate", 0.0),
            batch_size=training_state.get("batch_size", 0),
            epoch=training_state.get("epoch", 0),
            iteration=training_state.get("iteration", 0),
            throughput=training_state.get("throughput", 0.0),
            gradient_norm=training_state.get("gradient_norm", 0.0)
        )
    
    def collect_network_metrics(self, connections: List[Dict[str, Any]]) -> NetworkMetrics:
        """Collect network connectivity metrics"""
        worker_connections = {}
        latencies = []
        bandwidths = []
        packet_losses = []
        
        for conn in connections:
            worker_id = conn["worker_id"]
            latency = conn.get("latency", 0.0)
            bandwidth = conn.get("bandwidth", 0.0)
            packet_loss = conn.get("packet_loss", 0.0)
            
            worker_connections[worker_id] = {
                "latency": latency,
                "bandwidth": bandwidth,
                "packet_loss": packet_loss
            }
            
            latencies.append(latency)
            bandwidths.append(bandwidth)
            packet_losses.append(packet_loss)
        
        return NetworkMetrics(
            timestamp=time.time(),
            worker_connections=worker_connections,
            average_latency=statistics.mean(latencies) if latencies else 0.0,
            total_bandwidth=sum(bandwidths),
            packet_loss_rate=statistics.mean(packet_losses) if packet_losses else 0.0
        )
    
    def record_metrics(self, system_metrics: Optional[SystemMetrics],
                      training_metrics: Optional[TrainingMetrics],
                      network_metrics: Optional[NetworkMetrics]):
        """Record metrics to history"""
        with self.metrics_lock:
            metrics_record = {
                "timestamp": time.time(),
                "system": system_metrics,
                "training": training_metrics,
                "network": network_metrics
            }
            
            self.metrics_history.append(metrics_record)
            
            # Keep history within limits
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history = self.metrics_history[-self.max_history_size:]
            
            # Update current metrics
            if system_metrics:
                self.system_metrics = system_metrics
            if training_metrics:
                self.training_metrics = training_metrics
            if network_metrics:
                self.network_metrics = network_metrics
    
    def get_metrics_history(self, limit: Optional[int] = None,
                           start_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get metrics history with optional filtering"""
        with self.metrics_lock:
            history = self.metrics_history.copy()
        
        # Filter by time if specified
        if start_time:
            history = [m for m in history if m["timestamp"] >= start_time]
        
        # Limit results if specified
        if limit:
            history = history[-limit:]
        
        return history
    
    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics from history"""
        with self.metrics_lock:
            if not self.metrics_history:
                return {}
            
            # Aggregate system metrics
            system_metrics = {}
            system_values = {}
            
            for record in self.metrics_history:
                if record["system"]:
                    sys_metrics = record["system"]
                    for field in ["cpu_percent", "memory_percent", "memory_used_gb", "disk_used_gb"]:
                        if field not in system_values:
                            system_values[field] = []
                        system_values[field].append(getattr(sys_metrics, field))
            
            for field, values in system_values.items():
                if values:
                    system_metrics[field] = {
                        "avg": statistics.mean(values),
                        "min": min(values),
                        "max": max(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0.0
                    }
            
            # Aggregate training metrics
            training_metrics = {}
            training_values = {}
            
            for record in self.metrics_history:
                if record["training"]:
                    train_metrics = record["training"]
                    for field in ["loss", "accuracy", "throughput", "gradient_norm"]:
                        if field not in training_values:
                            training_values[field] = []
                        training_values[field].append(getattr(train_metrics, field))
            
            for field, values in training_values.items():
                if values:
                    training_metrics[field] = {
                        "avg": statistics.mean(values),
                        "min": min(values),
                        "max": max(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0.0
                    }
            
            return {
                "system": system_metrics,
                "training": training_metrics
            }
    
    async def start_collection(self, training_state: Optional[Dict[str, Any]] = None,
                              network_connections: Optional[List[Dict[str, Any]]] = None):
        """Start automatic metrics collection"""
        self.current_training_state = training_state
        self.current_network_connections = network_connections
        self._shutdown = False
        
        if self.collection_task is None:
            self.collection_task = asyncio.create_task(self._collection_loop())
        
        logger.info("Metrics collection started")
    
    async def stop_collection(self):
        """Stop automatic metrics collection"""
        self._shutdown = True
        
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
            self.collection_task = None
        
        logger.info("Metrics collection stopped")
    
    async def _collection_loop(self):
        """Main collection loop"""
        while not self._shutdown:
            try:
                # Collect system metrics
                system_metrics = self.collect_system_metrics()
                
                # Collect training metrics if available
                training_metrics = None
                if self.current_training_state:
                    training_metrics = self.collect_training_metrics(self.current_training_state)
                
                # Collect network metrics if available
                network_metrics = None
                if self.current_network_connections:
                    network_metrics = self.collect_network_metrics(self.current_network_connections)
                
                # Record metrics
                self.record_metrics(system_metrics, training_metrics, network_metrics)
                
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                logger.info("Metrics collection cancelled")
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(1.0)
    
    def update_training_state(self, training_state: Dict[str, Any]):
        """Update current training state"""
        self.current_training_state = training_state
    
    def update_network_connections(self, connections: List[Dict[str, Any]]):
        """Update current network connections"""
        self.current_network_connections = connections
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        return {
            "system": self.system_metrics,
            "training": self.training_metrics,
            "network": self.network_metrics
        }
    
    def clear_history(self):
        """Clear metrics history"""
        with self.metrics_lock:
            self.metrics_history.clear()
        
        logger.info("Metrics history cleared")