import asyncio
import time
import logging
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class FailureType(Enum):
    """Types of failures that can occur in the system"""
    WORKER_TIMEOUT = "worker_timeout"
    WORKER_CRASH = "worker_crash"
    COORDINATOR_FAILURE = "coordinator_failure"
    NETWORK_PARTITION = "network_partition"
    STORAGE_FAILURE = "storage_failure"
    GRADIENT_CORRUPTION = "gradient_corruption"

@dataclass
class FailureEvent:
    """Represents a failure event in the system"""
    timestamp: float
    failure_type: FailureType
    node_id: str
    details: str
    severity: str = "medium"  # low, medium, high, critical
    auto_recoverable: bool = True
    recovery_actions: List[str] = field(default_factory=list)

@dataclass
class RecoveryStrategy:
    """Defines how to recover from a specific type of failure"""
    failure_type: FailureType
    max_retries: int = 3
    retry_delay: float = 5.0
    escalation_threshold: int = 5
    recovery_timeout: float = 30.0
    actions: List[str] = field(default_factory=list)
    
class FaultToleranceManager:
    """Manages fault tolerance and recovery for the distributed training system"""
    
    def __init__(self, coordinator, cluster_manager, **kwargs):
        self.coordinator = coordinator
        self.cluster_manager = cluster_manager
        
        # Configuration
        self.heartbeat_timeout = kwargs.get("heartbeat_timeout", 30.0)
        self.max_worker_failures = kwargs.get("max_worker_failures", 2)
        self.recovery_timeout = kwargs.get("recovery_timeout", 60.0)
        self.checkpoint_interval = kwargs.get("checkpoint_interval", 300.0)  # 5 minutes
        
        # State tracking
        self.failed_nodes: Set[str] = set()
        self.recovering_nodes: Set[str] = set()
        self.failure_history: List[FailureEvent] = []
        self.last_heartbeats: Dict[str, float] = {}
        self.recovery_attempts: Dict[str, int] = {}
        
        # Recovery strategies
        self.recovery_strategies = self._initialize_recovery_strategies()
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.recovery_executor = ThreadPoolExecutor(max_workers=4)
        self._shutdown = False
        
        # Metrics
        self.metrics = {
            "total_failures": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "average_recovery_time": 0.0
        }
    
    def _initialize_recovery_strategies(self) -> Dict[FailureType, RecoveryStrategy]:
        """Initialize recovery strategies for different failure types"""
        return {
            FailureType.WORKER_TIMEOUT: RecoveryStrategy(
                failure_type=FailureType.WORKER_TIMEOUT,
                max_retries=3,
                retry_delay=5.0,
                actions=["restart_worker", "redistribute_work", "update_cluster_state"]
            ),
            FailureType.WORKER_CRASH: RecoveryStrategy(
                failure_type=FailureType.WORKER_CRASH,
                max_retries=2,
                retry_delay=10.0,
                actions=["restart_worker", "restore_checkpoint", "redistribute_work"]
            ),
            FailureType.COORDINATOR_FAILURE: RecoveryStrategy(
                failure_type=FailureType.COORDINATOR_FAILURE,
                max_retries=1,
                retry_delay=0.0,
                actions=["trigger_leader_election", "restore_state", "notify_workers"]
            ),
            FailureType.NETWORK_PARTITION: RecoveryStrategy(
                failure_type=FailureType.NETWORK_PARTITION,
                max_retries=5,
                retry_delay=15.0,
                actions=["wait_for_connectivity", "merge_partitions", "resolve_conflicts"]
            ),
            FailureType.GRADIENT_CORRUPTION: RecoveryStrategy(
                failure_type=FailureType.GRADIENT_CORRUPTION,
                max_retries=3,
                retry_delay=1.0,
                actions=["request_gradient_resend", "validate_gradients", "fallback_averaging"]
            )
        }
    
    async def start_monitoring(self):
        """Start fault tolerance monitoring"""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Fault tolerance monitoring started")
    
    async def stop_monitoring(self):
        """Stop fault tolerance monitoring"""
        self._shutdown = True
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
        
        self.recovery_executor.shutdown(wait=True)
        logger.info("Fault tolerance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop for fault detection"""
        while not self._shutdown:
            try:
                await self._check_worker_health()
                await self._check_coordinator_health()
                await self._check_network_health()
                await self._cleanup_old_failures()
                
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except asyncio.CancelledError:
                logger.info("Monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _check_worker_health(self):
        """Check health of all workers"""
        current_time = time.time()
        
        for node_id in self.cluster_manager.get_worker_nodes():
            if node_id in self.failed_nodes:
                continue
                
            last_heartbeat = self.last_heartbeats.get(node_id, 0)
            
            if current_time - last_heartbeat > self.heartbeat_timeout:
                logger.warning(f"Worker {node_id} missed heartbeat")
                await self._handle_worker_failure(node_id, FailureType.WORKER_TIMEOUT)
    
    async def _check_coordinator_health(self):
        """Check health of coordinator nodes"""
        # For now, we assume this coordinator is healthy
        # In a multi-coordinator setup, this would check other coordinators
        pass
    
    async def _check_network_health(self):
        """Check for network partitions and connectivity issues"""
        # Simple network health check
        active_workers = len(self.cluster_manager.get_active_workers())
        total_workers = len(self.cluster_manager.get_worker_nodes())
        
        if active_workers < total_workers * 0.5:  # Less than 50% workers active
            logger.warning("Potential network partition detected")
            await self._handle_network_partition()
    
    async def _cleanup_old_failures(self):
        """Clean up old failure events"""
        current_time = time.time()
        cutoff_time = current_time - 3600  # Keep failures for 1 hour
        
        self.failure_history = [
            event for event in self.failure_history
            if event.timestamp > cutoff_time
        ]
    
    async def _handle_worker_failure(self, node_id: str, failure_type: FailureType):
        """Handle worker failure"""
        if node_id in self.recovering_nodes:
            return  # Already recovering
        
        # Record failure event
        failure_event = FailureEvent(
            timestamp=time.time(),
            failure_type=failure_type,
            node_id=node_id,
            details=f"Worker {node_id} failed with {failure_type.value}",
            severity="high" if failure_type == FailureType.WORKER_CRASH else "medium"
        )
        
        self.failure_history.append(failure_event)
        self.failed_nodes.add(node_id)
        self.metrics["total_failures"] += 1
        
        logger.error(f"Worker failure detected: {node_id} ({failure_type.value})")
        
        # Start recovery process
        await self._start_recovery(node_id, failure_type)
    
    async def _handle_network_partition(self):
        """Handle network partition"""
        failure_event = FailureEvent(
            timestamp=time.time(),
            failure_type=FailureType.NETWORK_PARTITION,
            node_id="cluster",
            details="Network partition detected",
            severity="critical"
        )
        
        self.failure_history.append(failure_event)
        logger.critical("Network partition detected")
        
        # Implement partition recovery logic
        await self._recover_from_partition()
    
    async def _start_recovery(self, node_id: str, failure_type: FailureType):
        """Start recovery process for a failed node"""
        if node_id in self.recovering_nodes:
            return
        
        self.recovering_nodes.add(node_id)
        recovery_start_time = time.time()
        
        try:
            strategy = self.recovery_strategies.get(failure_type)
            if not strategy:
                logger.error(f"No recovery strategy for {failure_type}")
                return
            
            success = await self._execute_recovery_strategy(node_id, strategy)
            
            recovery_time = time.time() - recovery_start_time
            
            if success:
                self.failed_nodes.discard(node_id)
                self.recovery_attempts.pop(node_id, None)
                self.metrics["successful_recoveries"] += 1
                self.metrics["average_recovery_time"] = (
                    (self.metrics["average_recovery_time"] * (self.metrics["successful_recoveries"] - 1) + recovery_time) /
                    self.metrics["successful_recoveries"]
                )
                logger.info(f"Successfully recovered worker {node_id} in {recovery_time:.2f}s")
            else:
                self.metrics["failed_recoveries"] += 1
                logger.error(f"Failed to recover worker {node_id} after {recovery_time:.2f}s")
                
        except Exception as e:
            logger.error(f"Error during recovery of {node_id}: {e}")
            self.metrics["failed_recoveries"] += 1
        finally:
            self.recovering_nodes.discard(node_id)
    
    async def _execute_recovery_strategy(self, node_id: str, strategy: RecoveryStrategy) -> bool:
        """Execute recovery strategy for a failed node"""
        attempts = self.recovery_attempts.get(node_id, 0)
        
        if attempts >= strategy.max_retries:
            logger.error(f"Max recovery attempts ({strategy.max_retries}) reached for {node_id}")
            return False
        
        self.recovery_attempts[node_id] = attempts + 1
        
        logger.info(f"Attempting recovery of {node_id} (attempt {attempts + 1}/{strategy.max_retries})")
        
        # Execute recovery actions
        for action in strategy.actions:
            success = await self._execute_recovery_action(node_id, action)
            if not success:
                logger.warning(f"Recovery action '{action}' failed for {node_id}")
                
                # Wait before retry
                if attempts < strategy.max_retries - 1:
                    await asyncio.sleep(strategy.retry_delay)
                
                return False
        
        # Verify recovery
        return await self._verify_recovery(node_id)
    
    async def _execute_recovery_action(self, node_id: str, action: str) -> bool:
        """Execute a specific recovery action"""
        try:
            if action == "restart_worker":
                return await self._restart_worker(node_id)
            elif action == "redistribute_work":
                return await self._redistribute_work(node_id)
            elif action == "update_cluster_state":
                return await self._update_cluster_state(node_id)
            elif action == "restore_checkpoint":
                return await self._restore_checkpoint(node_id)
            elif action == "trigger_leader_election":
                return await self._trigger_leader_election()
            elif action == "wait_for_connectivity":
                return await self._wait_for_connectivity(node_id)
            else:
                logger.warning(f"Unknown recovery action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing recovery action '{action}' for {node_id}: {e}")
            return False
    
    async def _restart_worker(self, node_id: str) -> bool:
        """Restart a failed worker"""
        logger.info(f"Restarting worker {node_id}")
        
        # In a real implementation, this would:
        # 1. Terminate the failed worker process
        # 2. Start a new worker process
        # 3. Re-establish connections
        
        # For simulation, we'll just mark it as restarted
        await asyncio.sleep(2.0)  # Simulate restart time
        
        # Notify cluster manager
        success = await self.cluster_manager.restart_worker(node_id)
        
        if success:
            logger.info(f"Worker {node_id} restarted successfully")
        else:
            logger.error(f"Failed to restart worker {node_id}")
        
        return success
    
    async def _redistribute_work(self, node_id: str) -> bool:
        """Redistribute work from failed worker to other workers"""
        logger.info(f"Redistributing work from failed worker {node_id}")
        
        # Get active workers
        active_workers = self.cluster_manager.get_active_workers()
        
        if len(active_workers) < 1:
            logger.error("No active workers available for redistribution")
            return False
        
        # Redistribute training batches
        # In a real implementation, this would redistribute:
        # 1. Current training batches
        # 2. Model parameters
        # 3. Optimizer state
        
        # For simulation, we'll just update the cluster state
        await asyncio.sleep(1.0)
        
        logger.info(f"Work redistributed from {node_id} to {len(active_workers)} active workers")
        return True
    
    async def _update_cluster_state(self, node_id: str) -> bool:
        """Update cluster state after worker failure"""
        logger.info(f"Updating cluster state for failed worker {node_id}")
        
        # Remove worker from active list
        self.cluster_manager.mark_worker_failed(node_id)
        
        # Update gradient synchronization to exclude failed worker
        if hasattr(self.coordinator, 'gradient_coordinator'):
            self.coordinator.gradient_coordinator.remove_worker(node_id)
        
        return True
    
    async def _restore_checkpoint(self, node_id: str) -> bool:
        """Restore from checkpoint for recovered worker"""
        logger.info(f"Restoring checkpoint for worker {node_id}")
        
        # In a real implementation, this would:
        # 1. Find the latest checkpoint
        # 2. Restore model state
        # 3. Restore optimizer state
        # 4. Restore training progress
        
        await asyncio.sleep(1.5)  # Simulate restore time
        
        logger.info(f"Checkpoint restored for worker {node_id}")
        return True
    
    async def _trigger_leader_election(self) -> bool:
        """Trigger leader election for coordinator failure"""
        logger.info("Triggering leader election")
        
        # In a real implementation, this would trigger the Raft election
        if hasattr(self.coordinator, 'start_election'):
            await self.coordinator.start_election()
        
        return True
    
    async def _wait_for_connectivity(self, node_id: str) -> bool:
        """Wait for network connectivity to be restored"""
        logger.info(f"Waiting for connectivity to {node_id}")
        
        # Simple connectivity check
        max_wait = 30.0
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            if await self._check_node_connectivity(node_id):
                logger.info(f"Connectivity restored to {node_id}")
                return True
            
            await asyncio.sleep(2.0)
        
        logger.warning(f"Connectivity not restored to {node_id} within {max_wait}s")
        return False
    
    async def _check_node_connectivity(self, node_id: str) -> bool:
        """Check if node is reachable"""
        # Simple ping-like check
        # In real implementation, would use actual network connectivity test
        return node_id not in self.failed_nodes
    
    async def _recover_from_partition(self):
        """Recover from network partition"""
        logger.info("Attempting to recover from network partition")
        
        # In a real implementation, this would:
        # 1. Detect which nodes are in which partition
        # 2. Attempt to re-establish connections
        # 3. Merge cluster state
        # 4. Resolve any conflicts
        
        await asyncio.sleep(5.0)  # Simulate recovery time
        
        logger.info("Network partition recovery completed")
    
    async def _verify_recovery(self, node_id: str) -> bool:
        """Verify that a node has been successfully recovered"""
        # Check if node is responding to heartbeats
        if node_id in self.cluster_manager.get_active_workers():
            # Send a test message
            try:
                # In real implementation, would send actual test message
                await asyncio.sleep(0.5)  # Simulate test
                logger.info(f"Recovery verified for worker {node_id}")
                return True
            except Exception as e:
                logger.error(f"Recovery verification failed for {node_id}: {e}")
                return False
        
        return False
    
    def update_heartbeat(self, node_id: str):
        """Update heartbeat timestamp for a node"""
        self.last_heartbeats[node_id] = time.time()
    
    def report_failure(self, node_id: str, failure_type: FailureType, details: str = ""):
        """Report a failure event"""
        failure_event = FailureEvent(
            timestamp=time.time(),
            failure_type=failure_type,
            node_id=node_id,
            details=details,
            severity="high"
        )
        
        self.failure_history.append(failure_event)
        self.metrics["total_failures"] += 1
        
        # Trigger recovery asynchronously only if event loop is running
        try:
            asyncio.create_task(self._start_recovery(node_id, failure_type))
        except RuntimeError:
            # No event loop running, recovery will be handled elsewhere
            pass
    
    def get_failure_history(self, limit: int = 100) -> List[FailureEvent]:
        """Get recent failure history"""
        return self.failure_history[-limit:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get fault tolerance metrics"""
        return {
            **self.metrics,
            "failed_nodes": len(self.failed_nodes),
            "recovering_nodes": len(self.recovering_nodes),
            "recent_failures": len([
                f for f in self.failure_history
                if f.timestamp > time.time() - 3600  # Last hour
            ])
        }
    
    def get_cluster_health(self) -> Dict[str, Any]:
        """Get overall cluster health status"""
        total_workers = len(self.cluster_manager.get_worker_nodes())
        active_workers = len(self.cluster_manager.get_active_workers())
        failed_workers = len(self.failed_nodes)
        
        health_score = (active_workers / total_workers) * 100 if total_workers > 0 else 0
        
        return {
            "health_score": health_score,
            "total_workers": total_workers,
            "active_workers": active_workers,
            "failed_workers": failed_workers,
            "recovering_workers": len(self.recovering_nodes),
            "status": "healthy" if health_score >= 80 else "degraded" if health_score >= 60 else "critical"
        }
    
    def set_recovery_strategy(self, failure_type: FailureType, strategy: RecoveryStrategy):
        """Set custom recovery strategy for a failure type"""
        self.recovery_strategies[failure_type] = strategy
        logger.info(f"Updated recovery strategy for {failure_type.value}")
    
    def is_node_failed(self, node_id: str) -> bool:
        """Check if a node is currently failed"""
        return node_id in self.failed_nodes
    
    def is_node_recovering(self, node_id: str) -> bool:
        """Check if a node is currently recovering"""
        return node_id in self.recovering_nodes