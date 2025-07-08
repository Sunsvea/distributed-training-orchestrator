import asyncio
import grpc
import time
import logging
from typing import Optional, Dict, Any, List
import torch

from communication.cluster_pb2 import *
from communication.cluster_pb2_grpc import ClusterServiceStub
from worker.gradient_sync import GradientSynchronizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkerClient:
    def __init__(self, node_id: str, coordinator_address: str, port: int = None):
        self.node_id = node_id
        self.coordinator_address = coordinator_address
        self.port = port or 9001
        
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub: Optional[ClusterServiceStub] = None
        self.connected = False
        
        self.current_training_id: Optional[str] = None
        self.gradient_synchronizer: Optional[GradientSynchronizer] = None
        
        # Connection settings
        self.max_retries = 3
        self.retry_delay = 2.0
        self.heartbeat_interval = 5.0
        self.request_timeout = 30.0
        
        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._shutdown = False
    
    async def connect_to_coordinator(self, metadata: Optional[Dict[str, str]] = None) -> bool:
        """Connect to the coordinator and join the cluster"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Connecting to coordinator at {self.coordinator_address} (attempt {attempt + 1})")
                
                # Create gRPC channel
                self.channel = grpc.aio.insecure_channel(self.coordinator_address)
                self.stub = ClusterServiceStub(self.channel)
                
                # Test connection with join request
                join_request = JoinRequest(
                    node_id=self.node_id,
                    node_type="worker",
                    address="localhost",  # In real implementation, would get actual IP
                    port=self.port,
                    metadata=metadata or {}
                )
                
                response = await asyncio.wait_for(
                    self.stub.JoinCluster(join_request),
                    timeout=self.request_timeout
                )
                
                if response.success:
                    self.connected = True
                    logger.info(f"Successfully connected to coordinator. Leader: {response.leader_id}")
                    
                    # Start heartbeat task
                    self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                    
                    return True
                else:
                    logger.error(f"Failed to join cluster: {response.message}")
                    await self._cleanup_connection()
                    
            except asyncio.TimeoutError:
                logger.error(f"Connection timeout to {self.coordinator_address}")
            except grpc.aio.AioRpcError as e:
                logger.error(f"gRPC error connecting to coordinator: {e.code()}: {e.details()}")
            except Exception as e:
                logger.error(f"Unexpected error connecting to coordinator: {e}")
            
            if attempt < self.max_retries - 1:
                logger.info(f"Retrying connection in {self.retry_delay} seconds...")
                await asyncio.sleep(self.retry_delay)
                self.retry_delay *= 1.5  # Exponential backoff
        
        logger.error("Failed to connect to coordinator after all retries")
        return False
    
    async def disconnect_from_coordinator(self, reason: str = "shutdown") -> bool:
        """Disconnect from the coordinator and leave the cluster"""
        if not self.connected:
            return True
        
        try:
            self._shutdown = True
            
            # Cancel heartbeat task
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            # Send leave request
            if self.stub:
                leave_request = LeaveRequest(
                    node_id=self.node_id,
                    reason=reason
                )
                
                response = await asyncio.wait_for(
                    self.stub.LeaveCluster(leave_request),
                    timeout=self.request_timeout
                )
                
                if response.success:
                    logger.info("Successfully left cluster")
                else:
                    logger.warning(f"Leave cluster failed: {response.message}")
            
            await self._cleanup_connection()
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from coordinator: {e}")
            await self._cleanup_connection()
            return False
    
    async def send_heartbeat(self, status: NodeStatus) -> bool:
        """Send heartbeat to coordinator"""
        if not self.connected or not self.stub:
            return False
        
        try:
            heartbeat_request = HeartbeatRequest(
                node_id=self.node_id,
                term=0,  # Workers don't participate in consensus
                status=status
            )
            
            response = await asyncio.wait_for(
                self.stub.Heartbeat(heartbeat_request),
                timeout=self.request_timeout
            )
            
            if response.success:
                logger.debug("Heartbeat successful")
                return True
            else:
                logger.warning("Heartbeat failed")
                return False
                
        except asyncio.TimeoutError:
            logger.error("Heartbeat timeout")
            return False
        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                logger.error("Coordinator unavailable")
                self.connected = False
            else:
                logger.error(f"Heartbeat gRPC error: {e.code()}: {e.details()}")
            return False
        except Exception as e:
            logger.error(f"Unexpected heartbeat error: {e}")
            return False
    
    async def sync_gradients(self, training_id: str, iteration: int, 
                           gradients: Dict[str, torch.Tensor], 
                           loss: float, accuracy: float) -> Optional[GradientSyncResponse]:
        """Synchronize gradients with other workers"""
        if not self.connected or not self.stub:
            logger.error("Not connected to coordinator")
            return None
        
        try:
            # Initialize gradient synchronizer if needed
            if self.gradient_synchronizer is None:
                self.gradient_synchronizer = GradientSynchronizer("allreduce", self.node_id)
            
            # Serialize gradients
            serialized_gradients = self.gradient_synchronizer.serialize_gradients(gradients)
            
            # Create training metrics
            metrics = TrainingMetrics(
                loss=loss,
                accuracy=accuracy,
                iteration=iteration,
                learning_rate=0.001,  # Would get from actual optimizer
                timestamp=int(time.time()),
                custom_metrics={}
            )
            
            # Create sync request
            sync_request = GradientSyncRequest(
                training_id=training_id,
                node_id=self.node_id,
                iteration=iteration,
                gradients=serialized_gradients,
                metrics=metrics
            )
            
            # Send request with longer timeout for gradient sync
            response = await asyncio.wait_for(
                self.stub.SyncGradients(sync_request),
                timeout=self.request_timeout * 2  # Longer timeout for gradient sync
            )
            
            if response.success:
                logger.debug(f"Gradient sync successful for iteration {iteration}")
                return response
            else:
                logger.error(f"Gradient sync failed for iteration {iteration}")
                return response
                
        except asyncio.TimeoutError:
            logger.error(f"Gradient sync timeout for iteration {iteration}")
            return None
        except grpc.aio.AioRpcError as e:
            logger.error(f"Gradient sync gRPC error: {e.code()}: {e.details()}")
            return None
        except Exception as e:
            logger.error(f"Unexpected gradient sync error: {e}")
            return None
    
    async def get_cluster_status(self) -> Optional[StatusResponse]:
        """Get current cluster status"""
        if not self.connected or not self.stub:
            return None
        
        try:
            status_request = StatusRequest(node_id=self.node_id)
            
            response = await asyncio.wait_for(
                self.stub.GetClusterStatus(status_request),
                timeout=self.request_timeout
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting cluster status: {e}")
            return None
    
    async def get_training_assignment(self) -> Optional[Dict[str, Any]]:
        """Check if there's a training assignment from coordinator"""
        status = await self.get_cluster_status()
        if status and status.training_status == "active":
            return {
                "training_id": status.training_id,
                "status": status.training_status
            }
        return None
    
    async def report_training_completion(self, training_id: str, final_metrics: Dict[str, Any]) -> bool:
        """Report training completion to coordinator"""
        # This would typically be done through a separate RPC or by updating node status
        return await self.send_heartbeat(NodeStatus.ACTIVE)
    
    async def _heartbeat_loop(self):
        """Background task to send periodic heartbeats"""
        while not self._shutdown and self.connected:
            try:
                # Determine current status
                if self.current_training_id:
                    status = NodeStatus.TRAINING
                else:
                    status = NodeStatus.ACTIVE
                
                success = await self.send_heartbeat(status)
                
                if not success:
                    logger.warning("Heartbeat failed, checking connection...")
                    # Could implement reconnection logic here
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                logger.info("Heartbeat task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _cleanup_connection(self):
        """Clean up gRPC connection"""
        self.connected = False
        
        if self.channel:
            await self.channel.close()
            self.channel = None
        
        self.stub = None
    
    def set_training_id(self, training_id: Optional[str]):
        """Set current training session ID"""
        self.current_training_id = training_id
    
    def is_connected(self) -> bool:
        """Check if connected to coordinator"""
        return self.connected
    
    def get_node_info(self) -> Dict[str, Any]:
        """Get node information"""
        return {
            "node_id": self.node_id,
            "coordinator_address": self.coordinator_address,
            "port": self.port,
            "connected": self.connected,
            "current_training_id": self.current_training_id
        }

class WorkerClientPool:
    """Manages multiple worker clients for testing or multi-worker scenarios"""
    
    def __init__(self):
        self.clients: Dict[str, WorkerClient] = {}
    
    def add_worker(self, node_id: str, coordinator_address: str, port: int = None) -> WorkerClient:
        """Add a new worker client"""
        client = WorkerClient(node_id, coordinator_address, port)
        self.clients[node_id] = client
        return client
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect all workers to their coordinators"""
        results = {}
        tasks = []
        
        for node_id, client in self.clients.items():
            task = asyncio.create_task(client.connect_to_coordinator())
            tasks.append((node_id, task))
        
        for node_id, task in tasks:
            try:
                results[node_id] = await task
            except Exception as e:
                logger.error(f"Error connecting worker {node_id}: {e}")
                results[node_id] = False
        
        return results
    
    async def disconnect_all(self) -> Dict[str, bool]:
        """Disconnect all workers"""
        results = {}
        tasks = []
        
        for node_id, client in self.clients.items():
            task = asyncio.create_task(client.disconnect_from_coordinator())
            tasks.append((node_id, task))
        
        for node_id, task in tasks:
            try:
                results[node_id] = await task
            except Exception as e:
                logger.error(f"Error disconnecting worker {node_id}: {e}")
                results[node_id] = False
        
        return results
    
    def get_connected_workers(self) -> List[str]:
        """Get list of connected worker IDs"""
        return [node_id for node_id, client in self.clients.items() if client.is_connected()]
    
    def get_client(self, node_id: str) -> Optional[WorkerClient]:
        """Get specific worker client"""
        return self.clients.get(node_id)