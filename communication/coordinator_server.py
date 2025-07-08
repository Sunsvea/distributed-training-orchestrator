import asyncio
import grpc
import json
import time
import logging
from typing import Optional, Dict, List
from concurrent import futures

from communication.cluster_pb2 import *
from communication.cluster_pb2_grpc import ClusterServiceServicer, add_ClusterServiceServicer_to_server
from coordinator.raft_coordinator import RaftCoordinator
from coordinator.cluster_manager import ClusterManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoordinatorService(ClusterServiceServicer):
    def __init__(self, coordinator: RaftCoordinator, cluster_manager: ClusterManager):
        self.coordinator = coordinator
        self.cluster_manager = cluster_manager
        self.active_training_sessions: Dict[str, Dict] = {}
        self.training_counter = 0
    
    async def RequestVote(self, request: VoteRequest, context) -> VoteResponse:
        logger.info(f"Received vote request from {request.candidate_id} for term {request.term}")
        
        try:
            response = self.coordinator.handle_vote_request(request)
            logger.info(f"Vote response: granted={response.vote_granted}, term={response.term}")
            return response
        except Exception as e:
            logger.error(f"Error handling vote request: {e}")
            return VoteResponse(term=self.coordinator.current_term, vote_granted=False)
    
    async def AppendEntries(self, request: AppendEntriesRequest, context) -> AppendEntriesResponse:
        logger.debug(f"Received append entries from {request.leader_id}, term {request.term}")
        
        try:
            response = self.coordinator.handle_append_entries(request)
            return response
        except Exception as e:
            logger.error(f"Error handling append entries: {e}")
            return AppendEntriesResponse(term=self.coordinator.current_term, success=False)
    
    async def JoinCluster(self, request: JoinRequest, context) -> JoinResponse:
        logger.info(f"Node {request.node_id} requesting to join cluster")
        
        try:
            # Create node info
            node_info = NodeInfo(
                node_id=request.node_id,
                node_type=request.node_type,
                address=request.address,
                port=request.port,
                status=NodeStatus.JOINING,
                last_seen=int(time.time()),
                metadata=request.metadata
            )
            
            # Add to cluster if it's a worker node
            if request.node_type == "worker":
                success = self.cluster_manager.add_worker_node(node_info)
                if success:
                    # Update status to active
                    self.cluster_manager.update_node_status(request.node_id, NodeStatus.ACTIVE)
                    
                    logger.info(f"Worker {request.node_id} successfully joined cluster")
                    return JoinResponse(
                        success=True,
                        leader_id=self.coordinator.leader_id or self.coordinator.node_id,
                        cluster_nodes=self.cluster_manager.get_all_nodes(),
                        message="Successfully joined cluster"
                    )
                else:
                    return JoinResponse(
                        success=False,
                        message="Failed to add node to cluster"
                    )
            else:
                # For coordinator nodes, just acknowledge
                return JoinResponse(
                    success=True,
                    leader_id=self.coordinator.leader_id or self.coordinator.node_id,
                    cluster_nodes=[],
                    message="Coordinator node acknowledged"
                )
                
        except Exception as e:
            logger.error(f"Error joining node to cluster: {e}")
            return JoinResponse(
                success=False,
                message=f"Error: {str(e)}"
            )
    
    async def LeaveCluster(self, request: LeaveRequest, context) -> LeaveResponse:
        logger.info(f"Node {request.node_id} leaving cluster: {request.reason}")
        
        try:
            success = self.cluster_manager.remove_worker_node(request.node_id)
            if success:
                # If node was in training, handle cleanup
                for training_id, session in self.active_training_sessions.items():
                    if request.node_id in session.get("worker_nodes", []):
                        session["worker_nodes"].remove(request.node_id)
                        logger.info(f"Removed {request.node_id} from training session {training_id}")
                
                return LeaveResponse(
                    success=True,
                    message="Successfully left cluster"
                )
            else:
                return LeaveResponse(
                    success=False,
                    message="Node not found in cluster"
                )
                
        except Exception as e:
            logger.error(f"Error removing node from cluster: {e}")
            return LeaveResponse(
                success=False,
                message=f"Error: {str(e)}"
            )
    
    async def Heartbeat(self, request: HeartbeatRequest, context) -> HeartbeatResponse:
        logger.debug(f"Heartbeat from {request.node_id}")
        
        try:
            success = self.cluster_manager.handle_heartbeat(request.node_id, request.status)
            
            return HeartbeatResponse(
                success=success,
                term=self.coordinator.current_term,
                leader_id=self.coordinator.leader_id or self.coordinator.node_id
            )
            
        except Exception as e:
            logger.error(f"Error handling heartbeat: {e}")
            return HeartbeatResponse(
                success=False,
                term=self.coordinator.current_term,
                leader_id=self.coordinator.leader_id or self.coordinator.node_id
            )
    
    async def StartTraining(self, request: StartTrainingRequest, context) -> StartTrainingResponse:
        logger.info(f"Starting training with sync strategy: {request.sync_strategy}")
        
        try:
            # Generate unique training ID
            self.training_counter += 1
            training_id = f"training-{self.training_counter}-{int(time.time())}"
            
            # Get available worker nodes
            available_workers = self.cluster_manager.get_available_workers()
            
            if len(available_workers) == 0:
                return StartTrainingResponse(
                    success=False,
                    message="No available worker nodes"
                )
            
            # Parse configurations
            try:
                model_config = json.loads(request.model_config)
                dataset_config = json.loads(request.dataset_config)
            except json.JSONDecodeError as e:
                return StartTrainingResponse(
                    success=False,
                    message=f"Invalid JSON configuration: {str(e)}"
                )
            
            # Create training session
            training_session = {
                "training_id": training_id,
                "model_config": model_config,
                "dataset_config": dataset_config,
                "sync_strategy": request.sync_strategy,
                "hyperparameters": dict(request.hyperparameters),
                "worker_nodes": available_workers,
                "start_time": time.time(),
                "status": "active",
                "iterations": 0
            }
            
            self.active_training_sessions[training_id] = training_session
            
            # Mark workers as training
            for worker_id in available_workers:
                self.cluster_manager.update_node_status(worker_id, NodeStatus.TRAINING)
            
            # Log to Raft consensus
            log_entry_data = json.dumps({
                "action": "start_training",
                "training_id": training_id,
                "worker_nodes": available_workers
            }).encode()
            
            self.coordinator.append_log_entry("START_TRAINING", log_entry_data)
            
            logger.info(f"Training {training_id} started with {len(available_workers)} workers")
            
            return StartTrainingResponse(
                success=True,
                training_id=training_id,
                message=f"Training started with {len(available_workers)} workers"
            )
            
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            return StartTrainingResponse(
                success=False,
                message=f"Error: {str(e)}"
            )
    
    async def StopTraining(self, request: StopTrainingRequest, context) -> StopTrainingResponse:
        logger.info(f"Stopping training {request.training_id}")
        
        try:
            if request.training_id not in self.active_training_sessions:
                return StopTrainingResponse(
                    success=False,
                    message="Training session not found"
                )
            
            session = self.active_training_sessions[request.training_id]
            
            # Release worker nodes
            for worker_id in session["worker_nodes"]:
                self.cluster_manager.update_node_status(worker_id, NodeStatus.ACTIVE)
            
            # Update session status
            session["status"] = "stopped"
            session["end_time"] = time.time()
            
            # Log to Raft consensus
            log_entry_data = json.dumps({
                "action": "stop_training",
                "training_id": request.training_id,
                "reason": request.reason
            }).encode()
            
            self.coordinator.append_log_entry("STOP_TRAINING", log_entry_data)
            
            logger.info(f"Training {request.training_id} stopped")
            
            return StopTrainingResponse(
                success=True,
                message="Training stopped successfully"
            )
            
        except Exception as e:
            logger.error(f"Error stopping training: {e}")
            return StopTrainingResponse(
                success=False,
                message=f"Error: {str(e)}"
            )
    
    async def SyncGradients(self, request: GradientSyncRequest, context) -> GradientSyncResponse:
        logger.debug(f"Gradient sync from {request.node_id}, iteration {request.iteration}")
        
        try:
            if request.training_id not in self.active_training_sessions:
                return GradientSyncResponse(
                    success=False,
                    averaged_gradients=[],
                    should_continue=False
                )
            
            session = self.active_training_sessions[request.training_id]
            
            # Initialize gradient storage for this iteration if needed
            if "gradient_buffer" not in session:
                session["gradient_buffer"] = {}
            
            iteration_key = f"{request.iteration}"
            if iteration_key not in session["gradient_buffer"]:
                session["gradient_buffer"][iteration_key] = {
                    "gradients": {},
                    "metrics": {},
                    "received_from": set()
                }
            
            buffer = session["gradient_buffer"][iteration_key]
            
            # Store gradients and metrics from this worker
            buffer["gradients"][request.node_id] = request.gradients
            buffer["metrics"][request.node_id] = request.metrics
            buffer["received_from"].add(request.node_id)
            
            # Check if we have gradients from all workers
            expected_workers = set(session["worker_nodes"])
            if buffer["received_from"] == expected_workers:
                # All workers have sent gradients, compute average
                logger.info(f"All gradients received for iteration {request.iteration}, computing average")
                
                # Aggregate gradients (simplified averaging for now)
                averaged_gradients = self._average_gradients(buffer["gradients"])
                
                # Update session metrics
                session["iterations"] = max(session.get("iterations", 0), request.iteration)
                
                # Clean up old gradient buffers to save memory
                old_iterations = [k for k in session["gradient_buffer"].keys() 
                                if int(k) < request.iteration - 2]  # Keep last 2 iterations
                for old_iter in old_iterations:
                    del session["gradient_buffer"][old_iter]
                
                return GradientSyncResponse(
                    success=True,
                    averaged_gradients=averaged_gradients,
                    should_continue=True
                )
            else:
                # Still waiting for more workers
                return GradientSyncResponse(
                    success=True,
                    averaged_gradients=[],  # Empty until all received
                    should_continue=True
                )
                
        except Exception as e:
            logger.error(f"Error syncing gradients: {e}")
            return GradientSyncResponse(
                success=False,
                averaged_gradients=[],
                should_continue=False
            )
    
    def _average_gradients(self, gradient_dict: Dict[str, List[TensorData]]) -> List[TensorData]:
        """Average gradients from multiple workers"""
        if not gradient_dict:
            return []
        
        # Get parameter names from first worker
        worker_ids = list(gradient_dict.keys())
        first_worker_gradients = gradient_dict[worker_ids[0]]
        
        averaged_gradients = []
        
        for i, tensor_data in enumerate(first_worker_gradients):
            param_name = tensor_data.name
            
            # Collect all tensors for this parameter
            param_tensors = []
            for worker_id in worker_ids:
                worker_gradients = gradient_dict[worker_id]
                if i < len(worker_gradients) and worker_gradients[i].name == param_name:
                    param_tensors.append(worker_gradients[i])
            
            if param_tensors:
                # Simple averaging (in real implementation, would deserialize, average, and re-serialize)
                # For now, just return the first tensor as placeholder
                averaged_gradients.append(param_tensors[0])
        
        return averaged_gradients
    
    async def GetClusterStatus(self, request: StatusRequest, context) -> StatusResponse:
        logger.debug(f"Cluster status request from {request.node_id}")
        
        try:
            health = self.cluster_manager.get_cluster_health()
            all_nodes = self.cluster_manager.get_all_nodes()
            
            # Determine current training status
            training_status = "idle"
            training_id = None
            
            active_sessions = [s for s in self.active_training_sessions.values() 
                             if s["status"] == "active"]
            if active_sessions:
                training_status = "active"
                training_id = active_sessions[0]["training_id"]
            
            return StatusResponse(
                leader_id=self.coordinator.leader_id or self.coordinator.node_id,
                term=self.coordinator.current_term,
                nodes=all_nodes,
                training_status=training_status,
                training_id=training_id,
                health=health
            )
            
        except Exception as e:
            logger.error(f"Error getting cluster status: {e}")
            return StatusResponse(
                leader_id=self.coordinator.node_id,
                term=self.coordinator.current_term,
                nodes=[],
                training_status="error",
                health=ClusterHealth()
            )
    
    async def GetMetrics(self, request: MetricsRequest, context) -> MetricsResponse:
        logger.debug(f"Metrics request from {request.node_id}")
        
        try:
            # Collect training metrics from active sessions
            training_metrics = []
            node_metrics = []
            
            for session in self.active_training_sessions.values():
                if "gradient_buffer" in session:
                    # Extract metrics from gradient buffer
                    for iteration_key, buffer in session["gradient_buffer"].items():
                        for worker_id, metrics in buffer["metrics"].items():
                            training_metrics.append(metrics)
            
            # Create cluster metrics summary
            cluster_metrics = ClusterMetrics(
                total_throughput=0.0,  # Would calculate from real metrics
                average_loss=0.0,      # Would calculate from training metrics
                training_efficiency=1.0,
                total_iterations=sum(s.get("iterations", 0) for s in self.active_training_sessions.values())
            )
            
            return MetricsResponse(
                training_metrics=training_metrics,
                node_metrics=node_metrics,
                cluster_metrics=cluster_metrics
            )
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return MetricsResponse(
                training_metrics=[],
                node_metrics=[],
                cluster_metrics=ClusterMetrics()
            )

class CoordinatorServer:
    def __init__(self, coordinator: RaftCoordinator, cluster_manager: ClusterManager, port: int):
        self.coordinator = coordinator
        self.cluster_manager = cluster_manager
        self.port = port
        self.server: Optional[grpc.aio.Server] = None
        self.service = CoordinatorService(coordinator, cluster_manager)
    
    async def start(self):
        self.server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
        add_ClusterServiceServicer_to_server(self.service, self.server)
        
        listen_addr = f'[::]:{self.port}'
        self.server.add_insecure_port(listen_addr)
        
        logger.info(f"Starting coordinator server on {listen_addr}")
        await self.server.start()
        
        # Start background tasks
        asyncio.create_task(self._election_timeout_task())
        asyncio.create_task(self._heartbeat_task())
        asyncio.create_task(self._failure_detection_task())
    
    async def stop(self):
        if self.server:
            logger.info("Stopping coordinator server")
            await self.server.stop(grace=5)
    
    async def wait_for_termination(self):
        if self.server:
            await self.server.wait_for_termination()
    
    async def _election_timeout_task(self):
        """Handle Raft election timeouts"""
        while True:
            try:
                if (self.coordinator.state == "follower" and 
                    self.coordinator.is_election_timeout()):
                    logger.info("Election timeout, starting election")
                    self.coordinator.start_election()
                    # In real implementation, would send vote requests to other coordinators
                
                await asyncio.sleep(0.1)  # Check every 100ms
            except Exception as e:
                logger.error(f"Error in election timeout task: {e}")
                await asyncio.sleep(1)
    
    async def _heartbeat_task(self):
        """Send heartbeats if we're the leader"""
        while True:
            try:
                if self.coordinator.state == "leader":
                    # In real implementation, would send heartbeats to followers
                    logger.debug("Sending heartbeats to followers")
                
                await asyncio.sleep(self.coordinator.heartbeat_interval)
            except Exception as e:
                logger.error(f"Error in heartbeat task: {e}")
                await asyncio.sleep(1)
    
    async def _failure_detection_task(self):
        """Detect failed nodes"""
        while True:
            try:
                failed_nodes = self.cluster_manager.check_failed_nodes()
                for node_id in failed_nodes:
                    logger.warning(f"Detected failed node: {node_id}")
                    
                    # Handle training sessions with failed nodes
                    for training_id, session in self.service.active_training_sessions.items():
                        if node_id in session.get("worker_nodes", []):
                            logger.warning(f"Node {node_id} failed during training {training_id}")
                            # In real implementation, would handle recovery
                
                await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error in failure detection task: {e}")
                await asyncio.sleep(10)