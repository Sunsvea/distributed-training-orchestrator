import pytest
import asyncio
import torch
from unittest.mock import Mock, AsyncMock, patch

from communication.cluster_pb2 import *
from communication.coordinator_server import CoordinatorService
from communication.worker_client import WorkerClient
from coordinator.raft_coordinator import RaftCoordinator
from coordinator.cluster_manager import ClusterManager

class TestCommunicationIntegration:
    
    @pytest.fixture
    def coordinator_service(self):
        coordinator = RaftCoordinator(
            node_id="coordinator-1",
            address="localhost",
            port=8001,
            cluster_nodes=["localhost:8001"]
        )
        
        cluster_manager = ClusterManager(
            coordinator_nodes=["localhost:8001"]
        )
        
        return CoordinatorService(coordinator, cluster_manager)
    
    def test_join_cluster_flow(self, coordinator_service):
        """Test the complete join cluster flow"""
        request = JoinRequest(
            node_id="worker-1",
            node_type="worker",
            address="localhost",
            port=9001,
            metadata={"gpu_count": "1"}
        )
        
        # Mock context (not needed for this test)
        context = Mock()
        
        # Test the join cluster logic
        response = asyncio.run(coordinator_service.JoinCluster(request, context))
        
        assert response.success == True
        assert response.leader_id is not None
        assert len(response.cluster_nodes) == 1
        assert response.cluster_nodes[0].node_id == "worker-1"
    
    def test_gradient_sync_flow(self, coordinator_service):
        """Test gradient synchronization flow"""
        # First, add a worker to the cluster
        join_request = JoinRequest(
            node_id="worker-1",
            node_type="worker",
            address="localhost",
            port=9001
        )
        context = Mock()
        asyncio.run(coordinator_service.JoinCluster(join_request, context))
        
        # Start a training session
        start_request = StartTrainingRequest(
            model_config='{"model_type": "simple_fc", "input_size": 784, "num_classes": 10}',
            dataset_config='{"dataset": "mnist"}',
            sync_strategy="allreduce",
            hyperparameters={"learning_rate": "0.001"}
        )
        
        start_response = asyncio.run(coordinator_service.StartTraining(start_request, context))
        assert start_response.success == True
        training_id = start_response.training_id
        
        # Create mock gradients
        from worker.gradient_sync import GradientSynchronizer
        sync = GradientSynchronizer("allreduce", "worker-1")
        gradients = {
            "layer1.weight": torch.randn(10, 784),
            "layer1.bias": torch.randn(10)
        }
        serialized_gradients = sync.serialize_gradients(gradients)
        
        # Create metrics
        metrics = TrainingMetrics(
            loss=0.5,
            accuracy=0.8,
            iteration=1,
            learning_rate=0.001,
            timestamp=1234567890
        )
        
        # Send gradient sync request
        sync_request = GradientSyncRequest(
            training_id=training_id,
            node_id="worker-1",
            iteration=1,
            gradients=serialized_gradients,
            metrics=metrics
        )
        
        sync_response = asyncio.run(coordinator_service.SyncGradients(sync_request, context))
        assert sync_response.success == True
        # With only one worker, should get empty gradients back until more workers join
        assert sync_response.should_continue == True
    
    def test_cluster_status_flow(self, coordinator_service):
        """Test cluster status reporting"""
        context = Mock()
        
        # Add a worker first
        join_request = JoinRequest(
            node_id="worker-1",
            node_type="worker",
            address="localhost",
            port=9001
        )
        asyncio.run(coordinator_service.JoinCluster(join_request, context))
        
        # Get cluster status
        status_request = StatusRequest(node_id="coordinator-1")
        status_response = asyncio.run(coordinator_service.GetClusterStatus(status_request, context))
        
        assert status_response.leader_id is not None
        assert len(status_response.nodes) == 1
        assert status_response.health.total_nodes == 1
        assert status_response.health.healthy_nodes == 1
        assert status_response.training_status == "idle"
    
    def test_heartbeat_flow(self, coordinator_service):
        """Test heartbeat handling"""
        context = Mock()
        
        # Add a worker first
        join_request = JoinRequest(
            node_id="worker-1",
            node_type="worker",
            address="localhost",
            port=9001
        )
        asyncio.run(coordinator_service.JoinCluster(join_request, context))
        
        # Send heartbeat
        heartbeat_request = HeartbeatRequest(
            node_id="worker-1",
            term=0,
            status=NodeStatus.ACTIVE
        )
        
        heartbeat_response = asyncio.run(coordinator_service.Heartbeat(heartbeat_request, context))
        assert heartbeat_response.success == True
        assert heartbeat_response.leader_id is not None
    
    def test_training_lifecycle(self, coordinator_service):
        """Test complete training lifecycle"""
        context = Mock()
        
        # Add multiple workers
        for i in range(3):
            join_request = JoinRequest(
                node_id=f"worker-{i}",
                node_type="worker",
                address="localhost",
                port=9001 + i
            )
            response = asyncio.run(coordinator_service.JoinCluster(join_request, context))
            assert response.success == True
        
        # Start training
        start_request = StartTrainingRequest(
            model_config='{"model_type": "simple_fc", "input_size": 784, "num_classes": 10}',
            dataset_config='{"dataset": "mnist"}',
            sync_strategy="allreduce",
            hyperparameters={"learning_rate": "0.001", "epochs": "5"}
        )
        
        start_response = asyncio.run(coordinator_service.StartTraining(start_request, context))
        assert start_response.success == True
        training_id = start_response.training_id
        
        # Verify workers are in training status
        status_request = StatusRequest(node_id="coordinator-1")
        status_response = asyncio.run(coordinator_service.GetClusterStatus(status_request, context))
        assert status_response.training_status == "active"
        assert status_response.training_id == training_id
        
        # Stop training
        stop_request = StopTrainingRequest(
            training_id=training_id,
            reason="test_completion"
        )
        
        stop_response = asyncio.run(coordinator_service.StopTraining(stop_request, context))
        assert stop_response.success == True
        
        # Verify training stopped
        status_response = asyncio.run(coordinator_service.GetClusterStatus(status_request, context))
        # Training status should be idle after stopping
        assert status_response.training_status == "idle"

class TestWorkerClientUnit:
    
    def test_worker_client_initialization(self):
        """Test worker client initialization"""
        client = WorkerClient("worker-1", "localhost:8001", 9001)
        
        assert client.node_id == "worker-1"
        assert client.coordinator_address == "localhost:8001"
        assert client.port == 9001
        assert client.connected == False
        assert client.current_training_id is None
    
    @pytest.mark.asyncio
    async def test_worker_client_connection_mock(self):
        """Test worker client connection with mocked gRPC"""
        client = WorkerClient("worker-1", "localhost:8001")
        
        # Mock the gRPC components
        with patch('communication.worker_client.grpc.aio.insecure_channel') as mock_channel, \
             patch('communication.worker_client.ClusterServiceStub') as mock_stub_class:
            
            mock_stub = AsyncMock()
            mock_stub_class.return_value = mock_stub
            
            # Mock successful join response
            mock_stub.JoinCluster.return_value = JoinResponse(
                success=True,
                leader_id="coordinator-1",
                cluster_nodes=[],
                message="Successfully joined"
            )
            
            # Test connection
            success = await client.connect_to_coordinator({"test": "metadata"})
            assert success == True
            assert client.connected == True
            
            # Verify join request was called
            mock_stub.JoinCluster.assert_called_once()
            call_args = mock_stub.JoinCluster.call_args[0][0]
            assert call_args.node_id == "worker-1"
            assert call_args.node_type == "worker"
            assert call_args.metadata["test"] == "metadata"
    
    def test_gradient_sync_serialization(self):
        """Test gradient serialization for sync"""
        from worker.gradient_sync import GradientSynchronizer
        
        sync = GradientSynchronizer("allreduce", "worker-1")
        
        # Create test gradients
        gradients = {
            "conv1.weight": torch.randn(32, 3, 3, 3),
            "conv1.bias": torch.randn(32),
            "fc.weight": torch.randn(10, 128),
            "fc.bias": torch.randn(10)
        }
        
        # Serialize
        tensor_data_list = sync.serialize_gradients(gradients)
        assert len(tensor_data_list) == 4
        
        # Verify each tensor data
        for tensor_data in tensor_data_list:
            assert isinstance(tensor_data, TensorData)
            assert tensor_data.name in gradients
            assert len(tensor_data.shape) > 0
            assert len(tensor_data.data) > 0
            assert tensor_data.dtype.startswith("torch.")
        
        # Deserialize and verify
        deserialized = sync.deserialize_gradients(tensor_data_list)
        assert len(deserialized) == len(gradients)
        
        for name, original_tensor in gradients.items():
            assert name in deserialized
            assert torch.allclose(original_tensor, deserialized[name])

class TestMessageSerialization:
    
    def test_training_metrics_serialization(self):
        """Test training metrics message creation"""
        metrics = TrainingMetrics(
            loss=0.75,
            accuracy=0.92,
            iteration=150,
            learning_rate=0.0005,
            timestamp=1234567890,
            custom_metrics={
                "gradient_norm": 1.5,
                "batch_time": 0.25,
                "memory_usage": 2048.0
            }
        )
        
        # Verify all fields are properly set
        assert abs(metrics.loss - 0.75) < 1e-6
        assert abs(metrics.accuracy - 0.92) < 1e-6
        assert metrics.iteration == 150
        assert abs(metrics.learning_rate - 0.0005) < 1e-6
        assert metrics.timestamp == 1234567890
        assert len(metrics.custom_metrics) == 3
        assert abs(metrics.custom_metrics["gradient_norm"] - 1.5) < 1e-6
    
    def test_node_info_serialization(self):
        """Test node info message creation"""
        node_info = NodeInfo(
            node_id="worker-5",
            node_type="worker",
            address="192.168.1.100",
            port=9005,
            status=NodeStatus.TRAINING,
            last_seen=1234567890,
            metadata={
                "gpu_count": "2",
                "memory_gb": "16",
                "cpu_cores": "8"
            }
        )
        
        assert node_info.node_id == "worker-5"
        assert node_info.node_type == "worker"
        assert node_info.address == "192.168.1.100"
        assert node_info.port == 9005
        assert node_info.status == NodeStatus.TRAINING
        assert node_info.last_seen == 1234567890
        assert len(node_info.metadata) == 3
    
    def test_cluster_health_serialization(self):
        """Test cluster health message creation"""
        health = ClusterHealth(
            total_nodes=5,
            healthy_nodes=4,
            failed_nodes=1,
            consensus_healthy=True,
            training_active=True
        )
        
        assert health.total_nodes == 5
        assert health.healthy_nodes == 4
        assert health.failed_nodes == 1
        assert health.consensus_healthy == True
        assert health.training_active == True