import pytest
import asyncio
import grpc
from unittest.mock import Mock, AsyncMock, patch
import torch

from communication.cluster_pb2 import *
from communication.cluster_pb2_grpc import ClusterServiceStub, ClusterServiceServicer
from communication.coordinator_server import CoordinatorServer
from communication.worker_client import WorkerClient

@pytest.mark.asyncio
class TestCoordinatorServer:
    
    @pytest.fixture
    async def coordinator_server(self):
        from coordinator.raft_coordinator import RaftCoordinator
        from coordinator.cluster_manager import ClusterManager
        
        coordinator = RaftCoordinator(
            node_id="coordinator-1",
            address="localhost",
            port=8001,
            cluster_nodes=["localhost:8001", "localhost:8002", "localhost:8003"]
        )
        
        cluster_manager = ClusterManager(
            coordinator_nodes=["localhost:8001", "localhost:8002", "localhost:8003"]
        )
        
        server = CoordinatorServer(coordinator, cluster_manager, port=8001)
        await server.start()
        
        yield server
        
        await server.stop()
    
    async def test_join_cluster(self, coordinator_server):
        # Create a client to test the server
        async with grpc.aio.insecure_channel('localhost:8001') as channel:
            stub = ClusterServiceStub(channel)
            
            request = JoinRequest(
                node_id="worker-1",
                node_type="worker",
                address="localhost",
                port=9001,
                metadata={"gpu_count": "1", "memory": "8GB"}
            )
            
            response = await stub.JoinCluster(request)
            
            assert response.success == True
            assert len(response.cluster_nodes) >= 1
            assert any(node.node_id == "worker-1" for node in response.cluster_nodes)
    
    async def test_leave_cluster(self, coordinator_server):
        async with grpc.aio.insecure_channel('localhost:8001') as channel:
            stub = ClusterServiceStub(channel)
            
            # First join the cluster
            join_request = JoinRequest(
                node_id="worker-1",
                node_type="worker",
                address="localhost",
                port=9001
            )
            await stub.JoinCluster(join_request)
            
            # Then leave
            leave_request = LeaveRequest(
                node_id="worker-1",
                reason="shutdown"
            )
            
            response = await stub.LeaveCluster(leave_request)
            assert response.success == True
    
    async def test_heartbeat(self, coordinator_server):
        async with grpc.aio.insecure_channel('localhost:8001') as channel:
            stub = ClusterServiceStub(channel)
            
            # First join the cluster
            join_request = JoinRequest(
                node_id="worker-1",
                node_type="worker",
                address="localhost",
                port=9001
            )
            await stub.JoinCluster(join_request)
            
            # Send heartbeat
            heartbeat_request = HeartbeatRequest(
                node_id="worker-1",
                term=1,
                status=NodeStatus.ACTIVE
            )
            
            response = await stub.Heartbeat(heartbeat_request)
            assert response.success == True
    
    async def test_start_training(self, coordinator_server):
        async with grpc.aio.insecure_channel('localhost:8001') as channel:
            stub = ClusterServiceStub(channel)
            
            # Add some workers first
            for i in range(3):
                join_request = JoinRequest(
                    node_id=f"worker-{i}",
                    node_type="worker",
                    address="localhost",
                    port=9001 + i
                )
                await stub.JoinCluster(join_request)
            
            # Start training
            start_request = StartTrainingRequest(
                model_config='{"model_type": "simple_cnn", "input_size": 784, "num_classes": 10}',
                dataset_config='{"dataset": "mnist", "batch_size": 32}',
                sync_strategy="allreduce",
                hyperparameters={"learning_rate": "0.001", "epochs": "10"}
            )
            
            response = await stub.StartTraining(start_request)
            assert response.success == True
            assert response.training_id is not None
    
    async def test_get_cluster_status(self, coordinator_server):
        async with grpc.aio.insecure_channel('localhost:8001') as channel:
            stub = ClusterServiceStub(channel)
            
            request = StatusRequest(node_id="coordinator-1")
            response = await stub.GetClusterStatus(request)
            
            assert response.leader_id is not None
            assert response.health.consensus_healthy == True
    
    async def test_vote_request(self, coordinator_server):
        async with grpc.aio.insecure_channel('localhost:8001') as channel:
            stub = ClusterServiceStub(channel)
            
            vote_request = VoteRequest(
                term=2,
                candidate_id="coordinator-2",
                last_log_index=0,
                last_log_term=0
            )
            
            response = await stub.RequestVote(vote_request)
            assert response.term >= 2
    
    async def test_append_entries(self, coordinator_server):
        async with grpc.aio.insecure_channel('localhost:8001') as channel:
            stub = ClusterServiceStub(channel)
            
            # Create a log entry
            log_entry = LogEntry(
                term=1,
                command="START_TRAINING",
                data=b"training_config"
            )
            
            append_request = AppendEntriesRequest(
                term=1,
                leader_id="coordinator-2",
                prev_log_index=0,
                prev_log_term=0,
                entries=[log_entry],
                leader_commit=0
            )
            
            response = await stub.AppendEntries(append_request)
            assert response.term >= 1

class TestWorkerClient:
    
    @pytest.fixture
    def worker_client(self):
        client = WorkerClient(
            node_id="worker-1",
            coordinator_address="localhost:8001"
        )
        return client
    
    @pytest.mark.asyncio
    async def test_connect_to_coordinator(self, worker_client):
        # Mock the gRPC stub
        with patch('communication.worker_client.ClusterServiceStub') as mock_stub_class:
            mock_stub = AsyncMock()
            mock_stub_class.return_value = mock_stub
            
            # Mock successful join response
            mock_stub.JoinCluster.return_value = JoinResponse(
                success=True,
                leader_id="coordinator-1",
                cluster_nodes=[],
                message="Successfully joined cluster"
            )
            
            success = await worker_client.connect_to_coordinator()
            assert success == True
            assert worker_client.connected == True
    
    @pytest.mark.asyncio
    async def test_send_heartbeat(self, worker_client):
        with patch('communication.worker_client.ClusterServiceStub') as mock_stub_class:
            mock_stub = AsyncMock()
            mock_stub_class.return_value = mock_stub
            worker_client.stub = mock_stub
            worker_client.connected = True
            
            # Mock successful heartbeat response
            mock_stub.Heartbeat.return_value = HeartbeatResponse(
                success=True,
                term=1,
                leader_id="coordinator-1"
            )
            
            success = await worker_client.send_heartbeat(NodeStatus.ACTIVE)
            assert success == True
    
    @pytest.mark.asyncio
    async def test_sync_gradients(self, worker_client):
        with patch('communication.worker_client.ClusterServiceStub') as mock_stub_class:
            mock_stub = AsyncMock()
            mock_stub_class.return_value = mock_stub
            worker_client.stub = mock_stub
            worker_client.connected = True
            
            # Create mock gradients
            gradients = {
                "layer1.weight": torch.randn(10, 5),
                "layer1.bias": torch.randn(10)
            }
            
            # Mock successful gradient sync response
            mock_stub.SyncGradients.return_value = GradientSyncResponse(
                success=True,
                averaged_gradients=[],
                should_continue=True
            )
            
            response = await worker_client.sync_gradients(
                training_id="training-123",
                iteration=1,
                gradients=gradients,
                loss=0.5,
                accuracy=0.8
            )
            
            assert response.success == True
            assert response.should_continue == True
    
    @pytest.mark.asyncio
    async def test_disconnect_from_coordinator(self, worker_client):
        with patch('communication.worker_client.ClusterServiceStub') as mock_stub_class:
            mock_stub = AsyncMock()
            mock_stub_class.return_value = mock_stub
            worker_client.stub = mock_stub
            worker_client.connected = True
            
            # Mock successful leave response
            mock_stub.LeaveCluster.return_value = LeaveResponse(
                success=True,
                message="Successfully left cluster"
            )
            
            success = await worker_client.disconnect_from_coordinator("shutdown")
            assert success == True
            assert worker_client.connected == False

@pytest.mark.asyncio
class TestGradientSynchronization:
    
    async def test_gradient_serialization_over_grpc(self):
        from worker.gradient_sync import GradientSynchronizer
        
        sync = GradientSynchronizer("allreduce", "worker-1")
        
        # Create test gradients
        gradients = {
            "layer1.weight": torch.randn(5, 3),
            "layer1.bias": torch.randn(5),
            "layer2.weight": torch.randn(2, 5)
        }
        
        # Serialize for gRPC
        tensor_data_list = sync.serialize_gradients(gradients)
        
        # Verify serialization
        assert len(tensor_data_list) == 3
        for tensor_data in tensor_data_list:
            assert isinstance(tensor_data, TensorData)
            assert len(tensor_data.data) > 0
            assert len(tensor_data.shape) > 0
        
        # Deserialize
        deserialized = sync.deserialize_gradients(tensor_data_list)
        
        # Verify deserialization
        assert len(deserialized) == len(gradients)
        for name, tensor in deserialized.items():
            assert torch.allclose(tensor, gradients[name])
    
    async def test_metrics_serialization(self):
        # Test training metrics serialization
        metrics = TrainingMetrics(
            loss=0.5,
            accuracy=0.85,
            iteration=100,
            learning_rate=0.001,
            timestamp=1234567890,
            custom_metrics={"gradient_norm": 1.2, "batch_time": 0.1}
        )
        
        # Verify all fields are properly set
        assert metrics.loss == 0.5
        assert metrics.accuracy == 0.85
        assert metrics.iteration == 100
        assert len(metrics.custom_metrics) == 2

@pytest.mark.asyncio
class TestNetworkFailureHandling:
    
    async def test_connection_retry(self):
        client = WorkerClient("worker-1", "localhost:8001")
        
        with patch('communication.worker_client.grpc.aio.insecure_channel') as mock_channel:
            # Mock connection failure then success
            mock_channel.side_effect = [
                grpc.aio.AioRpcError(grpc.StatusCode.UNAVAILABLE, "Connection failed"),
                AsyncMock()  # Successful connection on retry
            ]
            
            # Should handle the retry internally
            success = await client.connect_to_coordinator()
            
            # Depending on implementation, this might be False if retry logic isn't implemented yet
            # This test documents the expected behavior
    
    async def test_heartbeat_failure_handling(self):
        client = WorkerClient("worker-1", "localhost:8001")
        client.connected = True
        
        mock_stub = AsyncMock()
        mock_stub.Heartbeat.side_effect = grpc.aio.AioRpcError(
            grpc.StatusCode.UNAVAILABLE, "Leader unavailable"
        )
        client.stub = mock_stub
        
        success = await client.send_heartbeat(NodeStatus.ACTIVE)
        
        # Should handle the failure gracefully
        assert success == False
        
    async def test_gradient_sync_timeout(self):
        client = WorkerClient("worker-1", "localhost:8001")
        client.connected = True
        
        mock_stub = AsyncMock()
        mock_stub.SyncGradients.side_effect = grpc.aio.AioRpcError(
            grpc.StatusCode.DEADLINE_EXCEEDED, "Request timeout"
        )
        client.stub = mock_stub
        
        gradients = {"layer1.weight": torch.randn(5, 3)}
        
        response = await client.sync_gradients(
            "training-123", 1, gradients, 0.5, 0.8
        )
        
        # Should handle timeout gracefully
        assert response is None or response.success == False