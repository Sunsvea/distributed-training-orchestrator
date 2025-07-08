import pytest
import torch
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from worker.training_worker import TrainingWorker
from worker.gradient_sync import GradientSynchronizer
from communication.cluster_pb2 import *

class TestTrainingWorker:
    
    @pytest.fixture
    def worker(self):
        return TrainingWorker(
            node_id="worker-1",
            coordinator_address="localhost:8001",
            port=9001
        )
    
    def test_worker_initialization(self, worker):
        assert worker.node_id == "worker-1"
        assert worker.coordinator_address == "localhost:8001"
        assert worker.port == 9001
        assert worker.status == NodeStatus.IDLE
        assert worker.current_training_id is None
        assert worker.model is None
        assert worker.optimizer is None
    
    def test_start_training(self, worker):
        model_config = {
            "model_type": "simple_cnn",
            "input_size": 784,
            "hidden_size": 128,
            "num_classes": 10
        }
        
        training_config = {
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 10,
            "sync_strategy": "allreduce"
        }
        
        success = worker.start_training("training-123", model_config, training_config)
        assert success == True
        assert worker.status == NodeStatus.TRAINING
        assert worker.current_training_id == "training-123"
        assert worker.model is not None
        assert worker.optimizer is not None
    
    def test_stop_training(self, worker):
        # First start training
        model_config = {"model_type": "simple_cnn", "input_size": 784, "hidden_size": 128, "num_classes": 10}
        training_config = {"batch_size": 32, "learning_rate": 0.001, "epochs": 10, "sync_strategy": "allreduce"}
        worker.start_training("training-123", model_config, training_config)
        
        # Then stop it
        success = worker.stop_training("training-123")
        assert success == True
        assert worker.status == NodeStatus.IDLE
        assert worker.current_training_id is None
    
    def test_stop_training_wrong_id(self, worker):
        # Start training with one ID
        model_config = {"model_type": "simple_cnn", "input_size": 784, "hidden_size": 128, "num_classes": 10}
        training_config = {"batch_size": 32, "learning_rate": 0.001, "epochs": 10, "sync_strategy": "allreduce"}
        worker.start_training("training-123", model_config, training_config)
        
        # Try to stop with wrong ID
        success = worker.stop_training("training-456")
        assert success == False
        assert worker.status == NodeStatus.TRAINING
        assert worker.current_training_id == "training-123"
    
    def test_compute_gradients(self, worker):
        model_config = {"model_type": "simple_cnn", "input_size": 784, "hidden_size": 128, "num_classes": 10}
        training_config = {"batch_size": 32, "learning_rate": 0.001, "epochs": 10, "sync_strategy": "allreduce"}
        worker.start_training("training-123", model_config, training_config)
        
        # Create mock batch
        batch_data = torch.randn(32, 784)
        batch_labels = torch.randint(0, 10, (32,))
        
        loss, gradients = worker.compute_gradients(batch_data, batch_labels)
        
        assert isinstance(loss, float)
        assert loss > 0
        assert isinstance(gradients, dict)
        assert len(gradients) > 0
        
        # Check that gradients have correct structure
        for param_name, grad in gradients.items():
            assert isinstance(grad, torch.Tensor)
            assert grad.requires_grad == False
    
    def test_apply_gradients(self, worker):
        model_config = {"model_type": "simple_cnn", "input_size": 784, "hidden_size": 128, "num_classes": 10}
        training_config = {"batch_size": 32, "learning_rate": 0.001, "epochs": 10, "sync_strategy": "allreduce"}
        worker.start_training("training-123", model_config, training_config)
        
        # Get initial model parameters
        initial_params = {}
        for name, param in worker.model.named_parameters():
            initial_params[name] = param.clone()
        
        # Create some gradients
        gradients = {}
        for name, param in worker.model.named_parameters():
            gradients[name] = torch.randn_like(param) * 0.1
        
        worker.apply_gradients(gradients)
        
        # Check that parameters changed
        for name, param in worker.model.named_parameters():
            assert not torch.equal(param, initial_params[name])
    
    def test_get_model_state(self, worker):
        model_config = {"model_type": "simple_cnn", "input_size": 784, "hidden_size": 128, "num_classes": 10}
        training_config = {"batch_size": 32, "learning_rate": 0.001, "epochs": 10, "sync_strategy": "allreduce"}
        worker.start_training("training-123", model_config, training_config)
        
        state = worker.get_model_state()
        assert isinstance(state, dict)
        assert len(state) > 0
        
        # Check that state contains model parameters
        for name, param in worker.model.named_parameters():
            assert name in state
            assert torch.equal(param, state[name])

class TestGradientSynchronizer:
    
    @pytest.fixture
    def sync_allreduce(self):
        return GradientSynchronizer(strategy="allreduce", node_id="worker-1")
    
    @pytest.fixture
    def sync_parameter_server(self):
        return GradientSynchronizer(strategy="parameter_server", node_id="worker-1")
    
    def test_allreduce_initialization(self, sync_allreduce):
        assert sync_allreduce.strategy == "allreduce"
        assert sync_allreduce.node_id == "worker-1"
    
    def test_parameter_server_initialization(self, sync_parameter_server):
        assert sync_parameter_server.strategy == "parameter_server"
        assert sync_parameter_server.node_id == "worker-1"
    
    def test_serialize_gradients(self, sync_allreduce):
        gradients = {
            "layer1.weight": torch.randn(10, 5),
            "layer1.bias": torch.randn(10),
            "layer2.weight": torch.randn(1, 10)
        }
        
        tensor_data_list = sync_allreduce.serialize_gradients(gradients)
        assert len(tensor_data_list) == 3
        
        for tensor_data in tensor_data_list:
            assert isinstance(tensor_data, TensorData)
            assert tensor_data.name in gradients
            assert len(tensor_data.shape) > 0
            assert len(tensor_data.data) > 0
    
    def test_deserialize_gradients(self, sync_allreduce):
        # First serialize some gradients
        original_gradients = {
            "layer1.weight": torch.randn(10, 5),
            "layer1.bias": torch.randn(10)
        }
        
        tensor_data_list = sync_allreduce.serialize_gradients(original_gradients)
        
        # Then deserialize them
        deserialized_gradients = sync_allreduce.deserialize_gradients(tensor_data_list)
        
        assert len(deserialized_gradients) == len(original_gradients)
        for name, tensor in deserialized_gradients.items():
            assert name in original_gradients
            assert torch.allclose(tensor, original_gradients[name])
    
    def test_allreduce_average_gradients(self, sync_allreduce):
        # Create multiple gradient sets
        gradients_list = []
        for i in range(3):
            gradients = {
                "layer1.weight": torch.ones(2, 2) * (i + 1),
                "layer1.bias": torch.ones(2) * (i + 1)
            }
            gradients_list.append(gradients)
        
        averaged = sync_allreduce.average_gradients(gradients_list)
        
        # Check that averaging worked correctly
        assert torch.allclose(averaged["layer1.weight"], torch.ones(2, 2) * 2.0)  # (1+2+3)/3 = 2
        assert torch.allclose(averaged["layer1.bias"], torch.ones(2) * 2.0)
    
    def test_compute_gradient_norm(self, sync_allreduce):
        gradients = {
            "layer1.weight": torch.ones(2, 2),
            "layer1.bias": torch.ones(2)
        }
        
        norm = sync_allreduce.compute_gradient_norm(gradients)
        expected_norm = torch.sqrt(torch.tensor(4.0 + 2.0))  # sqrt(4*1^2 + 2*1^2)
        assert torch.allclose(norm, expected_norm)
    
    def test_clip_gradients(self, sync_allreduce):
        gradients = {
            "layer1.weight": torch.ones(2, 2) * 10,
            "layer1.bias": torch.ones(2) * 10
        }
        
        clipped = sync_allreduce.clip_gradients(gradients, max_norm=1.0)
        
        # Check that gradient norm is now 1.0
        norm = sync_allreduce.compute_gradient_norm(clipped)
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-6)

@pytest.mark.asyncio
class TestWorkerIntegration:
    
    async def test_worker_training_loop(self):
        worker = TrainingWorker(
            node_id="worker-1",
            coordinator_address="localhost:8001",
            port=9001
        )
        
        model_config = {
            "model_type": "simple_cnn",
            "input_size": 784,
            "hidden_size": 128,
            "num_classes": 10
        }
        
        training_config = {
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 2,
            "sync_strategy": "allreduce"
        }
        
        # Start training
        success = worker.start_training("training-123", model_config, training_config)
        assert success == True
        
        # Simulate training iterations
        for iteration in range(5):
            # Create mock batch
            batch_data = torch.randn(32, 784)
            batch_labels = torch.randint(0, 10, (32,))
            
            # Compute gradients
            loss, gradients = worker.compute_gradients(batch_data, batch_labels)
            
            # Simulate gradient synchronization (normally would be done via gRPC)
            averaged_gradients = gradients  # In real scenario, this would be averaged
            
            # Apply gradients
            worker.apply_gradients(averaged_gradients)
            
            # Check that loss is reasonable
            assert loss > 0
            assert loss < 10  # Should not be too high
        
        # Stop training
        success = worker.stop_training("training-123")
        assert success == True
    
    async def test_gradient_synchronization_flow(self):
        # Create multiple workers
        workers = []
        for i in range(3):
            worker = TrainingWorker(
                node_id=f"worker-{i}",
                coordinator_address="localhost:8001",
                port=9001 + i
            )
            workers.append(worker)
        
        model_config = {
            "model_type": "simple_cnn",
            "input_size": 784,
            "hidden_size": 128,
            "num_classes": 10
        }
        
        training_config = {
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 2,
            "sync_strategy": "allreduce"
        }
        
        # Start training on all workers with same random seed
        torch.manual_seed(42)
        for worker in workers:
            torch.manual_seed(42)  # Ensure same initial weights
            success = worker.start_training("training-123", model_config, training_config)
            assert success == True
        
        # Simulate synchronization step
        batch_data = torch.randn(32, 784)
        batch_labels = torch.randint(0, 10, (32,))
        
        # Compute gradients on all workers
        all_gradients = []
        for worker in workers:
            loss, gradients = worker.compute_gradients(batch_data, batch_labels)
            all_gradients.append(gradients)
        
        # Average gradients (simulate allreduce)
        averaged_gradients = {}
        for param_name in all_gradients[0].keys():
            param_gradients = [grads[param_name] for grads in all_gradients]
            averaged_gradients[param_name] = torch.mean(torch.stack(param_gradients), dim=0)
        
        # Apply averaged gradients to all workers
        for worker in workers:
            worker.apply_gradients(averaged_gradients)
        
        # Check that all workers have the same model state
        states = [worker.get_model_state() for worker in workers]
        for param_name in states[0].keys():
            for i in range(1, len(states)):
                assert torch.equal(states[0][param_name], states[i][param_name])
        
        # Stop training on all workers
        for worker in workers:
            success = worker.stop_training("training-123")
            assert success == True