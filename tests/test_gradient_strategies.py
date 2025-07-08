import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import asyncio
from typing import List, Dict

from worker.gradient_sync import GradientSynchronizer
from worker.distributed_strategies import AllReduceStrategy, ParameterServerStrategy, CustomStrategy
from communication.gradient_coordinator import GradientCoordinator

class TestAllReduceStrategy:
    
    @pytest.fixture
    def allreduce_strategy(self):
        return AllReduceStrategy(node_id="worker-1", num_workers=3)
    
    def test_allreduce_initialization(self, allreduce_strategy):
        assert allreduce_strategy.node_id == "worker-1"
        assert allreduce_strategy.num_workers == 3
        assert allreduce_strategy.strategy_name == "allreduce"
    
    def test_allreduce_ring_topology(self, allreduce_strategy):
        # Test ring topology creation
        ring = allreduce_strategy.create_ring_topology(["worker-0", "worker-1", "worker-2"])
        assert len(ring) == 3
        assert ring["worker-0"]["next"] == "worker-1"
        assert ring["worker-1"]["next"] == "worker-2"
        assert ring["worker-2"]["next"] == "worker-0"
        assert ring["worker-0"]["prev"] == "worker-2"
        assert ring["worker-1"]["prev"] == "worker-0"
        assert ring["worker-2"]["prev"] == "worker-1"
    
    def test_allreduce_scatter_reduce(self, allreduce_strategy):
        # Create test gradients for multiple workers
        gradients = {
            "worker-0": {"param1": torch.tensor([1.0, 2.0, 3.0]), "param2": torch.tensor([4.0, 5.0])},
            "worker-1": {"param1": torch.tensor([2.0, 3.0, 4.0]), "param2": torch.tensor([5.0, 6.0])},
            "worker-2": {"param1": torch.tensor([3.0, 4.0, 5.0]), "param2": torch.tensor([6.0, 7.0])}
        }
        
        # Test scatter-reduce phase
        scattered = allreduce_strategy.scatter_reduce(gradients, "worker-1")
        
        # In scatter-reduce, each worker gets a subset of parameters
        # worker-1 has rank 1, so it gets param2 (since we have 2 params and 3 workers)
        assert len(scattered) == 1  # worker-1 gets 1 parameter
        assert "param2" in scattered
        
        # Check that gradients are properly reduced (averaged)
        expected_param2 = torch.mean(torch.stack([
            gradients["worker-0"]["param2"],
            gradients["worker-1"]["param2"], 
            gradients["worker-2"]["param2"]
        ]), dim=0)
        assert torch.allclose(scattered["param2"], expected_param2)
    
    def test_allreduce_allgather(self, allreduce_strategy):
        # Create test reduced gradients
        reduced_gradients = {
            "param1": torch.tensor([2.0, 3.0, 4.0]),  # Average of [1,2,3], [2,3,4], [3,4,5]
            "param2": torch.tensor([5.0, 6.0])        # Average of [4,5], [5,6], [6,7]
        }
        
        # Test allgather phase
        gathered = allreduce_strategy.allgather(reduced_gradients, ["worker-0", "worker-1", "worker-2"])
        
        # All workers should have the same final gradients
        assert len(gathered) == 3
        for worker_id, gradients in gathered.items():
            assert torch.allclose(gradients["param1"], reduced_gradients["param1"])
            assert torch.allclose(gradients["param2"], reduced_gradients["param2"])
    
    def test_allreduce_bandwidth_optimization(self, allreduce_strategy):
        # Test bandwidth-optimal allreduce
        gradients = {
            "worker-0": {"large_param": torch.randn(1000, 1000)},
            "worker-1": {"large_param": torch.randn(1000, 1000)},
            "worker-2": {"large_param": torch.randn(1000, 1000)}
        }
        
        # Should use ring allreduce for large tensors
        result = allreduce_strategy.optimize_communication(gradients)
        
        assert result["strategy"] == "ring_allreduce"
        assert result["estimated_bandwidth"] > 0
    
    def test_allreduce_numerical_stability(self, allreduce_strategy):
        # Test with very small and very large gradients
        gradients = {
            "worker-0": {"param": torch.tensor([1e-8, 1e8, 0.0])},
            "worker-1": {"param": torch.tensor([2e-8, 2e8, 1e-10])},
            "worker-2": {"param": torch.tensor([3e-8, 3e8, -1e-10])}
        }
        
        result = allreduce_strategy.reduce_with_numerical_stability(gradients)
        
        # Should handle numerical stability
        assert not torch.isnan(result["param"]).any()
        assert not torch.isinf(result["param"]).any()

class TestParameterServerStrategy:
    
    @pytest.fixture
    def ps_strategy(self):
        return ParameterServerStrategy(node_id="worker-1", server_nodes=["ps-0", "ps-1"])
    
    def test_parameter_server_initialization(self, ps_strategy):
        assert ps_strategy.node_id == "worker-1"
        assert len(ps_strategy.server_nodes) == 2
        assert ps_strategy.strategy_name == "parameter_server"
    
    def test_parameter_partitioning(self, ps_strategy):
        # Test parameter partitioning across servers
        parameters = {
            "layer1.weight": torch.randn(100, 50),
            "layer1.bias": torch.randn(100),
            "layer2.weight": torch.randn(10, 100),
            "layer2.bias": torch.randn(10)
        }
        
        partitions = ps_strategy.partition_parameters(parameters)
        
        # Should partition across available servers
        assert len(partitions) == 2
        assert "ps-0" in partitions
        assert "ps-1" in partitions
        
        # Each server should have some parameters
        assert len(partitions["ps-0"]) > 0
        assert len(partitions["ps-1"]) > 0
        
        # All parameters should be assigned
        total_params = sum(len(p) for p in partitions.values())
        assert total_params == len(parameters)
    
    def test_gradient_push(self, ps_strategy):
        # Test pushing gradients to parameter servers
        gradients = {
            "layer1.weight": torch.randn(100, 50),
            "layer1.bias": torch.randn(100)
        }
        
        push_requests = ps_strategy.create_push_requests(gradients)
        
        # Should create push requests for each server
        assert len(push_requests) <= len(ps_strategy.server_nodes)
        
        for server_id, request in push_requests.items():
            assert server_id in ps_strategy.server_nodes
            assert len(request["gradients"]) > 0
    
    def test_parameter_pull(self, ps_strategy):
        # Test pulling parameters from parameter servers
        parameter_requests = ps_strategy.create_pull_requests()
        
        # Should create pull requests for each server
        assert len(parameter_requests) <= len(ps_strategy.server_nodes)
        
        for server_id, request in parameter_requests.items():
            assert server_id in ps_strategy.server_nodes
            assert "parameter_keys" in request
    
    def test_staleness_handling(self, ps_strategy):
        # Test handling of stale gradients
        gradient_metadata = {
            "worker-0": {"timestamp": 1000, "iteration": 10},
            "worker-1": {"timestamp": 1100, "iteration": 11},
            "worker-2": {"timestamp": 900, "iteration": 9}  # Stale
        }
        
        # Set max_staleness to 1 (default is 3, so worker-2 with staleness 2 should be filtered)
        ps_strategy.max_staleness = 1
        filtered = ps_strategy.filter_stale_gradients(gradient_metadata)
        
        # Should filter out stale gradients
        assert "worker-2" not in filtered
        assert len(filtered) == 2
    
    def test_adaptive_learning_rate(self, ps_strategy):
        # Test adaptive learning rate based on staleness
        base_lr = 0.001
        staleness = 3
        
        adjusted_lr = ps_strategy.adjust_learning_rate(base_lr, staleness)
        
        # Should reduce learning rate for stale gradients
        assert adjusted_lr < base_lr

class TestCustomStrategy:
    
    @pytest.fixture
    def custom_strategy(self):
        return CustomStrategy(node_id="worker-1", topology="hierarchical")
    
    def test_custom_strategy_initialization(self, custom_strategy):
        assert custom_strategy.node_id == "worker-1"
        assert custom_strategy.topology == "hierarchical"
        assert custom_strategy.strategy_name == "custom"
    
    def test_hierarchical_aggregation(self, custom_strategy):
        # Test hierarchical gradient aggregation
        gradients = {
            "gpu-0": {"param": torch.tensor([1.0, 2.0])},
            "gpu-1": {"param": torch.tensor([3.0, 4.0])},
            "gpu-2": {"param": torch.tensor([5.0, 6.0])},
            "gpu-3": {"param": torch.tensor([7.0, 8.0])}
        }
        
        # Create hierarchy: 2 nodes with 2 GPUs each
        hierarchy = custom_strategy.create_hierarchy(gradients.keys(), gpus_per_node=2)
        
        assert len(hierarchy) == 2  # Two nodes
        assert len(hierarchy["node-0"]) == 2  # Two GPUs per node
        assert len(hierarchy["node-1"]) == 2
    
    def test_weighted_averaging(self, custom_strategy):
        # Test weighted averaging based on compute power
        gradients = {
            "worker-0": {"param": torch.tensor([1.0, 2.0])},
            "worker-1": {"param": torch.tensor([3.0, 4.0])},
            "worker-2": {"param": torch.tensor([5.0, 6.0])}
        }
        
        weights = {
            "worker-0": 0.5,  # Slower worker
            "worker-1": 1.0,  # Normal worker
            "worker-2": 1.5   # Faster worker
        }
        
        result = custom_strategy.weighted_average(gradients, weights)
        
        # Should compute weighted average
        expected = (0.5 * gradients["worker-0"]["param"] + 
                   1.0 * gradients["worker-1"]["param"] + 
                   1.5 * gradients["worker-2"]["param"]) / (0.5 + 1.0 + 1.5)
        
        assert torch.allclose(result["param"], expected)
    
    def test_gradient_compression(self, custom_strategy):
        # Test gradient compression
        gradients = {
            "param1": torch.randn(1000, 1000),
            "param2": torch.randn(500, 500)
        }
        
        compressed = custom_strategy.compress_gradients(gradients, compression_ratio=0.1)
        
        # Should compress gradients
        assert len(compressed) == len(gradients)
        for param_name, compressed_tensor in compressed.items():
            original_size = gradients[param_name].numel()
            compressed_size = compressed_tensor.numel()
            assert compressed_size < original_size
    
    def test_asynchronous_update(self, custom_strategy):
        # Test asynchronous gradient updates
        gradients = {
            "param": torch.tensor([1.0, 2.0, 3.0])
        }
        
        # Simulate async update with momentum
        momentum = 0.9
        previous_update = {"param": torch.tensor([0.1, 0.2, 0.3])}
        
        update = custom_strategy.compute_async_update(gradients, previous_update, momentum)
        
        # Should incorporate momentum
        assert update["param"].shape == gradients["param"].shape
        assert not torch.equal(update["param"], gradients["param"])

class TestGradientCoordinator:
    
    @pytest.fixture
    def gradient_coordinator(self):
        return GradientCoordinator(
            strategy="allreduce",
            worker_nodes=["worker-0", "worker-1", "worker-2"]
        )
    
    @pytest.mark.asyncio
    async def test_coordinate_gradient_sync(self, gradient_coordinator):
        # Test coordinating gradient synchronization
        gradients = {
            "worker-0": {"param": torch.tensor([1.0, 2.0])},
            "worker-1": {"param": torch.tensor([3.0, 4.0])},
            "worker-2": {"param": torch.tensor([5.0, 6.0])}
        }
        
        result = await gradient_coordinator.coordinate_sync(gradients, iteration=1)
        
        assert result["success"] == True
        assert "averaged_gradients" in result
        # The result should contain averaged gradients with same parameter names as input
        assert len(result["averaged_gradients"]) == len(gradients["worker-0"])
    
    @pytest.mark.asyncio
    async def test_fault_tolerant_sync(self, gradient_coordinator):
        # Test gradient sync with failed workers
        gradients = {
            "worker-0": {"param": torch.tensor([1.0, 2.0])},
            "worker-1": {"param": torch.tensor([3.0, 4.0])},
            # worker-2 failed
        }
        
        result = await gradient_coordinator.coordinate_sync(
            gradients, 
            iteration=1, 
            failed_workers=["worker-2"]
        )
        
        # Should handle missing workers gracefully
        assert result["success"] == True
        assert len(result["warnings"]) > 0  # Should warn about missing worker
    
    @pytest.mark.asyncio
    async def test_dynamic_strategy_selection(self, gradient_coordinator):
        # Test dynamic strategy selection based on conditions
        conditions = {
            "network_bandwidth": "high",
            "worker_count": 10,
            "model_size": "large"
        }
        
        selected_strategy = await gradient_coordinator.select_optimal_strategy(conditions)
        
        # Should select appropriate strategy
        assert selected_strategy in ["allreduce", "parameter_server", "custom"]
    
    def test_convergence_detection(self, gradient_coordinator):
        # Test gradient convergence detection
        gradient_history = [
            {"param": torch.tensor([1.0, 1.0])},
            {"param": torch.tensor([0.8, 0.8])},
            {"param": torch.tensor([0.6, 0.6])},
            {"param": torch.tensor([0.4, 0.4])},
            {"param": torch.tensor([0.2, 0.2])}
        ]
        
        converged = gradient_coordinator.detect_convergence(gradient_history)
        
        # Should detect convergence trend
        assert isinstance(converged, bool)
    
    def test_gradient_statistics(self, gradient_coordinator):
        # Test gradient statistics collection
        gradients = {
            "param1": torch.tensor([1.0, 2.0, 3.0]),
            "param2": torch.tensor([4.0, 5.0])
        }
        
        stats = gradient_coordinator.compute_gradient_statistics(gradients)
        
        # Should compute comprehensive statistics
        assert "gradient_norm" in stats
        assert "parameter_count" in stats
        assert "sparsity" in stats
        assert "distribution" in stats
        assert stats["parameter_count"] == 5  # 3 + 2 parameters

@pytest.mark.asyncio
class TestDistributedTrainingIntegration:
    
    async def test_full_training_cycle(self):
        # Test complete distributed training cycle
        from worker.training_worker import TrainingWorker
        from communication.gradient_coordinator import GradientCoordinator
        
        # Create workers
        workers = []
        for i in range(3):
            worker = TrainingWorker(
                node_id=f"worker-{i}",
                coordinator_address="localhost:8001",
                port=9001 + i
            )
            workers.append(worker)
        
        # Create gradient coordinator
        coordinator = GradientCoordinator(
            strategy="allreduce",
            worker_nodes=[f"worker-{i}" for i in range(3)]
        )
        
        # Initialize training
        model_config = {
            "model_type": "simple_fc",
            "input_size": 784,
            "hidden_size": 128,
            "num_classes": 10
        }
        
        training_config = {
            "batch_size": 32,
            "learning_rate": 0.001,
            "sync_strategy": "allreduce"
        }
        
        # Start training on all workers
        torch.manual_seed(42)
        for worker in workers:
            torch.manual_seed(42)  # Same initial weights
            success = worker.start_training("training-123", model_config, training_config)
            assert success == True
        
        # Simulate training iterations
        for iteration in range(3):
            # Create consistent batch data
            torch.manual_seed(42 + iteration)
            batch_data = torch.randn(32, 784)
            batch_labels = torch.randint(0, 10, (32,))
            
            # Compute gradients on all workers
            worker_gradients = {}
            for worker in workers:
                loss, gradients = worker.compute_gradients(batch_data, batch_labels)
                worker_gradients[worker.node_id] = gradients
            
            # Coordinate gradient synchronization
            sync_result = await coordinator.coordinate_sync(worker_gradients, iteration)
            assert sync_result["success"] == True
            
            # Apply synchronized gradients
            averaged_gradients = sync_result["averaged_gradients"]
            for worker in workers:
                worker.apply_gradients(averaged_gradients)
        
        # Verify workers converged to same model
        final_states = [worker.get_model_state() for worker in workers]
        for param_name in final_states[0].keys():
            for i in range(1, len(final_states)):
                assert torch.allclose(
                    final_states[0][param_name], 
                    final_states[i][param_name],
                    atol=1e-6
                )
        
        # Stop training
        for worker in workers:
            worker.stop_training("training-123")
    
    async def test_strategy_comparison(self):
        # Test comparison between different strategies
        strategies = ["allreduce", "parameter_server", "custom"]
        
        results = {}
        for strategy in strategies:
            coordinator = GradientCoordinator(
                strategy=strategy,
                worker_nodes=["worker-0", "worker-1", "worker-2"]
            )
            
            # Create test gradients
            gradients = {
                "worker-0": {"param": torch.randn(100, 100)},
                "worker-1": {"param": torch.randn(100, 100)},
                "worker-2": {"param": torch.randn(100, 100)}
            }
            
            start_time = torch.tensor(0.0)  # Mock timing
            result = await coordinator.coordinate_sync(gradients, iteration=1)
            end_time = torch.tensor(0.1)   # Mock timing
            
            results[strategy] = {
                "success": result["success"],
                "time": end_time - start_time,
                "gradient_norm": torch.norm(result["averaged_gradients"]["param"])
            }
        
        # All strategies should succeed
        for strategy, result in results.items():
            assert result["success"] == True