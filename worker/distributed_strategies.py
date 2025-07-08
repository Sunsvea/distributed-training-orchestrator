import torch
import torch.distributed as dist
import numpy as np
import time
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
import math

class DistributedStrategy(ABC):
    """Base class for distributed gradient synchronization strategies"""
    
    def __init__(self, node_id: str, **kwargs):
        self.node_id = node_id
        self.strategy_name = self.__class__.__name__.lower().replace("strategy", "")
        self.metrics = {
            "communication_time": 0.0,
            "computation_time": 0.0,
            "bandwidth_usage": 0.0,
            "sync_iterations": 0
        }
    
    @abstractmethod
    def synchronize_gradients(self, gradients: Dict[str, torch.Tensor], 
                            worker_gradients: Dict[str, Dict[str, torch.Tensor]],
                            **kwargs) -> Dict[str, torch.Tensor]:
        """Synchronize gradients across workers"""
        pass
    
    def update_metrics(self, metric_name: str, value: float):
        """Update strategy metrics"""
        self.metrics[metric_name] = value
        self.metrics["sync_iterations"] += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """Get strategy performance metrics"""
        return self.metrics.copy()

class AllReduceStrategy(DistributedStrategy):
    """Implementation of AllReduce gradient synchronization"""
    
    def __init__(self, node_id: str, num_workers: int, **kwargs):
        super().__init__(node_id, **kwargs)
        self.num_workers = num_workers
        self.strategy_name = "allreduce"
        self.ring_buffer = {}
        self.compression_enabled = kwargs.get("compression", False)
        self.compression_ratio = kwargs.get("compression_ratio", 0.1)
    
    def synchronize_gradients(self, gradients: Dict[str, torch.Tensor], 
                            worker_gradients: Dict[str, Dict[str, torch.Tensor]],
                            **kwargs) -> Dict[str, torch.Tensor]:
        """Perform AllReduce synchronization"""
        start_time = time.time()
        
        # Choose optimal AllReduce variant based on data size
        total_size = sum(tensor.numel() for tensor in gradients.values())
        
        if total_size > 1e6:  # Large models
            result = self._ring_allreduce(gradients, worker_gradients)
        elif self.num_workers > 8:  # Many workers
            result = self._tree_allreduce(gradients, worker_gradients)
        else:  # Small models or few workers
            result = self._butterfly_allreduce(gradients, worker_gradients)
        
        # Apply compression if enabled
        if self.compression_enabled:
            result = self._compress_gradients(result)
        
        self.update_metrics("communication_time", time.time() - start_time)
        return result
    
    def _ring_allreduce(self, gradients: Dict[str, torch.Tensor], 
                       worker_gradients: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Ring AllReduce implementation for bandwidth optimization"""
        worker_ids = list(worker_gradients.keys())
        ring_topology = self.create_ring_topology(worker_ids)
        
        # Phase 1: Scatter-Reduce
        scattered_gradients = self.scatter_reduce(worker_gradients, self.node_id)
        
        # Phase 2: AllGather
        result = self.allgather(scattered_gradients, worker_ids)
        
        return result[self.node_id]
    
    def _tree_allreduce(self, gradients: Dict[str, torch.Tensor], 
                       worker_gradients: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Tree AllReduce implementation for many workers"""
        # Build binary tree topology
        tree_topology = self._build_tree_topology(list(worker_gradients.keys()))
        
        # Reduce phase: aggregate gradients up the tree
        reduced_gradients = self._tree_reduce(worker_gradients, tree_topology)
        
        # Broadcast phase: distribute results down the tree
        result = self._tree_broadcast(reduced_gradients, tree_topology)
        
        return result
    
    def _butterfly_allreduce(self, gradients: Dict[str, torch.Tensor], 
                           worker_gradients: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Butterfly AllReduce implementation for low latency"""
        # Simple averaging for small number of workers
        return self._simple_average(worker_gradients)
    
    def create_ring_topology(self, worker_ids: List[str]) -> Dict[str, Dict[str, str]]:
        """Create ring topology for AllReduce"""
        n = len(worker_ids)
        ring = {}
        
        for i, worker_id in enumerate(worker_ids):
            ring[worker_id] = {
                "next": worker_ids[(i + 1) % n],
                "prev": worker_ids[(i - 1) % n],
                "rank": i
            }
        
        return ring
    
    def scatter_reduce(self, worker_gradients: Dict[str, Dict[str, torch.Tensor]], 
                      current_worker: str) -> Dict[str, torch.Tensor]:
        """Scatter-reduce phase of ring AllReduce"""
        # Get parameter names
        param_names = list(next(iter(worker_gradients.values())).keys())
        
        # Divide parameters among workers
        worker_ids = sorted(worker_gradients.keys())
        worker_rank = worker_ids.index(current_worker)
        num_workers = len(worker_ids)
        
        # Ensure each worker gets at least one parameter if possible
        if len(param_names) >= num_workers:
            params_per_worker = len(param_names) // num_workers
            start_idx = worker_rank * params_per_worker
            end_idx = start_idx + params_per_worker
            if worker_rank == num_workers - 1:  # Last worker gets remainder
                end_idx = len(param_names)
        else:
            # More workers than parameters - some workers get 1, others get 0
            if worker_rank < len(param_names):
                start_idx = worker_rank
                end_idx = worker_rank + 1
            else:
                start_idx = 0
                end_idx = 0
        
        my_params = param_names[start_idx:end_idx]
        
        # Reduce gradients for my parameters
        reduced_gradients = {}
        for param_name in my_params:
            param_gradients = [worker_gradients[wid][param_name] for wid in worker_ids]
            reduced_gradients[param_name] = torch.mean(torch.stack(param_gradients), dim=0)
        
        return reduced_gradients
    
    def allgather(self, reduced_gradients: Dict[str, torch.Tensor], 
                 worker_ids: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
        """AllGather phase of ring AllReduce"""
        # Simulate gathering all reduced gradients
        all_gradients = {}
        
        for worker_id in worker_ids:
            all_gradients[worker_id] = reduced_gradients.copy()
        
        return all_gradients
    
    def _build_tree_topology(self, worker_ids: List[str]) -> Dict[str, Any]:
        """Build binary tree topology for tree AllReduce"""
        n = len(worker_ids)
        tree = {}
        
        for i, worker_id in enumerate(worker_ids):
            tree[worker_id] = {
                "rank": i,
                "parent": worker_ids[(i - 1) // 2] if i > 0 else None,
                "left_child": worker_ids[2 * i + 1] if 2 * i + 1 < n else None,
                "right_child": worker_ids[2 * i + 2] if 2 * i + 2 < n else None
            }
        
        return tree
    
    def _tree_reduce(self, worker_gradients: Dict[str, Dict[str, torch.Tensor]], 
                    tree_topology: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Tree reduce phase"""
        # For simulation, return simple average
        return self._simple_average(worker_gradients)
    
    def _tree_broadcast(self, reduced_gradients: Dict[str, torch.Tensor], 
                       tree_topology: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Tree broadcast phase"""
        return reduced_gradients
    
    def _simple_average(self, worker_gradients: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Simple averaging of gradients"""
        if not worker_gradients:
            return {}
        
        # Get parameter names from first worker
        param_names = list(next(iter(worker_gradients.values())).keys())
        averaged_gradients = {}
        
        for param_name in param_names:
            param_gradients = [worker_gradients[wid][param_name] for wid in worker_gradients.keys()]
            averaged_gradients[param_name] = torch.mean(torch.stack(param_gradients), dim=0)
        
        return averaged_gradients
    
    def _compress_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply gradient compression"""
        compressed = {}
        
        for param_name, tensor in gradients.items():
            # Top-k sparsification
            flat_tensor = tensor.flatten()
            k = max(1, int(len(flat_tensor) * self.compression_ratio))
            
            # Get top-k values
            _, indices = torch.topk(torch.abs(flat_tensor), k)
            
            # Create sparse tensor
            sparse_tensor = torch.zeros_like(flat_tensor)
            sparse_tensor[indices] = flat_tensor[indices]
            
            compressed[param_name] = sparse_tensor.reshape(tensor.shape)
        
        return compressed
    
    def optimize_communication(self, gradients: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """Optimize communication strategy based on data characteristics"""
        total_size = sum(
            sum(tensor.numel() for tensor in worker_grads.values())
            for worker_grads in gradients.values()
        )
        
        # Estimate bandwidth requirements
        estimated_bandwidth = total_size * 4 * len(gradients)  # 4 bytes per float32
        
        if total_size > 1e6:
            strategy = "ring_allreduce"
        elif len(gradients) > 8:
            strategy = "tree_allreduce"
        else:
            strategy = "butterfly_allreduce"
        
        return {
            "strategy": strategy,
            "estimated_bandwidth": estimated_bandwidth,
            "compression_recommended": total_size > 1e5
        }
    
    def reduce_with_numerical_stability(self, gradients: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Reduce gradients with numerical stability"""
        if not gradients:
            return {}
        
        param_names = list(next(iter(gradients.values())).keys())
        stable_gradients = {}
        
        for param_name in param_names:
            param_gradients = [gradients[wid][param_name] for wid in gradients.keys()]
            
            # Use Kahan summation for better numerical stability
            result = torch.zeros_like(param_gradients[0])
            compensation = torch.zeros_like(param_gradients[0])
            
            for grad in param_gradients:
                y = grad - compensation
                t = result + y
                compensation = (t - result) - y
                result = t
            
            stable_gradients[param_name] = result / len(param_gradients)
        
        return stable_gradients

class ParameterServerStrategy(DistributedStrategy):
    """Implementation of Parameter Server gradient synchronization"""
    
    def __init__(self, node_id: str, server_nodes: List[str], **kwargs):
        super().__init__(node_id, **kwargs)
        self.server_nodes = server_nodes
        self.strategy_name = "parameter_server"
        self.max_staleness = kwargs.get("max_staleness", 3)
        self.adaptive_lr = kwargs.get("adaptive_lr", True)
        self.parameter_partitions = {}
    
    def synchronize_gradients(self, gradients: Dict[str, torch.Tensor], 
                            worker_gradients: Dict[str, Dict[str, torch.Tensor]],
                            **kwargs) -> Dict[str, torch.Tensor]:
        """Synchronize gradients using parameter server"""
        start_time = time.time()
        
        # Partition parameters across servers
        if not self.parameter_partitions:
            self.parameter_partitions = self.partition_parameters(gradients)
        
        # Handle staleness
        filtered_gradients = self.filter_stale_gradients(
            {wid: {"timestamp": time.time(), "iteration": kwargs.get("iteration", 0)} 
             for wid in worker_gradients.keys()}
        )
        
        # Aggregate gradients
        aggregated = self._aggregate_gradients(worker_gradients, filtered_gradients)
        
        # Apply adaptive learning rate
        if self.adaptive_lr:
            aggregated = self._apply_adaptive_lr(aggregated, kwargs.get("iteration", 0))
        
        self.update_metrics("communication_time", time.time() - start_time)
        return aggregated
    
    def partition_parameters(self, parameters: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        """Partition parameters across parameter servers"""
        param_names = list(parameters.keys())
        num_servers = len(self.server_nodes)
        
        # Calculate parameter sizes for balanced partitioning
        param_sizes = {name: parameters[name].numel() for name in param_names}
        sorted_params = sorted(param_names, key=lambda x: param_sizes[x], reverse=True)
        
        # Greedy partitioning to balance load
        partitions = {server: [] for server in self.server_nodes}
        server_loads = {server: 0 for server in self.server_nodes}
        
        for param_name in sorted_params:
            # Assign to least loaded server
            min_server = min(server_loads.keys(), key=lambda s: server_loads[s])
            partitions[min_server].append(param_name)
            server_loads[min_server] += param_sizes[param_name]
        
        return partitions
    
    def create_push_requests(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
        """Create push requests for parameter servers"""
        push_requests = {}
        
        for server_id, param_names in self.parameter_partitions.items():
            server_gradients = {name: gradients[name] for name in param_names if name in gradients}
            
            if server_gradients:
                push_requests[server_id] = {
                    "gradients": server_gradients,
                    "timestamp": time.time(),
                    "worker_id": self.node_id
                }
        
        return push_requests
    
    def create_pull_requests(self) -> Dict[str, Dict[str, Any]]:
        """Create pull requests for parameter servers"""
        pull_requests = {}
        
        for server_id, param_names in self.parameter_partitions.items():
            pull_requests[server_id] = {
                "parameter_keys": param_names,
                "worker_id": self.node_id,
                "timestamp": time.time()
            }
        
        return pull_requests
    
    def filter_stale_gradients(self, gradient_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Filter out stale gradients based on timestamp or iteration"""
        current_time = time.time()
        max_iteration = max(meta["iteration"] for meta in gradient_metadata.values())
        
        filtered = {}
        
        for worker_id, metadata in gradient_metadata.items():
            iteration_staleness = max_iteration - metadata["iteration"]
            
            if iteration_staleness <= self.max_staleness:
                filtered[worker_id] = metadata
        
        return filtered
    
    def adjust_learning_rate(self, base_lr: float, staleness: int) -> float:
        """Adjust learning rate based on gradient staleness"""
        if staleness == 0:
            return base_lr
        
        # Exponential decay for stale gradients
        return base_lr * (0.8 ** staleness)
    
    def _aggregate_gradients(self, worker_gradients: Dict[str, Dict[str, torch.Tensor]], 
                           filtered_workers: Dict[str, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Aggregate gradients from non-stale workers"""
        if not filtered_workers:
            return {}
        
        # Only use gradients from non-stale workers
        valid_gradients = {wid: worker_gradients[wid] for wid in filtered_workers.keys() if wid in worker_gradients}
        
        if not valid_gradients:
            return {}
        
        # Simple averaging
        param_names = list(next(iter(valid_gradients.values())).keys())
        aggregated = {}
        
        for param_name in param_names:
            param_gradients = [valid_gradients[wid][param_name] for wid in valid_gradients.keys()]
            aggregated[param_name] = torch.mean(torch.stack(param_gradients), dim=0)
        
        return aggregated
    
    def _apply_adaptive_lr(self, gradients: Dict[str, torch.Tensor], iteration: int) -> Dict[str, torch.Tensor]:
        """Apply adaptive learning rate to gradients"""
        # Simple adaptive scaling based on iteration
        scale_factor = 1.0 / (1.0 + 0.001 * iteration)
        
        scaled_gradients = {}
        for param_name, tensor in gradients.items():
            scaled_gradients[param_name] = tensor * scale_factor
        
        return scaled_gradients

class CustomStrategy(DistributedStrategy):
    """Custom gradient synchronization strategy with advanced features"""
    
    def __init__(self, node_id: str, topology: str = "hierarchical", **kwargs):
        super().__init__(node_id, **kwargs)
        self.topology = topology
        self.strategy_name = "custom"
        self.compression_enabled = kwargs.get("compression", True)
        self.quantization_bits = kwargs.get("quantization_bits", 8)
        self.momentum = kwargs.get("momentum", 0.9)
        self.previous_update = {}
    
    def synchronize_gradients(self, gradients: Dict[str, torch.Tensor], 
                            worker_gradients: Dict[str, Dict[str, torch.Tensor]],
                            **kwargs) -> Dict[str, torch.Tensor]:
        """Synchronize gradients using custom strategy"""
        start_time = time.time()
        
        # Determine weights for each worker
        weights = self._compute_worker_weights(worker_gradients, **kwargs)
        
        # Perform weighted averaging
        averaged = self.weighted_average(worker_gradients, weights)
        
        # Apply compression
        if self.compression_enabled:
            averaged = self.compress_gradients(averaged)
        
        # Apply momentum
        if self.momentum > 0:
            averaged = self.compute_async_update(averaged, self.previous_update, self.momentum)
            self.previous_update = averaged.copy()
        
        self.update_metrics("communication_time", time.time() - start_time)
        return averaged
    
    def create_hierarchy(self, worker_ids: List[str], gpus_per_node: int = 4) -> Dict[str, List[str]]:
        """Create hierarchical topology"""
        hierarchy = {}
        
        # Convert to list if needed
        if not isinstance(worker_ids, list):
            worker_ids = list(worker_ids)
        
        # Group workers by node
        num_nodes = len(worker_ids) // gpus_per_node + (1 if len(worker_ids) % gpus_per_node else 0)
        
        for node_id in range(num_nodes):
            start_idx = node_id * gpus_per_node
            end_idx = min(start_idx + gpus_per_node, len(worker_ids))
            hierarchy[f"node-{node_id}"] = worker_ids[start_idx:end_idx]
        
        return hierarchy
    
    def weighted_average(self, gradients: Dict[str, Dict[str, torch.Tensor]], 
                        weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Compute weighted average of gradients"""
        if not gradients:
            return {}
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            return {}
        
        normalized_weights = {wid: w / total_weight for wid, w in weights.items()}
        
        # Compute weighted average
        param_names = list(next(iter(gradients.values())).keys())
        weighted_gradients = {}
        
        for param_name in param_names:
            weighted_sum = torch.zeros_like(gradients[next(iter(gradients.keys()))][param_name])
            
            for worker_id, weight in normalized_weights.items():
                if worker_id in gradients:
                    weighted_sum += weight * gradients[worker_id][param_name]
            
            weighted_gradients[param_name] = weighted_sum
        
        return weighted_gradients
    
    def compress_gradients(self, gradients: Dict[str, torch.Tensor], 
                         compression_ratio: float = 0.1) -> Dict[str, torch.Tensor]:
        """Apply gradient compression using top-k sparsification"""
        compressed = {}
        
        for param_name, tensor in gradients.items():
            # Top-k compression
            flat_tensor = tensor.flatten()
            k = max(1, int(len(flat_tensor) * compression_ratio))
            
            # Get top-k values by magnitude
            _, indices = torch.topk(torch.abs(flat_tensor), k)
            
            # Create compressed tensor containing only top-k values
            # This actually reduces the tensor size by storing only the selected values
            compressed_values = flat_tensor[indices]
            
            compressed[param_name] = compressed_values
        
        return compressed
    
    def compute_async_update(self, gradients: Dict[str, torch.Tensor], 
                           previous_update: Dict[str, torch.Tensor], 
                           momentum: float) -> Dict[str, torch.Tensor]:
        """Compute asynchronous update with momentum"""
        update = {}
        
        for param_name, grad in gradients.items():
            if param_name in previous_update:
                # Apply momentum
                update[param_name] = momentum * previous_update[param_name] + (1 - momentum) * grad
            else:
                update[param_name] = grad
        
        return update
    
    def _compute_worker_weights(self, worker_gradients: Dict[str, Dict[str, torch.Tensor]], 
                              **kwargs) -> Dict[str, float]:
        """Compute weights for each worker based on various factors"""
        weights = {}
        
        # Base weight (equal for all workers)
        base_weight = 1.0 / len(worker_gradients)
        
        for worker_id, gradients in worker_gradients.items():
            weight = base_weight
            
            # Adjust based on gradient norm (smaller norms get higher weight)
            grad_norm = sum(torch.norm(tensor).item() for tensor in gradients.values())
            if grad_norm > 0:
                weight *= 1.0 / (1.0 + grad_norm * 0.1)
            
            # Adjust based on compute capacity (if available)
            compute_capacity = kwargs.get("compute_capacities", {}).get(worker_id, 1.0)
            weight *= compute_capacity
            
            weights[worker_id] = weight
        
        return weights
    
    def quantize_gradients(self, gradients: Dict[str, torch.Tensor], 
                          bits: int = 8) -> Dict[str, torch.Tensor]:
        """Quantize gradients to reduce communication overhead"""
        quantized = {}
        
        for param_name, tensor in gradients.items():
            # Simple uniform quantization
            tensor_min = tensor.min()
            tensor_max = tensor.max()
            
            if tensor_min == tensor_max:
                quantized[param_name] = tensor
                continue
            
            # Quantize to specified bits
            levels = 2 ** bits
            scale = (tensor_max - tensor_min) / (levels - 1)
            
            # Quantize and dequantize
            quantized_tensor = torch.round((tensor - tensor_min) / scale)
            dequantized_tensor = quantized_tensor * scale + tensor_min
            
            quantized[param_name] = dequantized_tensor
        
        return quantized