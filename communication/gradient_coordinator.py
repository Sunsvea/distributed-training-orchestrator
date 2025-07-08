import asyncio
import torch
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np

from worker.distributed_strategies import AllReduceStrategy, ParameterServerStrategy, CustomStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SyncResult:
    """Result of gradient synchronization"""
    success: bool
    averaged_gradients: Dict[str, torch.Tensor]
    sync_time: float
    strategy_used: str
    worker_count: int
    warnings: List[str]
    metrics: Dict[str, Any]

class GradientCoordinator:
    """Coordinates gradient synchronization across workers"""
    
    def __init__(self, strategy: str, worker_nodes: List[str], **kwargs):
        self.strategy = strategy
        self.worker_nodes = worker_nodes
        self.strategy_impl = self._create_strategy_impl(strategy, **kwargs)
        
        # Performance tracking
        self.sync_history = []
        self.convergence_threshold = kwargs.get("convergence_threshold", 1e-6)
        self.max_sync_time = kwargs.get("max_sync_time", 30.0)
        
        # Fault tolerance
        self.min_workers = kwargs.get("min_workers", len(worker_nodes) // 2)
        self.failed_workers = set()
        
        # Adaptive features
        self.adaptive_strategy = kwargs.get("adaptive_strategy", True)
        self.strategy_performance = {}
    
    def _create_strategy_impl(self, strategy: str, **kwargs):
        """Create strategy implementation"""
        if strategy == "allreduce":
            return AllReduceStrategy(
                node_id="coordinator",
                num_workers=len(self.worker_nodes),
                **kwargs
            )
        elif strategy == "parameter_server":
            server_nodes = kwargs.get("server_nodes", ["ps-0", "ps-1"])
            return ParameterServerStrategy(
                node_id="coordinator",
                server_nodes=server_nodes,
                **kwargs
            )
        elif strategy == "custom":
            return CustomStrategy(
                node_id="coordinator",
                topology=kwargs.get("topology", "hierarchical"),
                **kwargs
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    async def coordinate_sync(self, worker_gradients: Dict[str, Dict[str, torch.Tensor]], 
                            iteration: int, failed_workers: Optional[List[str]] = None,
                            **kwargs) -> Dict[str, Any]:
        """Coordinate gradient synchronization"""
        start_time = time.time()
        warnings = []
        
        # Update failed workers
        if failed_workers:
            self.failed_workers.update(failed_workers)
            warnings.append(f"Workers {failed_workers} marked as failed")
        
        # Remove failed workers from gradients
        active_gradients = {
            wid: grads for wid, grads in worker_gradients.items() 
            if wid not in self.failed_workers
        }
        
        # Check minimum worker threshold
        if len(active_gradients) < self.min_workers:
            return {
                "success": False,
                "averaged_gradients": {},
                "sync_time": time.time() - start_time,
                "strategy_used": self.strategy,
                "worker_count": len(active_gradients),
                "warnings": warnings + ["Insufficient workers for synchronization"],
                "metrics": {}
            }
        
        # Validate gradient consistency
        validation_result = self._validate_gradients(active_gradients)
        if not validation_result["valid"]:
            warnings.extend(validation_result["warnings"])
        
        # Perform synchronization
        try:
            # Get reference gradients (from first worker)
            reference_gradients = next(iter(active_gradients.values()))
            
            # Synchronize using selected strategy
            averaged_gradients = self.strategy_impl.synchronize_gradients(
                reference_gradients,
                active_gradients,
                iteration=iteration,
                **kwargs
            )
            
            # Apply post-processing
            averaged_gradients = self._post_process_gradients(averaged_gradients, iteration)
            
            # Update sync history
            sync_time = time.time() - start_time
            self._update_sync_history(averaged_gradients, sync_time, len(active_gradients))
            
            # Check for convergence
            if self._check_convergence():
                warnings.append("Gradient convergence detected")
            
            return {
                "success": True,
                "averaged_gradients": averaged_gradients,
                "sync_time": sync_time,
                "strategy_used": self.strategy,
                "worker_count": len(active_gradients),
                "warnings": warnings,
                "metrics": self.strategy_impl.get_metrics()
            }
            
        except Exception as e:
            logger.error(f"Gradient synchronization failed: {e}")
            return {
                "success": False,
                "averaged_gradients": {},
                "sync_time": time.time() - start_time,
                "strategy_used": self.strategy,
                "worker_count": len(active_gradients),
                "warnings": warnings + [f"Sync failed: {str(e)}"],
                "metrics": {}
            }
    
    async def select_optimal_strategy(self, conditions: Dict[str, Any]) -> str:
        """Select optimal strategy based on current conditions"""
        return self._select_optimal_strategy_sync(conditions)
    
    def _select_optimal_strategy_sync(self, conditions: Dict[str, Any]) -> str:
        """Select optimal strategy based on current conditions (sync version)"""
        if not self.adaptive_strategy:
            return self.strategy
        
        network_bandwidth = conditions.get("network_bandwidth", "medium")
        worker_count = conditions.get("worker_count", len(self.worker_nodes))
        model_size = conditions.get("model_size", "medium")
        
        # Decision logic for strategy selection
        if model_size == "large" and network_bandwidth == "low":
            return "parameter_server"
        elif worker_count > 16:
            return "allreduce"
        elif network_bandwidth == "high" and worker_count <= 8:
            return "custom"
        else:
            return "allreduce"
    
    def detect_convergence(self, gradient_history: List[Dict[str, torch.Tensor]]) -> bool:
        """Detect if gradients are converging"""
        if len(gradient_history) < 5:
            return False
        
        # Calculate gradient norms for recent history
        recent_norms = []
        for gradients in gradient_history[-5:]:
            total_norm = sum(torch.norm(tensor).item() for tensor in gradients.values())
            recent_norms.append(total_norm)
        
        # Check if gradient norms are decreasing
        if len(recent_norms) >= 2:
            trend = np.polyfit(range(len(recent_norms)), recent_norms, 1)[0]
            return bool(trend < -self.convergence_threshold)
        
        return False
    
    def compute_gradient_statistics(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compute comprehensive gradient statistics"""
        if not gradients:
            return {}
        
        stats = {}
        
        # Gradient norm
        total_norm = sum(torch.norm(tensor).item() for tensor in gradients.values())
        stats["gradient_norm"] = total_norm
        
        # Parameter count
        param_count = sum(tensor.numel() for tensor in gradients.values())
        stats["parameter_count"] = param_count
        
        # Sparsity
        total_zeros = sum((tensor == 0).sum().item() for tensor in gradients.values())
        stats["sparsity"] = total_zeros / param_count if param_count > 0 else 0
        
        # Distribution statistics
        all_values = torch.cat([tensor.flatten() for tensor in gradients.values()])
        stats["distribution"] = {
            "mean": all_values.mean().item(),
            "std": all_values.std().item(),
            "min": all_values.min().item(),
            "max": all_values.max().item(),
            "median": all_values.median().item()
        }
        
        # Per-parameter statistics
        stats["per_parameter"] = {}
        for param_name, tensor in gradients.items():
            stats["per_parameter"][param_name] = {
                "shape": list(tensor.shape),
                "norm": torch.norm(tensor).item(),
                "mean": tensor.mean().item(),
                "std": tensor.std().item()
            }
        
        return stats
    
    def _validate_gradients(self, worker_gradients: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """Validate gradient consistency across workers"""
        if not worker_gradients:
            return {"valid": False, "warnings": ["No gradients to validate"]}
        
        warnings = []
        
        # Check parameter consistency
        param_names_list = [set(grads.keys()) for grads in worker_gradients.values()]
        if len(set(frozenset(names) for names in param_names_list)) > 1:
            warnings.append("Parameter names inconsistent across workers")
        
        # Check shape consistency
        reference_shapes = next(iter(worker_gradients.values()))
        for worker_id, gradients in worker_gradients.items():
            for param_name, tensor in gradients.items():
                if param_name in reference_shapes:
                    if tensor.shape != reference_shapes[param_name].shape:
                        warnings.append(f"Shape mismatch for {param_name} in worker {worker_id}")
        
        # Check for NaN or Inf values
        for worker_id, gradients in worker_gradients.items():
            for param_name, tensor in gradients.items():
                if torch.isnan(tensor).any():
                    warnings.append(f"NaN values in {param_name} from worker {worker_id}")
                if torch.isinf(tensor).any():
                    warnings.append(f"Inf values in {param_name} from worker {worker_id}")
        
        return {"valid": len(warnings) == 0, "warnings": warnings}
    
    def _post_process_gradients(self, gradients: Dict[str, torch.Tensor], iteration: int) -> Dict[str, torch.Tensor]:
        """Post-process synchronized gradients"""
        processed = {}
        
        for param_name, tensor in gradients.items():
            # Gradient clipping
            max_norm = 1.0
            tensor_norm = torch.norm(tensor)
            if tensor_norm > max_norm:
                tensor = tensor * (max_norm / tensor_norm)
            
            # Numerical stability
            tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
            tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
            
            processed[param_name] = tensor
        
        return processed
    
    def _update_sync_history(self, gradients: Dict[str, torch.Tensor], sync_time: float, worker_count: int):
        """Update synchronization history"""
        # Keep only recent history
        max_history = 100
        
        sync_record = {
            "timestamp": time.time(),
            "gradients": gradients,
            "sync_time": sync_time,
            "worker_count": worker_count,
            "strategy": self.strategy
        }
        
        self.sync_history.append(sync_record)
        
        if len(self.sync_history) > max_history:
            self.sync_history = self.sync_history[-max_history:]
    
    def _check_convergence(self) -> bool:
        """Check if training is converging"""
        if len(self.sync_history) < 10:
            return False
        
        # Extract recent gradient norms
        recent_gradients = [record["gradients"] for record in self.sync_history[-10:]]
        return self.detect_convergence(recent_gradients)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the coordinator"""
        if not self.sync_history:
            return {}
        
        sync_times = [record["sync_time"] for record in self.sync_history]
        worker_counts = [record["worker_count"] for record in self.sync_history]
        
        return {
            "total_syncs": len(self.sync_history),
            "average_sync_time": np.mean(sync_times),
            "min_sync_time": np.min(sync_times),
            "max_sync_time": np.max(sync_times),
            "average_worker_count": np.mean(worker_counts),
            "strategy_used": self.strategy,
            "failed_workers": list(self.failed_workers),
            "convergence_detected": self._check_convergence()
        }
    
    def reset_failed_workers(self):
        """Reset failed workers set"""
        self.failed_workers.clear()
    
    def add_worker(self, worker_id: str):
        """Add a new worker to the coordinator"""
        if worker_id not in self.worker_nodes:
            self.worker_nodes.append(worker_id)
            self.failed_workers.discard(worker_id)
            logger.info(f"Added worker {worker_id} to coordinator")
    
    def remove_worker(self, worker_id: str):
        """Remove a worker from the coordinator"""
        if worker_id in self.worker_nodes:
            self.worker_nodes.remove(worker_id)
            self.failed_workers.add(worker_id)
            logger.info(f"Removed worker {worker_id} from coordinator")
    
    async def benchmark_strategies(self, test_gradients: Dict[str, Dict[str, torch.Tensor]], 
                                 iterations: int = 5) -> Dict[str, Dict[str, float]]:
        """Benchmark different strategies"""
        strategies = ["allreduce", "parameter_server", "custom"]
        results = {}
        
        for strategy in strategies:
            # Create temporary strategy implementation
            temp_strategy = self._create_strategy_impl(strategy)
            
            sync_times = []
            
            for i in range(iterations):
                start_time = time.time()
                
                # Perform synchronization
                reference_gradients = next(iter(test_gradients.values()))
                temp_strategy.synchronize_gradients(reference_gradients, test_gradients)
                
                sync_times.append(time.time() - start_time)
            
            results[strategy] = {
                "average_time": np.mean(sync_times),
                "min_time": np.min(sync_times),
                "max_time": np.max(sync_times),
                "std_time": np.std(sync_times)
            }
        
        return results
    
    def switch_strategy(self, new_strategy: str, **kwargs):
        """Switch to a different synchronization strategy"""
        if new_strategy != self.strategy:
            self.strategy = new_strategy
            self.strategy_impl = self._create_strategy_impl(new_strategy, **kwargs)
            logger.info(f"Switched to strategy: {new_strategy}")
    
    def get_strategy_recommendations(self, current_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Get strategy recommendations based on current conditions"""
        recommendations = {}
        
        # Analyze current performance
        recent_performance = self.get_performance_metrics()
        
        # Check if current strategy is performing well
        if recent_performance.get("average_sync_time", 0) > self.max_sync_time:
            recommendations["action"] = "switch_strategy"
            recommendations["reason"] = "Current strategy is too slow"
            # Use sync version since this is not an async function
            recommendations["suggested_strategy"] = self._select_optimal_strategy_sync(current_conditions)
        else:
            recommendations["action"] = "continue"
            recommendations["reason"] = "Current strategy performing well"
            recommendations["suggested_strategy"] = self.strategy
        
        # Additional optimizations
        optimizations = []
        
        if current_conditions.get("network_bandwidth") == "low":
            optimizations.append("Enable gradient compression")
        
        if current_conditions.get("worker_count", 0) > 8:
            optimizations.append("Consider parameter server strategy")
        
        if recent_performance.get("convergence_detected", False):
            optimizations.append("Reduce synchronization frequency")
        
        recommendations["optimizations"] = optimizations
        
        return recommendations