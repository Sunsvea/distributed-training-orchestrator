import torch
import numpy as np
from typing import Dict, List, Any
from communication.cluster_pb2 import TensorData

class GradientSynchronizer:
    def __init__(self, strategy: str, node_id: str):
        self.strategy = strategy
        self.node_id = node_id
        
        # Supported strategies
        self.supported_strategies = ["allreduce", "parameter_server", "custom"]
        if strategy not in self.supported_strategies:
            raise ValueError(f"Unsupported sync strategy: {strategy}")
    
    def serialize_gradients(self, gradients: Dict[str, torch.Tensor]) -> List[TensorData]:
        tensor_data_list = []
        
        for name, tensor in gradients.items():
            # Convert tensor to bytes
            tensor_np = tensor.cpu().numpy()
            tensor_bytes = tensor_np.tobytes()
            
            tensor_data = TensorData(
                name=name,
                shape=list(tensor.shape),
                data=tensor_bytes,
                dtype=str(tensor.dtype)
            )
            tensor_data_list.append(tensor_data)
        
        return tensor_data_list
    
    def deserialize_gradients(self, tensor_data_list: List[TensorData]) -> Dict[str, torch.Tensor]:
        gradients = {}
        
        for tensor_data in tensor_data_list:
            # Convert bytes back to tensor
            tensor_np = np.frombuffer(tensor_data.data, dtype=self._get_numpy_dtype(tensor_data.dtype))
            tensor_np = tensor_np.reshape(tensor_data.shape)
            tensor = torch.from_numpy(tensor_np)
            
            gradients[tensor_data.name] = tensor
        
        return gradients
    
    def _get_numpy_dtype(self, dtype_str: str) -> np.dtype:
        dtype_mapping = {
            "torch.float32": np.float32,
            "torch.float64": np.float64,
            "torch.int32": np.int32,
            "torch.int64": np.int64,
            "torch.float16": np.float16
        }
        return dtype_mapping.get(dtype_str, np.float32)
    
    def average_gradients(self, gradients_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not gradients_list:
            return {}
        
        if len(gradients_list) == 1:
            return gradients_list[0]
        
        # Initialize averaged gradients with zeros
        averaged_gradients = {}
        param_names = gradients_list[0].keys()
        
        for param_name in param_names:
            # Stack all gradients for this parameter
            param_gradients = [grads[param_name] for grads in gradients_list]
            stacked_gradients = torch.stack(param_gradients, dim=0)
            
            # Compute average
            averaged_gradients[param_name] = torch.mean(stacked_gradients, dim=0)
        
        return averaged_gradients
    
    def compute_gradient_norm(self, gradients: Dict[str, torch.Tensor]) -> torch.Tensor:
        total_norm = 0.0
        
        for grad in gradients.values():
            param_norm = grad.norm(dtype=torch.float32)
            total_norm += param_norm.item() ** 2
        
        return torch.sqrt(torch.tensor(total_norm))
    
    def clip_gradients(self, gradients: Dict[str, torch.Tensor], max_norm: float) -> Dict[str, torch.Tensor]:
        total_norm = self.compute_gradient_norm(gradients)
        
        if total_norm <= max_norm:
            return gradients
        
        # Clip gradients
        clip_coef = max_norm / (total_norm + 1e-6)
        clipped_gradients = {}
        
        for name, grad in gradients.items():
            clipped_gradients[name] = grad * clip_coef
        
        return clipped_gradients
    
    def apply_compression(self, gradients: Dict[str, torch.Tensor], compression_rate: float = 0.1) -> Dict[str, torch.Tensor]:
        if self.strategy != "custom":
            return gradients
        
        compressed_gradients = {}
        
        for name, grad in gradients.items():
            # Simple top-k compression
            flat_grad = grad.flatten()
            k = max(1, int(len(flat_grad) * compression_rate))
            
            # Get top-k values
            _, indices = torch.topk(torch.abs(flat_grad), k)
            
            # Create sparse gradient
            compressed_grad = torch.zeros_like(flat_grad)
            compressed_grad[indices] = flat_grad[indices]
            
            compressed_gradients[name] = compressed_grad.reshape(grad.shape)
        
        return compressed_gradients
    
    def aggregate_gradients(self, local_gradients: Dict[str, torch.Tensor], 
                          remote_gradients: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if self.strategy == "allreduce":
            return self._allreduce_aggregation(local_gradients, remote_gradients)
        elif self.strategy == "parameter_server":
            return self._parameter_server_aggregation(local_gradients, remote_gradients)
        elif self.strategy == "custom":
            return self._custom_aggregation(local_gradients, remote_gradients)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _allreduce_aggregation(self, local_gradients: Dict[str, torch.Tensor], 
                             remote_gradients: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # AllReduce: Average all gradients
        all_gradients = [local_gradients] + remote_gradients
        return self.average_gradients(all_gradients)
    
    def _parameter_server_aggregation(self, local_gradients: Dict[str, torch.Tensor], 
                                    remote_gradients: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Parameter Server: Simple averaging (in real implementation, this would be more complex)
        all_gradients = [local_gradients] + remote_gradients
        return self.average_gradients(all_gradients)
    
    def _custom_aggregation(self, local_gradients: Dict[str, torch.Tensor], 
                          remote_gradients: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Custom strategy: Weighted averaging based on gradient norms
        all_gradients = [local_gradients] + remote_gradients
        
        # Compute weights based on gradient norms
        weights = []
        for gradients in all_gradients:
            norm = self.compute_gradient_norm(gradients)
            weight = 1.0 / (norm + 1e-6)  # Inverse weighting
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Weighted averaging
        aggregated_gradients = {}
        param_names = local_gradients.keys()
        
        for param_name in param_names:
            weighted_sum = torch.zeros_like(local_gradients[param_name])
            
            for i, gradients in enumerate(all_gradients):
                weighted_sum += weights[i] * gradients[param_name]
            
            aggregated_gradients[param_name] = weighted_sum
        
        return aggregated_gradients
    
    def get_sync_info(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "node_id": self.node_id,
            "supported_strategies": self.supported_strategies
        }