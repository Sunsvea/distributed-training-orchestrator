import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple, Any
from communication.cluster_pb2 import NodeStatus
from worker.models import create_model
from worker.gradient_sync import GradientSynchronizer

class TrainingWorker:
    def __init__(self, node_id: str, coordinator_address: str, port: int):
        self.node_id = node_id
        self.coordinator_address = coordinator_address
        self.port = port
        self.status = NodeStatus.IDLE
        
        # Training state
        self.current_training_id: Optional[str] = None
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.loss_function: Optional[nn.Module] = None
        self.gradient_synchronizer: Optional[GradientSynchronizer] = None
        
        # Training config
        self.batch_size: int = 32
        self.learning_rate: float = 0.001
        self.epochs: int = 10
        self.current_epoch: int = 0
        self.current_iteration: int = 0
        
        # Metrics
        self.training_metrics = {
            "loss": [],
            "accuracy": [],
            "iteration": [],
            "epoch": []
        }
    
    def start_training(self, training_id: str, model_config: Dict[str, Any], training_config: Dict[str, Any]) -> bool:
        if self.status == NodeStatus.TRAINING:
            return False
        
        try:
            # Store training configuration
            self.current_training_id = training_id
            self.batch_size = training_config.get("batch_size", 32)
            self.learning_rate = training_config.get("learning_rate", 0.001)
            self.epochs = training_config.get("epochs", 10)
            
            # Create model
            self.model = create_model(model_config)
            if self.model is None:
                return False
            
            # Create optimizer
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate
            )
            
            # Create loss function
            self.loss_function = nn.CrossEntropyLoss()
            
            # Create gradient synchronizer
            sync_strategy = training_config.get("sync_strategy", "allreduce")
            self.gradient_synchronizer = GradientSynchronizer(sync_strategy, self.node_id)
            
            # Update status
            self.status = NodeStatus.TRAINING
            self.current_epoch = 0
            self.current_iteration = 0
            
            # Reset metrics
            self.training_metrics = {
                "loss": [],
                "accuracy": [],
                "iteration": [],
                "epoch": []
            }
            
            return True
            
        except Exception as e:
            print(f"Error starting training: {e}")
            return False
    
    def stop_training(self, training_id: str) -> bool:
        if self.current_training_id != training_id:
            return False
        
        self.current_training_id = None
        self.model = None
        self.optimizer = None
        self.loss_function = None
        self.gradient_synchronizer = None
        self.status = NodeStatus.IDLE
        
        return True
    
    def compute_gradients(self, batch_data: torch.Tensor, batch_labels: torch.Tensor) -> Tuple[float, Dict[str, torch.Tensor]]:
        if self.model is None or self.optimizer is None or self.loss_function is None:
            raise ValueError("Training not started")
        
        # Set model to training mode
        self.model.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(batch_data)
        loss = self.loss_function(outputs, batch_labels)
        
        # Backward pass
        loss.backward()
        
        # Extract gradients
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone().detach()
        
        self.current_iteration += 1
        
        return loss.item(), gradients
    
    def apply_gradients(self, gradients: Dict[str, torch.Tensor]):
        if self.model is None or self.optimizer is None:
            raise ValueError("Training not started")
        
        # Apply gradients to model parameters
        for name, param in self.model.named_parameters():
            if name in gradients:
                param.grad = gradients[name].clone()
        
        # Update parameters
        self.optimizer.step()
    
    def get_model_state(self) -> Dict[str, torch.Tensor]:
        if self.model is None:
            return {}
        
        state = {}
        for name, param in self.model.named_parameters():
            state[name] = param.clone().detach()
        
        return state
    
    def set_model_state(self, state: Dict[str, torch.Tensor]):
        if self.model is None:
            raise ValueError("Model not initialized")
        
        for name, param in self.model.named_parameters():
            if name in state:
                param.data = state[name].clone()
    
    def evaluate_batch(self, batch_data: torch.Tensor, batch_labels: torch.Tensor) -> Tuple[float, float]:
        if self.model is None or self.loss_function is None:
            raise ValueError("Training not started")
        
        # Set model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(batch_data)
            loss = self.loss_function(outputs, batch_labels)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == batch_labels).sum().item()
            accuracy = correct / batch_labels.size(0)
        
        return loss.item(), accuracy
    
    def record_metrics(self, loss: float, accuracy: float):
        self.training_metrics["loss"].append(loss)
        self.training_metrics["accuracy"].append(accuracy)
        self.training_metrics["iteration"].append(self.current_iteration)
        self.training_metrics["epoch"].append(self.current_epoch)
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "training_id": self.current_training_id,
            "current_epoch": self.current_epoch,
            "current_iteration": self.current_iteration,
            "metrics": self.training_metrics.copy()
        }
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status,
            "training_id": self.current_training_id,
            "current_epoch": self.current_epoch,
            "current_iteration": self.current_iteration,
            "coordinator_address": self.coordinator_address
        }