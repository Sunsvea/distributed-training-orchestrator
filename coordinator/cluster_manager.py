import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from communication.cluster_pb2 import *

@dataclass
class ClusterManager:
    coordinator_nodes: List[str]
    worker_nodes: Dict[str, NodeInfo] = field(default_factory=dict)
    leader_id: Optional[str] = None
    cluster_health: ClusterHealth = field(default_factory=lambda: ClusterHealth())
    
    def add_worker_node(self, node_info: NodeInfo) -> bool:
        if node_info.node_id in self.worker_nodes:
            return False
        
        node_info.last_seen = int(time.time())
        self.worker_nodes[node_info.node_id] = node_info
        self._update_cluster_health()
        return True
    
    def remove_worker_node(self, node_id: str) -> bool:
        if node_id not in self.worker_nodes:
            return False
        
        del self.worker_nodes[node_id]
        self._update_cluster_health()
        return True
    
    def update_node_status(self, node_id: str, status: NodeStatus):
        if node_id in self.worker_nodes:
            self.worker_nodes[node_id].status = status
            self.worker_nodes[node_id].last_seen = int(time.time())
            self._update_cluster_health()
    
    def get_node_info(self, node_id: str) -> Optional[NodeInfo]:
        return self.worker_nodes.get(node_id)
    
    def get_all_nodes(self) -> List[NodeInfo]:
        return list(self.worker_nodes.values())
    
    def get_available_workers(self) -> List[str]:
        available = []
        for node_id, node_info in self.worker_nodes.items():
            if node_info.status in [NodeStatus.ACTIVE, NodeStatus.TRAINING]:
                available.append(node_id)
        return available
    
    def get_cluster_health(self) -> ClusterHealth:
        return self.cluster_health
    
    def _update_cluster_health(self):
        total_nodes = len(self.worker_nodes)
        healthy_nodes = 0
        failed_nodes = 0
        
        for node_info in self.worker_nodes.values():
            if node_info.status == NodeStatus.FAILED:
                failed_nodes += 1
            elif node_info.status in [NodeStatus.ACTIVE, NodeStatus.TRAINING]:
                healthy_nodes += 1
        
        self.cluster_health = ClusterHealth(
            total_nodes=total_nodes,
            healthy_nodes=healthy_nodes,
            failed_nodes=failed_nodes,
            consensus_healthy=self.leader_id is not None,
            training_active=self._is_training_active()
        )
    
    def _is_training_active(self) -> bool:
        return any(
            node.status == NodeStatus.TRAINING 
            for node in self.worker_nodes.values()
        )
    
    def handle_heartbeat(self, node_id: str, status: NodeStatus) -> bool:
        if node_id not in self.worker_nodes:
            return False
        
        self.worker_nodes[node_id].status = status
        self.worker_nodes[node_id].last_seen = int(time.time())
        self._update_cluster_health()
        return True
    
    def check_failed_nodes(self, timeout_seconds: int = 30) -> List[str]:
        current_time = int(time.time())
        failed_nodes = []
        
        for node_id, node_info in self.worker_nodes.items():
            if current_time - node_info.last_seen > timeout_seconds:
                if node_info.status != NodeStatus.FAILED:
                    self.update_node_status(node_id, NodeStatus.FAILED)
                    failed_nodes.append(node_id)
        
        return failed_nodes
    
    def get_training_nodes(self) -> List[NodeInfo]:
        return [
            node for node in self.worker_nodes.values()
            if node.status == NodeStatus.TRAINING
        ]
    
    def assign_training_nodes(self, required_nodes: int) -> List[str]:
        available = self.get_available_workers()
        
        if len(available) < required_nodes:
            return []
        
        selected_nodes = available[:required_nodes]
        for node_id in selected_nodes:
            self.update_node_status(node_id, NodeStatus.TRAINING)
        
        return selected_nodes
    
    def release_training_nodes(self, node_ids: List[str]):
        for node_id in node_ids:
            if node_id in self.worker_nodes:
                self.update_node_status(node_id, NodeStatus.ACTIVE)