import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from coordinator.raft_coordinator import RaftCoordinator
from coordinator.cluster_manager import ClusterManager
from communication.cluster_pb2 import *

class TestRaftCoordinator:
    
    @pytest.fixture
    def coordinator(self):
        return RaftCoordinator(
            node_id="coordinator-1",
            address="localhost",
            port=8001,
            cluster_nodes=["localhost:8001", "localhost:8002", "localhost:8003"]
        )
    
    def test_coordinator_initialization(self, coordinator):
        assert coordinator.node_id == "coordinator-1"
        assert coordinator.address == "localhost"
        assert coordinator.port == 8001
        assert coordinator.state == "follower"
        assert coordinator.current_term == 0
        assert coordinator.voted_for is None
        assert coordinator.log == []
        assert coordinator.commit_index == 0
        assert coordinator.last_applied == 0
    
    def test_start_election(self, coordinator):
        coordinator.start_election()
        assert coordinator.state == "candidate"
        assert coordinator.current_term == 1
        assert coordinator.voted_for == coordinator.node_id
        assert coordinator.votes_received == 1
    
    def test_request_vote_valid(self, coordinator):
        vote_request = VoteRequest(
            term=1,
            candidate_id="coordinator-2",
            last_log_index=0,
            last_log_term=0
        )
        
        response = coordinator.handle_vote_request(vote_request)
        assert response.vote_granted == True
        assert response.term == 1
        assert coordinator.voted_for == "coordinator-2"
        assert coordinator.current_term == 1
    
    def test_request_vote_invalid_term(self, coordinator):
        coordinator.current_term = 2
        vote_request = VoteRequest(
            term=1,
            candidate_id="coordinator-2",
            last_log_index=0,
            last_log_term=0
        )
        
        response = coordinator.handle_vote_request(vote_request)
        assert response.vote_granted == False
        assert response.term == 2
    
    def test_request_vote_already_voted(self, coordinator):
        coordinator.current_term = 1
        coordinator.voted_for = "coordinator-3"
        vote_request = VoteRequest(
            term=1,
            candidate_id="coordinator-2",
            last_log_index=0,
            last_log_term=0
        )
        
        response = coordinator.handle_vote_request(vote_request)
        assert response.vote_granted == False
    
    def test_append_entries_heartbeat(self, coordinator):
        append_request = AppendEntriesRequest(
            term=1,
            leader_id="coordinator-2",
            prev_log_index=0,
            prev_log_term=0,
            entries=[],
            leader_commit=0
        )
        
        response = coordinator.handle_append_entries(append_request)
        assert response.success == True
        assert response.term == 1
        assert coordinator.state == "follower"
        assert coordinator.current_term == 1
        assert coordinator.leader_id == "coordinator-2"
    
    def test_append_entries_with_entries(self, coordinator):
        new_entry = LogEntry(
            term=1,
            command="START_TRAINING",
            data=b"model_config"
        )
        
        append_request = AppendEntriesRequest(
            term=1,
            leader_id="coordinator-2",
            prev_log_index=0,
            prev_log_term=0,
            entries=[new_entry],
            leader_commit=0
        )
        
        response = coordinator.handle_append_entries(append_request)
        assert response.success == True
        assert len(coordinator.log) == 1
        assert coordinator.log[0].command == "START_TRAINING"
    
    def test_become_leader(self, coordinator):
        coordinator.state = "candidate"
        coordinator.current_term = 1
        coordinator.votes_received = 2  # Majority of 3
        
        coordinator.become_leader()
        assert coordinator.state == "leader"
        assert coordinator.leader_id == coordinator.node_id
        assert len(coordinator.next_index) == 2  # Other nodes
        assert len(coordinator.match_index) == 2

class TestClusterManager:
    
    @pytest.fixture
    def cluster_manager(self):
        return ClusterManager(coordinator_nodes=["localhost:8001", "localhost:8002"])
    
    def test_cluster_manager_initialization(self, cluster_manager):
        assert len(cluster_manager.coordinator_nodes) == 2
        assert len(cluster_manager.worker_nodes) == 0
        assert cluster_manager.leader_id is None
        assert cluster_manager.cluster_health.total_nodes == 0
    
    def test_add_worker_node(self, cluster_manager):
        worker_info = NodeInfo(
            node_id="worker-1",
            node_type="worker",
            address="localhost",
            port=9001,
            status=NodeStatus.JOINING
        )
        
        success = cluster_manager.add_worker_node(worker_info)
        assert success == True
        assert len(cluster_manager.worker_nodes) == 1
        assert "worker-1" in cluster_manager.worker_nodes
    
    def test_remove_worker_node(self, cluster_manager):
        worker_info = NodeInfo(
            node_id="worker-1",
            node_type="worker",
            address="localhost",
            port=9001,
            status=NodeStatus.ACTIVE
        )
        
        cluster_manager.add_worker_node(worker_info)
        success = cluster_manager.remove_worker_node("worker-1")
        assert success == True
        assert len(cluster_manager.worker_nodes) == 0
    
    def test_update_node_status(self, cluster_manager):
        worker_info = NodeInfo(
            node_id="worker-1",
            node_type="worker",
            address="localhost",
            port=9001,
            status=NodeStatus.JOINING
        )
        
        cluster_manager.add_worker_node(worker_info)
        cluster_manager.update_node_status("worker-1", NodeStatus.ACTIVE)
        assert cluster_manager.worker_nodes["worker-1"].status == NodeStatus.ACTIVE
    
    def test_get_cluster_health(self, cluster_manager):
        # Add some nodes
        for i in range(3):
            worker_info = NodeInfo(
                node_id=f"worker-{i}",
                node_type="worker",
                address="localhost",
                port=9000 + i,
                status=NodeStatus.ACTIVE
            )
            cluster_manager.add_worker_node(worker_info)
        
        # Mark one as failed
        cluster_manager.update_node_status("worker-2", NodeStatus.FAILED)
        
        health = cluster_manager.get_cluster_health()
        assert health.total_nodes == 3
        assert health.healthy_nodes == 2
        assert health.failed_nodes == 1
    
    def test_get_available_workers(self, cluster_manager):
        # Add workers with different statuses
        statuses = [NodeStatus.ACTIVE, NodeStatus.TRAINING, NodeStatus.FAILED]
        for i, status in enumerate(statuses):
            worker_info = NodeInfo(
                node_id=f"worker-{i}",
                node_type="worker",
                address="localhost",
                port=9000 + i,
                status=status
            )
            cluster_manager.add_worker_node(worker_info)
        
        available = cluster_manager.get_available_workers()
        assert len(available) == 2  # ACTIVE and TRAINING
        assert "worker-0" in available
        assert "worker-1" in available
        assert "worker-2" not in available

@pytest.mark.asyncio
class TestCoordinatorIntegration:
    
    async def test_coordinator_election_process(self):
        # Create 3 coordinators
        coordinators = []
        for i in range(3):
            coord = RaftCoordinator(
                node_id=f"coordinator-{i}",
                address="localhost",
                port=8000 + i,
                cluster_nodes=[f"localhost:800{j}" for j in range(3)]
            )
            coordinators.append(coord)
        
        # Start election on first coordinator
        coordinators[0].start_election()
        
        # Simulate vote requests to other coordinators
        for i in range(1, 3):
            vote_request = VoteRequest(
                term=1,
                candidate_id="coordinator-0",
                last_log_index=0,
                last_log_term=0
            )
            response = coordinators[i].handle_vote_request(vote_request)
            if response.vote_granted:
                coordinators[0].votes_received += 1
        
        # Check if coordinator-0 becomes leader
        if coordinators[0].votes_received > len(coordinators) // 2:
            coordinators[0].become_leader()
        
        assert coordinators[0].state == "leader"
        assert coordinators[1].state == "follower"
        assert coordinators[2].state == "follower"
    
    async def test_log_replication(self):
        # Create leader and follower
        leader = RaftCoordinator(
            node_id="leader",
            address="localhost",
            port=8000,
            cluster_nodes=["localhost:8000", "localhost:8001"]
        )
        leader.state = "leader"
        leader.current_term = 1
        
        follower = RaftCoordinator(
            node_id="follower",
            address="localhost",
            port=8001,
            cluster_nodes=["localhost:8000", "localhost:8001"]
        )
        follower.current_term = 1
        
        # Leader creates log entry
        new_entry = LogEntry(
            term=1,
            command="START_TRAINING",
            data=b"model_config"
        )
        leader.log.append(new_entry)
        
        # Send append entries to follower
        append_request = AppendEntriesRequest(
            term=1,
            leader_id="leader",
            prev_log_index=0,
            prev_log_term=0,
            entries=[new_entry],
            leader_commit=0
        )
        
        response = follower.handle_append_entries(append_request)
        assert response.success == True
        assert len(follower.log) == 1
        assert follower.log[0].command == "START_TRAINING"