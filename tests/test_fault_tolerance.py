import pytest
import asyncio
import torch
import tempfile
import time
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from coordinator.fault_tolerance import FaultToleranceManager, FailureType, FailureEvent, RecoveryStrategy
from coordinator.checkpoint_manager import CheckpointManager, CheckpointMetadata
from coordinator.cluster_manager import ClusterManager
from coordinator.raft_coordinator import RaftCoordinator

class TestFaultToleranceManager:
    
    @pytest.fixture
    def mock_coordinator(self):
        coordinator = Mock()
        coordinator.gradient_coordinator = Mock()
        coordinator.gradient_coordinator.remove_worker = Mock()
        coordinator.start_election = AsyncMock()
        return coordinator
    
    @pytest.fixture
    def mock_cluster_manager(self):
        manager = Mock()
        manager.get_worker_nodes = Mock(return_value=["worker-1", "worker-2", "worker-3"])
        manager.get_active_workers = Mock(return_value=["worker-1", "worker-2", "worker-3"])
        manager.mark_worker_failed = Mock()
        manager.restart_worker = AsyncMock(return_value=True)
        return manager
    
    @pytest.fixture
    def fault_manager(self, mock_coordinator, mock_cluster_manager):
        return FaultToleranceManager(
            coordinator=mock_coordinator,
            cluster_manager=mock_cluster_manager,
            heartbeat_timeout=5.0,
            max_worker_failures=2
        )
    
    def test_fault_manager_initialization(self, fault_manager):
        """Test fault tolerance manager initialization"""
        assert fault_manager.heartbeat_timeout == 5.0
        assert fault_manager.max_worker_failures == 2
        assert len(fault_manager.failed_nodes) == 0
        assert len(fault_manager.recovering_nodes) == 0
        assert len(fault_manager.recovery_strategies) == 5
        assert FailureType.WORKER_TIMEOUT in fault_manager.recovery_strategies
    
    def test_recovery_strategy_initialization(self, fault_manager):
        """Test recovery strategy initialization"""
        worker_timeout_strategy = fault_manager.recovery_strategies[FailureType.WORKER_TIMEOUT]
        
        assert worker_timeout_strategy.max_retries == 3
        assert worker_timeout_strategy.retry_delay == 5.0
        assert "restart_worker" in worker_timeout_strategy.actions
        assert "redistribute_work" in worker_timeout_strategy.actions
    
    def test_heartbeat_update(self, fault_manager):
        """Test heartbeat update functionality"""
        fault_manager.update_heartbeat("worker-1")
        
        assert "worker-1" in fault_manager.last_heartbeats
        assert fault_manager.last_heartbeats["worker-1"] > 0
    
    def test_failure_reporting(self, fault_manager):
        """Test failure event reporting"""
        fault_manager.report_failure("worker-1", FailureType.WORKER_CRASH, "Process crashed")
        
        assert len(fault_manager.failure_history) == 1
        assert fault_manager.failure_history[0].node_id == "worker-1"
        assert fault_manager.failure_history[0].failure_type == FailureType.WORKER_CRASH
        assert fault_manager.metrics["total_failures"] == 1
    
    @pytest.mark.asyncio
    async def test_worker_failure_detection(self, fault_manager):
        """Test worker failure detection"""
        # Set old heartbeat for one worker, and recent heartbeats for others
        current_time = time.time()
        fault_manager.last_heartbeats["worker-1"] = current_time - 10.0  # Old heartbeat
        fault_manager.last_heartbeats["worker-2"] = current_time - 1.0   # Recent heartbeat
        fault_manager.last_heartbeats["worker-3"] = current_time - 1.0   # Recent heartbeat
        
        # Mock the handler to avoid actual recovery
        with patch.object(fault_manager, '_handle_worker_failure') as mock_handler:
            await fault_manager._check_worker_health()
            mock_handler.assert_called_once_with("worker-1", FailureType.WORKER_TIMEOUT)
    
    @pytest.mark.asyncio
    async def test_worker_failure_handling(self, fault_manager):
        """Test worker failure handling"""
        with patch.object(fault_manager, '_start_recovery') as mock_recovery:
            await fault_manager._handle_worker_failure("worker-1", FailureType.WORKER_CRASH)
            
            assert "worker-1" in fault_manager.failed_nodes
            assert len(fault_manager.failure_history) == 1
            assert fault_manager.metrics["total_failures"] == 1
            mock_recovery.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_recovery_action_restart_worker(self, fault_manager, mock_cluster_manager):
        """Test restart worker recovery action"""
        success = await fault_manager._restart_worker("worker-1")
        
        assert success == True
        mock_cluster_manager.restart_worker.assert_called_once_with("worker-1")
    
    @pytest.mark.asyncio
    async def test_recovery_action_redistribute_work(self, fault_manager, mock_cluster_manager):
        """Test work redistribution recovery action"""
        success = await fault_manager._redistribute_work("worker-1")
        
        assert success == True
        mock_cluster_manager.get_active_workers.assert_called()
    
    @pytest.mark.asyncio
    async def test_recovery_action_update_cluster_state(self, fault_manager, mock_cluster_manager, mock_coordinator):
        """Test cluster state update recovery action"""
        success = await fault_manager._update_cluster_state("worker-1")
        
        assert success == True
        mock_cluster_manager.mark_worker_failed.assert_called_once_with("worker-1")
        mock_coordinator.gradient_coordinator.remove_worker.assert_called_once_with("worker-1")
    
    @pytest.mark.asyncio
    async def test_recovery_strategy_execution(self, fault_manager):
        """Test recovery strategy execution"""
        strategy = RecoveryStrategy(
            failure_type=FailureType.WORKER_TIMEOUT,
            max_retries=2,
            retry_delay=0.1,
            actions=["restart_worker", "update_cluster_state"]
        )
        
        with patch.object(fault_manager, '_execute_recovery_action', return_value=True) as mock_action:
            with patch.object(fault_manager, '_verify_recovery', return_value=True) as mock_verify:
                success = await fault_manager._execute_recovery_strategy("worker-1", strategy)
                
                assert success == True
                assert mock_action.call_count == 2
                mock_verify.assert_called_once_with("worker-1")
    
    @pytest.mark.asyncio
    async def test_recovery_strategy_max_retries(self, fault_manager):
        """Test recovery strategy max retries"""
        fault_manager.recovery_attempts["worker-1"] = 3
        
        strategy = RecoveryStrategy(
            failure_type=FailureType.WORKER_TIMEOUT,
            max_retries=2,
            actions=["restart_worker"]
        )
        
        success = await fault_manager._execute_recovery_strategy("worker-1", strategy)
        assert success == False
    
    @pytest.mark.asyncio
    async def test_network_partition_detection(self, fault_manager, mock_cluster_manager):
        """Test network partition detection"""
        # Simulate network partition (only 1 out of 3 workers active)
        mock_cluster_manager.get_active_workers.return_value = ["worker-1"]
        
        with patch.object(fault_manager, '_handle_network_partition') as mock_handler:
            await fault_manager._check_network_health()
            mock_handler.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connectivity_check(self, fault_manager):
        """Test node connectivity check"""
        # Node not in failed nodes should be considered connected
        connected = await fault_manager._check_node_connectivity("worker-1")
        assert connected == True
        
        # Node in failed nodes should be considered disconnected
        fault_manager.failed_nodes.add("worker-2")
        connected = await fault_manager._check_node_connectivity("worker-2")
        assert connected == False
    
    @pytest.mark.asyncio
    async def test_recovery_verification(self, fault_manager, mock_cluster_manager):
        """Test recovery verification"""
        mock_cluster_manager.get_active_workers.return_value = ["worker-1", "worker-2"]
        
        verified = await fault_manager._verify_recovery("worker-1")
        assert verified == True
        
        verified = await fault_manager._verify_recovery("worker-3")
        assert verified == False
    
    def test_cluster_health_calculation(self, fault_manager, mock_cluster_manager):
        """Test cluster health calculation"""
        # All workers healthy
        health = fault_manager.get_cluster_health()
        
        assert health["health_score"] == 100.0
        assert health["total_workers"] == 3
        assert health["active_workers"] == 3
        assert health["status"] == "healthy"
        
        # One worker failed
        fault_manager.failed_nodes.add("worker-1")
        mock_cluster_manager.get_active_workers.return_value = ["worker-2", "worker-3"]
        
        health = fault_manager.get_cluster_health()
        assert health["health_score"] == (2/3) * 100
        assert health["failed_workers"] == 1
        assert health["status"] == "degraded"
    
    def test_failure_history_management(self, fault_manager):
        """Test failure history management"""
        # Add multiple failures
        for i in range(5):
            fault_manager.report_failure(f"worker-{i}", FailureType.WORKER_TIMEOUT, f"Test failure {i}")
        
        history = fault_manager.get_failure_history(limit=3)
        assert len(history) == 3
        assert all(isinstance(event, FailureEvent) for event in history)
    
    def test_metrics_collection(self, fault_manager):
        """Test metrics collection"""
        # Add some test data
        fault_manager.failed_nodes.add("worker-1")
        fault_manager.recovering_nodes.add("worker-2")
        fault_manager.metrics["total_failures"] = 5
        
        metrics = fault_manager.get_metrics()
        
        assert metrics["total_failures"] == 5
        assert metrics["failed_nodes"] == 1
        assert metrics["recovering_nodes"] == 1
        assert "recent_failures" in metrics
    
    def test_custom_recovery_strategy(self, fault_manager):
        """Test setting custom recovery strategy"""
        custom_strategy = RecoveryStrategy(
            failure_type=FailureType.WORKER_CRASH,
            max_retries=5,
            retry_delay=1.0,
            actions=["custom_action"]
        )
        
        fault_manager.set_recovery_strategy(FailureType.WORKER_CRASH, custom_strategy)
        
        assert fault_manager.recovery_strategies[FailureType.WORKER_CRASH] == custom_strategy
        assert fault_manager.recovery_strategies[FailureType.WORKER_CRASH].max_retries == 5
    
    def test_node_status_checks(self, fault_manager):
        """Test node status checking methods"""
        fault_manager.failed_nodes.add("worker-1")
        fault_manager.recovering_nodes.add("worker-2")
        
        assert fault_manager.is_node_failed("worker-1") == True
        assert fault_manager.is_node_failed("worker-2") == False
        assert fault_manager.is_node_recovering("worker-1") == False
        assert fault_manager.is_node_recovering("worker-2") == True
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, fault_manager):
        """Test monitoring start/stop lifecycle"""
        # Start monitoring
        await fault_manager.start_monitoring()
        assert fault_manager.monitoring_task is not None
        
        # Stop monitoring
        await fault_manager.stop_monitoring()
        assert fault_manager.monitoring_task is None
        assert fault_manager._shutdown == True

class TestCheckpointManager:
    
    @pytest.fixture
    def temp_checkpoint_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    
    @pytest.fixture
    def checkpoint_manager(self, temp_checkpoint_dir):
        return CheckpointManager(
            base_path=temp_checkpoint_dir,
            max_checkpoints=5,
            checkpoint_interval=60.0
        )
    
    def test_checkpoint_manager_initialization(self, checkpoint_manager, temp_checkpoint_dir):
        """Test checkpoint manager initialization"""
        assert checkpoint_manager.base_path == Path(temp_checkpoint_dir)
        assert checkpoint_manager.max_checkpoints == 5
        assert checkpoint_manager.checkpoint_interval == 60.0
        assert len(checkpoint_manager.checkpoints) == 0
    
    @pytest.mark.asyncio
    async def test_save_checkpoint(self, checkpoint_manager):
        """Test saving a checkpoint"""
        # Create test data
        model_state = {
            "layer1.weight": torch.randn(10, 5),
            "layer1.bias": torch.randn(10)
        }
        
        optimizer_state = {
            "state": {"param_groups": [{"lr": 0.001}]},
            "param_groups": [{"lr": 0.001}]
        }
        
        worker_states = {
            "worker-1": {"status": "active", "iteration": 100},
            "worker-2": {"status": "active", "iteration": 100}
        }
        
        metrics = {
            "loss": 0.5,
            "accuracy": 0.85,
            "epoch": 5
        }
        
        cluster_state = {
            "active_workers": ["worker-1", "worker-2"],
            "failed_workers": []
        }
        
        # Save checkpoint
        checkpoint_id = await checkpoint_manager.save_checkpoint(
            training_id="test_training",
            iteration=100,
            epoch=5,
            model_state=model_state,
            optimizer_state=optimizer_state,
            worker_states=worker_states,
            metrics=metrics,
            cluster_state=cluster_state
        )
        
        assert checkpoint_id is not None
        assert checkpoint_id in checkpoint_manager.checkpoints
        assert checkpoint_manager.metrics["successful_saves"] == 1
    
    @pytest.mark.asyncio
    async def test_restore_checkpoint(self, checkpoint_manager):
        """Test restoring a checkpoint"""
        # First save a checkpoint
        model_state = {"param": torch.randn(5, 5)}
        optimizer_state = {"lr": 0.001}
        worker_states = {"worker-1": {"status": "active"}}
        metrics = {"loss": 0.3}
        cluster_state = {"active_workers": ["worker-1"]}
        
        checkpoint_id = await checkpoint_manager.save_checkpoint(
            training_id="test_training",
            iteration=50,
            epoch=2,
            model_state=model_state,
            optimizer_state=optimizer_state,
            worker_states=worker_states,
            metrics=metrics,
            cluster_state=cluster_state
        )
        
        # Restore checkpoint
        restored_data = await checkpoint_manager.restore_checkpoint(checkpoint_id)
        
        assert "model_state" in restored_data
        assert "optimizer_state" in restored_data
        assert "worker_states" in restored_data
        assert "metrics" in restored_data
        assert "cluster_state" in restored_data
        assert restored_data["metadata"].iteration == 50
        assert checkpoint_manager.metrics["successful_restores"] == 1
    
    @pytest.mark.asyncio
    async def test_checkpoint_verification(self, checkpoint_manager):
        """Test checkpoint verification"""
        # Save a checkpoint
        model_state = {"param": torch.randn(3, 3)}
        optimizer_state = {"lr": 0.001}
        worker_states = {}
        metrics = {"loss": 0.1}
        cluster_state = {}
        
        checkpoint_id = await checkpoint_manager.save_checkpoint(
            training_id="test_training",
            iteration=25,
            epoch=1,
            model_state=model_state,
            optimizer_state=optimizer_state,
            worker_states=worker_states,
            metrics=metrics,
            cluster_state=cluster_state
        )
        
        # Verification should pass
        await checkpoint_manager._verify_checkpoint(checkpoint_id)
        
        # Corrupt checkpoint by modifying checksum
        metadata = checkpoint_manager.checkpoints[checkpoint_id]
        metadata.checksum = "invalid_checksum"
        
        # Verification should fail
        with pytest.raises(ValueError, match="checksum mismatch"):
            await checkpoint_manager._verify_checkpoint(checkpoint_id)
    
    def test_list_checkpoints(self, checkpoint_manager):
        """Test listing checkpoints"""
        # Add test checkpoints
        for i in range(3):
            checkpoint_id = f"test_checkpoint_{i}"
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                training_id="test_training",
                iteration=i * 10,
                epoch=i,
                timestamp=time.time() + i,
                model_state_size=1000,
                optimizer_state_size=500,
                worker_states={},
                loss=0.5 - i * 0.1,
                accuracy=0.7 + i * 0.1,
                metrics={},
                cluster_state={},
                file_paths={},
                checksum="test_checksum"
            )
            checkpoint_manager.checkpoints[checkpoint_id] = metadata
        
        # List all checkpoints
        all_checkpoints = checkpoint_manager.list_checkpoints()
        assert len(all_checkpoints) == 3
        
        # List checkpoints for specific training
        training_checkpoints = checkpoint_manager.list_checkpoints("test_training")
        assert len(training_checkpoints) == 3
        
        # List checkpoints for non-existent training
        no_checkpoints = checkpoint_manager.list_checkpoints("nonexistent_training")
        assert len(no_checkpoints) == 0
    
    def test_get_latest_checkpoint(self, checkpoint_manager):
        """Test getting latest checkpoint"""
        # Add test checkpoints with different timestamps
        for i in range(3):
            checkpoint_id = f"test_checkpoint_{i}"
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                training_id="test_training",
                iteration=i * 10,
                epoch=i,
                timestamp=time.time() + i,  # Later timestamps are newer
                model_state_size=1000,
                optimizer_state_size=500,
                worker_states={},
                loss=0.5,
                accuracy=0.8,
                metrics={},
                cluster_state={},
                file_paths={},
                checksum="test_checksum"
            )
            checkpoint_manager.checkpoints[checkpoint_id] = metadata
        
        latest = checkpoint_manager.get_latest_checkpoint("test_training")
        assert latest is not None
        assert latest.checkpoint_id == "test_checkpoint_2"  # Latest timestamp
        
        # Non-existent training should return None
        no_latest = checkpoint_manager.get_latest_checkpoint("nonexistent_training")
        assert no_latest is None
    
    def test_delete_checkpoint(self, checkpoint_manager):
        """Test deleting a checkpoint"""
        # Add test checkpoint
        checkpoint_id = "test_checkpoint"
        # Create a fake directory for the checkpoint
        checkpoint_dir = checkpoint_manager.base_path / checkpoint_id
        checkpoint_dir.mkdir(exist_ok=True)
        
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            training_id="test_training",
            iteration=10,
            epoch=1,
            timestamp=time.time(),
            model_state_size=1000,
            optimizer_state_size=500,
            worker_states={},
            loss=0.5,
            accuracy=0.8,
            metrics={},
            cluster_state={},
            file_paths={
                "model": str(checkpoint_dir / "model_state.pt"),
                "optimizer": str(checkpoint_dir / "optimizer_state.pkl"),
                "worker_states": str(checkpoint_dir / "worker_states.pkl"),
                "cluster_state": str(checkpoint_dir / "cluster_state.json"),
                "metrics": str(checkpoint_dir / "metrics.json")
            },
            checksum="test_checksum"
        )
        checkpoint_manager.checkpoints[checkpoint_id] = metadata
        
        # Delete checkpoint
        success = checkpoint_manager.delete_checkpoint(checkpoint_id)
        assert success == True
        assert checkpoint_id not in checkpoint_manager.checkpoints
        
        # Try to delete non-existent checkpoint
        success = checkpoint_manager.delete_checkpoint("nonexistent_checkpoint")
        assert success == False
    
    def test_checkpoint_metrics(self, checkpoint_manager):
        """Test checkpoint metrics collection"""
        # Add test data
        checkpoint_manager.metrics["total_checkpoints"] = 10
        checkpoint_manager.metrics["successful_saves"] = 8
        checkpoint_manager.metrics["failed_saves"] = 2
        
        metrics = checkpoint_manager.get_metrics()
        
        assert metrics["total_checkpoints"] == 10
        assert metrics["successful_saves"] == 8
        assert metrics["failed_saves"] == 2
        assert "total_checkpoints_stored" in metrics
        assert "disk_usage" in metrics
    
    def test_checkpoint_summary(self, checkpoint_manager):
        """Test checkpoint summary generation"""
        # Add test checkpoints for different trainings
        for training_id in ["training_1", "training_2"]:
            for i in range(2):
                checkpoint_id = f"{training_id}_checkpoint_{i}"
                metadata = CheckpointMetadata(
                    checkpoint_id=checkpoint_id,
                    training_id=training_id,
                    iteration=i * 10,
                    epoch=i,
                    timestamp=time.time() + i,
                    model_state_size=1000,
                    optimizer_state_size=500,
                    worker_states={},
                    loss=0.5,
                    accuracy=0.8,
                    metrics={},
                    cluster_state={},
                    file_paths={},
                    checksum="test_checksum"
                )
                checkpoint_manager.checkpoints[checkpoint_id] = metadata
        
        summary = checkpoint_manager.get_checkpoint_summary()
        
        assert summary["total"] == 4
        assert len(summary["by_training"]) == 2
        assert summary["by_training"]["training_1"]["count"] == 2
        assert summary["by_training"]["training_2"]["count"] == 2
        assert "latest" in summary["by_training"]["training_1"]
        assert "oldest" in summary["by_training"]["training_1"]
    
    @pytest.mark.asyncio
    async def test_automatic_checkpointing_lifecycle(self, checkpoint_manager):
        """Test automatic checkpointing start/stop"""
        # Start automatic checkpointing
        await checkpoint_manager.start_automatic_checkpointing("test_training")
        assert checkpoint_manager.current_training_id == "test_training"
        assert checkpoint_manager.checkpoint_task is not None
        
        # Stop automatic checkpointing
        await checkpoint_manager.stop_automatic_checkpointing()
        assert checkpoint_manager.checkpoint_task is None
        assert checkpoint_manager._shutdown == True