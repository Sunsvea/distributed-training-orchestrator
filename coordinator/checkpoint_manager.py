import asyncio
import json
import pickle
import os
import time
import logging
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class CheckpointMetadata:
    """Metadata for a training checkpoint"""
    checkpoint_id: str
    training_id: str
    iteration: int
    epoch: int
    timestamp: float
    model_state_size: int
    optimizer_state_size: int
    worker_states: Dict[str, Any]
    loss: float
    accuracy: float
    metrics: Dict[str, Any]
    cluster_state: Dict[str, Any]
    file_paths: Dict[str, str]
    checksum: str

class CheckpointManager:
    """Manages checkpoints for distributed training state"""
    
    def __init__(self, base_path: str = "./checkpoints", **kwargs):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Configuration
        self.max_checkpoints = kwargs.get("max_checkpoints", 10)
        self.checkpoint_interval = kwargs.get("checkpoint_interval", 300.0)  # 5 minutes
        self.compression_enabled = kwargs.get("compression", True)
        self.verification_enabled = kwargs.get("verification", True)
        
        # State
        self.checkpoints: Dict[str, CheckpointMetadata] = {}
        self.current_training_id: Optional[str] = None
        self.last_checkpoint_time: float = 0
        self.checkpoint_lock = threading.Lock()
        
        # Background tasks
        self.checkpoint_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._shutdown = False
        
        # Load existing checkpoints
        self._load_checkpoint_metadata()
        
        # Metrics
        self.metrics = {
            "total_checkpoints": 0,
            "successful_saves": 0,
            "successful_restores": 0,
            "failed_saves": 0,
            "failed_restores": 0,
            "average_save_time": 0.0,
            "average_restore_time": 0.0
        }
    
    def _load_checkpoint_metadata(self):
        """Load checkpoint metadata from disk"""
        metadata_file = self.base_path / "checkpoint_metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    
                self.checkpoints = {
                    checkpoint_id: CheckpointMetadata(**metadata)
                    for checkpoint_id, metadata in data.items()
                }
                
                logger.info(f"Loaded {len(self.checkpoints)} checkpoint metadata entries")
                
            except Exception as e:
                logger.error(f"Error loading checkpoint metadata: {e}")
                self.checkpoints = {}
    
    def _save_checkpoint_metadata(self):
        """Save checkpoint metadata to disk"""
        metadata_file = self.base_path / "checkpoint_metadata.json"
        
        try:
            with open(metadata_file, 'w') as f:
                data = {
                    checkpoint_id: asdict(metadata)
                    for checkpoint_id, metadata in self.checkpoints.items()
                }
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving checkpoint metadata: {e}")
    
    async def start_automatic_checkpointing(self, training_id: str):
        """Start automatic checkpointing for a training session"""
        self.current_training_id = training_id
        
        if self.checkpoint_task is None:
            self.checkpoint_task = asyncio.create_task(self._checkpoint_loop())
            
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
        logger.info(f"Started automatic checkpointing for training {training_id}")
    
    async def stop_automatic_checkpointing(self):
        """Stop automatic checkpointing"""
        self._shutdown = True
        
        if self.checkpoint_task:
            self.checkpoint_task.cancel()
            try:
                await self.checkpoint_task
            except asyncio.CancelledError:
                pass
            self.checkpoint_task = None
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            self.cleanup_task = None
        
        self.executor.shutdown(wait=True)
        logger.info("Stopped automatic checkpointing")
    
    async def _checkpoint_loop(self):
        """Main loop for automatic checkpointing"""
        while not self._shutdown:
            try:
                current_time = time.time()
                
                if current_time - self.last_checkpoint_time >= self.checkpoint_interval:
                    if self.current_training_id:
                        # This would be called by the training coordinator
                        # For now, we'll just log that it's time for a checkpoint
                        logger.info("Automatic checkpoint trigger (requires training state)")
                        self.last_checkpoint_time = current_time
                
                await asyncio.sleep(30.0)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in checkpoint loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _cleanup_loop(self):
        """Clean up old checkpoints"""
        while not self._shutdown:
            try:
                await self._cleanup_old_checkpoints()
                await asyncio.sleep(3600.0)  # Clean up every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60.0)
    
    async def save_checkpoint(self, training_id: str, iteration: int, epoch: int,
                            model_state: Dict[str, torch.Tensor],
                            optimizer_state: Dict[str, Any],
                            worker_states: Dict[str, Any],
                            metrics: Dict[str, Any],
                            cluster_state: Dict[str, Any]) -> str:
        """Save a complete training checkpoint"""
        save_start_time = time.time()
        
        try:
            # Generate checkpoint ID
            checkpoint_id = f"{training_id}_{iteration}_{int(time.time())}"
            checkpoint_dir = self.base_path / checkpoint_id
            checkpoint_dir.mkdir(exist_ok=True)
            
            # Save model state
            model_path = checkpoint_dir / "model_state.pt"
            await self._save_tensor_dict(model_state, model_path)
            
            # Save optimizer state
            optimizer_path = checkpoint_dir / "optimizer_state.pkl"
            await self._save_pickle(optimizer_state, optimizer_path)
            
            # Save worker states
            worker_states_path = checkpoint_dir / "worker_states.pkl"
            await self._save_pickle(worker_states, worker_states_path)
            
            # Save cluster state
            cluster_state_path = checkpoint_dir / "cluster_state.json"
            await self._save_json(cluster_state, cluster_state_path)
            
            # Save metrics
            metrics_path = checkpoint_dir / "metrics.json"
            await self._save_json(metrics, metrics_path)
            
            # Calculate sizes and checksum
            model_size = model_path.stat().st_size
            optimizer_size = optimizer_path.stat().st_size
            checksum = await self._calculate_checksum(checkpoint_dir)
            
            # Create metadata
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                training_id=training_id,
                iteration=iteration,
                epoch=epoch,
                timestamp=time.time(),
                model_state_size=model_size,
                optimizer_state_size=optimizer_size,
                worker_states=worker_states,
                loss=metrics.get("loss", 0.0),
                accuracy=metrics.get("accuracy", 0.0),
                metrics=metrics,
                cluster_state=cluster_state,
                file_paths={
                    "model": str(model_path),
                    "optimizer": str(optimizer_path),
                    "worker_states": str(worker_states_path),
                    "cluster_state": str(cluster_state_path),
                    "metrics": str(metrics_path)
                },
                checksum=checksum
            )
            
            # Store metadata
            with self.checkpoint_lock:
                self.checkpoints[checkpoint_id] = metadata
                self._save_checkpoint_metadata()
            
            save_time = time.time() - save_start_time
            
            # Update metrics
            self.metrics["total_checkpoints"] += 1
            self.metrics["successful_saves"] += 1
            self.metrics["average_save_time"] = (
                (self.metrics["average_save_time"] * (self.metrics["successful_saves"] - 1) + save_time) /
                self.metrics["successful_saves"]
            )
            
            logger.info(f"Checkpoint {checkpoint_id} saved in {save_time:.2f}s")
            return checkpoint_id
            
        except Exception as e:
            self.metrics["failed_saves"] += 1
            logger.error(f"Error saving checkpoint {checkpoint_id}: {e}")
            raise
    
    async def restore_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Restore a training checkpoint"""
        restore_start_time = time.time()
        
        try:
            if checkpoint_id not in self.checkpoints:
                raise ValueError(f"Checkpoint {checkpoint_id} not found")
            
            metadata = self.checkpoints[checkpoint_id]
            
            # Verify checkpoint integrity
            if self.verification_enabled:
                await self._verify_checkpoint(checkpoint_id)
            
            # Load model state
            model_state = await self._load_tensor_dict(metadata.file_paths["model"])
            
            # Load optimizer state
            optimizer_state = await self._load_pickle(metadata.file_paths["optimizer"])
            
            # Load worker states
            worker_states = await self._load_pickle(metadata.file_paths["worker_states"])
            
            # Load cluster state
            cluster_state = await self._load_json(metadata.file_paths["cluster_state"])
            
            # Load metrics
            metrics = await self._load_json(metadata.file_paths["metrics"])
            
            restore_time = time.time() - restore_start_time
            
            # Update metrics
            self.metrics["successful_restores"] += 1
            self.metrics["average_restore_time"] = (
                (self.metrics["average_restore_time"] * (self.metrics["successful_restores"] - 1) + restore_time) /
                self.metrics["successful_restores"]
            )
            
            logger.info(f"Checkpoint {checkpoint_id} restored in {restore_time:.2f}s")
            
            return {
                "metadata": metadata,
                "model_state": model_state,
                "optimizer_state": optimizer_state,
                "worker_states": worker_states,
                "cluster_state": cluster_state,
                "metrics": metrics
            }
            
        except Exception as e:
            self.metrics["failed_restores"] += 1
            logger.error(f"Error restoring checkpoint {checkpoint_id}: {e}")
            raise
    
    async def _save_tensor_dict(self, tensor_dict: Dict[str, torch.Tensor], path: Path):
        """Save tensor dictionary to disk"""
        def _save():
            torch.save(tensor_dict, path)
        
        await asyncio.get_event_loop().run_in_executor(self.executor, _save)
    
    async def _load_tensor_dict(self, path: str) -> Dict[str, torch.Tensor]:
        """Load tensor dictionary from disk"""
        def _load():
            return torch.load(path, map_location='cpu')
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _load)
    
    async def _save_pickle(self, obj: Any, path: Path):
        """Save object using pickle"""
        def _save():
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
        
        await asyncio.get_event_loop().run_in_executor(self.executor, _save)
    
    async def _load_pickle(self, path: str) -> Any:
        """Load object using pickle"""
        def _load():
            with open(path, 'rb') as f:
                return pickle.load(f)
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _load)
    
    async def _save_json(self, obj: Any, path: Path):
        """Save object as JSON"""
        def _save():
            with open(path, 'w') as f:
                json.dump(obj, f, indent=2)
        
        await asyncio.get_event_loop().run_in_executor(self.executor, _save)
    
    async def _load_json(self, path: str) -> Any:
        """Load object from JSON"""
        def _load():
            with open(path, 'r') as f:
                return json.load(f)
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _load)
    
    async def _calculate_checksum(self, checkpoint_dir: Path) -> str:
        """Calculate checksum for checkpoint directory"""
        def _calculate():
            hasher = hashlib.sha256()
            
            for file_path in sorted(checkpoint_dir.glob("*")):
                if file_path.is_file():
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hasher.update(chunk)
            
            return hasher.hexdigest()
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _calculate)
    
    async def _verify_checkpoint(self, checkpoint_id: str):
        """Verify checkpoint integrity"""
        metadata = self.checkpoints[checkpoint_id]
        checkpoint_dir = Path(metadata.file_paths["model"]).parent
        
        # Recalculate checksum
        current_checksum = await self._calculate_checksum(checkpoint_dir)
        
        if current_checksum != metadata.checksum:
            raise ValueError(f"Checkpoint {checkpoint_id} verification failed: checksum mismatch")
        
        # Verify all files exist
        for file_type, file_path in metadata.file_paths.items():
            if not Path(file_path).exists():
                raise ValueError(f"Checkpoint {checkpoint_id} verification failed: missing {file_type} file")
    
    async def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit"""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by timestamp (oldest first)
        sorted_checkpoints = sorted(
            self.checkpoints.items(),
            key=lambda x: x[1].timestamp
        )
        
        # Remove oldest checkpoints
        to_remove = len(self.checkpoints) - self.max_checkpoints
        
        for checkpoint_id, metadata in sorted_checkpoints[:to_remove]:
            try:
                # Remove checkpoint directory
                checkpoint_dir = Path(metadata.file_paths["model"]).parent
                
                if checkpoint_dir.exists():
                    await self._remove_directory(checkpoint_dir)
                
                # Remove from metadata
                with self.checkpoint_lock:
                    del self.checkpoints[checkpoint_id]
                
                logger.info(f"Removed old checkpoint {checkpoint_id}")
                
            except Exception as e:
                logger.error(f"Error removing checkpoint {checkpoint_id}: {e}")
        
        # Save updated metadata
        self._save_checkpoint_metadata()
    
    async def _remove_directory(self, directory: Path):
        """Remove directory and all contents"""
        def _remove():
            import shutil
            shutil.rmtree(directory)
        
        await asyncio.get_event_loop().run_in_executor(self.executor, _remove)
    
    def list_checkpoints(self, training_id: Optional[str] = None) -> List[CheckpointMetadata]:
        """List available checkpoints"""
        checkpoints = list(self.checkpoints.values())
        
        if training_id:
            checkpoints = [cp for cp in checkpoints if cp.training_id == training_id]
        
        # Sort by timestamp (newest first)
        return sorted(checkpoints, key=lambda x: x.timestamp, reverse=True)
    
    def get_latest_checkpoint(self, training_id: str) -> Optional[CheckpointMetadata]:
        """Get the latest checkpoint for a training session"""
        training_checkpoints = self.list_checkpoints(training_id)
        return training_checkpoints[0] if training_checkpoints else None
    
    def get_checkpoint_metadata(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        """Get metadata for a specific checkpoint"""
        return self.checkpoints.get(checkpoint_id)
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint"""
        if checkpoint_id not in self.checkpoints:
            return False
        
        try:
            metadata = self.checkpoints[checkpoint_id]
            checkpoint_dir = Path(metadata.file_paths["model"]).parent
            
            if checkpoint_dir.exists():
                # Try to schedule async removal if event loop is running
                try:
                    asyncio.create_task(self._remove_directory(checkpoint_dir))
                except RuntimeError:
                    # No event loop, remove synchronously
                    import shutil
                    shutil.rmtree(checkpoint_dir)
            
            with self.checkpoint_lock:
                del self.checkpoints[checkpoint_id]
                self._save_checkpoint_metadata()
            
            logger.info(f"Deleted checkpoint {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting checkpoint {checkpoint_id}: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get checkpoint manager metrics"""
        return {
            **self.metrics,
            "total_checkpoints_stored": len(self.checkpoints),
            "disk_usage": self._calculate_disk_usage()
        }
    
    def _calculate_disk_usage(self) -> int:
        """Calculate total disk usage of checkpoints"""
        total_size = 0
        
        for metadata in self.checkpoints.values():
            total_size += metadata.model_state_size
            total_size += metadata.optimizer_state_size
        
        return total_size
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """Get summary of all checkpoints"""
        if not self.checkpoints:
            return {"total": 0, "by_training": {}}
        
        by_training = {}
        
        for metadata in self.checkpoints.values():
            training_id = metadata.training_id
            
            if training_id not in by_training:
                by_training[training_id] = {
                    "count": 0,
                    "latest": None,
                    "oldest": None,
                    "total_size": 0
                }
            
            by_training[training_id]["count"] += 1
            by_training[training_id]["total_size"] += metadata.model_state_size + metadata.optimizer_state_size
            
            if (by_training[training_id]["latest"] is None or 
                metadata.timestamp > by_training[training_id]["latest"]["timestamp"]):
                by_training[training_id]["latest"] = {
                    "checkpoint_id": metadata.checkpoint_id,
                    "timestamp": metadata.timestamp,
                    "iteration": metadata.iteration,
                    "epoch": metadata.epoch
                }
            
            if (by_training[training_id]["oldest"] is None or 
                metadata.timestamp < by_training[training_id]["oldest"]["timestamp"]):
                by_training[training_id]["oldest"] = {
                    "checkpoint_id": metadata.checkpoint_id,
                    "timestamp": metadata.timestamp,
                    "iteration": metadata.iteration,
                    "epoch": metadata.epoch
                }
        
        return {
            "total": len(self.checkpoints),
            "by_training": by_training
        }