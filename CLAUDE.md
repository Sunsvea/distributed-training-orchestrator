# Distributed Training Orchestrator - Claude Development Guide

This document provides guidance for future Claude instances working on this distributed training orchestrator codebase.

## Project Overview

This is a fault-tolerant distributed machine learning training system that implements:
- Raft consensus algorithm for coordinator election and cluster management
- Multiple gradient synchronization strategies (AllReduce, Parameter Server, custom)
- Comprehensive fault tolerance and recovery mechanisms
- Real-time performance monitoring and alerting
- gRPC-based communication layer with Protocol Buffers

## Architecture Overview

### Core Components

#### Coordinator Layer (`coordinator/`)
- **`raft_coordinator.py`**: Implements Raft consensus algorithm for leader election and log replication
- **`cluster_manager.py`**: Manages cluster state, node lifecycle, and health monitoring
- **`fault_tolerance.py`**: Handles failure detection, recovery strategies, and system resilience
- **`checkpoint_manager.py`**: Manages model checkpointing, state persistence, and recovery

#### Worker Layer (`worker/`)
- **`training_worker.py`**: Main training worker that handles model training and gradient computation
- **`gradient_sync.py`**: Handles gradient serialization, compression, and synchronization
- **`distributed_strategies.py`**: Implements AllReduce, Parameter Server, and custom gradient strategies
- **`models.py`**: Model definitions and management

#### Communication Layer (`communication/`)
- **`cluster.proto`**: Protocol Buffer definitions for all system messages (23 RPC methods)
- **`gradient_coordinator.py`**: Coordinates gradient synchronization across the cluster
- **`worker_client.py`**: gRPC client for worker-to-coordinator communication
- **`coordinator_server.py`**: gRPC server implementation for coordinator

#### Monitoring Layer (`monitoring/`)
- **`metrics_collector.py`**: Collects system, training, and network metrics
- **`performance_monitor.py`**: Main monitoring system with trend analysis and health scoring
- **`alert_manager.py`**: Flexible alerting system with configurable rules
- **`dashboard_server.py`**: Web dashboard placeholder (needs implementation)

## Development Patterns

### Test-Driven Development
This project follows TDD principles with comprehensive test coverage:
- **Framework**: pytest with asyncio support
- **Pattern**: Write tests first, then implement functionality
- **Coverage**: All major components have corresponding test files
- **Mocking**: Extensive use of unittest.mock for external dependencies

### Async/Await Patterns
- All network operations use async/await
- Background tasks use asyncio.create_task()
- Proper cancellation handling with try/except asyncio.CancelledError

### Error Handling
- Comprehensive exception handling with appropriate recovery
- Structured logging with appropriate log levels
- Graceful degradation when possible

### Type Safety
- Comprehensive type annotations throughout
- Use of dataclasses for structured data
- Enum classes for constants and configuration

## Common Development Commands

### Setup and Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_coordinator.py

# Run tests with coverage
pytest tests/ --cov=.

# Run specific test method
pytest tests/test_coordinator.py::TestRaftCoordinator::test_leader_election
```

### Running the System
```bash
# Start coordinator
python coordinator/main.py

# Start workers (in separate terminals)
python worker/main.py --node-id worker-1
python worker/main.py --node-id worker-2
python worker/main.py --node-id worker-3

# Start dashboard (when implemented)
cd dashboard && npm start
```

### Protocol Buffer Compilation
```bash
# Regenerate protobuf files when cluster.proto changes
python -m grpc_tools.protoc -I communication/ --python_out=communication/ --grpc_python_out=communication/ communication/cluster.proto
```

## Key Implementation Details

### Gradient Synchronization Strategies

#### AllReduce Strategy
- Implements ring-based AllReduce topology
- Supports gradient compression with configurable ratios
- Handles worker failures gracefully with ring reconstruction

#### Parameter Server Strategy
- Implements centralized parameter server architecture
- Supports staleness tolerance for asynchronous updates
- Handles server failures with backup server promotion

#### Custom Strategy
- Weighted averaging based on worker performance
- Adaptive strategy selection based on cluster conditions
- Configurable aggregation functions

### Fault Tolerance Mechanisms

#### Failure Types Handled
- Worker timeouts and crashes
- Coordinator failures with leader election
- Network partitions and split-brain scenarios
- Gradual performance degradation

#### Recovery Strategies
- Automatic restart with exponential backoff
- Cluster reconfiguration for topology changes
- State reconstruction from checkpoints
- Manual intervention for critical failures

### Performance Monitoring

#### Metrics Collection
- System metrics: CPU, memory, disk, network I/O
- Training metrics: loss, accuracy, throughput, gradient norms
- Network metrics: latency, bandwidth, packet loss
- Custom metrics support for application-specific monitoring

#### Alerting System
- Configurable alert rules with multiple conditions
- Severity levels: INFO, WARNING, CRITICAL
- Alert deduplication and auto-resolution
- Flexible notification system (extensible)

## Common Patterns and Best Practices

### Dataclass Usage
```python
@dataclass
class TrainingMetrics:
    timestamp: float
    loss: float
    accuracy: float
    learning_rate: float
    # ... other fields
```

### Async Context Managers
```python
async def start_monitoring(self):
    self._shutdown = False
    await self.metrics_collector.start_collection()
    self.monitoring_task = asyncio.create_task(self._monitoring_loop())
```

### Thread-Safe Operations
```python
with self.state_lock:
    self.current_training_state = training_state
```

### Configuration Patterns
```python
def __init__(self, collection_interval: float = 5.0, 
             alert_evaluation_interval: float = 10.0):
    self.collection_interval = collection_interval
    self.alert_evaluation_interval = alert_evaluation_interval
```

## Testing Patterns

### Mock Usage
```python
@pytest.fixture
def mock_coordinator():
    coordinator = Mock(spec=RaftCoordinator)
    coordinator.get_leader.return_value = "coordinator-1"
    return coordinator
```

### Async Test Fixtures
```python
@pytest.mark.asyncio
async def test_monitoring_lifecycle(self, performance_monitor):
    await performance_monitor.start_monitoring()
    assert performance_monitor.monitoring_task is not None
    await performance_monitor.stop_monitoring()
```

### Test Data Generation
```python
def create_test_metrics(self, cpu_percent=50.0, memory_percent=60.0):
    return SystemMetrics(
        timestamp=time.time(),
        cpu_percent=cpu_percent,
        memory_percent=memory_percent,
        # ... other fields
    )
```

## Critical Files to Understand

### Core Implementation Files
1. **`coordinator/raft_coordinator.py`** - Raft consensus implementation
2. **`coordinator/fault_tolerance.py`** - Fault tolerance and recovery logic
3. **`worker/distributed_strategies.py`** - Gradient synchronization strategies
4. **`communication/cluster.proto`** - Protocol definitions for all system messages
5. **`monitoring/performance_monitor.py`** - Main monitoring system

### Key Test Files
1. **`tests/test_coordinator.py`** - Shows testing patterns and system behavior
2. **`tests/test_fault_tolerance.py`** - Comprehensive fault tolerance testing
3. **`tests/test_gradient_strategies.py`** - Distributed training strategy testing
4. **`tests/test_performance_monitoring.py`** - Performance monitoring testing (24 tests)

## Development Workflow

### Before Making Changes
1. Read the relevant test files to understand expected behavior
2. Check existing patterns in similar components
3. Understand the protocol definitions in `cluster.proto`
4. Review fault tolerance implications of changes

### When Adding New Features
1. **Follow TDD**: Write tests first, then implement
2. **Update Protocol**: Modify `cluster.proto` if new messages are needed
3. **Add Monitoring**: Include appropriate metrics and alerting
4. **Handle Failures**: Consider fault tolerance implications
5. **Document Changes**: Update relevant documentation

### Code Review Checklist
- [ ] Tests written and passing
- [ ] Proper error handling and logging
- [ ] Thread safety considered
- [ ] Async patterns used correctly
- [ ] Type hints provided
- [ ] Protocol buffer changes compiled
- [ ] Monitoring and alerting considered

## Known Issues and Limitations

### Current Implementation Status
- **Dashboard**: Placeholder implementation needs full web dashboard
- **GPU Monitoring**: System metrics collection needs nvidia-ml-py integration
- **Deployment**: Docker configurations need completion
- **Documentation**: API documentation could be expanded

### Performance Considerations
- Gradient compression trades accuracy for bandwidth
- AllReduce performance depends on network topology
- Monitor memory usage during large model training
- Consider batch size optimization for throughput

## Future Development Areas

### Pending Tasks
1. **Real-time Web Dashboard**: Implement full web dashboard with React/Vue
2. **Demo Scenarios**: Add fault injection and demonstration scenarios
3. **Performance Optimization**: Implement advanced gradient compression
4. **Advanced Monitoring**: Add more sophisticated alerting rules
5. **Documentation**: Create comprehensive API documentation

### Extension Points
- **New Gradient Strategies**: Add federated learning support
- **Advanced Fault Tolerance**: Implement Byzantine fault tolerance
- **Cloud Integration**: Add cloud provider integrations
- **Model Parallelism**: Extend beyond data parallelism
- **Security**: Add authentication and encryption

## Troubleshooting Common Issues

### Import Errors
- Check that protobuf files are generated: `python -m grpc_tools.protoc ...`
- Verify Python path includes project root
- Check that all dependencies are installed

### Test Failures
- Async tests: Ensure proper asyncio.CancelledError handling
- Timing issues: Use appropriate test timeouts and delays
- Mock issues: Verify mock specifications match actual interfaces

### gRPC Communication Issues
- Check that server is running before starting clients
- Verify port configurations match across components
- Check firewall and network connectivity

This guide should help future Claude instances understand the codebase structure, patterns, and best practices for working with this distributed training orchestrator system.