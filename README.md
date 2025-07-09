# Distributed Training Orchestrator

A fault-tolerant distributed machine learning training system with consensus-based coordination and real-time monitoring.

## Features

- **Consensus Algorithm**: Raft-based coordinator election and cluster management
- **Gradient Synchronization**: Multiple strategies (AllReduce, Parameter Server, custom)
- **Fault Tolerance**: Automatic recovery, checkpointing, split-brain detection
- **Dynamic Scaling**: Nodes can join/leave during training
- **Real-time Dashboard**: Live monitoring of training progress and cluster health

## Architecture

- **Coordinator Node**: Manages cluster state and consensus
- **Worker Nodes**: Execute training and gradient synchronization
- **Communication Layer**: gRPC for control plane, custom protocols for data plane
- **Web Dashboard**: React-based real-time visualization

## Quick Start

### Interactive Demo (Recommended for Demonstrations)
```bash
# Install dependencies
pip install -r requirements.txt

# Start the simple interactive demo (recommended)
python demo_simple.py

# OR start the full distributed training demo
python demo_orchestrator.py

# Access dashboard at http://localhost:8080
```

### Manual Setup (Development)
```bash
# Install dependencies
pip install -r requirements.txt

# Start coordinator
python coordinator/main.py

# Start workers (in separate terminals)
python worker/main.py --node-id worker-1
python worker/main.py --node-id worker-2
python worker/main.py --node-id worker-3

# Start dashboard only
python demo_dashboard.py
```

## Dashboard Features

The integrated web dashboard provides:

- **Real-time Monitoring**: Live system metrics, training progress, and cluster health
- **Cluster Visualization**: Interactive topology showing coordinator and worker nodes
- **Interactive Controls**: Add/remove workers, inject failures, switch strategies
- **Performance Charts**: Live CPU/memory trends and training progress curves
- **Smart Alerts**: Configurable alerting rules with multiple severity levels
- **Performance Insights**: Automated recommendations and optimization suggestions
- **WebSocket Updates**: Real-time data streaming for live dashboard updates
- **REST API**: Programmatic access to all metrics and cluster information
- **Export Capabilities**: Download metrics in JSON/CSV formats for analysis

Access the dashboard at `http://localhost:8080` after starting the demo.

## Interactive Demo Scenarios

The demo automatically cycles through different scenarios to showcase the system's capabilities:

1. **Baseline Training**: Normal distributed training across multiple workers
2. **Dynamic Scaling**: Add/remove workers during training with live cluster updates
3. **Fault Injection**: Worker failure simulation with automatic recovery
4. **Gradient Strategies**: Switch between AllReduce and Parameter Server algorithms
5. **Performance Analysis**: Real-time metrics and alerting demonstrations

### Interactive Controls
- **➕ Add Worker**: Dynamically scale up the cluster
- **➖ Remove Worker**: Scale down the cluster  
- **💥 Inject Failure**: Simulate worker failure to demonstrate recovery
- **🔄 Switch Strategy**: Change gradient synchronization method

See [DEMO_GUIDE.md](DEMO_GUIDE.md) for detailed demonstration instructions.