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

```bash
# Install dependencies
pip install -r requirements.txt

# Start coordinator
python coordinator/main.py

# Start workers (in separate terminals)
python worker/main.py --node-id worker-1
python worker/main.py --node-id worker-2
python worker/main.py --node-id worker-3

# Start real-time dashboard
python demo_dashboard.py
```

## Dashboard Features

The integrated web dashboard provides:

- **Real-time Monitoring**: Live system metrics, training progress, and cluster health
- **Interactive Charts**: Performance trends and resource utilization visualization
- **Smart Alerts**: Configurable alerting rules with multiple severity levels
- **Performance Insights**: Automated recommendations and optimization suggestions
- **WebSocket Updates**: Real-time data streaming for live dashboard updates
- **REST API**: Programmatic access to all metrics and cluster information
- **Export Capabilities**: Download metrics in JSON/CSV formats for analysis

Access the dashboard at `http://localhost:8080` after starting with `python demo_dashboard.py`.

## Demo Scenarios

1. **Baseline Training**: 3 nodes collaboratively training
2. **Dynamic Scaling**: Add/remove nodes during training
3. **Fault Injection**: Coordinator failure and recovery
4. **Network Partition**: Split-brain detection and healing
5. **Performance Analysis**: Scaling efficiency metrics