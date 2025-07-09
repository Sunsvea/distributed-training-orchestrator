"""
Main application runner for the distributed training orchestrator dashboard
Integrates performance monitoring with the real-time web dashboard
"""
import asyncio
import logging
import signal
import sys
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.performance_monitor import PerformanceMonitor, MetricType, AlertSeverity
from monitoring.alert_manager import AlertRule
from dashboard.server import DashboardServer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DashboardApplication:
    """Main application that coordinates the dashboard and monitoring system"""
    
    def __init__(self, dashboard_host: str = "0.0.0.0", dashboard_port: int = 8080):
        self.dashboard_host = dashboard_host
        self.dashboard_port = dashboard_port
        
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor(
            collection_interval=1.0,  # Collect metrics every second
            alert_evaluation_interval=5.0  # Evaluate alerts every 5 seconds
        )
        
        # Initialize dashboard server
        self.dashboard_server = DashboardServer(
            performance_monitor=self.performance_monitor,
            host=dashboard_host,
            port=dashboard_port
        )
        
        # Setup default alert rules
        self._setup_default_alerts()
        
        # Add some sample training state for demonstration
        self._setup_demo_state()
        
        # Shutdown flag
        self._shutdown = False
    
    def _setup_default_alerts(self):
        """Setup default alert rules for monitoring"""
        default_rules = [
            {
                "name": "high_cpu_usage",
                "metric_type": "system",
                "metric_name": "cpu_percent",
                "condition": "greater_than",
                "threshold": 80.0,
                "severity": "warning",
                "duration": 30.0,
                "message_template": "High CPU usage detected: {metric_value}%"
            },
            {
                "name": "critical_cpu_usage",
                "metric_type": "system",
                "metric_name": "cpu_percent",
                "condition": "greater_than",
                "threshold": 95.0,
                "severity": "critical",
                "duration": 10.0,
                "message_template": "Critical CPU usage detected: {metric_value}%"
            },
            {
                "name": "high_memory_usage",
                "metric_type": "system",
                "metric_name": "memory_percent",
                "condition": "greater_than",
                "threshold": 85.0,
                "severity": "warning",
                "duration": 30.0,
                "message_template": "High memory usage detected: {metric_value}%"
            },
            {
                "name": "critical_memory_usage",
                "metric_type": "system",
                "metric_name": "memory_percent",
                "condition": "greater_than",
                "threshold": 95.0,
                "severity": "critical",
                "duration": 10.0,
                "message_template": "Critical memory usage detected: {metric_value}%"
            },
            {
                "name": "training_loss_increase",
                "metric_type": "training",
                "metric_name": "loss",
                "condition": "greater_than",
                "threshold": 1.0,
                "severity": "warning",
                "duration": 60.0,
                "message_template": "Training loss is increasing: {metric_value}"
            },
            {
                "name": "low_training_accuracy",
                "metric_type": "training",
                "metric_name": "accuracy",
                "condition": "less_than",
                "threshold": 0.5,
                "severity": "warning",
                "duration": 120.0,
                "message_template": "Low training accuracy detected: {metric_value}"
            }
        ]
        
        # Convert to AlertRule objects and add to monitor
        for rule_config in default_rules:
            try:
                rule = AlertRule(
                    name=rule_config["name"],
                    metric_type=MetricType(rule_config["metric_type"]),
                    metric_name=rule_config["metric_name"],
                    condition=rule_config["condition"],
                    threshold=rule_config["threshold"],
                    severity=AlertSeverity(rule_config["severity"]),
                    duration=rule_config["duration"],
                    message_template=rule_config["message_template"]
                )
                self.performance_monitor.alert_manager.add_alert_rule(rule)
                logger.info(f"Added alert rule: {rule.name}")
            except Exception as e:
                logger.error(f"Failed to add alert rule {rule_config['name']}: {e}")
    
    def _setup_demo_state(self):
        """Setup demo training state and network connections for demonstration"""
        # Sample training state
        sample_training_state = {
            "loss": 0.45,
            "accuracy": 0.87,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epoch": 15,
            "iteration": 1500,
            "throughput": 245.0,
            "gradient_norm": 0.08
        }
        
        # Sample network connections
        sample_network_connections = [
            {
                "worker_id": "worker-1",
                "latency": 12.5,
                "bandwidth": 1000.0,
                "packet_loss": 0.01
            },
            {
                "worker_id": "worker-2",
                "latency": 15.2,
                "bandwidth": 950.0,
                "packet_loss": 0.0
            },
            {
                "worker_id": "worker-3",
                "latency": 18.7,
                "bandwidth": 800.0,
                "packet_loss": 0.02
            }
        ]
        
        # Register with performance monitor
        self.performance_monitor.register_training_state(sample_training_state)
        self.performance_monitor.register_network_connections(sample_network_connections)
        
        # Add some custom metrics
        self.performance_monitor.add_custom_metric("model_size_mb", 152.3)
        self.performance_monitor.add_custom_metric("dataset_size_gb", 2.8)
        self.performance_monitor.add_custom_metric("workers_active", 3)
    
    async def start(self):
        """Start the application"""
        logger.info("Starting Distributed Training Orchestrator Dashboard")
        
        try:
            # Start performance monitoring
            await self.performance_monitor.start_monitoring()
            logger.info("Performance monitoring started")
            
            # Start dashboard server (this will block)
            await self.dashboard_server.start()
            
        except Exception as e:
            logger.error(f"Failed to start application: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the application"""
        if self._shutdown:
            return
        
        self._shutdown = True
        logger.info("Stopping Distributed Training Orchestrator Dashboard")
        
        try:
            # Stop performance monitoring
            await self.performance_monitor.stop_monitoring()
            logger.info("Performance monitoring stopped")
            
            # Stop dashboard server
            await self.dashboard_server.stop()
            logger.info("Dashboard server stopped")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed Training Orchestrator Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Dashboard host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Dashboard port (default: 8080)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and start application
    app = DashboardApplication(dashboard_host=args.host, dashboard_port=args.port)
    app.setup_signal_handlers()
    
    try:
        await app.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)
    finally:
        await app.stop()


if __name__ == "__main__":
    asyncio.run(main())