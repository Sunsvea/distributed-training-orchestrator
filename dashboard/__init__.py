"""
Dashboard package for the distributed training orchestrator
Provides real-time web-based monitoring and visualization
"""

from .server import DashboardServer
from .main import DashboardApplication

__all__ = ["DashboardServer", "DashboardApplication"]