#!/usr/bin/env python3
"""
Demo script to run the distributed training orchestrator dashboard
This script demonstrates how to start the dashboard with sample data
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dashboard.main import DashboardApplication


async def main():
    """Run the dashboard demo"""
    print("🚀 Starting Distributed Training Orchestrator Dashboard Demo")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create and configure the dashboard application
        app = DashboardApplication(
            dashboard_host="0.0.0.0",
            dashboard_port=8080
        )
        
        # Setup signal handlers for graceful shutdown
        app.setup_signal_handlers()
        
        print("📊 Dashboard will be available at: http://localhost:8080")
        print("🔧 Features included:")
        print("  • Real-time system metrics monitoring")
        print("  • Training progress visualization")
        print("  • Performance alerts and insights")
        print("  • Interactive charts and graphs")
        print("  • WebSocket-based live updates")
        print("  • REST API for metrics access")
        print("\n⚡ Press Ctrl+C to stop the dashboard\n")
        
        # Start the dashboard
        await app.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Received keyboard interrupt, shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        sys.exit(1)
    finally:
        print("✅ Dashboard stopped")


if __name__ == "__main__":
    asyncio.run(main())