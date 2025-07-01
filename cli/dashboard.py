#!/usr/bin/env python3
###############################################################################
# Dashboard CLI — Terminal Category (User Interface)
# Command-line interface for the CORAL-X realtime dashboard
###############################################################################
import sys
import argparse
from pathlib import Path

# Add the current directory to Python path for coral imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    """Main CLI entry point for CORAL-X Dashboard."""
    parser = argparse.ArgumentParser(
        description="CORAL-X Evolution Dashboard - Real-time monitoring command center",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Use default config
  %(prog)s -c my_config.yaml                  # Use custom config  
  %(prog)s --refresh 1.0                      # 1-second refresh rate
  %(prog)s --snapshot                         # Single snapshot, no loop
  %(prog)s --simple                           # Simple text mode (no Rich)

Dashboard displays:
  🧬 Evolution Progress    📊 Performance Metrics    🎛️  DoRA Adapters
  🌊 CA Metrics           🧬 Genetic Operations     🌟 Emergent Behavior  
  💻 Infrastructure       🖥️  GPU Status           🎛️  Cheap Knobs Config
        """
    )
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='coral_x_codellama_config.yaml',
        help='Configuration file path (default: coral_x_codellama_config.yaml)'
    )
    
    parser.add_argument(
        '--refresh',
        type=float,
        default=2.0,
        help='Dashboard refresh interval in seconds (default: 2.0)'
    )
    
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='Create single snapshot instead of continuous monitoring'
    )
    
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Use simple text renderer (fallback mode)'
    )
    
    parser.add_argument(
        '--max-iterations',
        type=int,
        help='Maximum number of dashboard updates (for testing)'
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Error: Configuration file not found: {config_path}")
        print(f"💡 Make sure you're in the correct directory and the config file exists")
        sys.exit(1)
    
    try:
        # Start dashboard using functional composition
        run_dashboard(
            config_path=str(config_path),
            refresh_interval=args.refresh,
            snapshot_mode=args.snapshot,
            simple_mode=args.simple,
            max_iterations=args.max_iterations
        )
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Dashboard stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        sys.exit(1)


def run_dashboard(config_path: str,
                 refresh_interval: float = 2.0,
                 snapshot_mode: bool = False,
                 simple_mode: bool = False,
                 max_iterations: int = None):
    """
    Run the dashboard using functional composition.
    
    This is the main orchestration function that creates and coordinates
    all the components following the category theory architecture.
    """
    
    print(f"🎯 CORAL-X Dashboard starting...")
    print(f"   📄 Config: {config_path}")
    print(f"   🔄 Refresh: {refresh_interval}s")
    print(f"   📊 Mode: {'Snapshot' if snapshot_mode else 'Live monitoring'}")
    
    try:
        # Create dashboard orchestrator using functional composition
        orchestrator = create_dashboard_orchestrator(config_path, simple_mode)
        
        if snapshot_mode:
            # Single snapshot mode
            print(f"📸 Creating dashboard snapshot...")
            state = orchestrator.create_snapshot()
            orchestrator.renderer.clear_screen()
            orchestrator.renderer.render(state)
            print(f"✅ Snapshot complete")
        else:
            # Continuous monitoring mode
            print(f"🚀 Starting live dashboard monitoring...")
            print(f"   Press Ctrl+C to stop")
            print()
            
            orchestrator.run_dashboard_loop(
                refresh_interval=refresh_interval,
                max_iterations=max_iterations
            )
            
    except ImportError as e:
        if "rich" in str(e).lower() and not simple_mode:
            print(f"⚠️  Rich library not available, falling back to simple mode")
            print(f"💡 Install Rich for better display: pip install rich")
            run_dashboard(config_path, refresh_interval, snapshot_mode, True, max_iterations)
        else:
            raise


def create_dashboard_orchestrator(config_path: str, simple_mode: bool = False):
    """
    Create dashboard orchestrator using dependency injection and functional composition.
    
    This function demonstrates the category theory architecture:
    - Pure domain functions (dashboard state calculations)
    - Application layer (orchestrator coordination)  
    - Infrastructure layer (data sources and renderers)
    - Ports layer (abstract interfaces)
    """
    
    # Import infrastructure implementations (lazy loading)
    from coral.application.dashboard_orchestrator import DashboardFactory
    from infra.dashboard.file_data_source import FileBasedDataSource
    
    if simple_mode:
        from infra.dashboard.rich_renderer import SimpleDashboardRenderer
        renderer = SimpleDashboardRenderer()
    else:
        from infra.dashboard.rich_renderer import RichDashboardRenderer
        renderer = RichDashboardRenderer()
    
    # Load raw configuration dictionary instead of CoralConfig object
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    data_source = FileBasedDataSource(config)
    
    # Create orchestrator using functional composition (dependency injection)
    orchestrator = DashboardFactory.create(
        data_source=data_source,
        renderer=renderer,
        persistence=None  # No persistence for CLI dashboard
    )
    
    return orchestrator


def test_dashboard_components():
    """Test dashboard components in isolation (useful for debugging)."""
    print("🧪 Testing dashboard components...")
    
    try:
        # Test domain functions
        from coral.domain.dashboard_state import calculate_evolution_progress
        
        progress = calculate_evolution_progress(5, 50, None)
        print(f"✅ Domain functions: {progress['completion_percentage']:.1f}% complete")
        
        # Test data source
        from infra.dashboard.file_data_source import FileBasedDataSource
        
        if Path("coral_x_codellama_config.yaml").exists():
            data_source = FileBasedDataSource("coral_x_codellama_config.yaml")
            evolution_data = data_source.get_evolution_data()
            print(f"✅ Data source: Generation {evolution_data[0]}/{evolution_data[1]}")
        else:
            print("⚠️  No config file found for data source test")
        
        # Test renderer
        try:
            from infra.dashboard.rich_renderer import RichDashboardRenderer
            renderer = RichDashboardRenderer()
            print("✅ Rich renderer available")
        except ImportError:
            from infra.dashboard.rich_renderer import SimpleDashboardRenderer
            renderer = SimpleDashboardRenderer()
            print("✅ Simple renderer available (Rich not installed)")
        
        print("🎉 All components working correctly")
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    # Special test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_dashboard_components()
    else:
        main() 