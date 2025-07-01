###############################################################################
# Dashboard Orchestrator — Application Category (Business Logic Functors)
# Orchestrates dashboard state updates - NO direct I/O, pure coordination
###############################################################################
from typing import Optional
import time
from ..domain.dashboard_state import DashboardState, create_dashboard_state
from ..ports.dashboard_interfaces import DashboardDataSource, DashboardRenderer, DashboardPersistence


class DashboardOrchestrator:
    """Orchestrates dashboard state updates using pure function composition."""
    
    def __init__(self,
                 data_source: DashboardDataSource,
                 renderer: DashboardRenderer,
                 persistence: Optional[DashboardPersistence] = None):
        # Store interfaces (dependency injection)
        self.data_source = data_source
        self.renderer = renderer
        self.persistence = persistence
        
        # Evolution state (for tracking changes)
        self.last_state: Optional[DashboardState] = None
    
    def create_current_state(self) -> DashboardState:
        """Create current dashboard state by orchestrating data collection and pure calculations."""
        
        # Collect all data using injected interfaces (side effects isolated to ports)
        evolution_data = self.data_source.get_evolution_data()
        genetic_stats = self.data_source.get_genetic_stats()
        emergent_stats = self.data_source.get_emergent_behavior_stats()
        infrastructure_stats = self.data_source.get_infrastructure_stats()
        runtime_info = self.data_source.get_runtime_info()
        config = self.data_source.get_config()
        
        # FIX: Get real queue status from data source
        queue_status = {}
        if hasattr(self.data_source, 'get_queue_status'):
            queue_status = self.data_source.get_queue_status()
        
        # Unpack data tuples
        current_generation, max_generations, population = evolution_data
        adapters_trained, cache_hit_rate, models_cached, dataset_files = infrastructure_stats
        run_id, start_time, status, current_activity = runtime_info
        
        # Use pure domain function to create state (functional composition)
        state = create_dashboard_state(
            # Evolution data
            current_generation=current_generation,
            max_generations=max_generations,
            population=population,
            
            # Tracking data
            genetic_stats=genetic_stats,
            emergent_stats=emergent_stats,
            
            # Infrastructure data
            adapters_trained=adapters_trained,
            cache_hit_rate=cache_hit_rate,
            models_cached=models_cached,
            dataset_files=dataset_files,
            
            # Configuration
            config=config,
            
            # Runtime info
            run_id=run_id,
            start_time=start_time,
            status=status,
            current_activity=current_activity,
            
            # FIX: Pass real queue status
            queue_status=queue_status
        )
        
        return state
    
    def update_dashboard(self) -> DashboardState:
        """Update dashboard by creating new state and rendering it."""
        
        # Create new state using pure function composition
        current_state = self.create_current_state()
        
        # Check if state has changed (avoid unnecessary renders)
        if self._state_changed(current_state):
            # Clear and render new state
            self.renderer.clear_screen()
            self.renderer.render(current_state)
            
            # Persist state if configured
            if self.persistence:
                self.persistence.save_state(current_state)
            
            # Update tracking
            self.last_state = current_state
        
        return current_state
    
    def run_dashboard_loop(self, refresh_interval: float = 2.0, max_iterations: Optional[int] = None):
        """Run continuous dashboard update loop."""
        
        iteration = 0
        
        try:
            while True:
                # Update dashboard
                current_state = self.update_dashboard()
                
                # Check exit conditions
                if current_state.status in ['completed', 'failed']:
                    print(f"\n🏁 Evolution {current_state.status.upper()} - Dashboard stopping")
                    break
                
                if max_iterations and iteration >= max_iterations:
                    print(f"\n⏰ Maximum iterations ({max_iterations}) reached - Dashboard stopping")
                    break
                
                # Wait for next update
                time.sleep(refresh_interval)
                iteration += 1
                
        except KeyboardInterrupt:
            print(f"\n⏹️  Dashboard stopped by user")
        except Exception as e:
            print(f"\n❌ Dashboard error: {e}")
            raise
    
    def create_snapshot(self) -> DashboardState:
        """Create a single dashboard state snapshot without rendering."""
        return self.create_current_state()
    
    def _state_changed(self, new_state: DashboardState) -> bool:
        """Check if state has changed significantly (avoid unnecessary renders)."""
        if not self.last_state:
            return True
        
        # Compare key changing fields
        significant_changes = [
            new_state.current_generation != self.last_state.current_generation,
            new_state.adapters_trained != self.last_state.adapters_trained,
            new_state.total_behaviors != self.last_state.total_behaviors,
            new_state.status != self.last_state.status,
            abs(new_state.completion_percentage - self.last_state.completion_percentage) > 0.1,
        ]
        
        return any(significant_changes)


class DashboardFactory:
    """Factory for creating properly configured dashboard orchestrators."""
    
    @staticmethod
    def create(data_source: DashboardDataSource,
               renderer: DashboardRenderer,
               persistence: Optional[DashboardPersistence] = None) -> DashboardOrchestrator:
        """Create dashboard orchestrator with injected dependencies."""
        return DashboardOrchestrator(
            data_source=data_source,
            renderer=renderer,
            persistence=persistence
        )
    
    @staticmethod
    def create_live_dashboard(config_path: str = "coral_x_codellama_config.yaml",
                            refresh_interval: float = 2.0) -> DashboardOrchestrator:
        """Create live dashboard from configuration file - imports only here."""
        
        # Import implementations only when needed (lazy loading)
        from infra.dashboard.file_data_source import FileBasedDataSource
        from infra.dashboard.rich_renderer import RichDashboardRenderer
        
        # Create components
        data_source = FileBasedDataSource(config_path)
        renderer = RichDashboardRenderer()
        
        # Create orchestrator
        orchestrator = DashboardFactory.create(
            data_source=data_source,
            renderer=renderer,
            persistence=None  # No persistence for live dashboard
        )
        
        return orchestrator 