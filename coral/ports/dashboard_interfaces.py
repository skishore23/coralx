###############################################################################
# Dashboard Interfaces — Abstract Ports (Ports Category)
# Interface definitions for dashboard data sources - NO implementations
###############################################################################
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from ..domain.dashboard_state import DashboardState
from ..domain.neat import Population


class DashboardDataSource(ABC):
    """Abstract interface for dashboard data collection."""
    
    @abstractmethod
    def get_evolution_data(self) -> Tuple[int, int, Optional[Population]]:
        """Get (current_generation, max_generations, population)."""
        pass
    
    @abstractmethod
    def get_genetic_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get genetic operations statistics by generation."""
        pass
    
    @abstractmethod
    def get_emergent_behavior_stats(self) -> Dict[str, Any]:
        """Get emergent behavior detection statistics."""
        pass
    
    @abstractmethod
    def get_infrastructure_stats(self) -> Tuple[int, float, int, int]:
        """Get (adapters_trained, cache_hit_rate, models_cached, dataset_files)."""
        pass
    
    @abstractmethod
    def get_runtime_info(self) -> Tuple[str, float, str, str]:
        """Get (run_id, start_time, status, current_activity)."""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get complete configuration dictionary."""
        pass


class DashboardRenderer(ABC):
    """Abstract interface for dashboard rendering."""
    
    @abstractmethod
    def render(self, state: DashboardState) -> None:
        """Render dashboard state to output."""
        pass
    
    @abstractmethod
    def clear_screen(self) -> None:
        """Clear the display."""
        pass


class DashboardPersistence(ABC):
    """Abstract interface for dashboard state persistence."""
    
    @abstractmethod
    def save_state(self, state: DashboardState) -> None:
        """Save dashboard state."""
        pass
    
    @abstractmethod
    def load_last_state(self) -> Optional[DashboardState]:
        """Load last saved dashboard state."""
        pass 