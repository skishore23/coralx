###############################################################################
# Rich Dashboard Renderer — Infrastructure Category
# Terminal UI rendering using Rich library - handles all display logic
###############################################################################
import os
from datetime import datetime, timedelta
from typing import Optional

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.text import Text
    from rich.live import Live
    from rich.columns import Columns
    from rich.rule import Rule
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from coral.ports.dashboard_interfaces import DashboardRenderer
from coral.domain.dashboard_state import DashboardState


class RichDashboardRenderer(DashboardRenderer):
    """Rich terminal UI renderer for the CORAL-X dashboard."""
    
    def __init__(self):
        if not RICH_AVAILABLE:
            raise RuntimeError(
                "FAIL-FAST: Rich library not available. "
                "Install with: pip install rich"
            )
        
        self.console = Console()
        self.layout = self._create_layout()
    
    def render(self, state: DashboardState) -> None:
        """Render dashboard state using Rich layout."""
        
        # Update all panels with current state
        self.layout["header"].update(self._create_header(state))
        self.layout["left"].update(self._create_left_panels(state))
        self.layout["center"].update(self._create_center_panels(state))
        self.layout["right"].update(self._create_right_panels(state))
        self.layout["bottom"].update(self._create_bottom_panel(state))
        
        # Print the complete layout
        self.console.print(self.layout)
    
    def clear_screen(self) -> None:
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _create_layout(self) -> Layout:
        """Create the overall dashboard layout structure."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="bottom", size=1)
        )
        
        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="center", ratio=1),
            Layout(name="right", ratio=1)
        )
        
        return layout
    
    def _create_header(self, state: DashboardState) -> Panel:
        """Create the main header panel."""
        current_time = datetime.now().strftime('%H:%M:%S')
        
        # Status indicator
        status_indicators = {
            'starting': '🔄 STARTING',
            'evolving': '🧬 ACTIVE EVOLUTION',
            'completed': '✅ COMPLETED',
            'failed': '❌ FAILED'
        }
        status_display = status_indicators.get(state.status, '❓ UNKNOWN')
        
        header_text = Text()
        header_text.append("🎯 CORAL-X EVOLUTION DASHBOARD", style="bold cyan")
        header_text.append(f" | Run: {state.run_id}", style="dim")
        header_text.append(f" | {status_display}", style="bold green" if state.status == 'evolving' else "dim")
        header_text.append(f" | {current_time}", style="dim")
        
        return Panel(header_text, style="bright_blue")
    
    def _create_left_panels(self, state: DashboardState) -> Layout:
        """Create left column panels (Evolution Progress, Performance Metrics)."""
        left_layout = Layout()
        left_layout.split_column(
            Layout(name="evolution", size=8),
            Layout(name="performance", size=10),
            Layout(name="ca_metrics", size=8)
        )
        
        # Evolution Progress Panel
        evolution_panel = self._create_evolution_panel(state)
        left_layout["evolution"].update(evolution_panel)
        
        # Performance Metrics Panel
        performance_panel = self._create_performance_panel(state)
        left_layout["performance"].update(performance_panel)
        
        # CA Metrics Panel
        ca_panel = self._create_ca_panel(state)
        left_layout["ca_metrics"].update(ca_panel)
        
        return left_layout
    
    def _create_center_panels(self, state: DashboardState) -> Layout:
        """Create center column panels (Adapters, Genetic Operations)."""
        center_layout = Layout()
        center_layout.split_column(
            Layout(name="adapters", size=10),
            Layout(name="genetic", size=8),
            Layout(name="cheap_knobs", size=8)
        )
        
        # DoRA Adapters Panel
        adapters_panel = self._create_adapters_panel(state)
        center_layout["adapters"].update(adapters_panel)
        
        # Genetic Operations Panel
        genetic_panel = self._create_genetic_panel(state)
        center_layout["genetic"].update(genetic_panel)
        
        # Cheap Knobs Panel
        knobs_panel = self._create_cheap_knobs_panel(state)
        center_layout["cheap_knobs"].update(knobs_panel)
        
        return center_layout
    
    def _create_right_panels(self, state: DashboardState) -> Layout:
        """Create right column panels (Infrastructure, Queue Status, Emergent Behavior)."""
        right_layout = Layout()
        right_layout.split_column(
            Layout(name="infrastructure", size=8),
            Layout(name="queue_status", size=8),
            Layout(name="emergent", size=8),
            Layout(name="gpu_status", size=6)
        )
        
        # Infrastructure Panel
        infra_panel = self._create_infrastructure_panel(state)
        right_layout["infrastructure"].update(infra_panel)
        
        # Queue Status Panel
        queue_panel = self._create_queue_status_panel(state)
        right_layout["queue_status"].update(queue_panel)
        
        # Emergent Behavior Panel
        emergent_panel = self._create_emergent_panel(state)
        right_layout["emergent"].update(emergent_panel)
        
        # GPU Status Panel
        gpu_panel = self._create_gpu_panel(state)
        right_layout["gpu_status"].update(gpu_panel)
        
        return right_layout
    
    def _create_evolution_panel(self, state: DashboardState) -> Panel:
        """Create evolution progress panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="dim")
        table.add_column("Value")
        
        # Status
        status_icon = "🔥" if state.status == 'evolving' else "⏸️"
        table.add_row("Status:", f"{status_icon} {state.status.upper()}")
        
        # Generation
        table.add_row("Generation:", f"{state.current_generation}/{state.max_generations} ({state.completion_percentage:.1f}%)")
        
        # Progress bar
        progress_bar = self._create_progress_bar(state.completion_percentage)
        table.add_row("Progress:", progress_bar)
        
        # Population
        table.add_row("Population:", f"{state.population_size}")
        
        # Runtime
        runtime_str = self._format_duration(state.elapsed_time)
        table.add_row("Runtime:", runtime_str)
        
        return Panel(table, title="🧬 Evolution Progress", border_style="green")
    
    def _create_performance_panel(self, state: DashboardState) -> Panel:
        """Create performance metrics panel."""
        table = Table(show_header=True, box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Score", justify="center")
        table.add_column("Bar", width=20)
        table.add_column("Grade", justify="center")
        
        if state.best_scores:
            metrics = [
                ("🐛 Bugfix", state.best_scores.bugfix),
                ("🎨 Style", state.best_scores.style),
                ("🔒 Security", state.best_scores.security),
                ("⚡ Runtime", state.best_scores.runtime),
                ("📝 Syntax", getattr(state.best_scores, 'syntax', 0.0))
            ]
            
            for name, score in metrics:
                bar = self._create_score_bar(score)
                grade = self._score_to_grade(score)
                grade_style = self._get_grade_style(grade)
                table.add_row(
                    name, 
                    f"{score:.1%}", 
                    bar, 
                    Text(grade, style=grade_style)
                )
        else:
            table.add_row("No data", "---", "---", "---")
        
        return Panel(table, title="📊 Performance Metrics", border_style="yellow")
    
    def _create_ca_panel(self, state: DashboardState) -> Panel:
        """Create detailed CA metrics panel showing two-loop architecture."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Step", style="dim")
        table.add_column("Value")
        
        # Two-loop architecture status
        table.add_row("🔄 CA Evolution:", "✅ ACTIVE")
        table.add_row("🔍 Feature Extraction:", "✅ ACTIVE")
        table.add_row("🎛️ Cheap Knobs:", "✅ GENERATING")
        table.add_row("", "")  # Spacer
        
        # CA feature values (drives cheap knobs)
        table.add_row("🌀 Complexity:", f"{state.avg_ca_complexity:.3f} → temp")
        table.add_row("💫 Intensity:", f"{state.avg_ca_intensity:.3f} → top_p")
        table.add_row("🔄 Periodicity:", f"{state.avg_ca_period:.3f} → rep_penalty")
        table.add_row("📈 Convergence:", f"{state.avg_ca_convergence:.3f} → top_k")
        
        return Panel(table, title="🌊 Two-Loop CA → Cheap Knobs", border_style="blue")
    
    def _create_adapters_panel(self, state: DashboardState) -> Panel:
        """Create DoRA adapters panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="dim")
        table.add_column("Value")
        
        table.add_row("🎛️ Adapters Trained:", f"{state.adapters_trained}")
        table.add_row("🎯 Adapter Type:", "DoRA (Weight-Decomposed)")
        table.add_row("💾 Cache Hit Rate:", f"{state.cache_hit_rate:.1%}")
        table.add_row("⚡ Training Rate:", f"{state.training_rate:.1f} adapters/hour")
        table.add_row("📊 Parameter Space:", "240 combinations")
        
        return Panel(table, title="🎛️ DoRA Adapters", border_style="magenta")
    
    def _create_genetic_panel(self, state: DashboardState) -> Panel:
        """Create genetic operations panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Operation", style="dim")
        table.add_column("Count")
        table.add_column("Success Rate")
        
        table.add_row(
            "🔀 Crossovers:", 
            f"{state.crossover_count}",
            f"{state.crossover_success_rate:.1f}%"
        )
        table.add_row(
            "🧬 Mutations:", 
            f"{state.mutation_count}",
            f"{state.mutation_success_rate:.1f}%"
        )
        
        return Panel(table, title="🧬 Genetic Operations", border_style="cyan")
    
    def _create_cheap_knobs_panel(self, state: DashboardState) -> Panel:
        """Create cheap knobs configuration panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Parameter", style="dim")
        table.add_column("Range")
        
        for param, (min_val, max_val) in state.cheap_knobs.items():
            if param == 'temperature':
                display_name = "🌡️ Temperature:"
            elif param == 'top_p':
                display_name = "🎯 Top-P:"
            elif param == 'top_k':
                display_name = "🔢 Top-K:"
            else:
                display_name = f"{param.replace('_', ' ').title()}:"
            
            table.add_row(display_name, f"[{min_val}, {max_val}]")
        
        return Panel(table, title="🎛️ Cheap Knobs Configuration", border_style="yellow")
    
    def _create_infrastructure_panel(self, state: DashboardState) -> Panel:
        """Create infrastructure status panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Component", style="dim")
        table.add_column("Status")
        
        # Models status
        models_icon = "✅" if state.models_cached > 0 else "❌"
        table.add_row("🧠 Models:", f"{models_icon} {state.models_cached} cached")
        
        # Dataset status
        dataset_icon = "✅" if state.dataset_files > 0 else "❌"
        table.add_row("📚 Dataset:", f"{dataset_icon} {state.dataset_files} files")
        
        # Modal status
        modal_icon = "🟢" if state.modal_status == "READY" else "🔴"
        table.add_row("🌐 Modal App:", f"{modal_icon} {state.modal_status}")
        
        # Volume status
        table.add_row("💾 Volume:", "🟢 MOUNTED")
        
        return Panel(table, title="💻 Infrastructure", border_style="green")
    
    def _create_emergent_panel(self, state: DashboardState) -> Panel:
        """Create emergent behavior panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="dim")
        table.add_column("Value")
        
        # Status
        status_icon = "🟢" if state.emergent_active else "🔴"
        status_text = "ACTIVE" if state.emergent_active else "INACTIVE"
        table.add_row("Status:", f"{status_icon} {status_text}")
        
        # Detection stats
        table.add_row("🔍 Behaviors Detected:", f"{state.total_behaviors}")
        table.add_row("📊 Detection Rate:", f"{state.detection_rate:.1f}%")
        
        # Recent behaviors
        if state.recent_behaviors:
            table.add_row("", "")  # Spacer
            table.add_row("Recent Behaviors:", "")
            for behavior in state.recent_behaviors[-3:]:
                table.add_row("", f"• {behavior}")
        
        return Panel(table, title="🌟 Emergent Behavior", border_style="purple")
    
    def _create_queue_status_panel(self, state: DashboardState) -> Panel:
        """Create queue status panel for category theory queue monitoring."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Queue", style="dim")
        table.add_column("Status")
        
        # Queue status (if available in state)
        queue_data = getattr(state, 'queue_status', {})
        
        training_count = queue_data.get('training_queue', 0)
        test_count = queue_data.get('test_queue', 0)
        results_count = queue_data.get('results_queue', 0)
        pending_jobs = queue_data.get('pending_jobs', 0)
        
        table.add_row("🏗️ Training:", f"{training_count} jobs")
        table.add_row("🧪 Evaluation:", f"{test_count} jobs")
        table.add_row("📡 Results:", f"{results_count} ready")
        table.add_row("⏳ Pending:", f"{pending_jobs} waiting")
        
        # Queue health indicator
        total_active = training_count + test_count + pending_jobs
        if total_active > 0:
            health_icon = "🔥"
            health_status = "ACTIVE"
        else:
            health_icon = "💤"
            health_status = "IDLE"
        
        table.add_row("", "")  # Spacer
        table.add_row("🧮 Queue Health:", f"{health_icon} {health_status}")
        
        return Panel(table, title="📊 Queue Status", border_style="cyan")
    
    def _create_gpu_panel(self, state: DashboardState) -> Panel:
        """Create GPU status panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Component", style="dim")
        table.add_column("Status")
        
        table.add_row("🖥️ GPU Type:", state.gpu_type)
        table.add_row("🔋 Status:", "🟢 READY")
        table.add_row("📡 Monitoring:", "Awaiting generation...")
        
        return Panel(table, title="🖥️ GPU Status", border_style="bright_green")
    
    def _create_bottom_panel(self, state: DashboardState) -> Text:
        """Create bottom status line."""
        status_text = Text()
        status_text.append("🔄 Auto-refresh: 2s", style="dim")
        status_text.append(" | ", style="dim")
        status_text.append("Press Ctrl+C to exit", style="dim")
        return status_text
    
    # Helper methods
    
    def _create_progress_bar(self, percentage: float, width: int = 20) -> Text:
        """Create a text-based progress bar."""
        filled = int(percentage / 100 * width)
        bar = "█" * filled + "░" * (width - filled)
        
        progress_text = Text()
        progress_text.append("[", style="dim")
        progress_text.append(bar, style="green")
        progress_text.append("]", style="dim")
        progress_text.append(f" {percentage:.1f}%", style="bright_white")
        
        return progress_text
    
    def _create_score_bar(self, score: float, width: int = 15) -> Text:
        """Create a score visualization bar."""
        filled = int(score * width)
        empty = width - filled
        
        bar_text = Text()
        bar_text.append("█" * filled, style="green")
        bar_text.append("░" * empty, style="dim")
        
        return bar_text
    
    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 0.9:
            return "A+"
        elif score >= 0.85:
            return "A"
        elif score >= 0.8:
            return "A-"
        elif score >= 0.75:
            return "B+"
        elif score >= 0.7:
            return "B"
        elif score >= 0.65:
            return "B-"
        elif score >= 0.6:
            return "C"
        else:
            return "D"
    
    def _get_grade_style(self, grade: str) -> str:
        """Get style for grade display."""
        if grade.startswith('A'):
            return "bold green"
        elif grade.startswith('B'):
            return "yellow"
        elif grade.startswith('C'):
            return "orange1"
        else:
            return "red"
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"


class SimpleDashboardRenderer(DashboardRenderer):
    """Fallback simple text renderer when Rich is not available."""
    
    def render(self, state: DashboardState) -> None:
        """Render dashboard state as simple text."""
        print("=" * 60)
        print(f"🎯 CORAL-X EVOLUTION DASHBOARD | Run: {state.run_id}")
        print("=" * 60)
        
        print(f"📊 Generation: {state.current_generation}/{state.max_generations} ({state.completion_percentage:.1f}%)")
        print(f"🧬 Status: {state.status.upper()}")
        print(f"🎛️ Adapters: {state.adapters_trained} trained")
        print(f"🌟 Emergent: {state.total_behaviors} behaviors detected")
        print(f"⏱️ Runtime: {self._format_duration(state.elapsed_time)}")
        
        print("=" * 60)
    
    def clear_screen(self) -> None:
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        return f"{seconds:.0f}s" 