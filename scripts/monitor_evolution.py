#!/usr/bin/env python3
"""
Real-time CORAL-X Evolution Monitor
Watch live progress from Modal without checking logs
"""

import time
import json
from datetime import datetime
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn
from rich.layout import Layout

import modal

console = Console()

def get_modal_progress():
    """Get real-time progress from Modal."""
    try:
        app = modal.App.lookup("coral-x-production", create_if_missing=False)
        progress_fn = modal.Function.from_name("coral-x-production", "get_evolution_progress_modal")
        return progress_fn.remote({})
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Connection failed: {e}',
            'current_generation': 0,
            'max_generations': 0,
            'best_fitness': 0.0,
            'elapsed_time': 0.0
        }

def create_progress_panel(progress_data):
    """Create progress information panel."""
    status = progress_data.get('status', 'unknown')
    message = progress_data.get('message', 'No message')
    current_gen = progress_data.get('current_generation', 0)
    max_gen = progress_data.get('max_generations', 20)
    best_fitness = progress_data.get('best_fitness', 0.0)
    elapsed = progress_data.get('elapsed_time', 0.0)
    
    # Status color
    status_colors = {
        'initializing': 'yellow',
        'configuring': 'blue', 
        'loading_plugin': 'blue',
        'creating_population': 'cyan',
        'creating_engine': 'cyan',
        'evolving': 'green',
        'completed': 'bright_green',
        'failed': 'red',
        'error': 'red'
    }
    status_color = status_colors.get(status, 'white')
    
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Label", style="dim cyan")
    table.add_column("Value", style="white bold")
    
    table.add_row("Status:", f"[{status_color}]{status.upper().replace('_', ' ')}[/{status_color}]")
    table.add_row("Message:", message[:60] + "..." if len(message) > 60 else message)
    table.add_row("Generation:", f"{current_gen}/{max_gen}")
    
    # Progress bar
    if max_gen > 0:
        progress_pct = (current_gen / max_gen) * 100
        filled = int((current_gen / max_gen) * 20)
        bar = "█" * filled + "░" * (20 - filled)
        table.add_row("Progress:", f"[{bar}] {progress_pct:.1f}%")
    
    table.add_row("Best Fitness:", f"{best_fitness:.4f}" if best_fitness > 0 else "Not available")
    table.add_row("Elapsed Time:", f"{elapsed/60:.1f} minutes" if elapsed > 0 else "Not started")
    
    return Panel(table, title="🧬 CORAL-X Evolution Monitor", border_style="green")

def create_instruction_panel():
    """Create instructions panel."""
    instructions = [
        "📺 Real-time evolution monitoring",
        "🔄 Updates every 2 seconds",
        "⌨️  Press Ctrl+C to exit",
        "",
        "🎯 To start evolution:",
        "   ./coralx run config/test.yaml",
        "",
        "📊 To view dashboard:",
        "   ./coralx dashboard"
    ]
    
    text = "\n".join(instructions)
    return Panel(text, title="📋 Instructions", border_style="blue")

def main():
    """Main monitoring loop."""
    console.clear()
    console.print("[bold green]🚀 Starting CORAL-X Evolution Monitor...[/bold green]")
    console.print("[dim]Connecting to Modal...[/dim]\n")
    
    layout = Layout()
    layout.split_column(
        Layout(name="progress", size=15),
        Layout(name="instructions", size=12)
    )
    
    with Live(layout, console=console, screen=True, redirect_stderr=False) as live:
        try:
            while True:
                # Get latest progress
                progress_data = get_modal_progress()
                
                # Update panels
                layout["progress"].update(create_progress_panel(progress_data))
                layout["instructions"].update(create_instruction_panel())
                
                # Check if evolution completed
                status = progress_data.get('status', '')
                if status in ['completed', 'failed']:
                    time.sleep(5)  # Show final status for 5 seconds
                    break
                
                time.sleep(2)  # Update every 2 seconds
                
        except KeyboardInterrupt:
            console.print("\n[yellow]🛑 Monitor stopped by user[/yellow]")

if __name__ == "__main__":
    main() 