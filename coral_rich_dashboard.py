#!/usr/bin/env python3
"""
Simple Modal Dashboard - Clean log viewer for CORAL-X evolution
Uses Modal CLI directly for maximum simplicity
"""

import subprocess
import time
import re
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text

console = Console()

def get_modal_logs(app_name="coral-x-production", lines=50):
    """Get logs from Modal using CLI"""
    try:
        result = subprocess.run(
            ["modal", "app", "logs", app_name, "--lines", str(lines)],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip().split('\n')
        else:
            return [f"Error getting logs: {result.stderr}"]
    except subprocess.TimeoutExpired:
        return ["Timeout getting logs"]
    except subprocess.CalledProcessError as e:
        return [f"Modal CLI error: {e}"]
    except Exception as e:
        return [f"Error: {e}"]

def clean_log_line(line):
    """Remove timestamps and clean up log line"""
    # Remove Modal timestamps (YYYY-MM-DD HH:MM:SS format)
    line = re.sub(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} ', '', line)
    
    # Find first emoji and strip everything before it
    emoji_match = re.search(r'[🔸📈✅🔍🧬🌟🎛️⚡💾🎯🚀]', line)
    if emoji_match:
        return line[emoji_match.start():]
    
    return line

def categorize_log_line(line):
    """Categorize log lines by content"""
    if '🔸' in line or 'Problem' in line:
        return 'problem'
    elif '📈' in line or 'Scores:' in line:
        return 'scores'
    elif '✅' in line or 'Test results:' in line:
        return 'results'
    elif '🧬' in line or 'Genome' in line:
        return 'genome'
    elif '🎛️' in line or any(param in line for param in ['T=', 'p=', 'k=']):
        return 'knobs'
    elif '⚡' in line or 'Generation' in line:
        return 'generation'
    else:
        return 'other'

def create_dashboard():
    """Create the main dashboard layout"""
    layout = Layout()
    
    # Split into sections
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
    )
    
    layout["main"].split_row(
        Layout(name="problems", ratio=2),
        Layout(name="evolution", ratio=1),
    )
    
    return layout

def update_dashboard(layout, logs):
    """Update dashboard with latest logs"""
    # Clean and categorize logs
    cleaned_logs = [clean_log_line(line) for line in logs if line.strip()]
    
    # Create header
    header_text = Text("CORAL-X Evolution Dashboard", style="bold cyan")
    header_text.append(f" • {len(cleaned_logs)} log entries", style="dim")
    layout["header"].update(Panel(header_text, border_style="cyan"))
    
    # Problem/Results table
    problems_table = Table(title="Problems & Results", show_header=True, header_style="bold magenta")
    problems_table.add_column("Entry", style="cyan", no_wrap=True)
    
    # Evolution info table  
    evolution_table = Table(title="Evolution Info", show_header=True, header_style="bold green")
    evolution_table.add_column("Parameter", style="yellow")
    evolution_table.add_column("Value", style="white")
    
    # Track latest values
    latest_knobs = {}
    generation_info = []
    
    for line in cleaned_logs[-30:]:  # Show last 30 lines
        category = categorize_log_line(line)
        
        if category in ['problem', 'scores', 'results']:
            problems_table.add_row(line[:80] + "..." if len(line) > 80 else line)
        
        elif category == 'knobs':
            # Extract T=, p=, k= values
            temp_match = re.search(r'T=([0-9.]+)', line)
            p_match = re.search(r'p=([0-9.]+)', line)
            k_match = re.search(r'k=([0-9]+)', line)
            
            if temp_match:
                latest_knobs['Temperature'] = temp_match.group(1)
            if p_match:
                latest_knobs['Top-p'] = p_match.group(1)
            if k_match:
                latest_knobs['Top-k'] = k_match.group(1)
        
        elif category in ['genome', 'generation']:
            if len(generation_info) < 10:  # Keep last 10 generation entries
                generation_info.append(line[:50] + "..." if len(line) > 50 else line)
    
    # Add evolution parameters
    for param, value in latest_knobs.items():
        evolution_table.add_row(param, value)
    
    evolution_table.add_row("", "")  # Spacer
    evolution_table.add_row("[bold]Recent Generation Info[/bold]", "")
    for info in generation_info[-5:]:  # Show last 5
        evolution_table.add_row("", info)
    
    layout["problems"].update(problems_table)
    layout["evolution"].update(evolution_table)

def main():
    """Main dashboard loop"""
    console.print("[bold green]Starting CORAL-X Dashboard...[/bold green]")
    console.print("Press Ctrl+C to exit\n")
    
    layout = create_dashboard()
    
    with Live(layout, console=console, screen=True, redirect_stderr=False) as live:
        try:
            while True:
                logs = get_modal_logs()
                update_dashboard(layout, logs)
                time.sleep(2)  # Update every 2 seconds
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Dashboard stopped by user[/yellow]")

if __name__ == "__main__":
    main() 