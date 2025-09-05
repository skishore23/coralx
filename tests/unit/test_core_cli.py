"""Simple tests for simplified CLI."""

def test_cli_import():
    """Test that CLI main can be imported."""
    from core.cli.main import main
    assert main is not None

def test_cli_help():
    """Test that CLI shows help without arguments."""
    import subprocess
    import sys

    result = subprocess.run([
        sys.executable, '-m', 'core.cli.main'
    ], capture_output=True, text=True)

    assert result.returncode == 1  # Should exit with error code for no command
    assert "CORAL-X Evolution Framework" in result.stdout
