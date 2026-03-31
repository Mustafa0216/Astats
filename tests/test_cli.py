"""
CLI integration tests for AStats.
Tests basic invocation of the `simulate` and `explore` subcommands.
"""
import os
import subprocess
import sys
import pytest


PYTHON = sys.executable
CWD = os.path.join(os.path.dirname(__file__), "..")


def run_cli(*args, timeout=30):
    """Helper to invoke the AStats CLI and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [PYTHON, "-m", "astats.cli", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=CWD,
    )
    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# simulate command
# ---------------------------------------------------------------------------

class TestSimulateCLI:
    def test_simulate_creates_csv(self, tmp_path):
        """simulate command produces a valid CSV file."""
        out_file = tmp_path / "trial.csv"
        rc, stdout, stderr = run_cli(
            "simulate", "--rows", "50", "--output", str(out_file)
        )
        assert rc == 0, f"CLI error:\n{stderr}"
        assert out_file.exists(), "Output CSV file was not created."
        lines = out_file.read_text().strip().splitlines()
        # Header + 50 data rows
        assert len(lines) == 51, f"Expected 51 lines, got {len(lines)}"

    def test_simulate_default_output(self, tmp_path, monkeypatch):
        """simulate command works with default arguments."""
        monkeypatch.chdir(tmp_path)
        rc, stdout, stderr = run_cli("simulate")
        assert rc == 0, f"CLI error:\n{stderr}"


# ---------------------------------------------------------------------------
# explore command — help / interface only (no LLM required)
# ---------------------------------------------------------------------------

class TestExploreCLI:
    def test_explore_missing_args_exits_nonzero(self):
        """explore command with no arguments should exit with an error."""
        rc, stdout, stderr = run_cli("explore")
        assert rc != 0, "Expected non-zero exit for missing arguments."

    def test_explore_missing_file_exits_nonzero(self):
        """explore command with a non-existent file should fail gracefully."""
        rc, stdout, stderr = run_cli(
            "explore", "nonexistent_file.csv", "What is the mean?"
        )
        assert rc != 0, "Expected non-zero exit for missing data file."

    def test_simulate_then_explore_with_mock(self, tmp_path, monkeypatch):
        """
        End-to-end: simulate data, then call explore with a mocked LLM response.
        This ensures the CLI wires correctly without live Ollama.
        """
        from unittest.mock import patch, MagicMock

        out_file = tmp_path / "trial.csv"
        rc, _, err = run_cli("simulate", "--rows", "20", "--output", str(out_file))
        assert rc == 0, f"Simulate failed: {err}"

        # Patch AStatsAgent.ask to avoid real LLM call
        with patch("astats.agent.core.AStatsAgent.ask") as mock_ask:
            mock_ask.return_value = "Mock answer from specialist."
            rc2, stdout2, stderr2 = run_cli(
                "explore", str(out_file), "What is the mean recovery time?"
            )
        assert rc2 == 0, f"Explore CLI failed:\n{stderr2}"
