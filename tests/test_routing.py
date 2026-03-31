"""
Unit tests for the AStats Gen-3 Architecture.
Verifies Routing, Specialist loops, Critic verification, and HTML Reporting.
"""
import pytest
import os
from unittest.mock import patch, MagicMock, call
from astats.agent.core import AStatsAgent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(content: str):
    """Build a minimal router-compatible response object."""
    from astats.router import CompletionResponse
    raw = {
        "choices": [
            {"message": {"role": "assistant", "content": content, "tool_calls": None}}
        ]
    }
    return CompletionResponse(raw)

def _make_tool_response(fn_name: str, arguments: str, tool_id: str = "tc_001"):
    """Build a response that requests a tool call."""
    from astats.router import CompletionResponse
    raw = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_id,
                            "type": "function",
                            "function": {"name": fn_name, "arguments": arguments},
                        }
                    ],
                }
            }
        ]
    }
    return CompletionResponse(raw)

# ---------------------------------------------------------------------------
# Test Router Resolution (Gen-3 Multi-Provider)
# ---------------------------------------------------------------------------

class TestRouterResolution:
    def test_provider_prefixes(self):
        from astats.router import _resolve_provider
        
        # Ollama
        p, m, b = _resolve_provider("ollama/llama3", None)
        assert p == "ollama"
        assert m == "llama3"
        
        # Gemini
        p, m, b = _resolve_provider("gemini/gemini-pro", None)
        assert p == "gemini"
        assert "googleapis.com" in b
        
        # Groq
        p, m, b = _resolve_provider("groq/mixtral-8x7b", None)
        assert p == "groq"
        assert "groq.com" in b
        
        # Anthropic
        p, m, b = _resolve_provider("anthropic/claude-3-sonnet", None)
        assert p == "anthropic"
        assert "anthropic.com" in b

# ---------------------------------------------------------------------------
# Test Critic Agent & Reporting Logic
# ---------------------------------------------------------------------------

class TestAgentGen3Flow:
    def setup_method(self):
        # Create a dummy config
        self.config = {
            "agents": {
                "supervisor": "ollama/llama3.2:1b",
                "python_specialist": "openai/gpt-4o",
                "r_specialist": "openai/gpt-4o"
            }
        }
        self.agent = AStatsAgent()
        self.agent.models["supervisor"] = "ollama/llama3.2:1b"
        self.agent.models["python_specialist"] = "openai/gpt-4o"
        self.agent.models["r_specialist"] = "openai/gpt-4o"

    @patch("astats.agent.core.completion")
    @patch("astats.agent.core.discover_dataset", return_value="Summary")
    @patch("astats.agent.core.generate_html_report", return_value="report.html")
    def test_critic_rejects_then_approves(self, mock_report, mock_discover, mock_completion):
        """
        Tests the Supervisor -> Specialist -> Critic (Reject) -> Specialist (Fix) -> Critic (Approve) -> Report flow.
        """
        # 1. Supervisor says PYTHON
        resp_supervisor = _make_response("PYTHON")
        
        # 2. Specialist attempt 1 (Flawed)
        resp_specialist_1 = _make_response("Here is my p-value: 0.04.")
        
        # 3. Critic rejects attempt 1
        resp_critic_reject = _make_response("REJECT: You didn't check for normality.")
        
        # 4. Specialist attempt 2 (Fixed)
        resp_specialist_2 = _make_response("Checked normality (p=0.2). t-test p-value: 0.05.")
        
        # 5. Critic approves attempt 2
        resp_critic_approve = _make_response("APPROVE")

        mock_completion.side_effect = [
            resp_supervisor,    # Supervisor
            resp_specialist_1,  # Specialist 1
            resp_critic_reject, # Critic 1
            resp_specialist_2,  # Specialist 2
            resp_critic_approve # Critic 2
        ]

        self.agent.ask("test.csv", "Perform analysis")

        # Verify loop counts
        # Expected calls: 1 (supervisor) + 2 (specialist attempts) + 2 (critic attempts) = 5
        assert mock_completion.call_count == 5
        assert mock_report.called

    @patch("astats.agent.core.completion")
    @patch("astats.agent.core.discover_dataset", return_value="Summary")
    @patch("astats.agent.core.generate_html_report")
    def test_reporting_embeds_plots(self, mock_report, mock_discover, mock_completion):
        """Verify that the HTML reporter is called with the correct plot path if it exists."""
        mock_completion.side_effect = [
            _make_response("PYTHON"),
            _make_response("Done"), 
            _make_response("APPROVE")
        ]
        
        # Simulate plot generation
        with open("astats_plot.png", "w") as f: f.write("fake plot data")
        
        try:
            self.agent.ask("test.csv", "Plot data")
            call_args = mock_report.call_args[1]
            assert "astats_plot.png" in call_args["plot_path"]
        finally:
            if os.path.exists("astats_plot.png"): os.remove("astats_plot.png")

# ---------------------------------------------------------------------------
# Legacy compatibility (ensure core routing still works)
# ---------------------------------------------------------------------------

class TestLegacyRouting:
    def setup_method(self):
        self.agent = AStatsAgent()

    @patch("astats.agent.core.run_python_code", return_value="Error: NameError")
    @patch("astats.agent.core.completion")
    def test_error_feedback_loop(self, mock_completion, mock_run):
        """Tool errors should still be fed back to the specialist inside run_agent_loop."""
        step1 = _make_tool_response("run_python_code", '{"code": "bad code"}')
        step2 = _make_response("Fixed it.")
        mock_completion.side_effect = [step1, step2]

        result = self.agent.run_agent_loop(
            agent_name="Python-Specialist",
            model="ollama/llama3",
            system_prompt="System",
            user_prompt="User",
            tools=[self.agent.python_tool]
        )

        assert mock_completion.call_count == 2
        assert result == "Fixed it."
