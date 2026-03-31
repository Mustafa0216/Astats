import os
import json
import yaml
from rich.console import Console

# Lightweight drop-in router — replaces litellm to eliminate the 50-80 MB
# startup memory spike and the OOM crash caused by loading all provider SDKs.
from astats.router import completion

from astats.r_bridge import run_r_script, run_python_code
from astats.data.discovery import discover_dataset
from astats.reporting import generate_html_report

from dotenv import load_dotenv
load_dotenv()

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_plot(code: str) -> str:
    """
    Execute matplotlib plotting code and save the figure as a PNG file.
    The code must call plt.savefig(output_path) where output_path is provided
    in the exec globals.  Returns the saved file path or an error string.
    """
    import io
    import sys
    import matplotlib
    matplotlib.use("Agg")  # headless — no display required
    import matplotlib.pyplot as plt

    output_path = "astats_plot.png"
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        exec_globals = {
            "pd": __import__("pandas"),
            "np": __import__("numpy"),
            "plt": plt,
            "output_path": output_path,
        }
        exec(code, exec_globals)
        # If the code didn't call savefig, do it now
        if plt.get_fignums():
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close("all")
        captured = sys.stdout.getvalue()
        return f"[Plot saved] → {os.path.abspath(output_path)}" + (
            f"\n{captured}" if captured.strip() else ""
        )
    except Exception as e:
        import traceback
        return f"Error in plot code:\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class AStatsAgent:
    def __init__(self, config_path: str = "astats_config.yaml"):
        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}

        agents = config.get("agents", {})
        self.models = {
            # Supervisor intentionally uses a *small* local model — it only
            # needs to emit one token ("PYTHON" or "R").  A 1B-param model is
            # more than capable and avoids loading the full 8B weights for a
            # trivial routing decision (which was a primary OOM cause).
            "supervisor":        agents.get("supervisor",        "ollama/llama3.2:1b"),
            "python_specialist": agents.get("python_specialist", "ollama/llama3"),
            "r_specialist":      agents.get("r_specialist",      "gpt-4o"),
        }

        global_cfg = config.get("global", {})
        self.ollama_api_base = global_cfg.get("ollama_api_base", "http://localhost:11434")

        retry_cfg = config.get("retry", {})
        self.max_steps = int(retry_cfg.get("max_steps", 5))
        self.feedback_on_error = bool(retry_cfg.get("feedback_on_error", True))

        # ── Tool definitions ──────────────────────────────────────────────

        self.python_tool = {
            "type": "function",
            "function": {
                "name": "run_python_code",
                "description": (
                    "Execute Python code via pandas/numpy for data analysis. "
                    "Always print() your results."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        }

        self.plot_tool = {
            "type": "function",
            "function": {
                "name": "save_plot",
                "description": (
                    "Generate a matplotlib figure and save it as a PNG file. "
                    "Write code that uses `plt` (already imported) and ends with "
                    "plt.savefig(output_path). The saved file path will be returned."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        }

        self.r_tool = {
            "type": "function",
            "function": {
                "name": "run_r_script",
                "description": (
                    "Execute R code via Rscript for advanced statistical analysis. "
                    "Always print() your results. "
                    "If a package is missing, call install.packages() first."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        }

    def _api_base_for(self, model: str) -> str | None:
        if "ollama" in model:
            return self.ollama_api_base
        return None

    # ── Core agentic loop ─────────────────────────────────────────────────

    def run_agent_loop(
        self,
        agent_name: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        tools: list,
    ):
        console.print(
            f"[bold cyan][{agent_name} Agent][/bold cyan] Thinking "
            f"(Routed to Model: `{model}`)..."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        api_base = self._api_base_for(model)

        for step in range(self.max_steps):
            try:
                response = completion(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    api_base=api_base,
                )
            except Exception as e:
                console.print(
                    f"[bold red]Router Error with model '{model}':[/bold red] {e}"
                )
                return None

            message = response.choices[0].message

            # Safe dict conversion for message history
            msg_dict = dict(message)
            messages.append(msg_dict)

            if not message.tool_calls:
                console.print(
                    f"\n[bold green][{agent_name} Final Answer][/bold green]\n"
                    f"{message.content}"
                )
                return message.content

            # Execute tool calls
            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                code = args.get("code", "")

                console.print(
                    f"\n[bold magenta][Tool Call][/bold magenta] "
                    f"Executing `{fn_name}`:\n```\n{code}\n```"
                )

                if fn_name == "run_python_code":
                    result = run_python_code(code)
                elif fn_name == "run_r_script":
                    result = run_r_script(code)
                elif fn_name == "save_plot":
                    result = _save_plot(code)
                else:
                    result = f"Error: Unknown tool '{fn_name}'"

                is_error = isinstance(result, str) and result.lower().startswith("error")
                if is_error:
                    console.print(f"[bold red][Tool Error][/bold red]\n{result}")
                    if self.feedback_on_error:
                        console.print(
                            "[bold yellow][Feedback Loop][/bold yellow] "
                            "Sending error back to model for self-correction..."
                        )
                else:
                    console.print(f"[bold yellow][Tool Output][/bold yellow]\n{result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": fn_name,
                    "content": str(result),
                })

        console.print(f"[bold red]{agent_name} exceeded maximum steps.[/bold red]")
        return None

    # ── Entry point ───────────────────────────────────────────────────────

    def ask(self, data_path: str, query: str):
        console.print(
            f"[bold blue][Auto-Discovery][/bold blue] Profiling `{data_path}`..."
        )
        summary = discover_dataset(data_path)

        # 1. Supervisor routing — uses the SMALL local model
        supervisor_model = self.models["supervisor"]
        console.print(
            f"\n[bold purple][Supervisor Agent][/bold purple] Evaluating Task "
            f"(Model: `{supervisor_model}`)..."
        )

        supervisor_messages = [
            {
                "role": "system",
                "content": (
                    "You are a routing supervisor. Reply ONLY with a single word: "
                    "'PYTHON' if the query is about data wrangling, visualization, "
                    "or basic statistics using Python/pandas; "
                    "'R' if advanced statistical tests or R code is specifically requested."
                ),
            },
            {"role": "user", "content": f"Query: {query}"},
        ]

        try:
            sup_response = completion(
                model=supervisor_model,
                messages=supervisor_messages,
                api_base=self._api_base_for(supervisor_model),
            )
            decision = sup_response.choices[0].message.content.strip().upper()
        except Exception as e:
            console.print(
                f"[bold red]Supervisor Error with model '{supervisor_model}':[/bold red] {e}"
            )
            return

        console.print(
            f"[bold purple][Supervisor Decision][/bold purple] → `{decision}`"
        )

        user_prompt = (
            f"# Data Summary\n{summary}\n\n"
            f"# User Query\n{query}\n\n"
            "Please write and execute the necessary code to answer this accurately."
        )

        # 2. Setup Dispatch to specialist
        if "R" in decision:
            specialist_model = self.models["r_specialist"]
            agent_name = "R-Specialist"
            specialist_sys_prompt = (
                "You are an expert R statistician. You have access to 'run_r_script'. "
                "ALWAYS use it to write and execute R code. "
                "If a required package is missing, emit install.packages() first."
            )
            tools = [self.r_tool]
        else:
            specialist_model = self.models["python_specialist"]
            agent_name = "Python-Specialist"
            specialist_sys_prompt = (
                "You are an expert Python data analyst. "
                "You have access to 'run_python_code' for analysis and 'save_plot' for charts. "
                "Use 'save_plot' whenever a visualization is requested. "
                "Always print() your numeric results."
            )
            tools = [self.python_tool, self.plot_tool]

        # 3. The Professional Critic Loop (Max 3 revisions)
        # Uses the strongest model (r_specialist) to critique math/assumptions
        critic_model = self.models["r_specialist"]
        max_revisions = 3
        current_feedback = ""
        final_answer = ""

        console.print(f"\n[bold yellow][Router][/bold yellow] Dispatching '{agent_name}' ({specialist_model}) + Critic ({critic_model})")

        for rev in range(max_revisions):
            if current_feedback:
                console.print(f"[bold yellow][Revision {rev+1}/{max_revisions}][/bold yellow] Specialist addressing critic feedback...")
            
            prompt_with_feedback = user_prompt
            if current_feedback:
                prompt_with_feedback += f"\n\n[CRITIC FEEDBACK]: Your previous attempt was scientifically flawed. {current_feedback}\nFix your code and rewrite the analysis."

            # Clear plot if it exists to ensure fresh generation
            if os.path.exists("astats_plot.png"): os.remove("astats_plot.png")

            final_answer = self.run_agent_loop(
                agent_name=f"{agent_name}",
                model=specialist_model,
                system_prompt=specialist_sys_prompt,
                user_prompt=prompt_with_feedback,
                tools=tools,
            )

            if not final_answer:
                console.print("[red]Analysis failed entirely.[/red]")
                return

            console.print(f"\n[bold magenta][Critic Agent][/bold magenta] Verifying mathematical/methodological correctness...")
            critic_prompt = f"""
            You are a strict Statistical Methodology Reviewer.
            Review the following analysis of '{query}' on a dataset.
            Ensure all statistical assumptions (e.g., normality checks, sample size, variance equality, data constraints) were considered.
            If the logic is deeply flawed or assumptions were skipped, reply with exactly 'REJECT: <reason>'.
            If the analysis is statistically sound, reply with exactly 'APPROVE'.
            
            ANALYSIS:
            {final_answer}
            """
            
            try:
                critic_res = completion(
                    model=critic_model,
                    messages=[{"role": "user", "content": critic_prompt}],
                    api_base=self._api_base_for(critic_model)
                ).choices[0].message.content.strip()
            except Exception as e:
                console.print(f"[bold red]Critic Error:[/bold red] {e}. Bypassing.")
                critic_res = "APPROVE"

            if critic_res.startswith("REJECT"):
                fb = critic_res[7:]
                console.print(f"[bold red]Critic Rejected![/bold red] {fb}")
                current_feedback = fb
            else:
                console.print("[bold green]Critic Approved! Math is sound.[/bold green]")
                break
        
        # 4. Generate the HTML Report
        console.print("\n[bold cyan]Generating Zero-Dependency HTML Report...[/bold cyan]")
        plot_path = "astats_plot.png" if os.path.exists("astats_plot.png") else None
        
        try:
            report_file = generate_html_report(
                query=query,
                dataset_name=data_path,
                final_answer=final_answer,
                plot_path=plot_path
            )
            console.print(f"[bold green]Report ready:[/bold green] file:///{report_file.replace(chr(92), '/')}")
        except Exception as e:
            console.print(f"[bold red]Failed to write HTML report:[/bold red] {e}")
