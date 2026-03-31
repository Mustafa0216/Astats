import typer
from rich.console import Console
from rich.panel import Panel
from astats.simulator import simulate_clinical_trial
from astats.data.discovery import discover_dataset
from astats.agent.core import AStatsAgent
import yaml
import os
from rich.prompt import Prompt, Confirm

app = typer.Typer(help="AStats: Multi-Agent System for Applied Statistical Practitioner Workflows.")
console = Console()

@app.command()
def init():
    """Interactive setup wizard for LLM models and API Keys."""
    console.print(Panel("[bold cyan]Welcome to AStats setup![/bold cyan]\nLet's configure your agentic workflows.", border_style="cyan"))

    # Baseline Model Selection
    console.print("\n[bold yellow]Step 1: Baseline Route Model[/bold yellow]")
    console.print("Used for fast, cheap tasks like routing and data summarization. Local models are highly recommended here.")
    baseline_choices = {
        "1": "ollama/llama3.2:1b (100% Local, ~650MB Memory, Free)",
        "2": "gemini/gemini-2.5-flash (Lightning fast, requires API Key)",
        "3": "groq/llama3-8b-8192 (Fast open-weight, requires API Key)",
        "4": "openai/gpt-4o-mini (Cheap commercial, requires API Key)"
    }
    for k, v in baseline_choices.items():
        console.print(f"  [cyan]{k}[/cyan]: {v}")
    
    baseline_selection = Prompt.ask("Select Baseline Model", choices=["1", "2", "3", "4"], default="1")
    baseline_model_map = {"1": "ollama/llama3.2:1b", "2": "gemini/gemini-2.5-flash", "3": "groq/llama3-8b-8192", "4": "openai/gpt-4o-mini"}
    baseline_model = baseline_model_map[baseline_selection]

    # Professional Model Selection
    console.print("\n[bold yellow]Step 2: Professional Specialist Model[/bold yellow]")
    console.print("Used for heavy R mathematical modeling and Python data wrangling. Capable commercial models recommended.")
    pro_choices = {
        "1": "openai/gpt-4o (State-of-the-art, requires API Key)",
        "2": "anthropic/claude-3-5-sonnet-20240620 (Excellent at coding, requires API Key)",
        "3": "gemini/gemini-1.5-pro (High context, requires API Key)",
        "4": "ollama/llama3-70b (100% Local, high memory, Free)"
    }
    for k, v in pro_choices.items():
        console.print(f"  [cyan]{k}[/cyan]: {v}")
    
    pro_selection = Prompt.ask("Select Professional Model", choices=["1", "2", "3", "4"], default="1")
    pro_model_map = {"1": "openai/gpt-4o", "2": "anthropic/claude-3-5-sonnet-20240620", "3": "gemini/gemini-1.5-pro", "4": "ollama/llama3-70b"}
    pro_model = pro_model_map[pro_selection]

    # API Keys Setup
    console.print("\n[bold yellow]Step 3: Setup API Keys[/bold yellow]")
    env_updates = {}
    
    def prompt_key(provider: str, env_var: str):
        if Confirm.ask(f"Do you want to configure an API key for {provider}?"):
            key = Prompt.ask(f"Enter your {provider} API Key", password=True)
            if key: env_updates[env_var] = key

    # Check which keys we might need based on selections
    providers_needed = set([baseline_model.split('/')[0], pro_model.split('/')[0]])
    if "openai" in providers_needed: prompt_key("OpenAI", "OPENAI_API_KEY")
    if "gemini" in providers_needed: prompt_key("Google Gemini", "GEMINI_API_KEY")
    if "anthropic" in providers_needed: prompt_key("Anthropic", "ANTHROPIC_API_KEY")
    if "groq" in providers_needed: prompt_key("Groq", "GROQ_API_KEY")

    # Write to .env
    if env_updates:
        with open(".env", "w") as f:
            for k, v in env_updates.items():
                f.write(f"{k}={v}\n")
        console.print("[green]Saved API keys to .env[/green]")

    # Write to astats_config.yaml
    config_data = {
        "global": {
            "ollama_api_base": "http://localhost:11434"
        },
        "agents": {
            "supervisor": baseline_model,
            "python_specialist": pro_model,
            "r_specialist": pro_model
        },
        "retry": {
            "max_steps": 5,
            "feedback_on_error": True
        }
    }
    
    with open("astats_config.yaml", "w") as f:
        yaml.dump(config_data, f, default_flow_style=False)
    
    console.print(f"\n[bold green]Setup Complete![/bold green]")
    console.print(f"Supervisor (Baseline): {baseline_model}")
    console.print(f"Specialists (Professional): {pro_model}")
    console.print("\nYou are ready to analyze data using: `python -m astats.cli explore <dataset> <query>`")


@app.command()
def simulate(rows: int = typer.Option(1000, help="Number of rows to generate"),
             output: str = typer.Option("astats_demo.csv", help="Output file path")):
    """Simulates a baseline clinical trial dataset for testing."""
    path = simulate_clinical_trial(rows, output)
    console.print(Panel(f"Generated {rows} rows of simulated data.\\n[cyan]{path}[/cyan]", title="[bold green]Success[/bold green]"))

@app.command()
def discover(filepath: str):
    """Auto-discovers and summarizes a given dataset."""
    console.print(f"[bold blue]Running auto-discovery on {filepath}...[/bold blue]")
    summary = discover_dataset(filepath)
    console.print(Panel(summary, title="Data Summary Snapshot", border_style="blue"))

@app.command()
def explore(filepath: str, 
            query: str = typer.Argument(..., help="What do you want to ask the agent?"), 
            config: str = typer.Option("astats_config.yaml", help="Path to the multi-agent YAML configuration")):
    """Agentic workflow: Loads the dataset, invokes the Supervisor Agent, and routes to specialists."""
    try:
        agent = AStatsAgent(config_path=config)
        agent.ask(filepath, query)
    except Exception as e:
        console.print(f"[bold red]Execution Error:[/bold red] {e}")

if __name__ == "__main__":
    app()
