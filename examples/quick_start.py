import os
import sys
import time
import readline
import json
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from smolagents import (
    CodeAgent,
    VLLMServerModel,
    DuckDuckGoSearchTool
)

console = Console()

def typewriter(text, delay=0.02):
    """Print text like a typewriter effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def fancy_intro():
    console.print(Panel.fit("[bold cyan]Welcome to agent-distillation CLI[/bold cyan] 🤖", border_style="cyan"))
    typewriter("Your lightweight AI assistant is now online.")
    typewriter("Ask it anything — science, trivia, tasks, you name it.")
    console.print("\n[dim]Tip: Press [bold]Enter[/bold] without typing to try a default question.[/dim]\n")

def get_user_input():
    default_question = "How many times taller is the Empire State Building than the Eiffel Tower?"
    user_input = Prompt.ask("[green]Distilled Agent >[/green]", default="")
    return user_input.strip() or default_question

def setup_agent():
    model = VLLMServerModel(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        api_base="http://0.0.0.0:8000/v1",
        api_key="token-abc",
        lora_name="finetune",
        max_tokens=1024,
        n=2, temperature=0.4 # for SAG
    )

    agent = CodeAgent(
        tools=[
            DuckDuckGoSearchTool(),
        ],
        additional_authorized_imports=["numpy", "sympy"],
        model=model,
        max_steps=5
    )

    return agent

def main():
    fancy_intro()
    question = get_user_input()
    console.print(f"\n[bold yellow]You asked:[/bold yellow] {question}")

    agent = setup_agent()

    console.print("[blue]Thinking...[/blue] 🤔\n")
    result = agent.run(
        question,
        use_short_system_message=True
    )

    console.print("\n[bold green]Answer:[/bold green]")
    console.print(result)

if __name__ == "__main__":
    main()
