import os
import json
import yaml
import importlib
import re
from datasets import Dataset
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from smolagents import WikipediaRetrieverTool, FinalAnswerTool
from smolagents.models import remove_tool_call_from_messages, get_clean_message_list
from smolagents.agents import populate_template

PROMPT_TEMPLATES = yaml.safe_load(
    importlib.resources.files("smolagents.prompts").joinpath("code_agent.yaml").read_text()
)
INSTRUCTION1 = "\n\nIMPORTANT: Always provide a 'Thought:' sequence, and a 'Code:\n```py' sequence ending with '```<end_code>' sequence, else you will fail."
INSTRUCTION2 = "\n\nIMPORTANT: Always provide a 'Thought:' sequence, and a 'Code:\n```py' sequence ending with '```<end_code>' sequence, else you will fail. For final answer in math problems, always return the answer in LaTex format."
INSTRUCTION3 = "\n\nIMPORTANT: Always provide a 'Thought:' sequence, and a 'Code:\n```py' sequence ending with '```<end_code>' sequence, else you will fail. For math problems that are not multiple-choice, always output the final answer using LaTeX \\boxed{} format. Provide the exact value (e.g., \\boxed{\\frac{9}{14}}), not a decimal approximation (e.g., \\boxed{0.642857})."

INSTRUCTIONS = [
    INSTRUCTION3, INSTRUCTION2, INSTRUCTION1
]

ROLE_CONVERSION_DICT = {
    "MessageRole.SYSTEM": "system",
    "MessageRole.USER": "user",
    "MessageRole.ASSISTANT": "assistant",
    "MessageRole.TOOL_CALL": "tool-call",
    "MessageRole.TOOL_RESPONSE": "tool-response",
}


def print_pretty_messages(messages):
    console = Console()
    role_styles = {
        "system": ("System", "cyan"),
        "user": ("User", "green"),
        "assistant": ("Assistant", "magenta"),
    }

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        role_label, color = role_styles.get(role, ("Unknown", "red"))

        title_text = Text(f"{role_label}", style=f"bold {color}")
        content_text = Text(content, style="white")

        panel = Panel(content_text, title=title_text, border_style=color)
        console.print(panel)

# Only used for CoT models
def load_prompt() -> str:
    """Load the system prompt from YAML file"""
    prompt_file = Path(__file__).parent.parent.parent / "src" / "smolagents" / "prompts" / "teacher_model.yaml"
    with open(prompt_file, 'r') as f:
        prompt_data = yaml.safe_load(f)
    return prompt_data['system_prompt']

def clean_roles(messages):
    for message in messages:
        if message["role"] in ROLE_CONVERSION_DICT.keys():
            message["role"] = ROLE_CONVERSION_DICT[message["role"]]
    return messages

def remove_instruction_from_user_message(messages):
    for INSTRUCTION in INSTRUCTIONS:
        if INSTRUCTION in messages[1]["content"]:
            messages[1]["content"] = messages[1]["content"].replace(INSTRUCTION, "")
        assert INSTRUCTION not in messages[1]["content"]
    return messages

def remove_reference_tags(text):
    # <reference> ... </reference> 포함된 모든 내용을 삭제
    return re.sub(r'<reference>.*?</reference>', '', text, flags=re.DOTALL)

def clean_user_message(messages):
    user_message = messages[1]["content"]
    messages[1]["content"] = remove_reference_tags(user_message)
    return messages

def check_two_system_messages(messages):
    n_systems = 0
    for message in messages:
        role = message["role"]
        assert role in ['user', 'assistant', 'system', 'tool-call', 'tool-response']
        if role == "system":
            n_systems += 1

    if n_systems > 1:
        return True
    else:
        return False

def preprocess_sft_dataset(solution_type, datapath):
    if solution_type in ["cot", "reasoning"]:
        dataset = preprocess_cot_dataset(datapath)
    else:
        dataset = preprocess_logs(datapath)
    return dataset

def load_file_from_path(file_path):
    """
    Load a file from the given path.
    
    Args:
        file_path (str): Path to the file.
        
    Returns:
        dataset (list): loaded dataset
    """
    if os.path.exists(file_path):
        dataset = []
        with open(file_path, 'r') as f:
            for line in f:
                dataset.append(json.loads(line))
        return dataset
    else:
        # try to download from Hugging Face Hub
        from datasets import load_dataset
        dataset = load_dataset(file_path)
        dataset = dataset["train"].to_list()
        for data in dataset:
            data["log_data"] = json.loads(data["log_data"])
        return dataset

def preprocess_cot_dataset(datapath):
    system_prompt = load_prompt()
    dataset = load_file_from_path(datapath)

    processed_dataset = []
    for data in dataset:
        response = data["response"]
        messages = data['messages']
        messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )
        processed_dataset.append({"messages": messages})
    processed_dataset = Dataset.from_list(processed_dataset)
    return processed_dataset

def remove_last_user_message(
    messages
):
    if messages[-1]["role"] == "user":
        messages = messages[:-1]
    return messages

def clean_messages(messages):
    messages = clean_roles(messages)
    is_two_system = check_two_system_messages(messages)
    if is_two_system:
        import pdb; pdb.set_trace()
    messages = remove_tool_call_from_messages(messages)
    messages = get_clean_message_list(
        messages,
        role_conversions={
            "tool-response": "user",
            "tool-call": "assistant"
        },
        flatten_messages_as_text=True,
    )
    # messages = remove_instruction_from_user_message(messages)
    messages = clean_user_message(messages)
    messages = remove_last_user_message(messages)
    return messages

# Preprocess logs to messages only
def preprocess_logs(log_path):
    prompt_template = PROMPT_TEMPLATES["system_prompt_short"]

    tools = [WikipediaRetrieverTool()]
    tools = {tool.name: tool for tool in tools}
    tools.setdefault("final_answer", FinalAnswerTool())

    system_prompt = populate_template(
        prompt_template,
        variables={
            "tools": tools
        }
    )

    logs = load_file_from_path(log_path)

    dataset = []
    n_planning = 0
    for i, log in enumerate(logs):
        if not log["log_data"]:
            continue
        messages = log["log_data"]["messages"]
        messages = clean_messages(messages)
        messages[0]["content"] = system_prompt

        dataset.append({"messages": messages})
        if i == 0:
            print_pretty_messages(messages)
        # dataset.append(
        #     tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
        # )
        # Append additional messages
        for step in log["log_data"]["original_memory"]["steps"]:
            if len(step) > 0 and step.get("messages", None):
                additional_messages = clean_messages(step["messages"])
                # print("Additional messages...")
                # additional_messages = print_pretty_messages(additional_messages)
                dataset.append({"messages": additional_messages})
                n_planning += 1

    print("##### Planning data", n_planning)
    dataset = Dataset.from_list(dataset)
    return dataset

# Preprocess logs for the reward modeling
def preprocess_reward_dataset(log_path):
    rm_dataset = []
    with open(log_path) as f:
        for line in f:
            log = json.loads(line)
            for reward_data in log:
                rm_dataset.append(reward_data)

    dataset = []
    for rm_data in rm_dataset:
        messages = rm_data["messages"]
        messages = remove_tool_call_from_messages(messages)
        messages = get_clean_message_list(
            messages,
            role_conversions={
                "tool-response": "user",
                "tool-call": "assistant"
            },
            flatten_messages_as_text=True,
        )
        messages = remove_last_user_message(messages)
        # print(messages)
        if len(messages) == 1: continue
        reward = rm_data["step_reward"]
        dataset.append({
            "prompt": messages,
            "labels": reward
        })

    dataset = Dataset.from_list(dataset)
    return dataset


# Preprocess logs for the reward modeling
def preprocess_rollout_dataset(log_path):
    rm_dataset = []
    with open(log_path) as f:
        for line in f:
            log = json.loads(line)
            for reward_data in log:
                rm_dataset.append(reward_data)

    dataset = []
    for rm_data in rm_dataset:
        messages = rm_data["messages"]
        messages = remove_tool_call_from_messages(messages)
        messages = get_clean_message_list(
            messages,
            role_conversions={
                "tool-response": "user",
                "tool-call": "assistant"
            },
            flatten_messages_as_text=True,
        )
        messages = remove_last_user_message(messages)
        dataset.append({
            "prompt": messages,
        })

    dataset = Dataset.from_list(dataset)
    return dataset
