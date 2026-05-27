import os
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Any
from urllib import request
from urllib.error import HTTPError

from datetime import datetime
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    from .math_utils.qwen_math_parser import extract_answer
    from .math_utils.qwen_math_grader import math_equal
except ImportError:
    from math_utils.qwen_math_parser import extract_answer
    from math_utils.qwen_math_grader import math_equal

@dataclass
class JudgeResponse:
    content: str


class AnthropicJudgeModel:
    def __init__(self, model_id: str, api_key: str, temperature: float = 0.0, max_tokens: int = 512):
        self.model_id = model_id
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_input_token_count = 0
        self.last_output_token_count = 0

    def __call__(self, messages: List[Dict[str, str]], **kwargs) -> JudgeResponse:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        req = request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic judge request failed: {exc.code} {body}") from exc
        usage = data.get("usage", {})
        self.last_input_token_count = usage.get("input_tokens", 0)
        self.last_output_token_count = usage.get("output_tokens", 0)
        content = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        return JudgeResponse(content=content)

    def get_token_counts(self) -> Dict[str, int]:
        return {
            "input_token_count": self.last_input_token_count,
            "output_token_count": self.last_output_token_count,
        }


JUDGE_MODEL_ID = os.getenv("JUDGE_MODEL_ID", "claude-haiku-4-5-20251001")


def load_api_key(key_path: str = "keys/claude-key/key.env") -> str:
    """Load Claude API key from file"""
    with open(key_path) as f:
        return f.read().strip()


def setup_scoring_model() -> AnthropicJudgeModel:
    """Initialize the Claude model for LLM-as-a-judge scoring"""
    api_key = load_api_key()
    return AnthropicJudgeModel(
        model_id=JUDGE_MODEL_ID,
        api_key=api_key,
        temperature=0.0,
    )


def parse_judge_json(content: str) -> Dict[str, Any]:
    """Parse the judge JSON even if the model wraps it in Markdown fences."""
    import re
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`").removeprefix("json").strip()
    # JSON 문자열 내 raw control character(0x00-0x1f, 0x7f)는 invalid:
    # Claude judge 응답의 explanation 필드에 개행 등이 포함될 때 파싱 실패 방지
    content = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        start = content.find("{")
        while start != -1:
            try:
                result, _ = decoder.raw_decode(content[start:])
                return result
            except json.JSONDecodeError:
                start = content.find("{", start + 1)
        raise


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate cost based on Claude Haiku 4.5 pricing"""
    INPUT_COST_PER_1M = 1.0  # $1.00 per 1M input tokens
    OUTPUT_COST_PER_1M = 5.0  # $5.00 per 1M output tokens

    input_cost = (input_tokens / 1000000) * INPUT_COST_PER_1M
    output_cost = (output_tokens / 1000000) * OUTPUT_COST_PER_1M

    return input_cost + output_cost

def evaluate_factual_answer(
    model: AnthropicJudgeModel,
    predicted: str,
    gold: str,
    question: str,
    do_extract_answer: bool = False) -> Dict:
    """
    Evaluate if the predicted answer matches the gold answer using the model
    Returns dict with score and explanation
    """
    if len(str(predicted)) > 1000:
        predicted = str(predicted)[:1000]
    prompt = f"""Compare the following predicted answer with the gold (correct) answer and determine if they match in meaning given the question.
Be reasonable - phrasing differences are okay if the core meaning is the same. You should accept the aliases and the answer conveying the same conclusion.
If the predicted answer is overly verbose and fails to capture the key information found in the gold answer (i.e., low recall), consider it a false answer.

Question: {question}
Predicted answer: {predicted}
Gold answer: {gold}

Output your evaluation as a JSON object with two fields:
- explanation: Brief explanation of your scoring decision
- score: 0 or 1 indicating if answers match
"""

    response = model(
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )

    # Parse JSON response
    result = parse_judge_json(response.content)

    # Get token counts and calculate cost
    token_counts = model.get_token_counts()
    cost = calculate_cost(
        token_counts["input_token_count"],
        token_counts["output_token_count"]
    )

    return {
        "score": result["score"],
        "explanation": result["explanation"],
        "input_tokens": token_counts["input_token_count"],
        "output_tokens": token_counts["output_token_count"],
        "cost": cost
    }

def evaluate_math_answer(
    model: AnthropicJudgeModel,
    predicted: str,
    gold: str,
    question: str,
    do_extract_answer: bool
) -> Dict:
    """
    Evaluate if the predicted answer matches the gold answer using the model
    Returns dict with score and explanation
    """
    if type(gold) != str: gold = str(gold)

    # Does not need any model
    # if do_extract_answer:
    #     if not predicted:
    #         predicted = "No answer provided"
    #     if "\boxed" not in predicted and len(predicted.split("\n\n")) == 1:
    #         predicted = "\boxed{" + predicted + "}"
    #     pred_ans = extract_answer(predicted)
    # else:
    if type(predicted) == str and "boxed" in predicted:
        pred_ans = extract_answer(predicted)
    else:
        pred_ans = str(predicted)
    score = math_equal(pred_ans, gold, timeout=True)

    # Parse JSON response
    return {
        "score": score,
        "explanation": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "cost": 0.0
    }

def process_entry(args):
    """Process a single entry with its own model instance"""
    entry, model, task_type, do_extract_answer = args
    if 'error' in entry:
        result = deepcopy(entry)
        result["score"] = False
        result["explanation"] = ""
        result["cost"] = 0.0

    if "true_answer" in entry.keys():
        gold_key = "true_answer"
        pred_key = "generated_answer"
    else:
        gold_key = "answer"
        pred_key = "generated_answer"

    """
    [IMPLEMENT THE CORRECTNESS DETERMINING FUNCTION HERE]
    """
    if task_type == "fact":
        eval_func = evaluate_factual_answer
    elif task_type == "math":
        eval_func = evaluate_math_answer
    else:
        raise NotImplementedError

    evaluation = eval_func(
        model=model,
        predicted=entry.get(pred_key, None),
        gold=entry.get(gold_key, None),
        question=entry['question'],
        do_extract_answer=do_extract_answer,
    )

    result = deepcopy(entry)
    result["score"] = evaluation["score"]
    result["explanation"] = evaluation["explanation"]
    result["cost"] = evaluation["cost"]
    return result

def score_qa_results(
    log_file: str,
    max_workers: int = 4,
    task_type: str = "fact",
    do_extract_answer: bool = False,
    single_thread: bool = False
) -> Dict:
    """
    Score all QA results in the given folder using multiple threads
    Args:
        log_folder: Path to the folder containing the results
        max_workers: Maximum number of concurrent threads to use
    Returns dict with scores and statistics
    """
    results = []
    total_cost = 0

    log_folder = os.path.dirname(log_file)
    filename = os.path.basename(log_file)

    filepath = os.path.join(log_folder, filename)

    # Create output directory if it doesn't exist
    output_dir = os.path.join(log_folder, "evaluations")
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename based on input filename
    base_filename = os.path.splitext(filename)[0]
    output_file = os.path.join(output_dir, f"{base_filename}_scored.jsonl")

    # Read all entries first
    entries = []
    with open(filepath, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if type(entry) == str:
                continue # Invalid entry
            entries.append(entry)

    # Create a pool of models
    models = [setup_scoring_model() for _ in range(max_workers)]
    # Process entries in parallel
    if single_thread:
        # Process entries sequentially with a for-loop
        with open(output_file, 'w') as out_f:
            for i, entry in tqdm(enumerate(entries), total=len(entries), desc="Evaluating answers"):
                model = models[i % max_workers]  # 동일한 방식으로 모델 선택
                result = process_entry((entry, model, task_type, do_extract_answer))  # 바로 함수 호출

                if result:
                    results.append(result)
                    out_f.write(json.dumps(result) + '\n')
                    total_cost += result['cost']
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create arguments for each entry with a model
            args = [(entry, models[i % max_workers], task_type, do_extract_answer) for i, entry in enumerate(entries)]

            # Submit all tasks and get futures
            future_to_entry = {executor.submit(process_entry, arg): arg for arg in args}

            # Process results as they complete
            with open(output_file, 'w') as out_f:
                for future in tqdm(as_completed(future_to_entry), total=len(entries), desc="Evaluating answers"):
                    result = future.result()
                    if result:
                        results.append(result)
                        # Write individual result to output file
                        out_f.write(json.dumps(result) + '\n')
                        # Update totals
                        total_cost += result['cost']

    # Calculate statistics
    scores = [r['score'] for r in results]
    stats = {
        "log_file": log_file,
        'total_questions': len(results),
        'correct_answers': sum(scores),
        'accuracy': sum(scores) / len(scores) if scores else 0,
        # 'detailed_results': results, # to reduce the memory :)
        'costs': {
            'total_cost': total_cost,
            'average_cost_per_question': total_cost / len(results) if results else 0
        }
    }

    # Save summary statistics
    summary_file = os.path.join(output_dir, f"evaluation_summary_{base_filename}.json")
    with open(summary_file, 'w') as f:
        json.dump(stats, f, indent=2)

    return output_file, stats

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Score QA results with multi-threading support')
    parser.add_argument('--log_files', type=str, default=os.path.join("logs", "qa_results", "openai", "gpt-4o-mini"),
                      help='Path to the log folder containing results', nargs='+')
    parser.add_argument('--log_folder', type=str, help="score all files in the folder")
    parser.add_argument('--task_type', type=str, default='fact', choices=["fact", "math"])
    parser.add_argument('--do_extract_answer', action='store_true')
    parser.add_argument('--max_workers', type=int, default=8,
                      help='Maximum number of concurrent threads to use')
    parser.add_argument('--single_thread', action='store_true')

    args = parser.parse_args()

    if args.log_folder:
        all_paths = Path(args.log_folder).glob("*.jsonl")
        args.log_files = [str(s) for s in all_paths]

    if args.task_type == "fact":
        args.do_extract_answer = True

    for log_file in args.log_files:
        output_file, stats = score_qa_results(
            log_file,
            max_workers=args.max_workers,
            task_type=args.task_type,
            single_thread=args.single_thread,
            do_extract_answer=args.do_extract_answer
        )
        print(f"Accuracy: {stats['accuracy']:.2%}")
        print(f"Correct: {stats['correct_answers']}/{stats['total_questions']}")
        print(f"\nCost Summary:")
        print(f"Total Cost: ${stats['costs']['total_cost']:.4f}")