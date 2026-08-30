import json

from openai import OpenAI

from gpu_agent.prompts import (
    TRITON_GENERATION_PROMPT,
    TRITON_REPAIR_PROMPT,
    TRITON_GUIDELINES,
)


client = OpenAI()


def generate_triton_kernel(pytorch_code, gpu_specs):
    prompt = TRITON_GENERATION_PROMPT.format(
        pytorch_code=pytorch_code,
        gpu_specs=json.dumps(gpu_specs, indent=2),
    )

    response = client.responses.create(
        model="gpt-5.6-terra",
        input=prompt,
    )

    return response.output_text


def extract_python_code(response):
    response = response.strip()

    if response.startswith("```python"):
        response = response[len("```python"):]

    if response.startswith("```"):
        response = response[3:]

    if response.endswith("```"):
        response = response[:-3]

    return response.strip()


# gives the LLM short snippets of all previous failed versions
def format_refinement_history(refinement_history):
    if not refinement_history:
        return "None"

    history_context = []

    for attempt in refinement_history:
        entry = [
            f"Attempt {attempt['round']}:",
            f"Error type: {attempt['error_type']}",
            "Previous kernel snippet:",
            attempt["kernel_code"][:500],
            f"Error: {attempt['error']}",
        ]

        if "shape" in attempt:
            entry.append(f"Shape: {attempt['shape']}")

        history_context.append("\n".join(entry))

    return "\n\n".join(history_context)


def repair_triton_kernel(
    pytorch_code,
    triton_code,
    gpu_specs,
    error_type,
    error,
    refinement_history,
):
    prompt = TRITON_REPAIR_PROMPT.format(
        triton_guidelines=TRITON_GUIDELINES,
        pytorch_code=pytorch_code,
        triton_code=triton_code,
        error_type=error_type,
        error=error,
        refinement_history=format_refinement_history(
            refinement_history
        ),
    )

    response = client.responses.create(
        model="gpt-5.6-terra",
        input=prompt,
    )

    return extract_python_code(response.output_text)