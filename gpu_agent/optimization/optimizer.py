import os
import json
from openai import OpenAI

from gpu_agent.prompts import TRITON_OPTIMIZATION_PROMPT
from gpu_agent.generation.generator import extract_python_code

def optimize_triton_kernel(pytorch_code, triton_code, gpu_specs, benchmark, ncu_metrics, roofline,error=None):

    prompt = TRITON_OPTIMIZATION_PROMPT.format(
        gpu_specs=json.dumps(gpu_specs, indent=2),
        pytorch_code=pytorch_code,
        triton_code=triton_code,
        pytorch_ms=benchmark["pytorch_ms"],
        triton_ms=benchmark["triton_ms"],
        speedup=benchmark["speedup"],
        classification=roofline["classification"],
        compute_sol=roofline["compute_sol"],
        memory_sol=roofline["memory_sol"],
        efficiency=roofline["efficiency"],
        headroom=roofline["headroom"],
        ncu_metrics=json.dumps(ncu_metrics, indent=2),
        error=error or "None",
    )

    response = client.responses.create(
        model="gpt-5.6-terra",
        input=prompt,
    )

    return extract_python_code(response.output_text)