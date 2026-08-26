from gpu_agent.optimization.orchestrator import run_optimization


def main():
    pytorch_code = """
def pytorch_reference(x):
    return torch.relu(x)
"""

    run_optimization(pytorch_code)


if __name__ == "__main__":
    main()