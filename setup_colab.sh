#!/usr/bin/env bash
# Run in the Colab terminal with:
#   source /content/GPU-Agent/setup_colab.sh
# Sourcing retains PATH and LD_LIBRARY_PATH in that terminal. Running with bash
# performs setup, but a child process cannot update its parent's environment.

# Never trace environment expansions or authentication-related configuration.
set +x

_gpu_agent_prepend_path() {
    local variable=$1 directory=$2
    case ":${!variable-}:" in
        *":${directory}:"*) ;;
        *) export "$variable=$directory${!variable:+:${!variable}}" ;;
    esac
}

_gpu_agent_setup_colab() {
    local tool executable directory gpu_names
    local status=0

    _gpu_agent_prepend_path PATH "$HOME/.local/bin"
    # Colab's driver library locations; preserve existing library search paths.
    for directory in /usr/local/nvidia/lib64 /usr/local/nvidia/lib /usr/lib64-nvidia; do
        if [[ -d "$directory" ]]; then
            _gpu_agent_prepend_path LD_LIBRARY_PATH "$directory"
        fi
    done

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo 'ERROR: nvidia-smi is unavailable. Select a Colab GPU runtime.' >&2
        return 1
    fi
    if ! gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) || [[ -z "$gpu_names" ]]; then
        echo 'ERROR: No working NVIDIA GPU detected. Select a Colab GPU runtime.' >&2
        return 1
    fi
    printf 'Detected GPU(s):\n%s\n' "$gpu_names"
    if [[ "$gpu_names" != *A100* ]]; then
        echo 'Note: the detected GPU is not an A100.'
    fi

    # Prefer an already usable command. Otherwise discover executable files and
    # symlinks, including directories containing spaces, without version pins.
    for tool in nsys ncu; do
        if command -v "$tool" >/dev/null 2>&1 && "$tool" --version >/dev/null 2>&1; then
            continue
        fi
        while IFS= read -r -d '' executable; do
            if [[ -f "$executable" && -x "$executable" ]] && "$executable" --version >/dev/null 2>&1; then
                _gpu_agent_prepend_path PATH "${executable%/*}"
                break
            fi
        done < <(find /opt /usr/local -name "$tool" \( -type f -o -type l \) -print0 2>/dev/null | sort -zV -r)
    done

    if [[ -e /content/KernelBench || -L /content/KernelBench ]]; then
        if [[ ! -d /content/KernelBench ]]; then
            echo 'ERROR: /content/KernelBench exists but is not a directory.' >&2
            return 1
        fi
        echo 'Keeping existing /content/KernelBench unchanged.'
    else
        # Public clone only: disable credential helpers and interactive prompts.
        if ! GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/false git -c credential.helper= clone \
            https://github.com/ScalingIntelligence/KernelBench.git /content/KernelBench >/dev/null 2>&1; then
            echo 'ERROR: Public KernelBench clone failed; check network access and git availability.' >&2
            return 1
        fi
    fi

    # Actual runtime dependencies: torch in GPU-Agent, triton in generated kernels,
    # and openai for generation/diagnosis. Do not install KernelBench's full stack
    # or test-only pytest. No --upgrade: retain Colab's installed CUDA/PyTorch pair.
    echo 'Ensuring GPU-Agent Python dependencies: torch, triton, openai...'
    if ! python -m pip install --disable-pip-version-check --no-input torch triton openai >/dev/null 2>&1; then
        echo 'ERROR: Python dependency installation failed.' >&2
        return 1
    fi
    if ! python - <<'PY'
import torch
import triton
from openai import OpenAI

if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot access CUDA.")
print(f"Python dependencies: torch {torch.__version__}, triton {triton.__version__}; OpenAI import OK")
PY
    then
        return 1
    fi

    if ! command -v codex >/dev/null 2>&1; then
        echo 'Installing Codex CLI...'
        # Official standalone installer; no login or authentication commands.
        # https://learn.chatgpt.com/docs/codex/cli
        if ! (set -o pipefail; curl -fsSL https://chatgpt.com/codex/install.sh | sh) >/dev/null 2>&1; then
            echo 'ERROR: Codex CLI installation failed.' >&2
            return 1
        fi
        hash -r
    fi

    echo 'Setup versions/status:'
    nvidia-smi || status=1
    for tool in nsys ncu python codex; do
        if command -v "$tool" >/dev/null 2>&1; then
            printf '%s: ' "$tool"
            "$tool" --version || status=1
        else
            printf '%s: MISSING (required for complete setup)\n' "$tool"
            status=1
        fi
    done
    if (( status != 0 )); then
        echo 'Setup incomplete: resolve the missing or failing tools above.' >&2
    fi
    echo 'Reminder: OPENAI_API_KEY must be manually exported in your working terminal before running GPU-Agent.'
    return "$status"
}

_gpu_agent_setup_colab
