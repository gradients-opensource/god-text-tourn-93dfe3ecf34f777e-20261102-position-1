FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    vim zip tmux iotop nvtop bmon wget nano zsh htop \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace /cache /workspace/scripts/datasets /app/checkpoints
WORKDIR /workspace/scripts

COPY scripts/training_requirements.txt /workspace/scripts/training_requirements.txt
COPY scripts/grpo_requirements.txt /workspace/scripts/grpo_requirements.txt
COPY scripts/kernel_requirements.txt /workspace/scripts/kernel_requirements.txt
COPY scripts/ssm_kernels.py /workspace/scripts/ssm_kernels.py


RUN pip install uv && \
    pip install -U packaging==23.2 setuptools==75.8.0 wheel ninja && \
    uv pip install -r /workspace/scripts/training_requirements.txt --system && \
    pip install tenacity==9.1.2 tiktoken==0.9.0 && \
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl && \
    pip install --no-deps https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl && \
    pip install "fiber @ git+https://github.com/rayonlabs/fiber.git@2.4.0"

RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

RUN { pip install --no-deps --require-hashes -r /workspace/scripts/kernel_requirements.txt \
      && python /workspace/scripts/ssm_kernels.py --install \
      && python /workspace/scripts/ssm_kernels.py --selfcheck \
      && echo "ssm kernels: present"; } \
    || { pip uninstall -y mamba-ssm; \
         echo "========================================================================"; \
         echo "ssm kernels: ABSENT (fail-soft) - mamba-ssm removed, Mamba layers will run the torch path"; \
         echo "========================================================================"; \
         echo "absent: build step failed" > /workspace/scripts/.ssm_kernels_status; }

RUN { python -m venv /workspace/grpo_py \
      && /workspace/grpo_py/bin/pip install -U pip uv==0.12.19 \
      && /workspace/grpo_py/bin/python -m uv pip install \
          --python /workspace/grpo_py/bin/python \
          --torch-backend cu129 \
          --extra-index-url https://wheels.vllm.ai/0.26.0/cu129 \
          -r /workspace/scripts/grpo_requirements.txt \
      && /workspace/grpo_py/bin/python -c "import torch, vllm, trl, transformers; print('grpo venv', torch.__version__, torch.version.cuda, vllm.__version__, trl.__version__, transformers.__version__)" \
      && /workspace/grpo_py/bin/python -c "import torch, importlib.metadata as md; assert (torch.version.cuda or '').startswith('12.'), ('grpo torch not CUDA 12', torch.version.cuda); v = md.version('vllm'); assert v.endswith('+cu129'), ('grpo vllm not the CUDA 12 build', v); bad = sorted(d.metadata['Name'] for d in md.distributions() if d.metadata['Name'].lower().endswith('-cu13') or (d.metadata['Name'].lower() == 'cuda-toolkit' and d.version.startswith('13'))); assert not bad, ('CUDA 13 packages in grpo venv', bad); print('grpo venv CUDA', torch.version.cuda, 'vllm', v)" \
      && echo "grpo venv: present"; } \
    || { rm -rf /workspace/grpo_py; \
         echo "========================================================================"; \
         echo "grpo venv: ABSENT (fail-soft) - install or CUDA-12 check failed, /workspace/grpo_py removed;"; \
         echo "GRPO will run in the main env (torch 2.8) without vllm"; \
         echo "========================================================================"; }

RUN python -c "import transformers; v=transformers.__version__; assert v == '5.12.1', v" && \
    python -c "import torch; assert torch.__version__.startswith('2.8.0'), torch.__version__" && \
    python -c "import liger_kernel; from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction as F; assert hasattr(F.forward, '__call__')" && \
    python -c "import flash_attn, trl, peft, bitsandbytes" && \
    python -c "import torch, causal_conv1d_cuda; from causal_conv1d import causal_conv1d_fn, causal_conv1d_update" && \
    python -c "from deepspeed import DeepSpeedEngine; print('deps ok (deepspeed importable)')"

COPY scripts /workspace/scripts
RUN chmod +x /workspace/scripts/entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
