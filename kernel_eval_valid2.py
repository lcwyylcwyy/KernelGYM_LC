import asyncio
import os
from uuid import uuid4

import httpx


DEFAULT_SERVER_URLS = (
    os.getenv("KERNELGYM_SERVER_URL"),
    "http://192.168.31.68:8001",
    "http://192.168.31.68:8003",
    "http://localhost:10907",
)


async def resolve_server_url() -> str:
    probe_timeout = httpx.Timeout(5.0, connect=1.0)

    async with httpx.AsyncClient(timeout=probe_timeout) as client:
        for base_url in DEFAULT_SERVER_URLS:
            if not base_url:
                continue
            try:
                response = await client.get(f"{base_url}/health")
                response.raise_for_status()
                return base_url
            except httpx.HTTPError:
                continue

    raise RuntimeError(
        "Could not connect to a KernelGYM server. "
        "Set KERNELGYM_SERVER_URL to a reachable base URL."
    )


async def evaluate_reference_triton():
    server_url = await resolve_server_url()
    task_id = f"softmax-ref-{uuid4().hex[:8]}"
    reference_code = '''
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.softmax(x, dim=-1)

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(32, 512, device='cuda')]
'''

    # Minimal kernel stub: must be non-empty and correct, otherwise kernelbench
    # stops before running reference timing.
    kernel_code = '''
import torch

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.softmax(x, dim=-1)

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(32, 512, device='cuda')]
'''

    timeout = httpx.Timeout(300.0, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{server_url}/evaluate",
            json={
                "force_refresh": True,
                "task_id": task_id,
                "reference_code": reference_code,
                "kernel_code": kernel_code,
                "entry_point": "Model",
                "backend": "triton",
                "device_preference": "cuda:0",
                "reference_backend": "torch_compile",
                "return_reference_triton": True,
                "reference_triton_max_chars": 80000,
                "num_correct_trials": 5,
                "num_perf_trials": 100,
                "run_triton_detection": False,
            }
        )
        response.raise_for_status()
        return response.json()


workflow_response = asyncio.run(evaluate_reference_triton())
print(f"Status: {workflow_response.get('status')}")
print(f"Reference Runtime: {workflow_response['reference_runtime']:.4f} ms")

metadata = workflow_response.get("metadata") or {}
ref_triton = metadata.get("reference_triton_code", "")
if ref_triton:
    print("\n=== Reference Generated Triton Code ===")
    import pdb; pdb.set_trace()
    print(ref_triton)
else:
    print("\nNo reference Triton code captured. Metadata:")
    print(metadata)
