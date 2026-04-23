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


def _clean_reference_triton_output(raw_code: str) -> str:
    if not raw_code:
        return raw_code

    # The capture sometimes includes a duplicated kernel section split by this marker.
    marker = "\n# ---- triton-kernel ----\n"
    if marker in raw_code:
        raw_code = raw_code.split(marker, 1)[0].rstrip()

    return raw_code


def _extract_kernel_definition(code: str) -> str:
    """Extract just the @triton.jit kernel definition."""
    if not code:
        return ""
    if "@triton.jit" in code:
        start = code.find("@triton.jit")
        # Find the end of the function (next def or end of string)
        rest = code[start:]
        lines = rest.split('\n')
        
        # Collect until we hit another function or key markers
        result_lines = []
        indent_level = None
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith('def ') and result_lines:
                # Check if this is same level indent as @triton.jit decorator
                current_indent = len(line) - len(stripped)
                if current_indent == 0:
                    # New top-level function, stop here
                    break
            result_lines.append(line)
        
        return '\n'.join(result_lines).rstrip()
    return code


def _extract_invocation_code(code: str) -> str:
    """Extract the kernel invocation/launch code."""
    if not code:
        return ""
    
    invocation_parts = []
    lines = code.split('\n')
    
    # Look for grid definition and kernel.run() calls
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Capture grid definitions, launch parameters, and kernel calls
        if any(marker in stripped for marker in [
            'grid =', 'grid0 =', 'grid1 =',
            '.run(', '.call(', '.launch(',
            'cooperative_groups'
        ]) or ('triton_' in stripped and '(' in stripped):
            invocation_parts.append(line)
        # Also capture following lines that are continuations
        if invocation_parts and i > 0:
            prev_line = lines[i-1].rstrip()
            if prev_line.endswith('(') or prev_line.endswith(','):
                invocation_parts.append(line)
    
    if not invocation_parts:
        # Fallback: look for any lines after the kernel definition
        in_kernel = False
        for line in lines:
            if '@triton.jit' in line or 'def triton_' in line:
                in_kernel = True
                continue
            if in_kernel and line.strip() and not line.strip().startswith('def '):
                if '@triton.jit' not in line and 'libdevice' not in line:
                    invocation_parts.append(line)
            elif in_kernel and line.strip().startswith('def ') and 'def triton_' not in line:
                # End of kernel, rest is likely invocation
                in_kernel = False
    
    return '\n'.join(invocation_parts).strip() if invocation_parts else ""


workflow_response = asyncio.run(evaluate_reference_triton())
print(f"Status: {workflow_response.get('status')}")
print(f"Reference Runtime: {workflow_response['reference_runtime']:.4f} ms")

metadata = workflow_response.get("metadata") or {}
ref_triton = metadata.get("reference_triton_code", "")
ref_triton_full = metadata.get("reference_triton_full_context", "")

if ref_triton or ref_triton_full:
    print("\n" + "="*70)
    print("=== Reference Generated Triton Kernel Definition ===")
    print("="*70)
    cleaned = _clean_reference_triton_output(ref_triton)
    print(cleaned)
    
    print("\n" + "="*70)
    print("=== Reference Triton Invocation Code ===")
    print("="*70)
    if ref_triton_full:
        invocation = _extract_invocation_code(ref_triton_full)
        if invocation:
            print(invocation)
        else:
            print("[Invocation code not clearly separable from definition]")
            print("Full context:")
            print(ref_triton_full[:2000])  # Show first 2000 chars
    else:
        print("[Full context not available in this capture]")
else:
    print("\nNo reference Triton code captured. Metadata:")
    print(metadata)
