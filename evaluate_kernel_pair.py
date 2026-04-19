import argparse
import asyncio
import ast
import os
from pathlib import Path
from uuid import uuid4

import httpx


DEFAULT_SERVER_URLS = (
    os.getenv("KERNELGYM_SERVER_URL"),
    "http://192.168.31.68:8003",
    "http://localhost:10907",
)

REQUIRED_REFERENCE_FUNCTIONS = (
    "get_inputs",
    "get_init_inputs",
)


def describe_reference_mode(reference_backend: str) -> str:
    if reference_backend in {"compile", "torch_compile"}:
        return "compile"
    if reference_backend == "eager":
        return "eager"
    return "custom"


def build_reference_label(reference_backend: str) -> str:
    return (
        f"{reference_backend} "
        f"({describe_reference_mode(reference_backend)})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a KernelGYM reference/kernel file pair by path. "
            "The script copies get_inputs/get_init_inputs from the reference "
            "file into the kernel code when they are missing."
        )
    )
    parser.add_argument(
        "reference_path",
        type=Path,
        help="Path to the reference.py file containing the baseline Model.",
    )
    parser.add_argument(
        "kernel_path",
        type=Path,
        help="Path to the kernel file containing the optimized ModelNew.",
    )
    parser.add_argument(
        "--reference-backend",
        type=str,
        default="torch_compile",
        help=(
            "Reference execution backend. "
            "Use torch_compile for compile mode."
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="triton",
        help="Kernel backend name sent to KernelGYM.",
    )
    parser.add_argument(
        "--entry-point",
        type=str,
        default=None,
        help=(
            "Reference model class name. If omitted, infer the first "
            "top-level "
            "class name from the reference file."
        ),
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Optional task id. If omitted, generate a random one.",
    )
    parser.add_argument(
        "--num-correct-trials",
        type=int,
        default=5,
        help="Number of correctness trials.",
    )
    parser.add_argument(
        "--num-perf-trials",
        type=int,
        default=100,
        help="Number of performance trials.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help=(
            "Force the server to re-evaluate instead of using cached results."
        ),
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")
    return resolved.read_text(encoding="utf-8")


def parse_module(code: str, source_name: str) -> ast.Module:
    try:
        return ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Failed to parse {source_name}: {exc}") from exc


def infer_entry_point(
    reference_code: str,
    explicit_entry_point: str | None,
) -> str:
    if explicit_entry_point:
        return explicit_entry_point

    module = parse_module(reference_code, "reference code")
    for node in module.body:
        if isinstance(node, ast.ClassDef):
            return node.name

    raise ValueError(
        "Could not infer entry point from reference code: "
        "no class definition found."
    )


def get_top_level_function_sources(code: str) -> dict[str, str]:
    module = parse_module(code, "reference code")
    function_sources: dict[str, str] = {}

    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            source = ast.get_source_segment(code, node)
            if source:
                function_sources[node.name] = source.strip()

    return function_sources


def has_top_level_function(code: str, function_name: str) -> bool:
    module = parse_module(code, "kernel code")
    for node in module.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ):
            return True
    return False


def merge_reference_io_functions(reference_code: str, kernel_code: str) -> str:
    function_sources = get_top_level_function_sources(reference_code)
    merged_code = kernel_code.rstrip()

    for function_name in REQUIRED_REFERENCE_FUNCTIONS:
        if has_top_level_function(merged_code, function_name):
            continue
        function_source = function_sources.get(function_name)
        if function_source is None:
            raise ValueError(
                "Reference code does not define required function "
                f"'{function_name}'."
            )
        merged_code = f"{merged_code}\n\n{function_source}\n"

    return merged_code


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


async def evaluate_pair(args: argparse.Namespace) -> dict:
    reference_code = read_text(args.reference_path)
    raw_kernel_code = read_text(args.kernel_path)
    kernel_code = merge_reference_io_functions(reference_code, raw_kernel_code)
    entry_point = infer_entry_point(reference_code, args.entry_point)
    task_id = args.task_id or f"kernel-pair-{uuid4().hex[:8]}"
    server_url = await resolve_server_url()

    payload = {
        "task_id": task_id,
        "reference_code": reference_code,
        "reference_backend": args.reference_backend,
        "kernel_code": kernel_code,
        "entry_point": entry_point,
        "backend": args.backend,
        "num_correct_trials": args.num_correct_trials,
        "num_perf_trials": args.num_perf_trials,
        "force_refresh": args.force_refresh,
    }

    timeout = httpx.Timeout(300.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{server_url}/evaluate", json=payload)
        response.raise_for_status()
        return response.json()


def print_result(result: dict, reference_backend: str) -> None:
    print(f"Reference Label: {build_reference_label(reference_backend)}")
    print(f"Compiled: {result.get('compiled')}")
    print(f"Correctness: {result.get('correctness')}")

    speedup = result.get("speedup")
    if speedup is not None:
        print(f"Speedup: {speedup:.2f}x")

    reference_runtime = result.get("reference_runtime")
    if reference_runtime is not None:
        print(f"Reference Runtime: {reference_runtime:.4f} ms")

    kernel_runtime = result.get("kernel_runtime")
    if kernel_runtime is not None:
        print(f"Kernel Runtime: {kernel_runtime:.4f} ms")

    error = result.get("error") or result.get("error_message")
    if error:
        print(f"Error: {error}")


def main() -> int:
    args = parse_args()
    result = asyncio.run(evaluate_pair(args))
    print_result(result, args.reference_backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
