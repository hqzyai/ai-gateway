from importlib.util import find_spec
from pathlib import Path
import shutil


def preserve_runtime_assets(directory: str, names: list[str]) -> tuple[str, ...]:
    return tuple(
        name
        for name in names
        if name == "__pycache__"
        or name.endswith((".pyc", ".so"))
        or (Path(directory).name == "_experimental" and name == "out")
        or (Path(directory).name == "proxy" and name == "enterprise")
    )


def install() -> None:
    spec = find_spec("litellm")
    if spec is None or spec.origin is None:
        raise RuntimeError("The base image must contain the native LiteLLM package")
    destination = Path(spec.origin).parent
    shutil.copytree("/tmp/litellm-source", destination, dirs_exist_ok=True, ignore=preserve_runtime_assets)
    for cache in destination.rglob("__pycache__"):
        shutil.rmtree(cache)
    from litellm.proxy.management_endpoints.agentos_usage_endpoints import router

    assert any(route.path == "/agentos/usage/employee-budget" for route in router.routes)


if __name__ == "__main__":
    install()
