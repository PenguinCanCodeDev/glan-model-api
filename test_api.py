from fastapi.testclient import TestClient
import importlib.util
import pathlib
import sys

# Try to import the FastAPI app both as a package and as a standalone module so
# this test file can be executed via pytest (recommended) or directly with
# `python API_Calls/test_api.py` (useful for quick smoke checks).
try:
    # preferred when running under pytest or as part of the package
    from API_Calls.main import app  # type: ignore
except Exception:
    # fallback: load main.py by file path
    main_path = pathlib.Path(__file__).parent / "main.py"
    spec = importlib.util.spec_from_file_location("api_main", str(main_path))
    api_main = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(api_main)
    app = getattr(api_main, "app")


def test_health_root():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    # basic shape checks
    assert "status" in data
    assert data.get("status") == "ok"
    assert "api_endpoints" in data
    assert isinstance(data["api_endpoints"], list)


if __name__ == "__main__":
    # Simple smoke-run that doesn't require pytest. Prints status and JSON.
    client = TestClient(app)
    resp = client.get("/")
    print("status_code:", resp.status_code)
    try:
        print("json:", resp.json())
    except Exception:
        print("raw text:", resp.text)
