import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
import uvicorn
from threading import Thread
import time
from main import app
import requests
import uuid
from fastapi.responses import Response
import io
import imagehash
from PIL import Image
import os
from main import evict_from_local_storage

OG_IMAGE_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "og-image-fixture.png")


# This is the app we're going to take screenshots of OG images from.
html_app = FastAPI()


@html_app.get("/up")
async def up():
    return {"status": "up"}


@html_app.get("/random")
async def serve_random_page():
    random_uuid = uuid.uuid4()
    return Response(
        content=f"""
    <!DOCTYPE html>
    <html>
        <body style="margin: 0;">
            <div id="og-image" style="width: 1200px; height: 630px; background: #f0f0f0; display: flex; justify-content: center; align-items: center;">
                <span style="font-size: 48px;">Test OG Image {random_uuid}</span>
            </div>
        </body>
    </html>
    """,
        media_type="text/html",
    )


@html_app.get("/{path:path}")
async def serve_test_page():
    return Response(
        content="""
    <!DOCTYPE html>
    <html>
        <body style="margin: 0;">
            <div id="og-image" style="width: 1200px; height: 630px; background: #f0f0f0; display: flex; justify-content: center; align-items: center;">
                <span style="font-size: 48px;">Test OG Image</span>
            </div>
        </body>
    </html>
    """,
        media_type="text/html",
    )


def run_test_server():
    uvicorn.run(html_app, host="127.0.0.1", port=9001, log_level="error")


@pytest.fixture(autouse=True, scope="session")
def setup_html_server():
    client = TestClient(html_app)

    # Start the test HTML server
    server_thread = Thread(target=run_test_server, daemon=True)
    server_thread.start()

    # Wait for server to be ready by polling the /up endpoint
    timeout_seconds = 5
    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        try:
            response = client.get("/up")
            if response.status_code == 200:
                break
        except (ConnectionRefusedError, requests.exceptions.ConnectionError):
            time.sleep(0.1)
    else:
        raise RuntimeError(
            f"Test server failed to start within {timeout_seconds} seconds"
        )

    yield


def image_equal(image1: bytes, image2: bytes):
    # Convert bytes to PIL Images
    img1 = Image.open(io.BytesIO(image1))
    img2 = Image.open(io.BytesIO(image2))

    # Calculate difference hashes
    hash1 = imagehash.dhash(img1)
    hash2 = imagehash.dhash(img2)

    return hash1 == hash2


def test_e2e(monkeypatch, tmp_path):
    monkeypatch.setenv("BROWSER_POOL_SIZE", "2")
    monkeypatch.setenv("BASE_URL", "http://localhost:9001")
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path / "images"))
    monkeypatch.setenv("API_KEY", "test_api_key")
    with TestClient(app) as client:
        # Test the happy path -- an image we know is generated.
        response = client.get("/og-image/test")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        with open(OG_IMAGE_FIXTURE_PATH, "rb") as f:
            expected_image = f.read()
        assert image_equal(response.content, expected_image)

        # The first time we request a random page, it should be generated.
        response = client.get("/og-image/random")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        first_random_image = response.content

        # The second time we request a random page from the same path, it'll
        # be served from cache, so we should get the same image.
        response = client.get("/og-image/random")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert image_equal(response.content, first_random_image)

        # Purging and re-requesting the random page should generate a new image.
        response = client.post(
            "/purge-og-image",
            json={"request_uris": ["/random"]},
            headers={"X-API-Key": "test_api_key"},
        )
        assert response.status_code == 200
        response = client.get("/og-image/random")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert not image_equal(response.content, first_random_image)

        # When we add query params, it should be treated as a new image.
        response = client.get("/og-image/random?a=1&b=2")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert not image_equal(response.content, first_random_image)
        first_query_params_image = response.content

        # The order of query params should not matter. We should get the
        # same image served from cache.
        response = client.get("/og-image/random?b=2&a=1")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert image_equal(response.content, first_query_params_image)

        # When purging, the order of query params should not matter.
        response = client.post(
            "/purge-og-image",
            json={"request_uris": ["/random?b=2&a=1"]},
            headers={"X-API-Key": "test_api_key"},
        )
        assert response.status_code == 200
        response = client.get("/og-image/random?a=1&b=2")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert not image_equal(response.content, first_query_params_image)


def test_api_key_not_set_always_unauthorized(monkeypatch, tmp_path):
    monkeypatch.setenv("BROWSER_POOL_SIZE", "1")
    monkeypatch.setenv("BASE_URL", "http://localhost:9001")
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path / "images"))
    monkeypatch.delenv("API_KEY", raising=False)
    with TestClient(app) as client:
        response = client.post(
            "/purge-og-image",
            json={"request_uris": ["/test"]},
            headers={"X-API-Key": ""},
        )
        assert response.status_code == 401
        response = client.post(
            "/purge-og-image",
            json={"request_uris": ["/test"]},
            headers={"X-API-Key": "None"},
        )
        assert response.status_code == 401
        response = client.post(
            "/purge-og-image",
            json={"request_uris": ["/test"]},
            headers={"X-API-Key": "test_api_key"},
        )
        assert response.status_code == 401
        response = client.post(
            "/purge-og-image",
            json={"request_uris": ["/test"]},
        )
        assert response.status_code == 401


def test_api_key_when_set_is_authorized(monkeypatch, tmp_path):
    monkeypatch.setenv("BROWSER_POOL_SIZE", "1")
    monkeypatch.setenv("BASE_URL", "http://localhost:9001")
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path / "images"))
    monkeypatch.setenv("API_KEY", "my_test_api_key")
    with TestClient(app) as client:
        response = client.post(
            "/purge-og-image",
            json={"request_uris": ["/test"]},
            headers={"X-API-Key": "my_test_api_key"},
        )
        assert response.status_code == 200
        response = client.post(
            "/purge-og-image",
            json={"request_uris": ["/test"]},
            headers={"X-API-Key": "wrong_api_key"},
        )
        assert response.status_code == 401


@pytest.fixture
def storage_with_files(tmp_path):
    # Create test files with different access times
    files = {
        "old.png": (1000000, time.time() - 3600),  # 1MB, accessed 1 hour ago
        "medium.png": (2000000, time.time() - 1800),  # 2MB, accessed 30 mins ago
        "new.png": (1500000, time.time()),  # 1.5MB, accessed now
    }

    # Create the files
    for name, (size, atime) in files.items():
        path = tmp_path / name
        with open(path, "wb") as f:
            f.write(b"0" * size)
        os.utime(path, (atime, atime))

    return tmp_path


def test_no_eviction_needed(storage_with_files):
    # Test eviction with 4.5MB capacity (should keep all files)
    evicted = evict_from_local_storage(4.5, str(storage_with_files))
    assert len(evicted) == 0
    assert len(list(storage_with_files.glob("*.png"))) == 3


def test_evict_oldest_file(storage_with_files):
    # Test eviction with 3MB capacity (should evict oldest file)
    evicted = evict_from_local_storage(3.5, str(storage_with_files))
    assert len(evicted) == 1
    assert os.path.basename(evicted[0]) == "old.png"
    assert len(list(storage_with_files.glob("*.png"))) == 2
    assert not (storage_with_files / "old.png").exists()
    assert (storage_with_files / "medium.png").exists()
    assert (storage_with_files / "new.png").exists()


def test_evict_multiple_files(storage_with_files):
    # Test eviction with 1MB capacity (should evict all but newest)
    evicted = evict_from_local_storage(1.5, str(storage_with_files))
    assert len(evicted) == 2
    assert set(os.path.basename(p) for p in evicted) == {"old.png", "medium.png"}
    assert len(list(storage_with_files.glob("*.png"))) == 1
    assert (storage_with_files / "new.png").exists()


def test_evict_from_local_storage_empty_dir(tmp_path):
    # Test with empty directory
    evicted = evict_from_local_storage(1.0, str(tmp_path))
    assert len(evicted) == 0


def test_evict_from_local_storage_updates_on_access(tmp_path):
    # Create two test files
    files = {
        "old.png": (1000000, time.time() - 3600),
        "new.png": (1000000, time.time() - 1800),
    }

    for name, (size, atime) in files.items():
        path = tmp_path / name
        with open(path, "wb") as f:
            f.write(b"0" * size)
        os.utime(path, (atime, atime))

    # Update atime of old file to now
    old_path = tmp_path / "old.png"
    current_time = time.time()
    os.utime(old_path, (current_time, current_time))

    # Now evict with capacity that only allows one file
    evicted = evict_from_local_storage(1.0, str(tmp_path))
    assert len(evicted) == 1
    assert os.path.basename(evicted[0]) == "new.png"
    assert (tmp_path / "old.png").exists()
