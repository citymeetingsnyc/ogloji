import os
import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from starlette.status import HTTP_401_UNAUTHORIZED

from playwright._impl._browser import Browser
from playwright.async_api import async_playwright

import uvicorn

from contextlib import asynccontextmanager

import logging.config
from pythonjsonlogger.json import JsonFormatter

from urllib.parse import urlencode, urlparse, parse_qs
import hashlib

# ====================
# Logging configuration
# ====================

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": JsonFormatter,
        }
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "formatter": "json",
            "stream": "ext://sys.stdout",
        }
    },
    "loggers": {
        "fastapi": {"handlers": ["stdout"], "level": "DEBUG", "propagate": False},
        "uvicorn": {"handlers": ["stdout"], "level": "DEBUG", "propagate": False},
    },
    "root": {"level": "DEBUG", "handlers": ["stdout"]},
}
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger()


# =============
# Configuration
# =============


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server configuration
    BROWSER_POOL_SIZE: int = Field(
        default=5,
        description="Number of concurrent browser instances to screenshot OG images with",
    )
    API_KEY: str | None = Field(
        default=None, description="API key for authentication on protected endpoints"
    )

    # Base URL configuration
    BASE_URL: str = Field(
        description="Base URL for the website to generate OG images from"
    )

    # Local storage configuration
    LOCAL_STORAGE_PATH: str = Field(
        default="./images", description="Directory to store images locally"
    )


def get_settings() -> Settings:
    """Get the application settings."""
    return Settings()


class BrowserPool:
    """A pool of Playwright browser instances for concurrent screenshot operations.

    This class manages a fixed-size pool of browser instances using an async queue
    so we can reuse browser instances across multiple requests instead of spinning
    up a new browser instance for each request.
    """

    def __init__(self, size: int):
        """Initialize the browser pool.

        Args:
            size: The maximum number of concurrent browser instances to maintain
        """
        self.size = size
        self.pool = asyncio.Queue()

    async def init(self):
        """Initialize the Playwright instance and create the browser pool.

        Creates `size` number of headless Chromium browsers and adds them to the pool.
        Should be called during application startup.
        """
        self.playwright = await async_playwright().start()
        for _ in range(self.size):
            browser = await self.playwright.chromium.launch(headless=True)
            self.pool.put_nowait(browser)

    async def acquire(self) -> Browser:
        """Get an available browser instance from the pool.

        Returns:
            A Playwright Browser instance. If none are available, waits until one is returned.
        """
        return await self.pool.get()

    async def release(self, browser: Browser):
        """Return a browser instance to the pool.

        Args:
            browser: The browser instance to return to the pool
        """
        await self.pool.put(browser)

    async def shutdown(self):
        """Clean up all browser instances and stop Playwright."""
        while not self.pool.empty():
            browser = await self.pool.get()
            await browser.close()
        await self.playwright.stop()


browser_pool = None


async def init_browser_pool():
    """Initialize the browser pool with the configured size."""
    global browser_pool
    logger.info(
        f"Initializing browser pool with size {get_settings().BROWSER_POOL_SIZE}"
    )
    browser_pool = BrowserPool(get_settings().BROWSER_POOL_SIZE)
    await browser_pool.init()
    logger.info("Browser pool initialized successfully")


async def generate_image(path: str, filename: str) -> str:
    """Generate an OG image for the given path."""
    global browser_pool
    browser = await browser_pool.acquire()
    try:
        context = await browser.new_context(viewport={"width": 1200, "height": 630})
        page = await context.new_page()
        logger.info(f"Navigating to {get_settings().BASE_URL + path}")
        await page.goto(get_settings().BASE_URL + path)
        await page.wait_for_selector("#og-image", state="visible")
        fullpath = os.path.join(get_settings().LOCAL_STORAGE_PATH, filename)
        el = await page.query_selector("#og-image")
        await el.screenshot(path=fullpath)
        await context.close()
        return filename
    finally:
        await browser_pool.release(browser)


@asynccontextmanager
async def lifespan(app: FastAPI):
    class BucketNotFoundError(Exception):
        pass

    # Create the local image storage directory if it doesn't exist.
    if not os.path.exists(get_settings().LOCAL_STORAGE_PATH):
        logger.info(
            f"Creating local storage directory {get_settings().LOCAL_STORAGE_PATH}"
        )
        os.makedirs(get_settings().LOCAL_STORAGE_PATH, exist_ok=True)
    else:
        logger.info(
            f"Local storage directory {get_settings().LOCAL_STORAGE_PATH} already exists"
        )

    # Initialize our browser pool.
    await init_browser_pool()

    # Server is running and handling requests here
    yield

    # Shutdown
    await browser_pool.shutdown()


app = FastAPI(lifespan=lifespan)


def normalize_and_hash_path_and_query_params(path: str, query_params: dict) -> str:
    """Normalize the path and query parameters and create a consistent hash."""
    sorted_params = dict(sorted(query_params.items()))
    normalized_qs = urlencode(sorted_params, doseq=True)
    normalized_url = f"{path}?{normalized_qs}" if normalized_qs else path
    return hashlib.sha256(normalized_url.encode()).hexdigest()


@app.get("/og-image{path:path}")
async def serve_og_image(request: Request, path: str):
    # Get query parameters and normalize them
    query_params = dict(request.query_params)
    file_hash = normalize_and_hash_path_and_query_params(path, query_params)
    filename = f"{file_hash}.png"
    local_path = os.path.join(get_settings().LOCAL_STORAGE_PATH, filename)

    # Check local filesystem first
    if os.path.exists(local_path):
        return FileResponse(local_path)

    # Generate new image
    new_filename = await generate_image(path, filename)

    return FileResponse(os.path.join(get_settings().LOCAL_STORAGE_PATH, new_filename))


class PurgeRequest(BaseModel):
    request_uris: list[str]


@app.post("/purge-og-image")
async def purge_og_image(req: Request, body: PurgeRequest):
    # Check API key
    if (
        not get_settings().API_KEY
        or req.headers.get("X-API-Key") != get_settings().API_KEY
    ):
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    # Generate the filename using the same hashing logic
    for request_uri in body.request_uris:
        parsed_uri = urlparse(request_uri)
        path, query_params = parsed_uri.path, parse_qs(parsed_uri.query)
        file_hash = normalize_and_hash_path_and_query_params(path, query_params)
        filename = f"{file_hash}.png"

        # Delete from local storage if exists
        local_file = os.path.join(get_settings().LOCAL_STORAGE_PATH, filename)
        if os.path.exists(local_file):
            logger.info(f"Purging {local_file}")
            os.remove(local_file)
        else:
            logger.info(f"File {local_file} not found")

    return {"message": "Images purged successfully"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
