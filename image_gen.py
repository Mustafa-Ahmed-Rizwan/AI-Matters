# image_gen (updated)
import os
from dotenv import load_dotenv
import logging
from typing import Dict, Any
from gradio_client import Client
import asyncio
import random
import ssl
import httpx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
hf_api_token = os.getenv("HF_API_TOKEN")
if not hf_api_token:
    raise ValueError("HF_API_TOKEN not found in .env file")

# Create a persistent client once (reuse between calls)
# - model_space can be "stabilityai/stable-diffusion" or the full space URL
# - httpx_kwargs allows setting timeouts, proxies, verify, etc.
HTTPX_TIMEOUT = 300  # seconds for httpx client; tune as needed
_client = Client(
    "https://stabilityai-stable-diffusion.hf.space", 
    hf_token=hf_api_token,
    httpx_kwargs={"timeout": HTTPX_TIMEOUT},
    verbose=False
)

async def generate_image(prompt: str, model_space: str = None, max_retries: int = 3) -> Dict[str, Any]:
    """
    Generate an image with retry logic and proper timeout handling.
    Uses Client.submit() -> Job.result(timeout=...) so we can observe queue status and
    control waiting more reliably than wrapping predict() in an executor.
    """
    # pick client (either the persistent one or a new one if model_space specified)
    client = _client if model_space is None else Client(model_space, hf_token=hf_api_token, httpx_kwargs={"timeout": HTTPX_TIMEOUT})

    for attempt in range(max_retries):
        try:
            logger.info(f"Generating image for prompt: {prompt} (attempt {attempt + 1}/{max_retries})")

            # Optional: debug the remote API description (uncomment when debugging)
            # api_info = client.view_api(return_format="dict")
            # logger.debug(f"API info: {api_info}")

            # Submit prediction as a Job (non-blocking)
            job = client.submit(
                prompt,                 # positional args must match the space's input order
                negative="", 
                scale=9,
                api_name="/infer"       # use correct api_name for that Space
            )

            # Wait for job result with a generous timeout (seconds)
            # Job.result will raise TimeoutError if not finished in time.
            JOB_TIMEOUT = 180  # seconds; tune higher if images take longer
            result = job.result(timeout=JOB_TIMEOUT)

            # result handling - many Spaces return a list/dict; adjust to your space's shape
            if result and isinstance(result, list) and len(result) > 0 and "image" in result[0]:
                logger.info(f"Image generated successfully on attempt {attempt + 1}")
                return {"image_url": result[0]["image"], "error": None}
            else:
                logger.warning(f"Invalid response on attempt {attempt + 1}: {type(result)} {result}")
                if attempt == max_retries - 1:
                    return {"image_url": None, "error": "Failed to generate image: Invalid response from API"}

        except asyncio.TimeoutError:
            logger.error(f"Async timeout on attempt {attempt + 1}")
            if attempt == max_retries - 1:
                return {"image_url": None, "error": f"Image generation timed out after {max_retries} attempts"}
        except httpx.ReadTimeout as e:
            logger.error(f"HTTP read timeout on attempt {attempt + 1}: {e}")
            # will retry
        except (httpx.ConnectError, httpx.NetworkError) as e:
            logger.error(f"Network/connect error on attempt {attempt + 1}: {e}")
        except ssl.SSLError as e:
            logger.error(f"SSL error on attempt {attempt + 1}: {e}")
            # SSL handshake timeouts often need network / cert fixes; consider not retrying multiple times
            if attempt == max_retries - 1:
                return {"image_url": None, "error": f"Image generation failed after {max_retries} attempts: {e}"}
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {type(e).__name__}: {e}")
            if attempt == max_retries - 1:
                return {"image_url": None, "error": f"Image generation failed after {max_retries} attempts: {e}"}

        # backoff with jitter
        backoff = (2 ** attempt) + random.uniform(0, 1)
        logger.info(f"Sleeping {backoff:.1f}s before next attempt")
        await asyncio.sleep(backoff)

    return {"image_url": None, "error": "Maximum retry attempts exceeded"}