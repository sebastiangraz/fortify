"""Serverless GPU deploy on Modal (draft; not deployed yet).

  uv run --with modal modal deploy deploy/modal_app.py

One-time setup:
  modal secret create fortify FORTIFY_TOKEN=<random>
The weights volume caches LaMa and the Hugging Face models between cold starts.
Check Modal's current API names (gpu types, scaledown_window) before the first deploy.
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch>=2.8", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "numpy>=2.1",
        "pillow>=11",
        "transformers>=4.56",
        "timm>=1.0",
        "einops>=0.8",
        "accelerate>=1.0",
        "fastapi>=0.115",
        "uvicorn>=0.32",
    )
    .add_local_python_source("fortify")
)

app = modal.App("fortify")
weights = modal.Volume.from_name("fortify-weights", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    volumes={"/weights": weights},
    secrets=[modal.Secret.from_name("fortify")],
    env={"FORTIFY_WEIGHTS": "/weights", "HF_HOME": "/weights/hf"},
    scaledown_window=300,
    timeout=600,
)
@modal.asgi_app()
def web():
    from fortify.service import app as fastapi_app

    return fastapi_app
