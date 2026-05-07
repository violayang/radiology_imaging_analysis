# OCI Image Similarity Search Demo

This demo uses OCI Generative AI `cohere.embed-v4.0` to create image embeddings, stores them in an Oracle 26ai VECTOR table, and provides a Streamlit UI to search for visually similar images.

## Files

- `adb.sql`: creates the demo user and `IMAGE_VECTORS` table.
- `batch_vectorize_images.py`: reads images from OCI Object Storage, embeds them, and stores vectors in Oracle 26ai.
- `image_similarity_search.py`: searches the vector table from a local image or uploaded image bytes.
- `streamlit_app.py`: web UI for drag/drop image similarity search.
- `.env.example`: environment variable template.

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy the environment template and fill in your values:

```bash
cp .env.example .env
```

Unzip your ADB wallet locally and point `TNS_ADMIN` and `WALLET_LOCATION` to that directory. Do not commit `.env` or wallet files.

## Batch Vectorization

```bash
python batch_vectorize_images.py --limit 5
```

Remove `--limit` to ingest all images under the configured bucket and prefix.

## CLI Search

```bash
python image_similarity_search.py "/path/to/query-image.jpg" --top-k 5
```

## Streamlit UI

```bash
streamlit run streamlit_app.py
```

Open the local URL, upload one image, and click **Find Similar Images**.
