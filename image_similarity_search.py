import argparse
import array
import base64
import mimetypes
import os
import re
from pathlib import Path

import oci
import oracledb
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import EmbedTextDetails, OnDemandServingMode

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - lets the script explain missing deps later.
    load_dotenv = None


DEFAULT_MODEL_ID = "cohere.embed-v4.0"
DEFAULT_OUTPUT_DIMENSIONS = 1024


def load_environment():
    if load_dotenv:
        load_dotenv(override=True)


def env(name, default=None, required=False):
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def safe_table_name(table_name):
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_$#]*(\.[A-Za-z][A-Za-z0-9_$#]*)?", table_name):
        raise ValueError(f"Unsafe DB_TABLE value: {table_name!r}")
    return table_name


def get_oci_config():
    config_file = env("OCI_CONFIG_FILE", os.path.expanduser("~/.oci/config"))
    profile = env("OCI_PROFILE", "DEFAULT")
    config = oci.config.from_file(file_location=config_file, profile_name=profile)

    region = env("OCI_REGION")
    if region:
        config["region"] = region

    return config


def get_genai_client(config):
    endpoint = env("GENAI_ENDPOINT")
    if endpoint:
        return GenerativeAiInferenceClient(config=config, service_endpoint=endpoint)
    return GenerativeAiInferenceClient(config=config)


def get_db_connection():
    kwargs = {
        "user": env("DB_USERNAME", env("DB_USER"), required=True),
        "password": env("DB_PASSWORD", required=True),
        "dsn": env("DB_CONNECTION_STRING", env("DB_DSN"), required=True),
    }

    config_dir = env("TNS_ADMIN", env("WALLET_LOCATION"))
    if config_dir:
        kwargs["config_dir"] = config_dir

    wallet_location = env("WALLET_LOCATION")
    if wallet_location:
        kwargs["wallet_location"] = wallet_location

    wallet_password = env("WALLET_PASSWORD")
    if wallet_password:
        kwargs["wallet_password"] = wallet_password

    return oracledb.connect(**kwargs)


def image_to_data_uri(image_path):
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"Could not infer a supported image MIME type for: {path}")

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def image_bytes_to_data_uri(image_bytes, mime_type):
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image MIME type: {mime_type!r}")

    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def get_image_embedding(genai_client, image_path):
    image_data_uri = image_to_data_uri(image_path)
    return get_image_embedding_from_data_uri(genai_client, image_data_uri)


def get_image_embedding_from_data_uri(genai_client, image_data_uri):
    output_dimensions = int(env("OCI_EMBEDDING_DIMENSIONS", str(DEFAULT_OUTPUT_DIMENSIONS)))

    embed_details = EmbedTextDetails(
        inputs=[image_data_uri],
        compartment_id=env("OCI_COMPARTMENT_OCID", env("COMPARTMENT_OCID"), required=True),
        serving_mode=OnDemandServingMode(model_id=env("OCI_EMBEDDING_MODEL", DEFAULT_MODEL_ID)),
        input_type=EmbedTextDetails.INPUT_TYPE_IMAGE,
        embedding_types=[EmbedTextDetails.EMBEDDING_TYPES_FLOAT],
        output_dimensions=output_dimensions,
        truncate=EmbedTextDetails.TRUNCATE_NONE,
    )

    response = genai_client.embed_text(embed_details)
    embeddings = extract_float_embeddings(response.data)
    if not embeddings:
        raise RuntimeError(
            "OCI Generative AI returned no float embeddings for the query image. "
            f"Response fields: {response.data.attribute_map}"
        )

    return array.array("f", embeddings[0])


def extract_float_embeddings(embed_result):
    if embed_result.embeddings:
        return embed_result.embeddings

    by_type = embed_result.embeddings_by_type
    if not by_type:
        return None

    if isinstance(by_type, dict):
        return by_type.get("float") or by_type.get("FLOAT")

    float_embeddings = getattr(by_type, "float", None)
    if float_embeddings:
        return float_embeddings

    return None


def search_similar_images(image_path, top_k=5):
    load_environment()
    config = get_oci_config()
    genai_client = get_genai_client(config)
    query_embedding = get_image_embedding(genai_client, image_path)
    rows = search_by_embedding(query_embedding, top_k=top_k)
    return [row["object_name"] for row in rows]


def search_similar_image_bytes(image_bytes, mime_type, top_k=5):
    load_environment()
    config = get_oci_config()
    genai_client = get_genai_client(config)
    image_data_uri = image_bytes_to_data_uri(image_bytes, mime_type)
    query_embedding = get_image_embedding_from_data_uri(genai_client, image_data_uri)
    return search_by_embedding(query_embedding, top_k=top_k)


def search_by_embedding(query_embedding, top_k=5):
    table_name = safe_table_name(env("DB_TABLE", "IMAGE_VECTORS"))
    top_k = max(1, min(int(top_k), 100))

    sql = f"""
        SELECT object_name,
               bucket_name,
               namespace,
               VECTOR_DISTANCE(embedding, :query_embedding, COSINE) AS distance
        FROM {table_name}
        WHERE embedding IS NOT NULL
        ORDER BY distance
        FETCH FIRST {top_k} ROWS ONLY
    """

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, query_embedding=query_embedding)
            return [
                {
                    "object_name": row[0],
                    "bucket_name": row[1],
                    "namespace": row[2],
                    "distance": row[3],
                }
                for row in cursor.fetchall()
            ]


def main():
    parser = argparse.ArgumentParser(
        description="Search Oracle 26ai image vectors using a local query image."
    )
    parser.add_argument("image_path", help="Path to a local JPG, JPEG, PNG, GIF, or WEBP image.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of image names to return.")
    args = parser.parse_args()

    results = search_similar_images(args.image_path, top_k=args.top_k)
    for index, object_name in enumerate(results, start=1):
        print(f"{index}. {object_name}")


if __name__ == "__main__":
    main()
