import argparse
import array
import base64
import mimetypes
import os
import re

import oci
import oracledb
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import EmbedTextDetails, OnDemandServingMode
from oci.object_storage import ObjectStorageClient

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - lets the script explain missing deps later.
    load_dotenv = None


DEFAULT_MODEL_ID = "cohere.embed-v4.0"
DEFAULT_OUTPUT_DIMENSIONS = 1024
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp")


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


def get_namespace(object_storage_client):
    namespace = env("OBJECT_STORAGE_NAMESPACE", env("NAMESPACE"))
    if namespace:
        return namespace
    return object_storage_client.get_namespace().data


def iter_image_objects(object_storage_client, namespace, bucket_name, prefix):
    start = None
    while True:
        response = object_storage_client.list_objects(
            namespace_name=namespace,
            bucket_name=bucket_name,
            prefix=prefix or None,
            start=start,
        )

        for obj in response.data.objects:
            if obj.name.lower().endswith(IMAGE_EXTENSIONS):
                yield obj.name

        start = response.data.next_start_with
        if not start:
            break


def object_to_data_uri(object_storage_client, namespace, bucket_name, object_name):
    response = object_storage_client.get_object(
        namespace_name=namespace,
        bucket_name=bucket_name,
        object_name=object_name,
    )
    mime_type, _ = mimetypes.guess_type(object_name)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"

    encoded = base64.b64encode(response.data.content).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def get_image_embedding(genai_client, image_data_uri):
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
            "OCI Generative AI returned no float embeddings for an image. "
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


def upsert_image_vector(cursor, table_name, namespace, bucket_name, object_name, embedding):
    cursor.execute(
        f"""
        UPDATE {table_name}
        SET embedding = :embedding
        WHERE namespace = :namespace
          AND bucket_name = :bucket_name
          AND object_name = :object_name
        """,
        embedding=embedding,
        namespace=namespace,
        bucket_name=bucket_name,
        object_name=object_name,
    )

    if cursor.rowcount:
        return "updated"

    cursor.execute(
        f"""
        INSERT INTO {table_name} (object_name, bucket_name, namespace, embedding)
        VALUES (:object_name, :bucket_name, :namespace, :embedding)
        """,
        object_name=object_name,
        bucket_name=bucket_name,
        namespace=namespace,
        embedding=embedding,
    )
    return "inserted"


def ingest_bucket_images(limit=None, commit_every=25):
    load_environment()
    config = get_oci_config()
    object_storage_client = ObjectStorageClient(config)
    genai_client = get_genai_client(config)

    namespace = get_namespace(object_storage_client)
    bucket_name = env("BUCKET_NAME", required=True)
    prefix = env("PREFIX", "")
    table_name = safe_table_name(env("DB_TABLE", "IMAGE_VECTORS"))

    inserted = 0
    updated = 0
    processed = 0

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            for object_name in iter_image_objects(object_storage_client, namespace, bucket_name, prefix):
                if limit is not None and processed >= limit:
                    break

                print(f"Embedding {object_name}")
                image_data_uri = object_to_data_uri(object_storage_client, namespace, bucket_name, object_name)
                embedding = get_image_embedding(genai_client, image_data_uri)
                action = upsert_image_vector(
                    cursor,
                    table_name,
                    namespace,
                    bucket_name,
                    object_name,
                    embedding,
                )

                processed += 1
                inserted += int(action == "inserted")
                updated += int(action == "updated")

                if processed % commit_every == 0:
                    conn.commit()
                    print(f"Committed {processed} images")

            conn.commit()

    return {"processed": processed, "inserted": inserted, "updated": updated}


def main():
    parser = argparse.ArgumentParser(
        description="Vectorize images from OCI Object Storage into an Oracle 26ai VECTOR table."
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of images to ingest.")
    parser.add_argument("--commit-every", type=int, default=25, help="Commit after this many images.")
    args = parser.parse_args()

    summary = ingest_bucket_images(limit=args.limit, commit_every=args.commit_every)
    print(
        "Done: "
        f"{summary['processed']} processed, "
        f"{summary['inserted']} inserted, "
        f"{summary['updated']} updated"
    )


if __name__ == "__main__":
    main()
