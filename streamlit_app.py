import mimetypes

import streamlit as st
from oci.object_storage import ObjectStorageClient

from image_similarity_search import (
    env,
    get_oci_config,
    load_environment,
    search_similar_image_bytes,
)


st.set_page_config(
    page_title="Image Similarity Search",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_object_storage_client():
    load_environment()
    return ObjectStorageClient(get_oci_config())


@st.cache_data(show_spinner=False)
def fetch_object_image(namespace, bucket_name, object_name):
    client = get_object_storage_client()
    response = client.get_object(
        namespace_name=namespace,
        bucket_name=bucket_name,
        object_name=object_name,
    )
    mime_type, _ = mimetypes.guess_type(object_name)
    return response.data.content, mime_type or "image/jpeg"


def result_namespace(row):
    namespace = row.get("namespace") or env("OBJECT_STORAGE_NAMESPACE", env("NAMESPACE"))
    if namespace:
        return namespace
    return get_object_storage_client().get_namespace().data


def result_bucket(row):
    return row.get("bucket_name") or env("BUCKET_NAME", required=True)


def render_result_card(index, row):
    object_name = row["object_name"]
    namespace = result_namespace(row)
    bucket_name = result_bucket(row)

    image_bytes, _ = fetch_object_image(namespace, bucket_name, object_name)
    st.image(image_bytes, use_container_width=True)
    st.markdown(f"**{index}. {object_name}**")

    distance = row.get("distance")
    if distance is not None:
        st.caption(f"Cosine distance: {float(distance):.4f}")


def main():
    load_environment()

    st.title("Image Similarity Search")

    uploaded_file = st.file_uploader(
        "Upload query image",
        type=["jpg", "jpeg", "png", "gif", "webp"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        return

    query_bytes = uploaded_file.getvalue()

    preview_col, action_col = st.columns([1, 2], gap="large")
    with preview_col:
        st.image(query_bytes, caption=f"Query: {uploaded_file.name}", use_container_width=True)

    with action_col:
        st.subheader("Search")
        run_search = st.button("Find Similar Images", type="primary", use_container_width=True)

    if not run_search:
        return

    with st.spinner("Embedding query image and searching vector store..."):
        results = search_similar_image_bytes(
            query_bytes,
            uploaded_file.type or "image/jpeg",
            top_k=5,
        )

    st.subheader("Top 5 Similar Images")

    if not results:
        st.warning("No matching vectors were found. Run batch vectorization first.")
        return

    columns = st.columns(min(5, len(results)))
    for index, row in enumerate(results, start=1):
        with columns[(index - 1) % len(columns)]:
            render_result_card(index, row)


if __name__ == "__main__":
    main()
