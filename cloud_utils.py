# cloud_utils.py
"""Small GCS helpers used by the training runner.

These helpers defer importing `google.cloud.storage` until the function is
called so importing this module doesn't fail in environments that don't have
the GCS client installed (useful for local linting or tests).
"""
from urllib.parse import urlparse
from typing import Optional, Tuple
import json
import tempfile
from pathlib import Path


def _parse_gs_uri(gcs_uri: str) -> Tuple[str, str]:
    p = urlparse(gcs_uri)
    if p.scheme != "gs":
        raise ValueError("Expect gs:// URI")
    bucket = p.netloc
    blob_path = p.path.lstrip("/")
    return bucket, blob_path


def upload_file(local_path: str, gcs_uri: str, client: Optional[object] = None) -> None:
    """Upload a local file to a `gs://` URI.

    The `client` param may be a `google.cloud.storage.Client` instance; if not
    provided the function will create one lazily.
    """
    from google.cloud import storage

    client = client or storage.Client()
    bucket_name, blob_path = _parse_gs_uri(gcs_uri)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)


def upload_json(obj, gcs_uri: str, client: Optional[object] = None) -> None:
    """Upload an object as pretty JSON to `gs://`.

    This writes the JSON string directly to the blob (no local temp file).
    """
    from google.cloud import storage

    client = client or storage.Client()
    bucket_name, blob_path = _parse_gs_uri(gcs_uri)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_string(json.dumps(obj, indent=2), content_type="application/json")