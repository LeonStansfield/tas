#
# TEE Attestation Service - OpenBao Plugin Integration
#
# Copyright 2026 Hewlett Packard Enterprise Development LP.
# SPDX-License-Identifier: MIT
#
# This file is part of the TEE Attestation Service.
#
# This plugin provides integration with OpenBao for key management and
# secret retrieval for the TEE Attestation Service (TAS) via HTTP REST.

from __future__ import annotations

import base64
import json
import math
import os
import posixpath
import re
import secrets as _secrets
import threading
from typing import Any, Dict, Optional
from urllib.parse import quote, urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

try:
    import yaml  # PyYAML (listed in requirements.txt)
except Exception:
    yaml = None

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding as asympadding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.serialization import (
    load_der_public_key,
    load_pem_public_key,
)

from tas.exceptions import KBMResponseError, KBMUnavailableError
from tas.tas_logging import get_logger

# Setup logging for the OpenBao KBM plugin
logger = get_logger("tas.plugins.tas_kbm_openbao")

# Declare host dependencies (opt-in): currently no extra host kwargs required
KBM_HOST_KWARGS = set()

AES_KEY_LEN = 32  # AES-256
IV_LEN = 12  # AES-GCM IV size


# Crypto Helpers


def _b64(b: bytes) -> str:
    """Encode bytes to ASCII Base64 string."""
    return base64.b64encode(b).decode("ascii")


def _load_rsa_public_key(raw: bytes) -> rsa.RSAPublicKey:
    """Load RSA public key from PEM or DER format and verify key type."""
    key = None
    try:
        key = load_pem_public_key(raw)
    except Exception:
        pass

    if key is None:
        try:
            key = load_der_public_key(raw)
        except ValueError as e:
            raise ValueError("Invalid RSA public key format") from e

    if not isinstance(key, rsa.RSAPublicKey):
        raise ValueError(
            f"Invalid public key type: expected an RSA public key (RSAPublicKey), got {type(key).__name__}. "
            "Only RSA public keys are supported for secret wrapping."
        )

    if key.key_size < 2048:
        raise ValueError(
            f"RSA public key size ({key.key_size} bits) is insufficient. "
            "Minimum required RSA key size is 2048 bits."
        )

    return key


def _aes_gcm_encrypt(key: bytes, iv: bytes, plaintext: bytes) -> tuple[bytes, bytes]:
    """Encrypt plaintext using AES-256-GCM, returning ciphertext and auth tag."""
    cipher = Cipher(algorithms.AES(key), modes.GCM(iv))
    enc = cipher.encryptor()
    ciphertext = enc.update(plaintext) + enc.finalize()
    return ciphertext, enc.tag


def _secret_to_bytes(v: Any) -> bytes:
    """Normalize a secret value to bytes."""
    if isinstance(v, bytes):
        return v
    if isinstance(v, bytearray):
        return bytes(v)
    if isinstance(v, str):
        return v.encode("utf-8")
    return json.dumps(v, separators=(",", ":")).encode("utf-8")


# Configuration File Handling


def _load_config_file(config_file: Optional[str]) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file with environment variable substitution.

    Raises ValueError if an explicit config_file path was supplied but does not exist,
    cannot be read, or fails to parse.
    """
    if not config_file:
        logger.debug("No config file specified for OpenBao KBM")
        return {}

    path = os.path.abspath(config_file)
    if not os.path.exists(path):
        raise ValueError(f"OpenBao config file not found: {path}")
    if not os.path.isfile(path) or not os.access(path, os.R_OK):
        raise ValueError(f"OpenBao config file is not a readable regular file: {path}")

    logger.info(f"Loading OpenBao KBM config from: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        raise ValueError(f"Failed to read OpenBao config file '{path}': {e}") from e

    def env_replacer(match):
        env_var = match.group(1)
        env_value = os.getenv(env_var)
        if env_value is None:
            logger.warning(
                f"Environment variable {env_var} not found, replacing with empty string"
            )
            return ""
        return env_value

    content = re.sub(r"\$\{([^}]+)\}", env_replacer, content)

    data = None
    _, ext = os.path.splitext(path.lower())
    if ext in (".yaml", ".yml"):
        if not yaml:
            raise RuntimeError("PyYAML is required to parse YAML config files")
        try:
            data = yaml.safe_load(content)
        except Exception as e:
            raise ValueError(
                f"Failed to parse OpenBao YAML config '{path}': {e}"
            ) from e
    else:
        try:
            data = json.loads(content)
        except Exception as e:
            raise ValueError(
                f"Failed to parse OpenBao JSON config '{path}': {e}"
            ) from e

    if data is None:
        data = {}
    elif not isinstance(data, dict):
        raise ValueError(
            f"Config file '{path}' must parse to a key-value dictionary, got {type(data).__name__}"
        )

    return data


def _validate_config(
    url: Optional[str],
    verify_ssl: bool,
    ca_bundle: Optional[str],
    kv_version: int,
    requests_timeout: int,
    pool_connections: int,
    pool_maxsize: int,
    retry_total: int,
    retry_backoff_factor: float,
) -> None:
    """Validate all OpenBao configuration parameters at startup.

    Raises:
        ValueError: If any configuration parameter fails validation.
    """
    # Require explicit URL
    if not url or not isinstance(url, str) or not url.strip():
        raise ValueError(
            "OpenBao URL is required. Specify 'url' in configuration or set BAO_ADDR/VAULT_ADDR."
        )

    # Validate URL scheme
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(
            f"Invalid OpenBao URL '{url}': URL must include a valid scheme ('http' or 'https') and host."
        )

    if parsed.scheme == "http" and verify_ssl:
        raise ValueError(
            f"Insecure scheme 'http://' is not permitted when verify_ssl=True. "
            f"Use HTTPS or set verify_ssl=False for local development environments."
        )

    # Validate CA bundle
    if ca_bundle:
        bundle_path = os.path.abspath(ca_bundle)
        if not os.path.exists(bundle_path):
            raise ValueError(f"Configured ca_bundle does not exist: {bundle_path}")
        if not os.path.isfile(bundle_path) or not os.access(bundle_path, os.R_OK):
            raise ValueError(
                f"Configured ca_bundle is not a readable regular file: {bundle_path}"
            )

    # Validate kv_version
    if kv_version not in (1, 2):
        raise ValueError(
            f"Invalid kv_version: {kv_version}. OpenBao KV engine version must be 1 or 2."
        )

    # Validate requests_timeout
    if (
        not isinstance(requests_timeout, (int, float))
        or not math.isfinite(requests_timeout)
        or requests_timeout <= 0
    ):
        raise ValueError(
            f"Invalid requests_timeout: {requests_timeout}. Timeout must be a positive finite number."
        )

    # Validate connection pools
    if (
        isinstance(pool_connections, bool)
        or not isinstance(pool_connections, int)
        or pool_connections <= 0
    ):
        raise ValueError(
            f"Invalid pool_connections: {pool_connections}. Must be an integer greater than 0."
        )
    if (
        isinstance(pool_maxsize, bool)
        or not isinstance(pool_maxsize, int)
        or pool_maxsize <= 0
    ):
        raise ValueError(
            f"Invalid pool_maxsize: {pool_maxsize}. Must be an integer greater than 0."
        )

    # Validate retries
    if (
        isinstance(retry_total, bool)
        or not isinstance(retry_total, int)
        or retry_total < 0
    ):
        raise ValueError(
            f"Invalid retry_total: {retry_total}. Must be an integer >= 0."
        )
    if (
        isinstance(retry_backoff_factor, bool)
        or not isinstance(retry_backoff_factor, (int, float))
        or not math.isfinite(retry_backoff_factor)
        or retry_backoff_factor < 0.0
    ):
        raise ValueError(
            f"Invalid retry_backoff_factor: {retry_backoff_factor}. Must be a finite number >= 0.0."
        )


# Validation and URL Safety Helpers


def _validate_key_id(key_id: str) -> None:
    """
    Validate that key_id is safe and cannot alter the intended OpenBao path.

    Rejects:
    - Empty, whitespace, or non-string values
    - Path traversal segments ('.' and '..') and empty segments
    - Characters '?', '#', and '\\'
    - ASCII control characters
    - URL-encoded traversal sequences (%2e, %2f, %5c)
    """
    if not isinstance(key_id, str) or not key_id.strip():
        raise ValueError("key_id must be a non-empty string")

    # Reject ?, #, and \
    if any(c in key_id for c in ("?", "#", "\\")):
        raise ValueError("key_id contains invalid characters ('?', '#', or '\\')")

    # Reject ASCII control characters (0x00-0x1F and 0x7F)
    if any(ord(c) < 32 or ord(c) == 127 for c in key_id):
        raise ValueError("key_id contains control characters")

    # Reject URL-encoded and double-URL-encoded traversal attempts (%2e, %2f, %5c, %252e, etc.)
    if re.search(r"(?i)%(?:25)*(?:2e|2f|5c)", key_id):
        raise ValueError("key_id contains URL-encoded traversal sequences")

    # Reject empty segments and '.' / '..' path components
    segments = key_id.split("/")
    for segment in segments:
        if segment in ("", ".", ".."):
            raise ValueError(f"key_id contains invalid path segment: '{segment}'")


def _build_url(
    base_url: str,
    mount_point: str,
    kv_version: int,
    key_id: str,
) -> str:
    """
    Validate key_id, safely quote path components, and construct endpoint URL.

    Verifies the resulting URL path remains strictly scoped to the configured mount point.
    """
    _validate_key_id(key_id)

    # Split key_id into segments, quote each, and reconstruct
    segments = key_id.split("/")
    encoded_path = "/".join(quote(seg, safe="") for seg in segments)

    mount = mount_point.strip("/")
    if int(kv_version) == 2:
        expected_prefix = f"/v1/{mount}/data/"
    else:
        expected_prefix = f"/v1/{mount}/"

    endpoint = f"{expected_prefix}{encoded_path}"
    url = urljoin(f"{base_url.rstrip('/')}/", endpoint.lstrip("/"))

    # Verify URL Scope: ensure path does not escape the configured mount path
    parsed = urlparse(url)
    norm_path = posixpath.normpath(parsed.path)
    norm_prefix = posixpath.normpath(expected_prefix)
    if not (norm_path == norm_prefix or norm_path.startswith(norm_prefix + "/")):
        raise ValueError(
            f"URL path '{norm_path}' escapes configured mount path '{norm_prefix}'"
        )

    return url


def _parse_bool(value: Any, name: str = "value") -> bool:
    """
    Strictly parse a boolean configuration value.

    Accepts:
        True, False (bool)
        "true", "false", "1", "0" (case-insensitive str)
        1, 0 (int)

    Raises:
        ValueError for any other value (does not silently fall back).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1"):
            return True
        if v in ("false", "0"):
            return False
    elif isinstance(value, int) and value in (0, 1):
        return bool(value)

    raise ValueError(
        f"Invalid boolean value for '{name}': {value!r}. "
        f"Expected true/false or 1/0."
    )


__all__ = [
    # Public KBM plugin API
    "kbm_open_client_connection",
    "kbm_close_client_connection",
    "kbm_get_secret",
    "OpenBaoUnavailableError",
    "OpenBaoPoolTimeoutError",
    "OpenBaoResponseError",
]

# OpenBao Client Implementation


class OpenBaoUnavailableError(KBMUnavailableError):
    """Raised when the OpenBao backend is temporarily unavailable."""

    public_message = "Secret service is temporarily unavailable"

    def __init__(self, *, retry_after: int = 1):
        super().__init__(retry_after=retry_after)


class OpenBaoPoolTimeoutError(OpenBaoUnavailableError):
    """Raised when OpenBao connection acquisition or pool times out."""

    pass


class OpenBaoResponseError(KBMResponseError):
    """Raised when OpenBao returns an invalid response."""

    public_message = "Secret service returned an invalid response"

    pass


class _OpenBaoClient:
    """OpenBao Key Management Backend client using HTTP REST."""

    def __init__(
        self,
        url: str = "http://127.0.0.1:8200",
        token: Optional[str] = None,
        mount_point: str = "secret",
        kv_version: int = 2,
        secret_field: str = "secret",
        verify_ssl: bool = True,
        ca_bundle: Optional[str] = None,
        requests_timeout: int = 30,
        retry_total: int = 3,
        retry_backoff_factor: float = 0.05,
        pool_connections: int = 10,
        pool_maxsize: int = 20,
    ):
        self.url = url.rstrip("/")
        self.token = token
        self.mount_point = mount_point.strip("/")
        self.kv_version = kv_version
        self.secret_field = secret_field
        self.verify_ssl = verify_ssl
        self.ca_bundle = ca_bundle
        self.requests_timeout = requests_timeout
        self.retry_total = retry_total
        self.retry_backoff_factor = retry_backoff_factor
        self.pool_connections = pool_connections
        self.pool_maxsize = pool_maxsize

        # Bounded semaphore to strictly bound connection acquisition queue times.
        # This prevents threads from blocking indefinitely when pool_maxsize is reached.
        self._pool_semaphore = threading.BoundedSemaphore(value=self.pool_maxsize)

        # Configure SSL verify parameter for requests library
        # verify can be: False (disable verification), True (use system CAs), or str (path to CA bundle)
        if not self.verify_ssl:
            self.verify_param: Any = False
            logger.warning(
                "TLS verification is DISABLED for OpenBao connections (verify_ssl=false). "
                "This should only be used for development/debug environments."
            )
        else:
            # verify_ssl is True: use CA bundle if provided, otherwise use system defaults
            if self.ca_bundle and os.path.isfile(self.ca_bundle):
                self.verify_param = self.ca_bundle
                logger.debug(f"Using custom CA bundle: {self.ca_bundle}")
            else:
                self.verify_param = True
                logger.debug("Using system default CA certificates")

        # HTTP session for REST communication
        self.session = requests.Session()
        if self.token:
            self.session.headers.update({"X-Vault-Token": self.token})

        # Retry strategy
        retry_strategy = Retry(
            total=self.retry_total,
            connect=self.retry_total,
            read=self.retry_total,
            backoff_factor=self.retry_backoff_factor,  # exponential backoff factor for retries
            status_forcelist=[
                429,
                500,
                502,
                503,
                504,
            ],  # HTTP status codes (429: Rate limits, 500: Internal Server Error, 502: Bad Gateway, 503: Service Unavailable, 504: Gateway Timeout)
            allowed_methods=frozenset(["GET", "HEAD", "OPTIONS"]),
            raise_on_status=False,  # allows the application layer to parse non-transient HTTP errors cleanly
        )

        # Connection pool adapter configured with pool_block=True:
        # - pool_connections: Controls the number of distinct host connection pools cached by urllib3.
        # - pool_maxsize: Hard upper bound on concurrent connections maintained per host pool.
        # - pool_block=True: Enforces pool_maxsize as a strict ceiling. When all connections in the
        #   pool are in use, subsequent connection acquisition requests block until an active connection
        #   is returned to the pool (or until timeout), preventing connection storms and descriptor churn.
        # - Process scope: These limits apply per TAS worker process. Total concurrent connections
        #   to OpenBao across a deployment will be (number of worker processes * pool_maxsize).
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.pool_connections,
            pool_maxsize=self.pool_maxsize,
            pool_block=True,
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        logger.info(
            f"OpenBao KBM client initialized for {self.url} (mount: {self.mount_point}, KV v{self.kv_version})"
        )

    def close(self) -> None:
        """Close the underlying HTTP session."""
        if self.session:
            self.session.close()

    def get_secret(self, key_id: str) -> bytes:
        """
        Retrieve secret bytes for the given key_id from OpenBao via REST API.

        Supports both KV version 1 and KV version 2 engines.
        """
        logger.debug(f"Fetching secret from OpenBao for key_id: {key_id}")

        url = _build_url(
            base_url=self.url,
            mount_point=self.mount_point,
            kv_version=self.kv_version,
            key_id=key_id,
        )

        # Bound connection checkout: block up to requests_timeout seconds
        acquired = self._pool_semaphore.acquire(timeout=self.requests_timeout)
        if not acquired:
            logger.error(
                f"Connection acquisition timed out after {self.requests_timeout}s "
                f"for key_id: {key_id} (pool maxsize: {self.pool_maxsize})"
            )
            raise OpenBaoPoolTimeoutError()

        try:
            resp = self.session.get(
                url,
                verify=self.verify_param,
                timeout=self.requests_timeout,
            )
        except requests.RequestException as exc:
            logger.error("OpenBao request failed", exc_info=True)
            raise OpenBaoUnavailableError() from exc
        finally:
            self._pool_semaphore.release()

        if resp.status_code == 404:
            logger.error(f"Secret not found in OpenBao: {key_id}")
            raise ValueError("Secret not found")
        elif resp.status_code != 200:
            logger.error(
                f"OpenBao secret retrieval failed ({resp.status_code}) for key_id {key_id}: {resp.text}"
            )
            raise OpenBaoResponseError()

        try:
            payload = resp.json()
        except Exception as e:
            logger.error(
                f"Failed to parse OpenBao response as JSON for key_id {key_id}: {e}"
            )
            raise OpenBaoResponseError() from e

        if not isinstance(payload, dict):
            logger.error(
                f"Invalid OpenBao response structure: expected JSON object, got {type(payload).__name__}"
            )
            raise OpenBaoResponseError()

        if self.kv_version == 2:
            top_data = payload.get("data")
            if top_data is None:
                raise OpenBaoResponseError()
            if not isinstance(top_data, dict):
                raise OpenBaoResponseError()
            data = top_data.get("data")
        else:
            data = payload.get("data")

        if data is None:
            logger.error(
                f"OpenBao response contains null secret data for key_id: {key_id}"
            )
            raise OpenBaoResponseError()

        if not isinstance(data, dict):
            logger.error(
                f"Invalid OpenBao response structure for key_id {key_id}: expected secret data dictionary, got {type(data).__name__}"
            )
            raise OpenBaoResponseError()

        if self.secret_field not in data:
            logger.error(
                f"Secret field '{self.secret_field}' not found in OpenBao secret data for key_id: {key_id}"
            )
            raise ValueError("Secret field not found in OpenBao secret data")

        val = data[self.secret_field]
        if val is None:
            logger.error(
                "OpenBao returned a null value for the configured secret field"
            )
            raise OpenBaoResponseError()
        return _secret_to_bytes(val)


# KBM Plugin Interface


def kbm_open_client_connection(config_file: Optional[str] = None) -> _OpenBaoClient:
    """
    Initialize and return the OpenBao KBM client handle.

    Args:
        config_file: Path to plugin configuration file (YAML or JSON)

    Returns:
        _OpenBaoClient handle for use with kbm_get_secret
    """
    logger.info("Initializing OpenBao KBM client connection")
    cfg = _load_config_file(config_file)

    def _get_required_int(key: str, env_var: str, default: int) -> int:
        val = cfg.get(key)
        if val is None:
            val = os.getenv(env_var)
        if val is None:
            return default

        # Reject booleans
        if isinstance(val, bool):
            raise ValueError(
                f"Boolean not allowed for integer setting '{key}' / '{env_var}': {val!r}"
            )

        # If it's already an int (and not a bool)
        if isinstance(val, int):
            return val

        # Reject float types that have fractional parts or non-finite values
        if isinstance(val, float):
            if not math.isfinite(val) or not val.is_integer():
                raise ValueError(
                    f"Invalid non-integer or non-finite float value for '{key}' / '{env_var}': {val!r}"
                )
            return int(val)

        # For string values (e.g. from environment variables or YAML)
        if isinstance(val, str):
            s = val.strip()
            # Strict regex for optional sign and digits only (rejects decimal dots, 'nan', 'inf')
            if not re.fullmatch(r"[+-]?\d+", s):
                raise ValueError(
                    f"Invalid integer value for '{key}' / '{env_var}': {val!r}"
                )
            return int(s)

        raise ValueError(
            f"Invalid type for '{key}' / '{env_var}': {type(val).__name__}"
        )

    def _get_required_float(key: str, env_var: str, default: float) -> float:
        val = cfg.get(key)
        if val is None:
            val = os.getenv(env_var)
        if val is None:
            return default

        # Reject booleans
        if isinstance(val, bool):
            raise ValueError(
                f"Boolean not allowed for numeric setting '{key}' / '{env_var}': {val!r}"
            )

        try:
            f_val = float(val)
        except (ValueError, TypeError):
            raise ValueError(
                f"Invalid numeric value for '{key}' / '{env_var}': {val!r}"
            )

        if not math.isfinite(f_val):
            raise ValueError(
                f"Numeric value for '{key}' / '{env_var}' must be finite (got {val!r})"
            )

        return f_val

    # Resolve connection & security parameters
    url = cfg.get("url") or os.getenv("BAO_ADDR") or os.getenv("VAULT_ADDR")

    raw_verify_ssl = cfg.get("verify_ssl")
    if raw_verify_ssl is None:
        raw_verify_ssl = os.getenv("BAO_VERIFY_SSL")
    if raw_verify_ssl is None:
        raw_verify_ssl = os.getenv("VAULT_VERIFY_SSL")
    if raw_verify_ssl is None:
        verify_ssl = True
    else:
        verify_ssl = _parse_bool(raw_verify_ssl, "verify_ssl")

    ca_bundle = (
        cfg.get("ca_bundle") or os.getenv("BAO_CA_BUNDLE") or os.getenv("VAULT_CACERT")
    )

    # Resolve engine, timeouts, pool and retry settings
    kv_version = _get_required_int("kv_version", "BAO_KV_VERSION", 2)
    requests_timeout = _get_required_int("requests_timeout", "BAO_REQUESTS_TIMEOUT", 30)
    pool_connections = _get_required_int("pool_connections", "BAO_POOL_CONNECTIONS", 10)
    pool_maxsize = _get_required_int("pool_maxsize", "BAO_POOL_MAXSIZE", 20)
    retry_total = _get_required_int("retry_total", "BAO_RETRY_TOTAL", 3)
    retry_backoff_factor = _get_required_float(
        "retry_backoff_factor", "BAO_RETRY_BACKOFF_FACTOR", 0.05
    )

    # Validate entire configuration before continuing
    _validate_config(
        url=url,
        verify_ssl=verify_ssl,
        ca_bundle=ca_bundle,
        kv_version=kv_version,
        requests_timeout=requests_timeout,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        retry_total=retry_total,
        retry_backoff_factor=retry_backoff_factor,
    )

    # Token precedence: config token -> BAO_TOKEN -> VAULT_TOKEN -> token_file
    token = (
        (cfg.get("token") or "").strip()
        or os.getenv("BAO_TOKEN")
        or os.getenv("VAULT_TOKEN")
    )

    if not token:
        token_file = (
            cfg.get("token_file")
            or os.getenv("BAO_TOKEN_FILE")
            or os.getenv("VAULT_TOKEN_FILE")
        )
        if token_file:
            token_path = os.path.abspath(token_file)
            if not os.path.exists(token_path):
                raise ValueError(f"Configured token_file not found: {token_path}")
            if not os.path.isfile(token_path) or not os.access(token_path, os.R_OK):
                raise ValueError(
                    f"Configured token_file is not a readable regular file: {token_path}"
                )
            try:
                with open(token_path, "r", encoding="utf-8") as f:
                    token = f.read().strip()
            except Exception as e:
                raise ValueError(
                    f"Failed to read token from token_file '{token_path}': {e}"
                ) from e
            if not token:
                raise ValueError(f"Configured token_file '{token_path}' is empty")

    client = _OpenBaoClient(
        url=url,
        token=token,
        mount_point=cfg.get("mount_point", "secret"),
        kv_version=kv_version,
        secret_field=cfg.get("secret_field", "secret"),
        verify_ssl=verify_ssl,
        ca_bundle=ca_bundle,
        requests_timeout=requests_timeout,
        retry_total=retry_total,
        retry_backoff_factor=retry_backoff_factor,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
    )
    return client


def kbm_close_client_connection(client: Any) -> None:
    """
    Clean up the OpenBao client connection.

    Args:
        client: Client handle to close
    """
    logger.info("Closing OpenBao KBM client connection")
    if hasattr(client, "close"):
        client.close()


def kbm_get_secret(client: Any, key_id: str, wrapping_key: bytes) -> Dict[str, str]:
    """
    Retrieve secret from OpenBao and wrap with client RSA public key.

    Args:
        client: _OpenBaoClient handle from kbm_open_client_connection
        key_id: Identifier for the secret to retrieve
        wrapping_key: Client RSA public key for wrapping the secret

    Returns:
        Dictionary with keys: wrapped_key, blob, iv, tag (all base64-encoded)
    """
    logger.info(f"OpenBao KBM get_secret request for key_id: {key_id}")

    if not isinstance(client, _OpenBaoClient):
        logger.error("Invalid client handle provided")
        raise ValueError("Invalid client handle")
    if not key_id:
        logger.error("key_id is required but not provided")
        raise ValueError("key_id required")
    if not wrapping_key:
        logger.error("wrapping_key is required but not provided")
        raise ValueError("wrapping_key (client RSA public key) is required")

    # Load and validate client RSA public key
    pub = _load_rsa_public_key(wrapping_key)

    # Retrieve secret bytes from OpenBao
    secret_bytes = client.get_secret(key_id)

    # Generate ephemeral AES-256 key and 12-byte IV
    aes_key = _secrets.token_bytes(AES_KEY_LEN)
    iv = _secrets.token_bytes(IV_LEN)

    # Encrypt secret using AES-256-GCM
    blob, tag = _aes_gcm_encrypt(aes_key, iv, secret_bytes)

    # Wrap ephemeral AES key using client's RSA public key (RSA-OAEP SHA-256)
    wrapped_key = pub.encrypt(
        aes_key,
        asympadding.OAEP(
            mgf=asympadding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    result = {
        "wrapped_key": _b64(wrapped_key),
        "blob": _b64(blob),
        "iv": _b64(iv),
        "tag": _b64(tag),
    }

    logger.info(f"Successfully wrapped secret for key_id: {key_id}")
    return result
