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
import os
import re
import secrets as _secrets
import threading
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

try:
    import yaml  # PyYAML (listed in requirements.txt)
except Exception:
    yaml = None

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding as asympadding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.serialization import (
    load_der_public_key,
    load_pem_public_key,
)

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


def _load_rsa_public_key(raw: bytes):
    """Load RSA public key from PEM or DER format."""
    try:
        return load_pem_public_key(raw)
    except Exception:
        pass
    try:
        return load_der_public_key(raw)
    except Exception as e:
        raise ValueError("Invalid RSA public key format") from e


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
    """Load configuration from YAML or JSON file with environment variable substitution."""
    if not config_file:
        logger.debug("No config file specified for OpenBao KBM")
        return {}
    path = os.path.abspath(config_file)
    if not os.path.isfile(path):
        logger.warning(f"Config file not found: {path}")
        return {}

    logger.info(f"Loading OpenBao KBM config from: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    def env_replacer(match):
        env_var = match.group(1)
        env_value = os.getenv(env_var)
        if env_value is None:
            logger.warning(
                f"Environment variable {env_var} not found, using placeholder"
            )
            return match.group(0)
        return env_value

    content = re.sub(r"\$\{([^}]+)\}", env_replacer, content)

    _, ext = os.path.splitext(path.lower())
    if ext in (".yaml", ".yml") and yaml:
        try:
            data = yaml.safe_load(content) or {}
            if isinstance(data, dict):
                return data
        except Exception as e:
            logger.warning(f"Failed to parse config as YAML: {e}")

    try:
        data = json.loads(content) or {}
        return data
    except Exception as e:
        logger.warning(f"Failed to parse config as JSON: {e}")
        return {}


# OpenBao Client Implementation


class _OpenBaoClient:
    """OpenBao Key Management Backend client using HTTP REST."""

    def __init__(
        self,
        url: str = "http://127.0.0.1:8200",
        token: Optional[str] = None,
        auth_method: str = "token",
        role_id: Optional[str] = None,
        secret_id: Optional[str] = None,
        secret_id_file: Optional[str] = None,
        approle_mount: str = "approle",
        token_renew_on_401: bool = True,
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
        self.auth_method = auth_method.lower()
        self.role_id = role_id
        self.secret_id = secret_id
        self.secret_id_file = secret_id_file
        self.approle_mount = approle_mount.strip("/")
        self.token_renew_on_401 = token_renew_on_401
        self.mount_point = mount_point.strip("/")
        self.kv_version = int(kv_version)
        self.secret_field = secret_field
        self.verify_ssl = verify_ssl
        self.ca_bundle = ca_bundle
        self.requests_timeout = requests_timeout
        self.retry_total = max(0, int(retry_total))
        self.retry_backoff_factor = max(0.0, float(retry_backoff_factor))
        self.pool_connections = max(1, int(pool_connections))
        self.pool_maxsize = max(1, int(pool_maxsize))

        # Guards token updates across threads
        self._auth_lock = threading.Lock()

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

        # Connection pool adapter
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.pool_connections,
            pool_maxsize=self.pool_maxsize,
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Initial authentication if using AppRole
        if self.auth_method == "approle":
            self.authenticate()

        logger.info(
            f"OpenBao KBM client initialized for {self.url} (mount: {self.mount_point}, auth: {self.auth_method})"
        )

    def authenticate(self) -> str:
        """Authenticate or renew token with OpenBao and update session headers."""
        if self.auth_method == "approle":
            secret_id = self.secret_id
            if self.secret_id_file and os.path.isfile(self.secret_id_file):
                with open(self.secret_id_file, "r", encoding="utf-8") as f:
                    secret_id = f.read().strip()

            if not self.role_id or not secret_id:
                raise ValueError(
                    "role_id and secret_id (or secret_id_file) are required for AppRole authentication"
                )

            login_url = urljoin(f"{self.url}/", f"v1/auth/{self.approle_mount}/login")
            payload = {"role_id": self.role_id, "secret_id": secret_id}

            try:
                resp = self.session.post(
                    login_url,
                    json=payload,
                    verify=self.verify_param,
                    timeout=self.requests_timeout,
                )
            except requests.RequestException as e:
                raise RuntimeError(
                    f"OpenBao AppRole login connection error: {e}"
                ) from e

            if resp.status_code != 200:
                raise RuntimeError(
                    f"OpenBao AppRole login failed ({resp.status_code}): {resp.text}"
                )

            token = resp.json().get("auth", {}).get("client_token")
            if not token:
                raise RuntimeError(
                    "OpenBao AppRole login response missing client_token"
                )

            self.token = token
            self.session.headers["X-Vault-Token"] = token
            logger.info("Successfully authenticated to OpenBao via AppRole")
            return token

        elif self.auth_method == "token":
            if not self.token_renew_on_401 or not self.token:
                raise RuntimeError("Static OpenBao token rejected (401 Unauthorized)")

            renew_url = urljoin(f"{self.url}/", "v1/auth/token/renew-self")
            try:
                resp = self.session.post(
                    renew_url,
                    headers={"X-Vault-Token": self.token},
                    verify=self.verify_param,
                    timeout=self.requests_timeout,
                )
            except requests.RequestException as e:
                raise RuntimeError(
                    f"OpenBao token renewal connection error: {e}"
                ) from e

            if resp.status_code != 200:
                raise RuntimeError(
                    f"OpenBao token renewal failed ({resp.status_code}): {resp.text}"
                )

            logger.info("Successfully renewed OpenBao token")
            return self.token

        else:
            raise ValueError(f"Unsupported auth_method: {self.auth_method}")

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make an authenticated request to OpenBao with re-authentication on 401."""
        url = urljoin(f"{self.url}/", endpoint.lstrip("/"))
        kwargs.setdefault("verify", self.verify_param)
        kwargs.setdefault("timeout", self.requests_timeout)

        current_token = self.token
        try:
            resp = self.session.request(method, url, **kwargs)
        except requests.RequestException as e:
            logger.error(f"Failed to connect to OpenBao at {url}: {e}")
            raise RuntimeError(f"OpenBao connection error: {e}") from e

        # Token was rejected (expired or revoked); re-authenticate once under lock and retry
        if resp.status_code == 401:
            logger.info(
                "Received 401 from OpenBao; attempting re-authentication and retry"
            )
            with self._auth_lock:
                # Re-authenticate only if another concurrent thread has not already refreshed the token
                if self.token == current_token:
                    self.authenticate()

            if (
                "headers" in kwargs
                and isinstance(kwargs["headers"], dict)
                and "X-Vault-Token" in kwargs["headers"]
            ):
                kwargs["headers"]["X-Vault-Token"] = self.token

            try:
                resp = self.session.request(method, url, **kwargs)
            except requests.RequestException as e:
                logger.error(f"Failed to connect to OpenBao on retry at {url}: {e}")
                raise RuntimeError(f"OpenBao connection error on retry: {e}") from e

        return resp

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

        if self.kv_version == 2:
            endpoint = f"/v1/{self.mount_point}/data/{key_id}"
        else:
            endpoint = f"/v1/{self.mount_point}/{key_id}"

        resp = self._make_request("GET", endpoint)

        if resp.status_code == 404:
            logger.error(f"Secret not found in OpenBao: {key_id}")
            raise ValueError(f"Secret not found: {key_id}")
        elif resp.status_code != 200:
            logger.error(
                f"OpenBao secret retrieval failed ({resp.status_code}): {resp.text}"
            )
            raise RuntimeError(f"OpenBao API error ({resp.status_code}): {resp.text}")

        payload = resp.json()
        if self.kv_version == 2:
            data = payload.get("data", {}).get("data", {})
        else:
            data = payload.get("data", {})

        if isinstance(data, dict):
            if self.secret_field in data:
                val = data[self.secret_field]
            elif len(data) == 1:
                val = next(iter(data.values()))
            else:
                val = data
        else:
            val = data

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

    def _get_conf(key, env_var, default, caster=str):
        val = cfg.get(key)

        if val is None:
            val = os.getenv(env_var)

        if val is None:
            return default

        try:
            if caster is bool:
                return str(val).strip().lower() in ("true", "1", "yes")

            return caster(val)

        except (ValueError, TypeError):
            logger.warning(
                f"Invalid value for {key}/{env_var}: '{val}', using default {default}"
            )
            return default

    url = (
        cfg.get("url")
        or os.getenv("BAO_ADDR")
        or os.getenv("VAULT_ADDR")
        or "http://127.0.0.1:8200"
    )
    token = cfg.get("token") or os.getenv("BAO_TOKEN") or os.getenv("VAULT_TOKEN")

    # Authentication options
    auth_method = _get_conf("auth_method", "BAO_AUTH_METHOD", "token")
    role_id = _get_conf("role_id", "BAO_ROLE_ID", None)
    secret_id = _get_conf("secret_id", "BAO_SECRET_ID", None)
    secret_id_file = _get_conf("secret_id_file", "BAO_SECRET_ID_FILE", None)
    approle_mount = _get_conf("approle_mount", "BAO_APPROLE_MOUNT", "approle")
    token_renew_on_401 = _get_conf(
        "token_renew_on_401", "BAO_TOKEN_RENEW_ON_401", True, bool
    )

    mount_point = _get_conf("mount_point", "BAO_MOUNT_POINT", "secret")
    kv_version = _get_conf("kv_version", "BAO_KV_VERSION", 2, int)
    secret_field = _get_conf("secret_field", "BAO_SECRET_FIELD", "secret")
    verify_ssl = _get_conf("verify_ssl", "BAO_VERIFY_SSL", True, bool)
    ca_bundle = _get_conf("ca_bundle", "BAO_CACERT", os.getenv("VAULT_CACERT"))
    requests_timeout = _get_conf("requests_timeout", "BAO_REQUESTS_TIMEOUT", 30, int)

    # Connection pooling and retry options (config file takes precedence)
    retry_total = _get_conf("retry_total", "BAO_RETRY_TOTAL", 3, int)
    retry_backoff_factor = _get_conf(
        "retry_backoff_factor", "BAO_RETRY_BACKOFF_FACTOR", 0.05, float
    )
    pool_connections = _get_conf("pool_connections", "BAO_POOL_CONNECTIONS", 10, int)
    pool_maxsize = _get_conf("pool_maxsize", "BAO_POOL_MAXSIZE", 20, int)

    client = _OpenBaoClient(
        url=url,
        token=token,
        auth_method=auth_method,
        role_id=role_id,
        secret_id=secret_id,
        secret_id_file=secret_id_file,
        approle_mount=approle_mount,
        token_renew_on_401=token_renew_on_401,
        mount_point=mount_point,
        kv_version=kv_version,
        secret_field=secret_field,
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


__all__ = [
    # Public KBM plugin API
    "kbm_open_client_connection",
    "kbm_close_client_connection",
    "kbm_get_secret",
]
