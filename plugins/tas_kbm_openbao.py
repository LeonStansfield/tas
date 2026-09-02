

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
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests

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
        mount_point: str = "secret",
        kv_version: int = 2,
        secret_field: str = "secret",
        verify_ssl: bool = True,
        ca_bundle: Optional[str] = None,
        requests_timeout: int = 30,
    ):
        self.url = url.rstrip("/")
        self.token = token
        self.mount_point = mount_point.strip("/")
        self.kv_version = int(kv_version)
        self.secret_field = secret_field
        self.verify_ssl = verify_ssl
        self.ca_bundle = ca_bundle
        self.requests_timeout = requests_timeout

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

        if self.kv_version == 2:
            endpoint = f"/v1/{self.mount_point}/data/{key_id}"
        else:
            endpoint = f"/v1/{self.mount_point}/{key_id}"

        url = urljoin(f"{self.url}/", endpoint.lstrip("/"))
        try:
            resp = self.session.get(
                url,
                verify=self.verify_param,
                timeout=self.requests_timeout,
            )
        except requests.RequestException as e:
            logger.error(f"Failed to connect to OpenBao at {url}: {e}")
            raise RuntimeError(f"OpenBao connection error: {e}") from e

        if resp.status_code == 404:
            logger.error(f"Secret not found in OpenBao: {key_id}")
            raise ValueError(f"Secret not found: {key_id}")
        elif resp.status_code != 200:
            logger.error(
                f"OpenBao secret retrieval failed ({resp.status_code}): {resp.text}"
            )
            raise RuntimeError(
                f"OpenBao API error ({resp.status_code}): {resp.text}"
            )

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


# Public KBM Plugin Interface


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

    url = (
        cfg.get("url")
        or os.getenv("BAO_ADDR")
        or os.getenv("VAULT_ADDR")
        or "http://127.0.0.1:8200"
    )
    token = cfg.get("token") or os.getenv("BAO_TOKEN") or os.getenv("VAULT_TOKEN")
    mount_point = cfg.get("mount_point", "secret")
    kv_version = cfg.get("kv_version", 2)
    secret_field = cfg.get("secret_field", "secret")
    verify_ssl = cfg.get("verify_ssl", True)
    ca_bundle = cfg.get("ca_bundle")
    requests_timeout = cfg.get("requests_timeout", 30)

    client = _OpenBaoClient(
        url=url,
        token=token,
        mount_point=mount_point,
        kv_version=kv_version,
        secret_field=secret_field,
        verify_ssl=verify_ssl,
        ca_bundle=ca_bundle,
        requests_timeout=requests_timeout,
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
