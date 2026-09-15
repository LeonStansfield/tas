#
# TEE Attestation Service - OpenBao KBM Plugin Unit Tests
#
# Copyright 2026 Hewlett Packard Enterprise Development LP.
# SPDX-License-Identifier: MIT
#

import base64
import io
import json
import os
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import requests
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import padding as asympadding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from requests.exceptions import ConnectionError
from urllib3.exceptions import EmptyPoolError
from urllib3.response import HTTPResponse

from plugins.tas_kbm_openbao import (
    _build_url,
    _load_config_file,
    _load_rsa_public_key,
    _OpenBaoClient,
    _parse_bool,
    _validate_config,
    _validate_key_id,
    kbm_close_client_connection,
    kbm_get_secret,
    kbm_open_client_connection,
)


class TestOpenBaoKBM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rsa_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        cls.rsa_pub_pem = cls.rsa_key.public_key().public_bytes(
            encoding=Encoding.PEM,
            format=PublicFormat.SubjectPublicKeyInfo,
        )
        # Provide default valid test environment for tests invoking kbm_open_client_connection()
        cls.env_patcher = patch.dict(os.environ, {"BAO_ADDR": "https://127.0.0.1:8200"})
        cls.env_patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.env_patcher.stop()

    def _decrypt_wrapped_secret(self, wrapped_dict: dict) -> bytes:
        """Helper to unwrap and decrypt the payload using the test RSA private key."""
        enc_aes_key = base64.b64decode(wrapped_dict["wrapped_key"])
        blob = base64.b64decode(wrapped_dict["blob"])
        iv = base64.b64decode(wrapped_dict["iv"])
        tag = base64.b64decode(wrapped_dict["tag"])

        aes_key = self.rsa_key.decrypt(
            enc_aes_key,
            asympadding.OAEP(
                mgf=asympadding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

        cipher = Cipher(algorithms.AES(aes_key), modes.GCM(iv, tag))
        dec = cipher.decryptor()
        return dec.update(blob) + dec.finalize()

    def test_load_config_file_with_env_substitution(self):
        with patch.dict(os.environ, {"TEST_BAO_TOKEN": "my-secret-token"}):
            with (
                patch("os.path.exists", return_value=True),
                patch("os.path.isfile", return_value=True),
                patch("os.access", return_value=True),
                patch(
                    "builtins.open",
                    unittest.mock.mock_open(
                        read_data="url: https://localhost:8200\ntoken: ${TEST_BAO_TOKEN}\n"
                    ),
                ),
            ):
                cfg = _load_config_file("fake_config.yaml")
                self.assertEqual(cfg["url"], "https://localhost:8200")
                self.assertEqual(cfg["token"], "my-secret-token")

    def test_load_config_file_not_found_raises(self):
        """None returns empty dict, but explicit non-existent path raises ValueError."""
        self.assertEqual(_load_config_file(None), {})
        with self.assertRaises(ValueError) as ctx:
            _load_config_file("/nonexistent/path/config.yaml")
        self.assertIn("not found", str(ctx.exception).lower())

    def test_load_config_file_unreadable_raises(self):
        """Explicit unreadable config file raises ValueError."""
        import tempfile

        with tempfile.NamedTemporaryFile("w", delete=False) as tf:
            tf.write("url: https://127.0.0.1:8200\n")
            path = tf.name
        try:
            with patch("os.access", return_value=False):
                with self.assertRaises(ValueError) as ctx:
                    _load_config_file(path)
                self.assertIn("readable", str(ctx.exception).lower())
        finally:
            os.remove(path)

    def test_load_config_file_malformed_yaml_raises(self):
        """Malformed YAML syntax raises ValueError immediately."""
        import tempfile

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as tf:
            tf.write("url: [unclosed yaml list\n")
            path = tf.name
        try:
            with self.assertRaises(ValueError) as ctx:
                _load_config_file(path)
            self.assertIn("yaml", str(ctx.exception).lower())
        finally:
            os.remove(path)

    def test_load_config_file_malformed_json_raises(self):
        """Malformed JSON syntax raises ValueError immediately."""
        import tempfile

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tf:
            tf.write('{"url": "https://localhost:8200", unclosed json')
            path = tf.name
        try:
            with self.assertRaises(ValueError) as ctx:
                _load_config_file(path)
            self.assertIn("json", str(ctx.exception).lower())
        finally:
            os.remove(path)

    def test_load_config_file_not_a_mapping_raises(self):
        """Config file root that does not evaluate to a mapping raises ValueError."""
        import tempfile

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as tf:
            tf.write("- item1\n- item2\n")
            path = tf.name
        try:
            with self.assertRaises(ValueError) as ctx:
                _load_config_file(path)
            self.assertIn("dictionary", str(ctx.exception).lower())
        finally:
            os.remove(path)

    def test_client_connection_pooling_and_retry_config(self):
        client = _OpenBaoClient(
            retry_total=4,
            retry_backoff_factor=0.1,
            pool_connections=15,
            pool_maxsize=25,
        )
        self.assertEqual(client.retry_total, 4)
        self.assertEqual(client.retry_backoff_factor, 0.1)
        self.assertEqual(client.pool_connections, 15)
        self.assertEqual(client.pool_maxsize, 25)

        adapter = client.session.get_adapter("http://localhost:8200")
        self.assertEqual(adapter._pool_connections, 15)
        self.assertEqual(adapter._pool_maxsize, 25)
        self.assertTrue(adapter._pool_block)
        self.assertEqual(adapter.max_retries.total, 4)
        self.assertEqual(adapter.max_retries.backoff_factor, 0.1)

        # Verify urllib3 pool reflects the block=True configuration
        pool = adapter.poolmanager.connection_from_url("http://localhost:8200")
        self.assertTrue(pool.block)
        self.assertEqual(pool.pool.maxsize, 25)
        client.close()

    def test_missing_url_fails_startup(self):
        """Plugin fails startup if URL is omitted from both config and environment."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                kbm_open_client_connection()
            self.assertIn("url is required", str(ctx.exception).lower())

    def test_invalid_url_scheme_fails(self):
        """URL with non-HTTP/HTTPS scheme fails startup."""
        with patch.dict(os.environ, {"BAO_ADDR": "ftp://127.0.0.1:8200"}):
            with self.assertRaises(ValueError) as ctx:
                kbm_open_client_connection()
            self.assertIn("scheme", str(ctx.exception).lower())

    def test_insecure_http_with_verify_ssl_true_fails(self):
        """Insecure http:// scheme fails validation when verify_ssl is True."""
        with patch.dict(
            os.environ, {"BAO_ADDR": "http://127.0.0.1:8200", "BAO_VERIFY_SSL": "true"}
        ):
            with self.assertRaises(ValueError) as ctx:
                kbm_open_client_connection()
            self.assertIn("insecure", str(ctx.exception).lower())

    def test_insecure_http_with_verify_ssl_false_succeeds(self):
        """Insecure http:// scheme is permitted only when verify_ssl is False."""
        with patch.dict(
            os.environ, {"BAO_ADDR": "http://127.0.0.1:8200", "BAO_VERIFY_SSL": "false"}
        ):
            client = kbm_open_client_connection()
            self.assertEqual(client.url, "http://127.0.0.1:8200")
            self.assertFalse(client.verify_ssl)
            kbm_close_client_connection(client)

    def test_https_with_verify_ssl_true_succeeds(self):
        """HTTPS URL with verify_ssl True succeeds."""
        with patch.dict(
            os.environ, {"BAO_ADDR": "https://127.0.0.1:8200", "BAO_VERIFY_SSL": "true"}
        ):
            client = kbm_open_client_connection()
            self.assertEqual(client.url, "https://127.0.0.1:8200")
            self.assertTrue(client.verify_ssl)
            kbm_close_client_connection(client)

    def test_retry_on_server_error_and_success(self):
        """Verify that transient 5xx errors are retried automatically until successful."""
        client = _OpenBaoClient(retry_total=2, retry_backoff_factor=0.01)

        raw_503 = HTTPResponse(
            body=io.BytesIO(b"Service Unavailable"),
            headers={},
            status=503,
            reason="Service Unavailable",
            preload_content=False,
            request_method="GET",
        )
        body_200 = json.dumps({"data": {"data": {"secret": "retry-success"}}}).encode(
            "utf-8"
        )
        raw_200 = HTTPResponse(
            body=io.BytesIO(body_200),
            headers={"Content-Type": "application/json"},
            status=200,
            reason="OK",
            preload_content=False,
            request_method="GET",
        )

        with (
            patch("urllib3.connectionpool.HTTPConnectionPool._get_conn"),
            patch(
                "urllib3.connectionpool.HTTPConnectionPool._make_request",
                side_effect=[raw_503, raw_200],
            ) as mock_make_request,
        ):
            secret_bytes = client.get_secret("test-retry")
            self.assertEqual(secret_bytes, b"retry-success")
            self.assertEqual(mock_make_request.call_count, 2)

        client.close()

    @patch("requests.Session.get")
    def test_kbm_get_secret_success_kv2(self, mock_get):
        """Verify full retrieval and decryption flow for KV v2."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": {"data": {"secret": "super-secret-payload"}}
        }
        mock_get.return_value = mock_resp

        client = kbm_open_client_connection()
        result = kbm_get_secret(client, "my-key-v2", self.rsa_pub_pem)

        for key in ("wrapped_key", "blob", "iv", "tag"):
            self.assertIn(key, result)

        decrypted = self._decrypt_wrapped_secret(result)
        self.assertEqual(decrypted, b"super-secret-payload")
        kbm_close_client_connection(client)

    @patch("requests.Session.get")
    def test_kbm_get_secret_success_kv1(self, mock_get):
        """Verify full retrieval and decryption flow for KV v1."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"secret": "kv1-secret-payload"}}
        mock_get.return_value = mock_resp

        client = _OpenBaoClient(kv_version=1)
        result = kbm_get_secret(client, "my-key-v1", self.rsa_pub_pem)
        decrypted = self._decrypt_wrapped_secret(result)
        self.assertEqual(decrypted, b"kv1-secret-payload")
        client.close()

    @patch("requests.Session.get")
    def test_kbm_get_secret_not_found(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        client = kbm_open_client_connection()
        with self.assertRaises(ValueError):
            kbm_get_secret(client, "nonexistent-key", self.rsa_pub_pem)
        kbm_close_client_connection(client)

    @patch("requests.Session.get")
    def test_kbm_get_secret_server_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_get.return_value = mock_resp

        client = kbm_open_client_connection()
        with self.assertRaises(RuntimeError):
            kbm_get_secret(client, "err-key", self.rsa_pub_pem)
        kbm_close_client_connection(client)

    def test_kbm_get_secret_invalid_arguments(self):
        client = kbm_open_client_connection()

        with self.assertRaises(ValueError):
            kbm_get_secret("not-a-client", "key", self.rsa_pub_pem)

        with self.assertRaises(ValueError):
            kbm_get_secret(client, "", self.rsa_pub_pem)

        with self.assertRaises(ValueError):
            kbm_get_secret(client, "key", b"")

        with self.assertRaises(ValueError):
            kbm_get_secret(client, "key", b"invalid-pem-data")

        kbm_close_client_connection(client)

    def test_shipped_config_file_loading(self):
        """Verify that the shipped openbao.yaml configuration parses correctly and sets requests_timeout."""
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "openbao", "openbao.yaml"
        )
        self.assertTrue(
            os.path.isfile(config_path), f"Config file not found at {config_path}"
        )

        cfg = _load_config_file(config_path)
        self.assertIn(
            "requests_timeout",
            cfg,
            "requests_timeout key is missing or misspelled in config",
        )
        self.assertEqual(cfg["requests_timeout"], 30)
        self.assertIn("mount_point", cfg)
        self.assertIn("kv_version", cfg)
        self.assertEqual(cfg["url"], "https://127.0.0.1:8200")

    def test_token_file_reading(self):
        """Verify token is correctly read from token_file when env token is not set."""
        import tempfile

        with tempfile.NamedTemporaryFile("w", delete=False) as tf:
            tf.write("secret-from-token-file\n")
            token_file_path = tf.name

        cfg_content = f"url: https://localhost:8200\ntoken_file: {token_file_path}\n"
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as cf:
            cf.write(cfg_content)
            config_yaml_path = cf.name

        try:
            with patch.dict(os.environ, {}, clear=True):
                client = kbm_open_client_connection(config_yaml_path)
                self.assertEqual(client.token, "secret-from-token-file")
                kbm_close_client_connection(client)
        finally:
            os.remove(token_file_path)
            os.remove(config_yaml_path)

    def test_validate_key_id_positive(self):
        """Verify valid key_id patterns pass validation without error."""
        valid_keys = [
            "customer/key1",
            "prod/app/secret",
            "customer/app key",
            "my-key-v2",
        ]
        for key in valid_keys:
            with self.subTest(key=key):
                # Should not raise
                _validate_key_id(key)

    def test_validate_key_id_negative(self):
        """Verify invalid key_id values, traversal attempts, and special characters raise ValueError."""
        invalid_keys = [
            "..",
            "../secret",
            "secret/../admin",
            "abc?test=1",
            "abc#fragment",
            "abc\\test",
            "%2e%2e",
            "%2e%2e%2fadmin",
            "%252e%252e",
            "%252e%252e%252fadmin",
            "%252f",
            "%252F",
            "%255c",
            "%255C",
            "",
            "   ",
            "/leading/slash",
            "trailing/slash/",
            "double//slash",
            "secret\x00null",
            "secret\nnewline",
        ]
        for key in invalid_keys:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    _validate_key_id(key)

    def test_build_url_encoding_and_scope(self):
        """Verify URL encoding of path segments and scope verification."""
        # KV v2 encoding
        url_v2 = _build_url(
            base_url="http://127.0.0.1:8200",
            mount_point="secret",
            kv_version=2,
            key_id="customer/app key",
        )
        self.assertEqual(
            url_v2,
            "http://127.0.0.1:8200/v1/secret/data/customer/app%20key",
        )

        # KV v1 encoding
        url_v1 = _build_url(
            base_url="http://127.0.0.1:8200",
            mount_point="secret",
            kv_version=1,
            key_id="customer/app key",
        )
        self.assertEqual(
            url_v1,
            "http://127.0.0.1:8200/v1/secret/customer/app%20key",
        )

    def test_parse_bool_valid(self):
        """Verify valid truthy and falsy boolean inputs."""
        self.assertTrue(_parse_bool(True, "test"))
        self.assertTrue(_parse_bool("true", "test"))
        self.assertTrue(_parse_bool("True", "test"))
        self.assertTrue(_parse_bool("1", "test"))
        self.assertTrue(_parse_bool(1, "test"))

        self.assertFalse(_parse_bool(False, "test"))
        self.assertFalse(_parse_bool("false", "test"))
        self.assertFalse(_parse_bool("FALSE", "test"))
        self.assertFalse(_parse_bool("0", "test"))
        self.assertFalse(_parse_bool(0, "test"))

    def test_parse_bool_invalid(self):
        """Verify invalid boolean values raise ValueError immediately."""
        invalid_values = ["yes", "no", "enabled", "on", "off", None, 2, -1, "", []]
        for val in invalid_values:
            with self.subTest(val=val):
                with self.assertRaises(ValueError):
                    _parse_bool(val, "verify_ssl")

    def test_verify_ssl_invalid_fails_startup(self):
        """Verify startup fails immediately on invalid verify_ssl configuration."""
        with patch.dict(os.environ, {"BAO_VERIFY_SSL": "invalid_val"}):
            with self.assertRaises(ValueError):
                kbm_open_client_connection()

    def test_ca_bundle_nonexistent_fails(self):
        """Configuring a non-existent CA bundle raises ValueError."""
        with patch.dict(
            os.environ,
            {
                "BAO_ADDR": "https://127.0.0.1:8200",
                "BAO_CA_BUNDLE": "/path/to/missing-ca.pem",
            },
        ):
            with self.assertRaises(ValueError) as ctx:
                kbm_open_client_connection()
            self.assertIn("ca_bundle", str(ctx.exception).lower())

    def test_ca_bundle_valid_file_succeeds(self):
        """Configuring an existing readable CA bundle passes validation."""
        import tempfile

        with tempfile.NamedTemporaryFile("w", delete=False) as tf:
            tf.write("-----BEGIN CERTIFICATE-----\n...\n-----END CERTIFICATE-----\n")
            ca_path = tf.name
        try:
            with patch.dict(
                os.environ,
                {
                    "BAO_ADDR": "https://127.0.0.1:8200",
                    "BAO_CA_BUNDLE": ca_path,
                },
            ):
                client = kbm_open_client_connection()
                self.assertEqual(client.ca_bundle, ca_path)
                self.assertEqual(client.verify_param, ca_path)
                kbm_close_client_connection(client)
        finally:
            os.remove(ca_path)

    def test_invalid_kv_version_fails(self):
        """kv_version values other than 1 or 2 raise ValueError."""
        invalid_versions = [0, 3, -1, 999, "invalid"]
        for ver in invalid_versions:
            with self.subTest(ver=ver):
                with patch.dict(
                    os.environ,
                    {
                        "BAO_ADDR": "https://127.0.0.1:8200",
                        "BAO_KV_VERSION": str(ver),
                    },
                ):
                    with self.assertRaises(ValueError) as ctx:
                        kbm_open_client_connection()
                    self.assertIn("kv_version", str(ctx.exception).lower())

    def test_valid_kv_version_succeeds(self):
        """kv_version 1 and 2 pass validation."""
        for ver in (1, 2):
            with self.subTest(ver=ver):
                with patch.dict(
                    os.environ,
                    {
                        "BAO_ADDR": "https://127.0.0.1:8200",
                        "BAO_KV_VERSION": str(ver),
                    },
                ):
                    client = kbm_open_client_connection()
                    self.assertEqual(client.kv_version, ver)
                    kbm_close_client_connection(client)

    def test_invalid_requests_timeout_fails(self):
        """requests_timeout <= 0 or non-numeric raises ValueError."""
        invalid_timeouts = [0, -1, -30, "not-a-number"]
        for timeout in invalid_timeouts:
            with self.subTest(timeout=timeout):
                with patch.dict(
                    os.environ,
                    {
                        "BAO_ADDR": "https://127.0.0.1:8200",
                        "BAO_REQUESTS_TIMEOUT": str(timeout),
                    },
                ):
                    with self.assertRaises(ValueError) as ctx:
                        kbm_open_client_connection()
                    self.assertIn("requests_timeout", str(ctx.exception).lower())

    def test_invalid_pool_connections_fails(self):
        """pool_connections <= 0 or invalid type raises ValueError."""
        for val in [0, -5, "abc"]:
            with self.subTest(val=val):
                with patch.dict(
                    os.environ,
                    {
                        "BAO_ADDR": "https://127.0.0.1:8200",
                        "BAO_POOL_CONNECTIONS": str(val),
                    },
                ):
                    with self.assertRaises(ValueError) as ctx:
                        kbm_open_client_connection()
                    self.assertIn("pool_connections", str(ctx.exception).lower())

    def test_invalid_pool_maxsize_fails(self):
        """pool_maxsize <= 0 or invalid type raises ValueError."""
        for val in [0, -10, "xyz"]:
            with self.subTest(val=val):
                with patch.dict(
                    os.environ,
                    {
                        "BAO_ADDR": "https://127.0.0.1:8200",
                        "BAO_POOL_MAXSIZE": str(val),
                    },
                ):
                    with self.assertRaises(ValueError) as ctx:
                        kbm_open_client_connection()
                    self.assertIn("pool_maxsize", str(ctx.exception).lower())

    def test_invalid_retry_total_fails(self):
        """retry_total < 0 or invalid type raises ValueError."""
        for val in [-1, -5, "bad"]:
            with self.subTest(val=val):
                with patch.dict(
                    os.environ,
                    {
                        "BAO_ADDR": "https://127.0.0.1:8200",
                        "BAO_RETRY_TOTAL": str(val),
                    },
                ):
                    with self.assertRaises(ValueError) as ctx:
                        kbm_open_client_connection()
                    self.assertIn("retry_total", str(ctx.exception).lower())

    def test_invalid_retry_backoff_factor_fails(self):
        """retry_backoff_factor < 0.0 or invalid type raises ValueError."""
        for val in [-0.1, -1.0, "bad"]:
            with self.subTest(val=val):
                with patch.dict(
                    os.environ,
                    {
                        "BAO_ADDR": "https://127.0.0.1:8200",
                        "BAO_RETRY_BACKOFF_FACTOR": str(val),
                    },
                ):
                    with self.assertRaises(ValueError) as ctx:
                        kbm_open_client_connection()
                    self.assertIn("retry_backoff_factor", str(ctx.exception).lower())

    def test_validate_config_helper_all_valid(self):
        """_validate_config passes cleanly with valid parameters."""
        # Should not raise
        _validate_config(
            url="https://bao.internal:8200",
            verify_ssl=True,
            ca_bundle=None,
            kv_version=2,
            requests_timeout=30,
            pool_connections=10,
            pool_maxsize=20,
            retry_total=3,
            retry_backoff_factor=0.05,
        )

    def test_validate_config_helper_catches_zero_timeout(self):
        """_validate_config raises ValueError if requests_timeout is zero."""
        with self.assertRaises(ValueError):
            _validate_config(
                url="https://bao.internal:8200",
                verify_ssl=True,
                ca_bundle=None,
                kv_version=2,
                requests_timeout=0,
                pool_connections=10,
                pool_maxsize=20,
                retry_total=3,
                retry_backoff_factor=0.05,
            )

    def test_rsa_public_key_succeeds(self):
        """1. Verify valid RSA public key succeeds and returns RSAPublicKey instance."""
        pub_key = _load_rsa_public_key(self.rsa_pub_pem)
        self.assertIsInstance(pub_key, rsa.RSAPublicKey)

    def test_non_rsa_public_key_raises(self):
        """2. Verify non-RSA (EC) public key is rejected with ValueError."""
        ec_key = ec.generate_private_key(ec.SECP256R1())
        ec_pub_pem = ec_key.public_key().public_bytes(
            encoding=Encoding.PEM,
            format=PublicFormat.SubjectPublicKeyInfo,
        )

        with self.assertRaises(ValueError) as ctx:
            _load_rsa_public_key(ec_pub_pem)
        self.assertIn("rsa public key", str(ctx.exception).lower())

        client = kbm_open_client_connection()
        try:
            with self.assertRaises(ValueError) as ctx:
                kbm_get_secret(client, "my-key", ec_pub_pem)
            self.assertIn("rsa public key", str(ctx.exception).lower())
        finally:
            kbm_close_client_connection(client)

    @patch("requests.Session.get")
    def test_missing_secret_field_raises(self, mock_get):
        """3. Verify missing secret_field raises ValueError without returning sibling data."""
        # KV v2 with single sibling field that should NOT be returned
        mock_resp_v2 = MagicMock()
        mock_resp_v2.status_code = 200
        mock_resp_v2.json.return_value = {
            "data": {"data": {"sibling_secret": "sibling-value"}}
        }
        mock_get.return_value = mock_resp_v2

        client = kbm_open_client_connection()
        try:
            with self.assertRaises(ValueError) as ctx:
                kbm_get_secret(client, "key-v2", self.rsa_pub_pem)
            self.assertIn("not found", str(ctx.exception).lower())
        finally:
            kbm_close_client_connection(client)

        # KV v1 with multi-field data missing configured secret_field
        mock_resp_v1 = MagicMock()
        mock_resp_v1.status_code = 200
        mock_resp_v1.json.return_value = {
            "data": {"field_a": "val_a", "field_b": "val_b"}
        }
        mock_get.return_value = mock_resp_v1

        client_v1 = _OpenBaoClient(kv_version=1, secret_field="target_key")
        try:
            with self.assertRaises(ValueError) as ctx:
                client_v1.get_secret("key-v1")
            self.assertIn("target_key", str(ctx.exception))
        finally:
            client_v1.close()

    @patch("requests.Session.get")
    def test_null_response_data_raises(self, mock_get):
        """4. Verify null response data raises ValueError."""
        # KV v2 outer data null
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": None}
        mock_get.return_value = mock_resp

        client = kbm_open_client_connection()
        try:
            with self.assertRaises(ValueError) as ctx:
                client.get_secret("key-null-v2-outer")
            self.assertIn("null", str(ctx.exception).lower())

            # KV v2 inner secret data null
            mock_resp.json.return_value = {"data": {"data": None}}
            with self.assertRaises(ValueError) as ctx:
                client.get_secret("key-null-v2-inner")
            self.assertIn("null", str(ctx.exception).lower())
        finally:
            kbm_close_client_connection(client)

        # KV v1 data null
        mock_resp.json.return_value = {"data": None}
        client_v1 = _OpenBaoClient(kv_version=1)
        try:
            with self.assertRaises(ValueError) as ctx:
                client_v1.get_secret("key-null-v1")
            self.assertIn("null", str(ctx.exception).lower())
        finally:
            client_v1.close()

    @patch("requests.Session.get")
    def test_invalid_response_structure_raises(self, mock_get):
        """5. Verify malformed payloads and invalid response structures raise ValueError."""
        client = kbm_open_client_connection()
        try:
            # Malformed JSON in response
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.side_effect = ValueError("Expecting value: line 1 column 1")
            mock_get.return_value = mock_resp
            with self.assertRaises(ValueError) as ctx:
                client.get_secret("key-malformed-json")
            self.assertIn("malformed", str(ctx.exception).lower())

            # Top-level is not a dictionary (e.g. JSON list)
            mock_resp.json.side_effect = None
            mock_resp.json.return_value = ["not", "a", "dict"]
            with self.assertRaises(ValueError) as ctx:
                client.get_secret("key-list-payload")
            self.assertIn("structure", str(ctx.exception).lower())

            # KV v2 outer data is not a dictionary
            mock_resp.json.return_value = {"data": "not-a-dict"}
            with self.assertRaises(ValueError) as ctx:
                client.get_secret("key-invalid-v2-outer")
            self.assertIn("structure", str(ctx.exception).lower())

            # KV v2 inner secret data is not a dictionary
            mock_resp.json.return_value = {"data": {"data": "not-a-dict"}}
            with self.assertRaises(ValueError) as ctx:
                client.get_secret("key-invalid-v2-inner")
            self.assertIn("structure", str(ctx.exception).lower())

            # KV v1 secret data is not a dictionary
            client_v1 = _OpenBaoClient(kv_version=1)
            mock_resp.json.return_value = {"data": [1, 2, 3]}
            try:
                with self.assertRaises(ValueError) as ctx:
                    client_v1.get_secret("key-invalid-v1")
                self.assertIn("structure", str(ctx.exception).lower())
            finally:
                client_v1.close()
        finally:
            kbm_close_client_connection(client)

    def test_connection_pool_bounded_concurrency(self):
        """Demonstrate that pool_block=True bounds concurrent checkouts to pool_maxsize."""
        pool_maxsize = 2
        num_workers = 6
        client = _OpenBaoClient(
            url="http://127.0.0.1:8200",
            pool_connections=1,
            pool_maxsize=pool_maxsize,
            retry_total=0,
        )

        adapter = client.session.get_adapter("http://127.0.0.1:8200")
        pool = adapter.poolmanager.connection_from_url("http://127.0.0.1:8200")

        active_conns = 0
        peak_active_conns = 0
        lock = threading.Lock()
        orig_get_conn = pool._get_conn
        orig_put_conn = pool._put_conn

        def slow_get_conn(timeout=None):
            nonlocal active_conns, peak_active_conns
            conn = orig_get_conn(timeout=timeout)
            with lock:
                active_conns += 1
                if active_conns > peak_active_conns:
                    peak_active_conns = active_conns
            return conn

        def slow_put_conn(conn):
            nonlocal active_conns
            time.sleep(0.03)
            with lock:
                active_conns -= 1
            orig_put_conn(conn)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": {"data": {"secret": "concurrent-secret"}}
        }

        with (
            patch.object(pool, "_get_conn", side_effect=slow_get_conn),
            patch.object(pool, "_put_conn", side_effect=slow_put_conn),
            patch.object(client.session, "send", return_value=mock_resp),
        ):
            results = []
            threads = []

            def worker():
                try:
                    res = client.get_secret("test-concurrency")
                    results.append(res)
                except Exception as e:
                    results.append(e)

            for _ in range(num_workers):
                t = threading.Thread(target=worker)
                threads.append(t)
                t.start()

            for t in threads:
                t.join(timeout=5)

            self.assertEqual(len(results), num_workers)
            for res in results:
                self.assertEqual(res, b"concurrent-secret")

            # With pool_block=True, simultaneous checkouts cannot exceed pool_maxsize
            self.assertLessEqual(peak_active_conns, pool_maxsize)

        client.close()

    def test_connection_pool_blocks_and_exhausts_on_timeout(self):
        """Demonstrate that pool_block=True causes callers to block when exhausted and raise EmptyPoolError."""
        client = _OpenBaoClient(
            url="http://127.0.0.1:8200",
            pool_connections=1,
            pool_maxsize=1,
            requests_timeout=0.1,
            retry_total=0,
        )

        adapter = client.session.get_adapter("http://127.0.0.1:8200")
        pool = adapter.poolmanager.connection_from_url("http://127.0.0.1:8200")
        err = EmptyPoolError(pool, "Pool is full and request timed out")
        req_err = ConnectionError(err)

        with patch.object(client.session, "send", side_effect=req_err):
            with self.assertRaises(RuntimeError) as ctx:
                client.get_secret("test-exhaustion")

            self.assertIn("OpenBao connection error", str(ctx.exception))
            cause = ctx.exception.__cause__
            self.assertIsInstance(cause, ConnectionError)
            self.assertIn("Pool is full", str(cause))

        client.close()

    def test_connection_pool_semaphore_exhaustion_timeout(self):
        """Demonstrate that semaphore acquisition bounds pool exhaustion wait time."""
        client = _OpenBaoClient(
            url="http://127.0.0.1:8200",
            pool_connections=1,
            pool_maxsize=1,
            requests_timeout=0.05,
            retry_total=0,
        )

        # Acquire the single semaphore permit so any call to get_secret must block
        self.assertTrue(client._pool_semaphore.acquire(blocking=False))
        try:
            with self.assertRaises(RuntimeError) as ctx:
                client.get_secret("test-semaphore-exhaustion")
            self.assertIn("timed out after 0.05s", str(ctx.exception))
            self.assertIn("connection pool exhausted", str(ctx.exception))
        finally:
            client._pool_semaphore.release()
            client.close()

    def test_rsa_public_key_under_2048_bits_raises(self):
        """Verify RSA public keys smaller than 2048 bits are rejected with ValueError."""
        weak_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=1024,
        )
        weak_pub_pem = weak_key.public_key().public_bytes(
            encoding=Encoding.PEM,
            format=PublicFormat.SubjectPublicKeyInfo,
        )

        with self.assertRaises(ValueError) as ctx:
            _load_rsa_public_key(weak_pub_pem)
        self.assertIn(
            "minimum required rsa key size is 2048 bits", str(ctx.exception).lower()
        )

        client = kbm_open_client_connection()
        try:
            with self.assertRaises(ValueError) as ctx:
                kbm_get_secret(client, "my-key", weak_pub_pem)
            self.assertIn(
                "minimum required rsa key size is 2048 bits", str(ctx.exception).lower()
            )
        finally:
            kbm_close_client_connection(client)

    @patch("requests.Session.post")
    def test_approle_login_success(self, mock_post):
        """Verify successful AppRole login obtains and configures client token."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "auth": {
                "client_token": "approle-test-token-12345",
                "policies": ["default"],
            }
        }
        mock_post.return_value = mock_resp

        client = _OpenBaoClient(
            auth_method="approle",
            role_id="my-role-id",
            secret_id="my-secret-id",
            approle_mount="approle",
        )

        self.assertEqual(client.token, "approle-test-token-12345")
        self.assertEqual(
            client.session.headers.get("X-Vault-Token"),
            "approle-test-token-12345",
        )
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertTrue(args[0].endswith("/v1/auth/approle/login"))
        self.assertEqual(
            kwargs["json"],
            {"role_id": "my-role-id", "secret_id": "my-secret-id"},
        )
        client.close()

    @patch("requests.Session.post")
    def test_approle_login_failure_raises(self, mock_post):
        """Verify failed AppRole login raises RuntimeError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "invalid role or secret ID"
        mock_post.return_value = mock_resp

        with self.assertRaises(RuntimeError) as ctx:
            _OpenBaoClient(
                auth_method="approle",
                role_id="bad-role",
                secret_id="bad-secret",
            )
        self.assertIn("AppRole login failed", str(ctx.exception))

    def test_approle_missing_role_id_raises(self):
        """Verify missing role_id raises ValueError when auth_method=approle."""
        with self.assertRaises(ValueError) as ctx:
            _OpenBaoClient(
                auth_method="approle",
                role_id=None,
                secret_id="some-secret",
            )
        self.assertIn("role_id is required", str(ctx.exception))

    def test_approle_missing_secret_id_raises(self):
        """Verify missing secret_id and secret_id_file raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            _OpenBaoClient(
                auth_method="approle",
                role_id="some-role",
                secret_id=None,
                secret_id_file=None,
            )
        self.assertIn("Either secret_id or secret_id_file", str(ctx.exception))

    @patch("requests.Session.post")
    def test_approle_secret_id_file_reading(self, mock_post):
        """Verify secret_id is read from secret_id_file."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"auth": {"client_token": "token-from-file"}}
        mock_post.return_value = mock_resp

        import tempfile

        with tempfile.NamedTemporaryFile("w", delete=False) as tf:
            tf.write("file-injected-secret-id\n")
            secret_file_path = tf.name

        try:
            client = _OpenBaoClient(
                auth_method="approle",
                role_id="my-role",
                secret_id_file=secret_file_path,
            )
            self.assertEqual(client.token, "token-from-file")
            _, kwargs = mock_post.call_args
            self.assertEqual(kwargs["json"]["secret_id"], "file-injected-secret-id")
            client.close()
        finally:
            os.remove(secret_file_path)

    @patch("requests.Session.post")
    def test_approle_secret_id_file_precedence_over_secret_id(self, mock_post):
        """Verify secret_id_file takes precedence over inline secret_id."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"auth": {"client_token": "token-precedence"}}
        mock_post.return_value = mock_resp

        import tempfile

        with tempfile.NamedTemporaryFile("w", delete=False) as tf:
            tf.write("priority-secret-id\n")
            secret_file_path = tf.name

        try:
            client = _OpenBaoClient(
                auth_method="approle",
                role_id="my-role",
                secret_id="fallback-secret-id",
                secret_id_file=secret_file_path,
            )
            _, kwargs = mock_post.call_args
            self.assertEqual(kwargs["json"]["secret_id"], "priority-secret-id")
            client.close()
        finally:
            os.remove(secret_file_path)

    def test_approle_empty_secret_id_file_raises(self):
        """Verify empty secret_id_file raises ValueError."""
        import tempfile

        with tempfile.NamedTemporaryFile("w", delete=False) as tf:
            tf.write("   \n")
            secret_file_path = tf.name

        try:
            with self.assertRaises(ValueError) as ctx:
                _OpenBaoClient(
                    auth_method="approle",
                    role_id="my-role",
                    secret_id_file=secret_file_path,
                )
            self.assertIn("empty", str(ctx.exception).lower())
        finally:
            os.remove(secret_file_path)

    def test_approle_missing_secret_id_file_raises(self):
        """Verify non-existent secret_id_file raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            _OpenBaoClient(
                auth_method="approle",
                role_id="my-role",
                secret_id_file="/nonexistent/secret_id_file.txt",
            )
        self.assertIn("not found", str(ctx.exception).lower())

    @patch("requests.Session.post")
    @patch("requests.Session.get")
    def test_approle_401_triggers_reauthentication_and_retry(self, mock_get, mock_post):
        """Verify HTTP 401 triggers reauthentication, updates token, and retries the request."""
        # Initial login
        login_resp = MagicMock()
        login_resp.status_code = 200
        login_resp.json.return_value = {"auth": {"client_token": "initial-token"}}

        # Renew login after 401
        renew_resp = MagicMock()
        renew_resp.status_code = 200
        renew_resp.json.return_value = {"auth": {"client_token": "renewed-token-456"}}
        mock_post.side_effect = [login_resp, renew_resp]

        # First request returns 401; retried request returns 200
        req_401 = MagicMock()
        req_401.status_code = 401
        req_401.text = "permission denied: token expired"

        req_200 = MagicMock()
        req_200.status_code = 200
        req_200.json.return_value = {
            "data": {"data": {"secret": "secret-after-renewal"}}
        }
        mock_get.side_effect = [req_401, req_200]

        client = _OpenBaoClient(
            auth_method="approle",
            role_id="my-role",
            secret_id="my-secret",
            token_renew_on_401=True,
        )
        self.assertEqual(client.token, "initial-token")

        secret = client.get_secret("my-key")
        self.assertEqual(secret, b"secret-after-renewal")
        self.assertEqual(client.token, "renewed-token-456")
        self.assertEqual(
            client.session.headers.get("X-Vault-Token"), "renewed-token-456"
        )
        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(mock_post.call_count, 2)
        client.close()

    @patch("requests.Session.post")
    @patch("requests.Session.get")
    def test_approle_401_retry_failure_raises(self, mock_get, mock_post):
        """Verify that when a retried request still returns 401, RuntimeError is raised."""
        login_resp = MagicMock()
        login_resp.status_code = 200
        login_resp.json.return_value = {"auth": {"client_token": "initial-token"}}
        mock_post.return_value = login_resp

        req_401 = MagicMock()
        req_401.status_code = 401
        req_401.text = "still unauthorized"
        mock_get.side_effect = [req_401, req_401]

        client = _OpenBaoClient(
            auth_method="approle",
            role_id="my-role",
            secret_id="my-secret",
            token_renew_on_401=True,
        )

        with self.assertRaises(RuntimeError) as ctx:
            client.get_secret("my-key")
        self.assertIn("reauthentication failed", str(ctx.exception).lower())
        client.close()

    @patch("requests.Session.post")
    @patch("requests.Session.get")
    def test_approle_token_renew_on_401_false(self, mock_get, mock_post):
        """Verify token_renew_on_401=False does not attempt reauthentication."""
        login_resp = MagicMock()
        login_resp.status_code = 200
        login_resp.json.return_value = {"auth": {"client_token": "initial-token"}}
        mock_post.return_value = login_resp

        req_401 = MagicMock()
        req_401.status_code = 401
        req_401.text = "unauthorized"
        mock_get.return_value = req_401

        client = _OpenBaoClient(
            auth_method="approle",
            role_id="my-role",
            secret_id="my-secret",
            token_renew_on_401=False,
        )

        with self.assertRaises(RuntimeError) as ctx:
            client.get_secret("my-key")
        self.assertIn("OpenBao API error (401)", str(ctx.exception))
        # No reauthentication POST was performed beyond startup
        self.assertEqual(mock_post.call_count, 1)
        client.close()

    @patch("requests.Session.post")
    @patch("requests.Session.get")
    def test_approle_concurrent_401_single_reauth(self, mock_get, mock_post):
        """Verify multiple simultaneous 401 responses trigger only a single reauthentication."""
        login_count = 0
        lock = threading.Lock()

        def login_handler(*args, **kwargs):
            nonlocal login_count
            with lock:
                login_count += 1
                curr = login_count
            time.sleep(0.02)
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"auth": {"client_token": f"token-v{curr}"}}
            return resp

        mock_post.side_effect = login_handler

        def get_handler(*args, **kwargs):
            # Check token on incoming request
            current_token = client.session.headers.get("X-Vault-Token")
            resp = MagicMock()
            if current_token == "token-v1":
                # Simulated expired token returns 401
                resp.status_code = 401
                resp.text = "token expired"
            else:
                # Refreshed token returns 200
                resp.status_code = 200
                resp.json.return_value = {
                    "data": {"data": {"secret": "concurrent-approle-secret"}}
                }
            return resp

        mock_get.side_effect = get_handler

        client = _OpenBaoClient(
            auth_method="approle",
            role_id="my-role",
            secret_id="my-secret",
            token_renew_on_401=True,
            pool_maxsize=10,
        )
        self.assertEqual(login_count, 1)  # Initial login

        num_threads = 6
        threads = []
        results = []

        def worker():
            try:
                res = client.get_secret("test-concurrency")
                results.append(res)
            except Exception as e:
                results.append(e)

        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=5)

        self.assertEqual(len(results), num_threads)
        for r in results:
            self.assertEqual(r, b"concurrent-approle-secret")

        # Total logins: 1 at startup + 1 on 401 surge = 2
        self.assertEqual(login_count, 2)
        client.close()

    @patch("requests.Session.post")
    def test_kbm_open_client_connection_approle_env(self, mock_post):
        """Verify kbm_open_client_connection initializes AppRole mode from environment variables."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "auth": {"client_token": "token-from-env-approle"}
        }
        mock_post.return_value = mock_resp

        env_vars = {
            "BAO_ADDR": "https://127.0.0.1:8200",
            "BAO_AUTH_METHOD": "approle",
            "BAO_ROLE_ID": "env-role-123",
            "BAO_SECRET_ID": "env-secret-456",
            "BAO_APPROLE_MOUNT": "custom-approle",
        }
        with patch.dict(os.environ, env_vars):
            client = kbm_open_client_connection()
            self.assertEqual(client.auth_method, "approle")
            self.assertEqual(client.role_id, "env-role-123")
            self.assertEqual(client.secret_id, "env-secret-456")
            self.assertEqual(client.approle_mount, "custom-approle")
            self.assertEqual(client.token, "token-from-env-approle")
            kbm_close_client_connection(client)
