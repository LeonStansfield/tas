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
import unittest
from unittest.mock import MagicMock, patch

import requests
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding as asympadding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from urllib3.response import HTTPResponse

from plugins.tas_kbm_openbao import (
    _load_config_file,
    _OpenBaoClient,
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
            with patch("os.path.isfile", return_value=True):
                with patch(
                    "builtins.open",
                    unittest.mock.mock_open(
                        read_data="url: http://localhost:8200\ntoken: ${TEST_BAO_TOKEN}\n"
                    ),
                ):
                    cfg = _load_config_file("fake_config.yaml")
                    self.assertEqual(cfg["url"], "http://localhost:8200")
                    self.assertEqual(cfg["token"], "my-secret-token")

    def test_load_config_file_not_found(self):
        self.assertEqual(_load_config_file(None), {})
        self.assertEqual(_load_config_file("/nonexistent/path/config.yaml"), {})

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
        self.assertEqual(adapter.max_retries.total, 4)
        self.assertEqual(adapter.max_retries.backoff_factor, 0.1)
        client.close()

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

    def test_approle_initial_login(self):
        """Verify AppRole authentication succeeds and sets X-Vault-Token header."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"auth": {"client_token": "approle-test-token"}}

        with patch("requests.Session.post", return_value=mock_resp):
            client = _OpenBaoClient(
                auth_method="approle",
                role_id="test-role",
                secret_id="test-secret",
            )
            self.assertEqual(client.token, "approle-test-token")
            self.assertEqual(
                client.session.headers.get("X-Vault-Token"), "approle-test-token"
            )
            client.close()

    def test_reauth_on_401_approle(self):
        """Verify 401 triggers AppRole re-authentication and request retry."""
        mock_login = MagicMock()
        mock_login.status_code = 200
        mock_login.json.return_value = {"auth": {"client_token": "token-v2"}}

        resp_401 = MagicMock()
        resp_401.status_code = 401

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {
            "data": {"data": {"secret": "secret-after-reauth"}}
        }

        with patch("requests.Session.post", return_value=mock_login):
            client = _OpenBaoClient(
                auth_method="approle",
                role_id="test-role",
                secret_id="test-secret",
            )

        # GET secret returns 401 then 200 after re-authenticating
        with (
            patch.object(client.session, "request", side_effect=[resp_401, resp_200]),
            patch.object(
                client, "authenticate", wraps=client.authenticate
            ) as mock_auth,
            patch("requests.Session.post", return_value=mock_login),
        ):
            secret_bytes = client.get_secret("test-key")
            self.assertEqual(secret_bytes, b"secret-after-reauth")
            self.assertEqual(mock_auth.call_count, 1)

        client.close()

    def test_renew_on_401_token(self):
        """Verify 401 triggers token renewal and request retry for renewable tokens."""
        resp_401 = MagicMock()
        resp_401.status_code = 401

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {
            "data": {"data": {"secret": "secret-after-renew"}}
        }

        mock_renew = MagicMock()
        mock_renew.status_code = 200

        client = _OpenBaoClient(
            auth_method="token",
            token="initial-token",
            token_renew_on_401=True,
        )

        with (
            patch.object(client.session, "request", side_effect=[resp_401, resp_200]),
            patch("requests.Session.post", return_value=mock_renew),
        ):
            secret_bytes = client.get_secret("test-key")
            self.assertEqual(secret_bytes, b"secret-after-renew")

        client.close()

    def test_401_failure_raises_runtime_error(self):
        """Verify unrecoverable 401 raises RuntimeError without retry loops."""
        resp_401 = MagicMock()
        resp_401.status_code = 401

        client = _OpenBaoClient(
            auth_method="token",
            token="static-token",
            token_renew_on_401=False,
        )

        with patch.object(client.session, "request", return_value=resp_401):
            with self.assertRaises(RuntimeError):
                client.get_secret("test-key")

        client.close()

    @patch("requests.Session.request")
    def test_kbm_get_secret_kv2_success(self, mock_req):
        # Mock OpenBao KV v2 response
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": {
                "data": {"secret": "secret-text"},
                "metadata": {"version": 1},
            }
        }
        mock_req.return_value = mock_resp

        client = kbm_open_client_connection()

        result = kbm_get_secret(client, "test-key-1", self.rsa_pub_pem)

        for key in ("wrapped_key", "blob", "iv", "tag"):
            self.assertIn(key, result)

        decrypted = self._decrypt_wrapped_secret(result)
        self.assertEqual(decrypted, b"secret-text")
        kbm_close_client_connection(client)

    @patch("requests.Session.request")
    def test_kbm_get_secret_kv2_not_found(self, mock_req):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_req.return_value = mock_resp

        client = kbm_open_client_connection()

        with self.assertRaises(ValueError):
            kbm_get_secret(client, "nonexistent-key", self.rsa_pub_pem)

        kbm_close_client_connection(client)

    def test_init_create_key_if_absent_requires_redis(self):
        """create_key_if_absent=True without redis_client must raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            _OpenBaoClient(create_key_if_absent=True, redis_client=None)
        self.assertIn("Redis client is required", str(ctx.exception))

    def test_init_create_key_if_absent_checks_redis_ping(self):
        """create_key_if_absent=True pings Redis and raises RuntimeError if unhealthy."""
        mock_redis = MagicMock()
        mock_redis.ping.side_effect = Exception("Redis unreachable")

        with self.assertRaises(RuntimeError) as ctx:
            _OpenBaoClient(create_key_if_absent=True, redis_client=mock_redis)
        self.assertIn("Redis client connection failed", str(ctx.exception))

    @patch("plugins.tas_kbm_openbao.redis_lock.Lock")
    def test_get_secret_create_if_absent_kv2_success(self, mock_lock_cls):
        """Verify key creation under lock for KV v2 when key does not exist initially."""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True

        mock_lock = MagicMock()
        mock_lock.__enter__.return_value = mock_lock
        mock_lock.__exit__.return_value = False
        mock_lock_cls.return_value = mock_lock

        client = _OpenBaoClient(
            create_key_if_absent=True,
            redis_client=mock_redis,
            kv_version=2,
            mount_point="secret",
            secret_field="secret",
            requests_timeout=10,
        )

        resp_404 = MagicMock(status_code=404)
        resp_200_create = MagicMock(status_code=200, text="ok")

        # Calls: 1) Initial check -> 404, 2) In-lock double-check -> 404, 3) Write secret -> 200
        with patch.object(
            client, "_make_request", side_effect=[resp_404, resp_404, resp_200_create]
        ) as mock_req:
            secret = client.get_secret("new-key")
            self.assertEqual(len(secret), 32)
            mock_lock_cls.assert_called_once_with(
                mock_redis,
                "tas:openbao:create_key:new-key",
                timeout=70,
                blocking=True,
                blocking_timeout=60,
            )
            self.assertEqual(mock_req.call_count, 3)
            # Verify the write call payload has cas: 0
            create_call_args = mock_req.call_args_list[2]
            self.assertEqual(create_call_args[0], ("POST", "/v1/secret/data/new-key"))
            payload = create_call_args[1]["json"]
            self.assertIn("data", payload)
            self.assertEqual(payload.get("options"), {"cas": 0})

        client.close()

    @patch("plugins.tas_kbm_openbao.redis_lock.Lock")
    def test_get_secret_create_if_absent_double_check_race(self, mock_lock_cls):
        """Verify that if another process created the key during lock wait, stored value is returned."""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True

        mock_lock = MagicMock()
        mock_lock.__enter__.return_value = mock_lock
        mock_lock.__exit__.return_value = False
        mock_lock_cls.return_value = mock_lock

        client = _OpenBaoClient(
            create_key_if_absent=True,
            redis_client=mock_redis,
            kv_version=2,
        )

        resp_404 = MagicMock(status_code=404)
        resp_200_existing = MagicMock(
            status_code=200,
            json=lambda: {"data": {"data": {"secret": "created-by-competitor"}}},
        )

        # Calls: 1) Initial check -> 404, 2) In-lock double-check -> 200 (created by competitor)
        with patch.object(
            client, "_make_request", side_effect=[resp_404, resp_200_existing]
        ) as mock_req:
            secret = client.get_secret("race-key")
            self.assertEqual(secret, b"created-by-competitor")
            self.assertEqual(mock_req.call_count, 2)

        client.close()

    @patch("plugins.tas_kbm_openbao.redis_lock.Lock")
    def test_get_secret_create_cas_conflict_fallback(self, mock_lock_cls):
        """Verify fallback when write fails with CAS conflict (400 check-and-set)."""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True

        mock_lock = MagicMock()
        mock_lock.__enter__.return_value = mock_lock
        mock_lock.__exit__.return_value = False
        mock_lock_cls.return_value = mock_lock

        client = _OpenBaoClient(
            create_key_if_absent=True,
            redis_client=mock_redis,
            kv_version=2,
        )

        resp_404 = MagicMock(status_code=404)
        resp_400_cas = MagicMock(
            status_code=400, text="check-and-set parameter did not match"
        )
        resp_200_existing = MagicMock(
            status_code=200,
            json=lambda: {"data": {"data": {"secret": "winner-secret"}}},
        )

        # Calls: 1) Initial check -> 404, 2) In-lock double-check -> 404, 3) Write -> 400 CAS, 4) Re-fetch -> 200
        with patch.object(
            client,
            "_make_request",
            side_effect=[resp_404, resp_404, resp_400_cas, resp_200_existing],
        ) as mock_req:
            secret = client.get_secret("cas-key")
            self.assertEqual(secret, b"winner-secret")
            self.assertEqual(mock_req.call_count, 4)

        client.close()

    def test_kbm_open_client_connection_with_redis(self):
        """Test kbm_open_client_connection passes redis_client and parses create_key_if_absent."""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True

        with patch(
            "plugins.tas_kbm_openbao._load_config_file",
            return_value={"create_key_if_absent": True},
        ):
            client = kbm_open_client_connection(redis_client=mock_redis)
            self.assertTrue(client.create_key_if_absent)
            self.assertIs(client.redis_client, mock_redis)
            client.close()
