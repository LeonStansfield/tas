#
# TEE Attestation Service - OpenBao KBM Plugin Unit Tests
#
# Copyright 2026 Hewlett Packard Enterprise Development LP.
# SPDX-License-Identifier: MIT
#

import base64
import json
import os
import unittest
from unittest.mock import MagicMock, patch

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding as asympadding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from plugins.tas_kbm_openbao import (
    _load_config_file,
    kbm_close_client_connection,
    kbm_get_secret,
    kbm_open_client_connection,
)


class TestOpenBaoKBM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate an RSA private key for decrypting and testing wrapped secrets
        cls.rsa_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        cls.rsa_pub_pem = cls.rsa_key.public_key().public_bytes(
            encoding=Encoding.PEM,
            format=PublicFormat.SubjectPublicKeyInfo,
        )

    def test_load_config_file_with_env_substitution(self):
        with patch.dict(os.environ, {"TEST_BAO_TOKEN": "my-secret-token"}):
            # Test inline loading with mock or temp config
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

    @patch("requests.Session.get")
    def test_kbm_get_secret_kv2_success(self, mock_get):
        # Mock OpenBao KV v2 response
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": {
                "data": {"secret": "secret-text"},
                "metadata": {"version": 1},
            }
        }
        mock_get.return_value = mock_resp

        client = kbm_open_client_connection()

        result = kbm_get_secret(client, "test-key-1", self.rsa_pub_pem)

        self.assertIn("wrapped_key", result)
        self.assertIn("blob", result)
        self.assertIn("iv", result)
        self.assertIn("tag", result)

        # Decrypt wrapped AES key with RSA private key
        wrapped_key_bytes = base64.b64decode(result["wrapped_key"])
        aes_key = self.rsa_key.decrypt(
            wrapped_key_bytes,
            asympadding.OAEP(
                mgf=asympadding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

        # Decrypt AES-GCM blob
        blob_bytes = base64.b64decode(result["blob"])
        iv_bytes = base64.b64decode(result["iv"])
        tag_bytes = base64.b64decode(result["tag"])

        cipher = Cipher(algorithms.AES(aes_key), modes.GCM(iv_bytes, tag_bytes))
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(blob_bytes) + decryptor.finalize()

        self.assertEqual(plaintext, b"secret-text")
        kbm_close_client_connection(client)

    @patch("requests.Session.get")
    def test_kbm_get_secret_not_found(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        client = kbm_open_client_connection()

        with self.assertRaises(ValueError):
            kbm_get_secret(client, "nonexistent-key", self.rsa_pub_pem)

        kbm_close_client_connection(client)
