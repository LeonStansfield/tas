#
# TEE Attestation Service - Client route KBM error handling tests
#
# Copyright 2026 Hewlett Packard Enterprise Development LP.
# SPDX-License-Identifier: MIT
#

import base64
from unittest.mock import MagicMock, patch

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from flask import Flask

from tas.client_routes import client_bp
from tas.error_handlers import register_error_handlers
from tas.exceptions import KBMResponseError, KBMUnavailableError


@pytest.fixture()
def app():
    test_app = Flask(__name__)
    test_app.config["TESTING"] = True
    test_app.extensions["redis"] = MagicMock(name="redis")
    test_app.extensions["kbm_client"] = MagicMock(name="kbm_client")
    test_app.extensions["kbm_get_secret"] = MagicMock(name="kbm_get_secret")
    test_app.register_blueprint(client_bp)
    register_error_handlers(test_app)
    return test_app


def _request_body():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return {
        "tee-type": "amd-sev-snp",
        "nonce": "nonce-value",
        "tee-evidence": "evidence",
        "policy-id": "policy:test",
        "report-data-binding": True,
        "wrapping-key": base64.b64encode(public_key).decode("ascii"),
    }


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def patched_route_dependencies():
    with (
        patch("tas.client_routes.authenticate_request", return_value=None),
        patch("tas.client_routes.validate_nonce", return_value=(True, None)),
        patch("tas.client_routes.vm_verify", return_value=(True, "key-id", None)),
    ):
        yield


def _post_secret(client):
    return client.post("/kb/v0/get_secret", json=_request_body())


def _assert_no_internal_details(response, details):
    response_text = response.get_data(as_text=True)
    for detail in details:
        assert detail not in response_text


def test_unavailable_exception_returns_503(app, client, patched_route_dependencies):
    app.extensions["kbm_get_secret"].side_effect = KBMUnavailableError(retry_after=17)

    response = _post_secret(client)

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "17"
    assert response.get_json() == {"error": "Secret service is temporarily unavailable"}
    _assert_no_internal_details(
        response, ["http://", "key-id", "KBMUnavailableError", "connection"]
    )


def test_response_exception_returns_502(app, client, patched_route_dependencies):
    app.extensions["kbm_get_secret"].side_effect = KBMResponseError()

    response = _post_secret(client)

    assert response.status_code == 502
    assert response.get_json() == {
        "error": "Secret service returned an invalid response"
    }
    assert "Retry-After" not in response.headers
    _assert_no_internal_details(
        response,
        [
            "http://",
            "key-id",
            "KBMResponseError",
            "response body",
            "parser",
            "protocol",
        ],
    )


@pytest.mark.parametrize(
    "exception",
    [
        RuntimeError("internal implementation detail"),
        RuntimeError("legacy plugin failure"),
    ],
)
def test_unrelated_exceptions_return_generic_500(
    app, client, patched_route_dependencies, exception
):
    app.extensions["kbm_get_secret"].side_effect = exception

    response = _post_secret(client)

    assert response.status_code == 500
    assert response.get_json() == {"error": "Internal server error"}
    _assert_no_internal_details(
        response, [str(exception), exception.__class__.__name__]
    )


def test_value_error_remains_404(app, client, patched_route_dependencies):
    app.extensions["kbm_get_secret"].side_effect = ValueError("Secret not found")

    response = _post_secret(client)

    assert response.status_code == 404
    assert response.get_json() == {"error": "Secret retrieval failed"}
    _assert_no_internal_details(response, ["Secret not found"])
