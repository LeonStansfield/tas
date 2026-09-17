#
# TEE Attestation Service - Shared KBM Exceptions
#
# Copyright 2026 Hewlett Packard Enterprise Development LP.
# SPDX-License-Identifier: MIT
#
# This file is part of the TEE Attestation Service.


class KBMUnavailableError(RuntimeError):
    """Raised when the KBM backend is temporarily unavailable."""

    public_message = "Secret service is temporarily unavailable"

    def __init__(self, retry_after=1):
        super().__init__(self.public_message)
        self.message = self.public_message
        self.retry_after = retry_after


class KBMResponseError(RuntimeError):
    """Raised when the KBM backend returns an invalid response."""

    public_message = "Secret service returned an invalid response"

    def __init__(self):
        super().__init__(self.public_message)
