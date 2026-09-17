# TAS developer guide

## KBM plugin error handling

KBM plugins should raise the shared `tas.exceptions.KBMUnavailableError` when
the backend is temporarily unavailable, and `tas.exceptions.KBMResponseError`
when the backend returns a malformed or invalid response. The API route owns
HTTP translation: unavailable backends become HTTP 503 responses with a
`Retry-After` header, and invalid backend responses become HTTP 502 responses.
Plugins must keep diagnostic details such as key identifiers, timeout values,
and pool sizes in server logs rather than including them in client responses.

Plugins written against the old exception-extension contract that exported
`KBM_UNAVAILABLE_EXCEPTION` or `KBM_RESPONSE_EXCEPTION` must be updated to raise
the shared exception classes directly; the old export names are no longer read
by TAS and will be ignored.