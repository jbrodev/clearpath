"""
Async FHIR client for ClearPath.
Fetches minimal resources in parallel.
Validates JWT expiry before any fetch.
"""

import json
import time
from base64 import b64decode
from datetime import datetime, timedelta, timezone

import httpx

from clearpath.models.a2a import FHIRContext


class TokenExpiredError(Exception):
    pass


class FHIRClientError(Exception):
    pass


def _is_token_expired(token: str) -> bool:
    """Return True iff the JWT's exp claim is in the past. Best-effort parse."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return False
        payload_b64 = parts[1]
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        payload = json.loads(b64decode(payload_b64).decode("utf-8"))
        exp = payload.get("exp")
        return bool(exp and time.time() > exp)
    except Exception:
        return False


def check_token_expiry(token: str) -> None:
    """Raise TokenExpiredError if the JWT exp claim is in the past."""
    if _is_token_expired(token):
        raise TokenExpiredError("FHIR access token has expired (exp claim)")


class FHIRClient:
    def __init__(self, context: FHIRContext):
        self.base_url = context.fhirUrl.rstrip("/")
        self.token = context.fhirToken
        self.patient_id = context.patientId
        self.refresh_token = context.fhirRefreshToken
        self.refresh_url = context.fhirRefreshTokenUrl

        # Only enforce expiry up-front when we have no way to recover.
        # If refresh creds are present, we'll try to refresh in fetch_all().
        if self.token and not (self.refresh_token and self.refresh_url):
            check_token_expiry(self.token)

        self._set_headers()

    def _set_headers(self) -> None:
        headers = {"Accept": "application/fhir+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self._headers = headers

    async def _refresh_token_if_expired(self, client: httpx.AsyncClient) -> None:
        """If the access token is expired and we have refresh credentials, swap it in place."""
        if not self.token or not _is_token_expired(self.token):
            return
        if not (self.refresh_token and self.refresh_url):
            raise TokenExpiredError("FHIR access token has expired (exp claim)")
        resp = await client.post(
            self.refresh_url,
            json={"refreshToken": self.refresh_token},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        new_access = data.get("accessToken")
        if not new_access:
            raise FHIRClientError("refresh response missing accessToken")
        self.token = new_access
        self.refresh_token = data.get("refreshToken", self.refresh_token)
        self._set_headers()

    async def _get(self, client: httpx.AsyncClient, path: str) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            resp = await client.get(url, headers=self._headers, timeout=10.0)
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException:
            return {"resourceType": "Bundle", "entry": [], "_error": "timeout"}
        except httpx.HTTPStatusError as e:
            return {"resourceType": "Bundle", "entry": [], "_error": str(e)}
        except Exception as e:
            return {"resourceType": "Bundle", "entry": [], "_error": str(e)}

    def _six_months_ago(self) -> str:
        return (datetime.now(timezone.utc) - timedelta(days=180)).strftime("%Y-%m-%d")

    def _twelve_months_ago(self) -> str:
        return (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")

    async def fetch_all(self) -> dict:
        """Fetch all required resources in parallel. Returns raw FHIR bundles keyed by resource type."""
        if not self.patient_id:
            return {}

        pid = self.patient_id
        six_mo = self._six_months_ago()
        twelve_mo = self._twelve_months_ago()

        async with httpx.AsyncClient() as client:
            await self._refresh_token_if_expired(client)

            import asyncio
            results = await asyncio.gather(
                self._get(client, f"Patient/{pid}"),
                self._get(client, f"Condition?patient={pid}&clinical-status=active&_count=100"),
                self._get(client, f"MedicationRequest?patient={pid}&status=active&_count=100"),
                self._get(client, f"Procedure?patient={pid}&date=gt{six_mo}&_count=50"),
                self._get(client, f"DocumentReference?patient={pid}&date=gt{twelve_mo}&_count=20&_sort=-date"),
                self._get(client, f"Observation?patient={pid}&category=vital-signs&date=gt{six_mo}&_sort=-date&_count=10"),
                self._get(client, f"Observation?patient={pid}&category=laboratory&date=gt{six_mo}&_sort=-date&_count=30"),
                self._get(client, f"Encounter?patient={pid}&date=gt{twelve_mo}&_sort=-date&_count=20"),
                self._get(client, f"AllergyIntolerance?patient={pid}&_count=50"),
            )

        keys = [
            "patient", "conditions", "medications", "procedures",
            "documents", "vitals", "labs", "encounters", "allergies"
        ]
        return dict(zip(keys, results))
