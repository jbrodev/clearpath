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


def check_token_expiry(token: str) -> None:
    """Parse JWT and raise TokenExpiredError if exp claim is in the past."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return
        payload_b64 = parts[1]
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        payload = json.loads(b64decode(payload_b64).decode("utf-8"))
        exp = payload.get("exp")
        if exp and time.time() > exp:
            raise TokenExpiredError("FHIR access token has expired (exp claim)")
    except TokenExpiredError:
        raise
    except Exception:
        pass


class FHIRClient:
    def __init__(self, context: FHIRContext):
        self.base_url = context.fhirUrl.rstrip("/")
        self.token = context.fhirToken
        self.patient_id = context.patientId

        if self.token:
            check_token_expiry(self.token)

        headers = {"Accept": "application/fhir+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        self._headers = headers

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
