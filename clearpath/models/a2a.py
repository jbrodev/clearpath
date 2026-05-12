"""
A2A protocol models for ClearPath.
JSON-RPC 2.0 over HTTP; agent card follows Prompt Opinion's A2A v1 schema
(https://docs.promptopinion.ai/a2a-v1-migration). FHIR context arrives in
message.metadata keyed by FHIR_CONTEXT_EXTENSION_URI.
"""

from typing import Any
from pydantic import BaseModel, Field


FHIR_CONTEXT_EXTENSION_URI = "https://app.promptopinion.ai/schemas/a2a/v1/fhir-context"


class FHIRContext(BaseModel):
    fhirUrl: str
    fhirToken: str | None = None
    patientId: str | None = None
    fhirRefreshToken: str | None = None
    fhirRefreshTokenUrl: str | None = None


class A2AMessage(BaseModel):
    role: str = "user"
    parts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None

    def get_text(self) -> str:
        texts = []
        for part in self.parts:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
        return " ".join(texts).strip()

    def get_fhir_context(self) -> FHIRContext | None:
        if not self.metadata:
            return None
        raw = self.metadata.get(FHIR_CONTEXT_EXTENSION_URI)
        if not raw:
            return None
        return FHIRContext(**raw)


class A2AMessageSendParams(BaseModel):
    message: A2AMessage
    sessionId: str | None = None
    metadata: dict[str, Any] | None = None


class JSONRPCRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: str | int | None = None
    method: str
    params: A2AMessageSendParams | dict[str, Any] | None = None


class JSONRPCError(BaseModel):
    code: int
    message: str
    data: Any | None = None


class A2ATaskStatus(BaseModel):
    state: str = "TASK_STATE_COMPLETED"


class A2AArtifact(BaseModel):
    artifactId: str
    name: str | None = None
    parts: list[dict[str, Any]] = Field(default_factory=list)


class A2ATask(BaseModel):
    id: str
    contextId: str
    status: A2ATaskStatus = Field(default_factory=A2ATaskStatus)
    artifacts: list[A2AArtifact] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None


class JSONRPCResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: str | int | None = None
    result: A2ATask | None = None
    error: JSONRPCError | None = None
