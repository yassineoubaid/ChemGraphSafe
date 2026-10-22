"""Public extension contracts. Reference labels never enter Request."""

import math
from typing import Literal, Protocol, get_args

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

Decision = Literal["ALLOW", "REVIEW", "BLOCK"]
Task = Literal["triage", "scisafetybench_risk_type"]
RiskType = Literal[
    "Direct Malicious User",
    "Indirect Malicious User",
    "Task-Intrinsic Risk",
    "Unintentional Consequences",
]
RISK_TYPES = get_args(RiskType)
Status = Literal["ok", "abstain", "error"]
ProcessingLocation = Literal["local", "hosted"]
Reason = Literal["legitimate_use", "harm_facilitation", "insufficient_context", "operational_risk"]
LABELS = ("ALLOW", "REVIEW", "BLOCK")
BINARY_LABELS = ("ALLOW", "BLOCK")
PROTOCOL_VERSION = "triage-1"


def policy_task(policy: dict | None = None) -> Task:
    task = policy.get("task", "triage") if policy is not None else "triage"
    if task not in get_args(Task):
        raise ValueError("Unknown classification task")
    return task


def question_key(policy: dict | None = None) -> str:
    return "risk_type" if policy_task(policy) == "scisafetybench_risk_type" else "triage"


def task_protocol(task: Task) -> str:
    return "scisafetybench-risk-type-1" if task == "scisafetybench_risk_type" else PROTOCOL_VERSION


def risk_baselines() -> dict[str, str]:
    return {"always-" + label.lower().replace(" ", "-"): label for label in RISK_TYPES}


def policy_labels(policy: dict) -> tuple[str, ...]:
    """Return a supported label set in canonical order, independent of JSON key order."""
    criteria = policy.get("criteria")
    if policy_task(policy) == "scisafetybench_risk_type":
        if not isinstance(criteria, dict) or set(criteria) != set(RISK_TYPES):
            raise ValueError(
                "Risk-type policy must define exactly the four SciSafetyBench categories"
            )
        if any(not isinstance(value, str) or not value.strip() for value in criteria.values()):
            raise ValueError("Risk-type criteria require nonempty definitions")
        if not isinstance(policy.get("instructions"), str) or not policy["instructions"].strip():
            raise ValueError("Risk-type policy requires classification instructions")
        return RISK_TYPES
    if not isinstance(criteria, dict) or set(criteria) not in (set(LABELS), set(BINARY_LABELS)):
        raise ValueError("Policy must define ALLOW/BLOCK or ALLOW/REVIEW/BLOCK")
    return tuple(label for label in LABELS if label in criteria)


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Request(Record):
    text: str = Field(min_length=1)
    context: str = ""


class Example(Record):
    id: str
    request: Request
    source: str
    source_id: str
    source_revision: str
    source_split: str
    source_label: Literal["benign", "harmful"] | None
    source_native_harmful: bool | None = None
    domain: str = "Chemistry"
    category: str = "unspecified"
    family: str | None = None
    group: str = ""
    related_source_ids: list[str] = Field(default_factory=list)
    license: str
    hosted_processing_allowed: bool = False
    rights_evidence: str


class RotationComponent(Record):
    """One request-free Laya choice result used in a balanced probability average."""

    option_order: tuple[int, ...]
    choice: Decision | RiskType
    probabilities: dict[str, float]
    choice_argmax_mismatch: bool = False

    @model_validator(mode="after")
    def consistent(self):
        p = self.probabilities
        if set(p) not in (set(LABELS), set(BINARY_LABELS), set(RISK_TYPES)) or self.choice not in p:
            raise ValueError("A rotation component requires a supported matching label set")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in p.values()):
            raise ValueError("Invalid rotation-component probability")
        if abs(sum(p.values()) - 1) > 1e-3:
            raise ValueError("Rotation-component probabilities must sum to one within 1e-3")
        if tuple(sorted(self.option_order)) != tuple(range(len(p))):
            raise ValueError("Rotation-component option order must be a label permutation")
        self.choice_argmax_mismatch = p[self.choice] < max(p.values())
        return self


class Prediction(Record):
    status: Status
    decision: Decision | RiskType | None = None
    probabilities: dict[str, float] | None = None
    probability_origin: Literal["native", "rotation_averaged", "mock"] | None = None
    rotation_components: dict[str, RotationComponent] | None = None
    choice_argmax_mismatch: bool = False
    returned_model: str | None = None
    reason: Reason | None = None
    error_code: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    provider_cost_usd: float | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    latency_ms: float = Field(default=0, ge=0)
    retry_overhead_ms: float = Field(default=0, ge=0)
    attempts: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def consistent(self):
        if (self.status == "ok") != (self.decision is not None):
            raise ValueError("Only successful decisions may carry a policy label")
        if self.probabilities is not None:
            p = self.probabilities
            if (
                self.status != "ok"
                or set(p) not in (set(LABELS), set(BINARY_LABELS), set(RISK_TYPES))
                or self.decision not in p
            ):
                raise ValueError("A distribution requires a matching supported decision label set")
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in p.values()):
                raise ValueError("Invalid probability")
            if abs(sum(p.values()) - 1) > 1e-3:
                raise ValueError("Probability sum must be within 1e-3 of one")
            if self.probability_origin is None:
                raise ValueError("Probability provenance is required")
            self.choice_argmax_mismatch = p[self.decision] < max(p.values())
        if self.rotation_components is not None:
            components = self.rotation_components
            if (
                self.status != "ok"
                or self.probability_origin != "rotation_averaged"
                or self.probabilities is None
                or len(components) != len(self.probabilities)
                or any(not isinstance(key, str) or not key for key in components)
            ):
                raise ValueError("Rotation components require a complete averaged prediction")
            labels = set(self.probabilities)
            if any(set(component.probabilities) != labels for component in components.values()):
                raise ValueError("Rotation-component labels differ from the averaged distribution")
            orders = [component.option_order for component in components.values()]
            if len(set(orders)) != len(orders) or any(
                {order[position] for order in orders} != set(range(len(labels)))
                for position in range(len(labels))
            ):
                raise ValueError("Rotation components do not balance every option position")
            for label, probability in self.probabilities.items():
                mean = math.fsum(
                    component.probabilities[label] for component in components.values()
                ) / len(components)
                if not math.isclose(mean, probability, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError("Rotation-component mean differs from averaged probabilities")
        elif self.probability_origin == "rotation_averaged":
            raise ValueError("Rotation-averaged probabilities require auditable components")
        return self

    @model_serializer(mode="wrap")
    def serialize_without_empty_rotation_components(self, handler):
        result = handler(self)
        if self.rotation_components is None:
            result.pop("rotation_components", None)
        return result

    def validate_policy(self, policy: dict | None = None) -> "Prediction":
        """Validate the active experiment's label set; omitted policy retains the legacy default."""
        labels = policy_labels(policy) if policy is not None else LABELS
        if self.status == "ok":
            if self.decision not in labels:
                raise ValueError("Decision is outside the active policy labels")
            if self.probabilities is not None and set(self.probabilities) != set(labels):
                raise ValueError("Probability labels differ from the active policy labels")
            if self.rotation_components is not None and any(
                set(component.probabilities) != set(labels)
                for component in self.rotation_components.values()
            ):
                raise ValueError("Rotation-component labels differ from the active policy labels")
            if self.rotation_components is not None:
                expected = {
                    f"{question_key(policy)}_rotation_{rotation}": tuple(
                        (index + rotation) % len(labels) for index in range(len(labels))
                    )
                    for rotation in range(len(labels))
                }
                if set(self.rotation_components) != set(expected):
                    raise ValueError("Rotation-component IDs differ from the active task")
                if any(
                    self.rotation_components[key].option_order != option_order
                    for key, option_order in expected.items()
                ):
                    raise ValueError("Rotation-component order differs from its question ID")
                if self.decision != max(labels, key=self.probabilities.__getitem__):
                    raise ValueError("Rotation-averaged decision is not the canonical argmax")
            if (
                policy is not None
                and self.reason is not None
                and self.reason not in policy.get("reason_codes", [])
            ):
                raise ValueError("Reason is outside the active policy reason codes")
        return self


class Annotation(Record):
    example_id: str
    policy_hash: str
    config_hash: str
    origin: Literal["llm_provisional", "owned_fixture"]
    requested_model: str
    prediction: Prediction
    source_disagreement: bool = False


class RuleAnnotation(Record):
    """A reproducible policy reference, without an invented model/curator identity."""

    example_id: str
    policy_hash: str
    origin: Literal["rule_derived"] = "rule_derived"
    rule_id: str
    rule_hash: str
    provenance: dict[str, str]
    prediction: Prediction


def parse_annotation(row: dict) -> Annotation | RuleAnnotation:
    cls = RuleAnnotation if row.get("origin") == "rule_derived" else Annotation
    return cls.model_validate(row)


class PredictionRow(Record):
    example_id: str
    model_name: str
    split: Literal["dev", "eval", "smoke"]
    run_key: str
    prediction: Prediction


class DatasetLoader(Protocol):
    def load(self, config) -> list[Example]: ...


class DecisionModel(Protocol):
    processing_location: ProcessingLocation

    def predict(self, request: Request, policy: dict) -> Prediction: ...


class Labeler(Protocol):
    def annotate(self, example: Example, policy: dict) -> Annotation: ...
