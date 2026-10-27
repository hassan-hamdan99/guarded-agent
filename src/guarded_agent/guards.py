"""The three guards around every write: approval, verification and an audit trail."""
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from .tools import requested_changes


@dataclass
class PendingAction:
    """A write the model asked for. Nothing has been sent to the API yet."""

    id: str
    tool_use_id: str
    tool_name: str
    booking_id: str
    changes: dict
    before: dict
    preview: str


def propose(api: httpx.Client, tool_use_id: str, name: str, args: dict) -> PendingAction:
    """Build the pending action and its preview from the API's current state, never from model text."""
    changes = requested_changes(name, args)
    r = api.get(f"/bookings/{args['booking_id']}")
    r.raise_for_status()
    current = r.json()
    before = {k: current.get(k) for k in changes}
    lines = [f"{name} on {args['booking_id']} ({current['customer_name']})"]
    lines += [f"  {k}: {before[k]!r} -> {v!r}" for k, v in changes.items()]
    if name == "cancel_booking":
        lines.append(f"  reason: {args.get('reason', '')!r}")
    return PendingAction(
        id=uuid.uuid4().hex[:12], tool_use_id=tool_use_id, tool_name=name,
        booking_id=args["booking_id"], changes=changes, before=before, preview="\n".join(lines),
    )


@dataclass
class Verification:
    ok: bool
    mismatches: dict = field(default_factory=dict)  # field -> {"expected": ..., "actual": ...}


def apply_and_verify(api: httpx.Client, action: PendingAction) -> Verification:
    """Send the write, then read the record back. The API's own `success` flag is not evidence."""
    r = api.patch(f"/bookings/{action.booking_id}", json=action.changes)
    r.raise_for_status()
    after = api.get(f"/bookings/{action.booking_id}").json()
    mismatches = {
        k: {"expected": v, "actual": after.get(k)} for k, v in action.changes.items() if after.get(k) != v
    }
    return Verification(ok=not mismatches, mismatches=mismatches)


class AuditLog:
    """Append-only JSONL trail of every proposal, decision and verification."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.entries: list[dict] = []

    def record(self, event: str, **data) -> None:
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **data}
        self.entries.append(entry)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def action_to_dict(action: PendingAction | None) -> dict | None:
    return asdict(action) if action else None
