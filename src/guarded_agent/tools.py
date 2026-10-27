"""Tool definitions and their executors.

Read tools run immediately. Write tools never run from here directly: the agent turns
them into a PendingAction that a person must approve (see guards.py).
"""
import httpx

# Only these fields ever reach the model. Email, phone, driver's licence and price stay out
# of the context window: less PII exposure and fewer tokens per tool result.
MODEL_VISIBLE_FIELDS = ("id", "customer_name", "car_class", "pickup_date", "return_date", "status", "notes")

# What each write tool is allowed to change, enforced in code (the JSON schema is not trusted alone).
UPDATABLE_FIELDS = {"return_date", "car_class", "notes"}
CAR_CLASSES = ["economy", "compact", "suv", "van"]

READ_TOOLS = {"search_bookings", "get_booking"}
WRITE_TOOLS = {"update_booking", "cancel_booking"}

TOOL_SPECS = [
    {
        "name": "search_bookings",
        "description": "Find bookings by customer name (partial match). Returns a list of bookings.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"customer_name": {"type": "string"}},
            "required": ["customer_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_booking",
        "description": "Get one booking by its id (e.g. BK-1001).",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"booking_id": {"type": "string"}},
            "required": ["booking_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "update_booking",
        "description": (
            "Change a booking's return date, car class or notes. A human operator reviews every "
            "change before it is applied; the tool result tells you whether it was approved and "
            "whether the change was verified as saved."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "booking_id": {"type": "string"},
                "return_date": {"type": "string", "description": "YYYY-MM-DD"},
                "car_class": {"type": "string", "enum": CAR_CLASSES},
                "notes": {"type": "string"},
            },
            "required": ["booking_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "cancel_booking",
        "description": "Cancel a booking. A human operator reviews it before it is applied.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"booking_id": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["booking_id", "reason"],
            "additionalProperties": False,
        },
    },
]


def visible(booking: dict) -> dict:
    return {k: booking[k] for k in MODEL_VISIBLE_FIELDS if k in booking}


def run_read_tool(api: httpx.Client, name: str, args: dict) -> dict | list:
    if name == "search_bookings":
        r = api.get("/bookings", params={"customer": args["customer_name"]})
        r.raise_for_status()
        return [visible(b) for b in r.json()]
    if name == "get_booking":
        r = api.get(f"/bookings/{args['booking_id']}")
        r.raise_for_status()
        return visible(r.json())
    raise ValueError(f"not a read tool: {name}")


def requested_changes(name: str, args: dict) -> dict:
    """The exact field changes a write tool asks for, after code-level validation."""
    if name == "update_booking":
        changes = {k: v for k, v in args.items() if k != "booking_id" and v is not None}
        illegal = set(changes) - UPDATABLE_FIELDS
        if illegal:
            raise ValueError(f"fields not updatable: {sorted(illegal)}")
        if not changes:
            raise ValueError("no changes requested")
        if "car_class" in changes and changes["car_class"] not in CAR_CLASSES:
            raise ValueError(f"unknown car class: {changes['car_class']}")
        return changes
    if name == "cancel_booking":
        return {"status": "cancelled"}
    raise ValueError(f"not a write tool: {name}")
