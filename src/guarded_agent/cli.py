"""Interactive demo: chat with the agent against the in-process mock API.

    python -m guarded_agent.cli                       # normal API
    python -m guarded_agent.cli --fault silent_drop:return_date   # API that lies about saving
    python -m guarded_agent.cli --dry-run

Needs ANTHROPIC_API_KEY (or an `ant auth login` profile).
"""
import argparse

from fastapi.testclient import TestClient

from .agent import GuardedAgent
from .guards import AuditLog
from .llm import ClaudeLLM
from .mock_api import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fault", action="append", default=[], help="e.g. silent_drop:return_date")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit", default="audit.jsonl")
    args = parser.parse_args()

    api = TestClient(create_app(set(args.fault)))
    agent = GuardedAgent(ClaudeLLM(), api, audit=AuditLog(args.audit), dry_run=args.dry_run)
    print("Booking assistant. Try: 'Extend Maria Silva's rental to 12 November'. Ctrl+C to exit.\n")

    while True:
        state = agent.start(input("you> "))
        while state.status == "needs_approval":
            print("\n--- approval needed ---\n" + state.pending["preview"] + "\n-----------------------")
            approved = input("approve? [y/N] ").strip().lower() == "y"
            state = agent.resume(state, approved=approved)
        print(f"agent> {state.final_text or '[' + state.status + ']'}\n")


if __name__ == "__main__":
    main()
