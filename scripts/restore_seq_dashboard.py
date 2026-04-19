"""Restore the Seq observability dashboard and user preferences.

Idempotent — safe to run on every startup. Skips creation if a dashboard
with the same title already exists. Always updates the default dashboard
preference to point at the correct ID.

Usage:
    python scripts/restore_seq_dashboard.py
    python scripts/restore_seq_dashboard.py --force   # recreate even if exists
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from base64 import b64encode
from pathlib import Path

DASHBOARD_JSON = Path(__file__).parent / "seq_dashboard.json"
DASHBOARD_TITLE = "Document Classification — Observability"


def _auth_header(username: str, password: str) -> str:
    return "Basic " + b64encode(f"{username}:{password}".encode()).decode()


def _request(method: str, url: str, body: dict | None, auth: str) -> dict:
    data = json.dumps(body).encode() if body else None
    headers = {"Authorization": auth, "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def wait_for_seq(base_url: str, auth: str, retries: int = 10) -> None:
    import time
    for i in range(retries):
        try:
            urllib.request.urlopen(
                urllib.request.Request(f"{base_url}/api/dashboards?shared=true", headers={"Authorization": auth}),
                timeout=3,
            )
            return
        except Exception:
            if i == retries - 1:
                raise RuntimeError(f"Seq not reachable at {base_url} after {retries} attempts")
            print(f"  Seq not ready, retrying ({i + 1}/{retries})...")
            time.sleep(3)


def find_existing(base_url: str, auth: str) -> str | None:
    resp = _request("GET", f"{base_url}/api/dashboards?shared=true", None, auth)
    for d in resp:
        if d.get("Title") == DASHBOARD_TITLE:
            return d["Id"]
    return None


def create_dashboard(base_url: str, auth: str) -> str:
    payload = json.loads(DASHBOARD_JSON.read_text())
    # Strip server-assigned IDs so Seq generates fresh ones
    payload.pop("Id", None)
    payload.pop("Links", None)
    for chart in payload.get("Charts", []):
        chart.pop("Id", None)
        for q in chart.get("Queries", []):
            q.pop("Id", None)
    result = _request("POST", f"{base_url}/api/dashboards", payload, auth)
    return result["Id"]


def set_default(base_url: str, auth: str, dashboard_id: str) -> None:
    user = _request("GET", f"{base_url}/api/users/user-admin", None, auth)
    user.setdefault("Preferences", {})["DefaultDashboardId"] = dashboard_id
    user.pop("Links", None)
    _request("PUT", f"{base_url}/api/users/user-admin", user, auth)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-url", default="http://localhost:5341")
    parser.add_argument("--username", default="admin")
    parser.add_argument("--password", default="stellar1234")
    parser.add_argument("--force", action="store_true", help="Recreate dashboard even if it exists")
    args = parser.parse_args()

    auth = _auth_header(args.username, args.password)

    print(f"Connecting to Seq at {args.seq_url}...")
    try:
        wait_for_seq(args.seq_url, auth)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    existing_id = find_existing(args.seq_url, auth)

    if existing_id and not args.force:
        print(f"Dashboard already exists: {existing_id} — skipping creation")
        dashboard_id = existing_id
    else:
        if existing_id and args.force:
            print(f"--force: recreating dashboard (was {existing_id})")
        else:
            print("Dashboard not found — creating...")
        dashboard_id = create_dashboard(args.seq_url, auth)
        print(f"Created dashboard: {dashboard_id}")

    set_default(args.seq_url, auth, dashboard_id)
    print(f"Default dashboard set to: {dashboard_id}")
    print("Done.")


if __name__ == "__main__":
    main()
