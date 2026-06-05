"""rhypedb HTTP/JSON client (urllib-based, no external deps)."""

import json
import urllib.request
import urllib.error


class RhypedbHttpClient:
    def __init__(self, base_url: str = "http://127.0.0.1:4200"):
        self.base_url = base_url

    def query(self, q: str) -> dict:
        """Execute a query and return the parsed JSON response."""
        body = json.dumps({"query": q}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/query",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            return {"error": f"HTTP {e.code}: {e.read().decode()}"}

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health") as resp:
                return resp.status == 200
        except Exception:
            return False

    def status(self) -> dict:
        with urllib.request.urlopen(f"{self.base_url}/status") as resp:
            return json.loads(resp.read().decode())

    def close(self) -> None:
        pass  # urllib has no persistent connection state
