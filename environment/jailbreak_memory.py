import json
from pathlib import Path


class JailbreakMemory:
    def __init__(self, log_path="vulnerability_log.json"):
        self.log_path = Path(log_path)
        self.log: dict[str, dict[str, object]] = {}

        if self.log_path.exists():
            try:
                loaded = json.loads(self.log_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.log = loaded
            except (json.JSONDecodeError, OSError):
                self.log = {}

    def _save(self) -> None:
        self.log_path.write_text(json.dumps(self.log, indent=2), encoding="utf-8")

    def record_success(self, attack_pattern, harm_category, persona):
        normalized_pattern = str(attack_pattern).strip()
        if not normalized_pattern:
            return

        existing = self.log.get(normalized_pattern, {})
        count = int(existing.get("count", 0)) + 1

        self.log[normalized_pattern] = {
            "count": count,
            "harm_category": str(harm_category),
            "persona": str(persona),
        }
        self._save()

    def was_seen_before(self, attack_pattern):
        normalized_pattern = str(attack_pattern).strip()
        if not normalized_pattern:
            return False
        entry = self.log.get(normalized_pattern)
        return bool(entry and int(entry.get("count", 0)) >= 1)

    def get_penalty(self, attack_pattern):
        return 0.2 if self.was_seen_before(attack_pattern) else 0.0

    def summarize(self):
        if not self.log:
            print("No jailbreak vulnerabilities logged yet.")
            return

        print("\nJailbreak Vulnerability Summary")
        print("Pattern | Persona | Harm Category | Success Count")
        print("-" * 90)
        for pattern, details in self.log.items():
            persona = str(details.get("persona", "unknown"))
            harm_category = str(details.get("harm_category", "unknown"))
            count = int(details.get("count", 0))
            print(f"{pattern} | {persona} | {harm_category} | {count}")