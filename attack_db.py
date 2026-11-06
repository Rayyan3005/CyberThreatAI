"""
MITRE ATT&CK STIX database loader and query index for CyberThreatAI.
Reads data/enterprise-attack.json and builds a lightweight fast cache.
"""

from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parent
STIX_PATH = ROOT / "data" / "enterprise-attack.json"
CACHE_PATH = ROOT / "data" / "mitre_index.json"

# Human friendly tactic names mapping
TACTIC_MAP = {
    "reconnaissance": "Reconnaissance",
    "resource-development": "Resource Development",
    "initial-access": "Initial Access",
    "execution": "Execution",
    "persistence": "Persistence",
    "privilege-escalation": "Privilege Escalation",
    "defense-evasion": "Defense Evasion",
    "credential-access": "Credential Access",
    "discovery": "Discovery",
    "lateral-movement": "Lateral Movement",
    "collection": "Collection",
    "command-and-control": "Command and Control",
    "exfiltration": "Exfiltration",
    "impact": "Impact",
}


def build_attack_index():
    """Parse enterprise-attack.json and create a lightweight lookup index."""
    if not STIX_PATH.exists():
        return {}

    with open(STIX_PATH, "r", encoding="utf-8") as f:
        stix_data = json.load(f)

    techniques = {}
    mitigations = {}
    tactics = {}

    for obj in stix_data.get("objects", []):
        obj_type = obj.get("type")
        if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
            continue

        if obj_type == "course-of-action":
            mitigations[obj.get("id")] = {
                "name": obj.get("name", ""),
                "description": obj.get("description", ""),
            }

        elif obj_type == "x-mitre-tactic":
            short_name = obj.get("x_mitre_shortname", "")
            tactics[short_name] = {
                "name": obj.get("name", ""),
                "description": obj.get("description", ""),
                "id": next(
                    (ref.get("external_id") for ref in obj.get("external_references", []) if ref.get("source_name") == "mitre-attack"),
                    ""
                )
            }

        elif obj_type == "attack-pattern":
            external_id = None
            url = ""
            for ref in obj.get("external_references", []):
                if ref.get("source_name") == "mitre-attack":
                    external_id = ref.get("external_id")
                    url = ref.get("url", "")
                    break

            if not external_id:
                continue

            kill_chains = [
                TACTIC_MAP.get(kc.get("phase_name"), kc.get("phase_name").replace("-", " ").title())
                for kc in obj.get("kill_chain_phases", [])
                if kc.get("kill_chain_name") == "mitre-attack"
            ]

            desc = obj.get("description", "")
            # Clean citations like (Citation: ...)
            clean_desc = re.sub(r"\(Citation:[^\)]+\)", "", desc).strip()
            # Summary snippet
            first_para = clean_desc.split("\n\n")[0] if "\n\n" in clean_desc else clean_desc
            if len(first_para) > 400:
                first_para = first_para[:397] + "..."

            detection = obj.get("x_mitre_detection", "")
            clean_detection = re.sub(r"\(Citation:[^\)]+\)", "", detection).strip()

            techniques[external_id] = {
                "id": external_id,
                "name": obj.get("name", ""),
                "tactics": kill_chains,
                "platforms": obj.get("x_mitre_platforms", []),
                "is_subtechnique": obj.get("x_mitre_is_subtechnique", False),
                "url": url,
                "summary": first_para,
                "full_description": clean_desc,
                "detection": clean_detection,
            }

    index_data = {
        "techniques": techniques,
        "tactics": tactics,
        "total_techniques": len(techniques),
    }

    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)
    except Exception as exc:
        print(f"Failed to cache index: {exc}")

    return index_data


def get_attack_index():
    """Load cached index or build if missing."""
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return build_attack_index()


def get_technique_details(technique_id: str):
    """Retrieve details for a specific MITRE technique ID."""
    index = get_attack_index()
    techniques = index.get("techniques", {})
    return techniques.get(technique_id)


def search_techniques(query: str, limit: int = 15):
    """Search techniques by ID, name, tactic, or description."""
    index = get_attack_index()
    techniques = index.get("techniques", {})
    query_lower = query.lower().strip()

    matches = []
    for tech_id, item in techniques.items():
        score = 0
        if tech_id.lower() == query_lower:
            score += 100
        elif tech_id.lower().startswith(query_lower):
            score += 80
        elif query_lower in tech_id.lower():
            score += 50

        if query_lower in item["name"].lower():
            score += 40

        for t in item.get("tactics", []):
            if query_lower in t.lower():
                score += 30

        if query_lower in item.get("summary", "").lower():
            score += 10

        if score > 0:
            matches.append((score, item))

    matches.sort(key=lambda x: x[0], reverse=True)
    return [m[1] for m in matches[:limit]]


if __name__ == "__main__":
    print("Building ATT&CK index...")
    idx = build_attack_index()
    print(f"Indexed {idx['total_techniques']} techniques.")
    t1566 = get_technique_details("T1566.001")
    print("Sample lookup T1566.001:", t1566["name"], "| Tactics:", t1566["tactics"])
