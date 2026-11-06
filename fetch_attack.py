from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
DEST = ROOT / "data" / "enterprise-attack.json"

def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading the official MITRE ATT&CK Enterprise STIX bundle...")
    response = requests.get(URL, timeout=90)
    response.raise_for_status()
    DEST.write_bytes(response.content)
    print(f"Saved {len(response.content):,} bytes to {DEST}")
    print("Use this file as a reference taxonomy. Verify technique IDs/names and record the bundle retrieval date/version.")

if __name__ == "__main__":
    main()
