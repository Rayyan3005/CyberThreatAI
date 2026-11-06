"""
Indicator of Compromise (IOC) extractor and defanging utility for SOC analysts.
Extracts IPs, Domains, URLs, Emails, Hashes, and CVEs from threat intelligence text.
"""

import re
import ipaddress


def defang(text: str) -> str:
    """Safely defang text/IOCs for security reports (e.g. http -> hxxp, . -> [.])"""
    text = re.sub(r"https?://", lambda m: "hxxps://" if "https" in m.group(0) else "hxxp://", text, flags=re.IGNORECASE)
    text = re.sub(r"\.(com|org|net|io|ru|cn|xyz|top|info|biz|cc|pw)\b", r"[.]\1", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\d)\.(?=\d)", "[.]", text)  # Defang IPs: 1.1.1.1 -> 1[.]1[.]1[.]1
    return text


def refang(text: str) -> str:
    """Restore defanged text to standard format."""
    text = re.sub(r"hxxps?://", lambda m: "https://" if "hxxps" in m.group(0).lower() else "http://", text, flags=re.IGNORECASE)
    text = text.replace("[.]", ".").replace("(.)", ".").replace("{.}", ".")
    text = text.replace("[@]", "@").replace("(@)", "@")
    text = text.replace("[:]", ":")
    return text


def extract_iocs(text: str) -> dict:
    """
    Extract structured Indicators of Compromise from raw text.
    Handles both normal and defanged IOCs.
    """
    normalized = refang(text)

    # 1. IPv4 extraction with validation
    raw_ips = set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", normalized))
    valid_ips = []
    for ip in raw_ips:
        try:
            parsed = ipaddress.ip_address(ip)
            # Exclude loopback/standard subnet masks if obvious
            if ip not in ["255.255.255.0", "0.0.0.0"]:
                valid_ips.append({
                    "indicator": ip,
                    "defanged": defang(ip),
                    "is_private": parsed.is_private,
                })
        except ValueError:
            pass

    # 2. SHA256 / SHA1 / MD5 Hashes
    sha256 = list(set(re.findall(r"\b[A-Fa-f0-9]{64}\b", normalized)))
    sha1 = list(set(re.findall(r"\b[A-Fa-f0-9]{40}\b", normalized)))
    md5 = list(set(re.findall(r"\b[A-Fa-f0-9]{32}\b", normalized)))

    # 3. URLs
    urls = list(set(re.findall(r"\bhttps?://[^\s<>\"\'\)]+", normalized)))

    # 4. Email addresses
    emails = list(set(re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", normalized)))

    # 5. Domains (excluding parts of URLs or emails, and standard false positives)
    candidate_domains = set(re.findall(r"\b[a-zA-Z0-9][-a-zA-Z0-9]{1,61}[a-zA-Z0-9]\.(?:com|org|net|io|ru|cn|xyz|top|info|biz|cc|pw|co|me|online|live|tech|site|store|app|dev)\b", normalized, flags=re.IGNORECASE))
    cleaned_domains = []
    for d in candidate_domains:
        d_lower = d.lower()
        if not any(d_lower in u.lower() for u in urls) and not any(d_lower in e.lower() for e in emails):
            # Exclude benign common references
            if d_lower not in ["mitre.org", "attack.mitre.org", "github.com", "microsoft.com", "google.com"]:
                cleaned_domains.append({
                    "indicator": d,
                    "defanged": defang(d),
                })

    # 6. CVE identifiers
    cves = list(set(re.findall(r"\bCVE-\d{4}-\d{4,7}\b", normalized, flags=re.IGNORECASE)))

    # 7. MITRE Technique IDs mentioned in text
    mitre_ids = list(set(re.findall(r"\bT\d{4}(?:\.\d{3})?\b", normalized)))

    total_count = len(valid_ips) + len(sha256) + len(sha1) + len(md5) + len(urls) + len(emails) + len(cleaned_domains) + len(cves)

    return {
        "ips": valid_ips,
        "sha256": [{"indicator": h, "defanged": h} for h in sha256],
        "sha1": [{"indicator": h, "defanged": h} for h in sha1],
        "md5": [{"indicator": h, "defanged": h} for h in md5],
        "urls": [{"indicator": u, "defanged": defang(u)} for u in urls],
        "emails": [{"indicator": e, "defanged": defang(e)} for e in emails],
        "domains": cleaned_domains,
        "cves": cves,
        "mitre_ids": mitre_ids,
        "total_count": total_count,
    }


if __name__ == "__main__":
    test_text = """
    Incident Note: Phishing email from billing@fake-internal[.]com urged victim to open invoice_sept.xlsm.
    Payload downloaded from hxxp://malicious-gateway[.]top/payload.exe connecting to C2 at 185.220.101.5.
    File SHA256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    Vulnerability exploited: CVE-2023-38831. Technique referenced: T1566.001.
    """
    res = extract_iocs(test_text)
    print("Found IOCs:", res["total_count"])
    print("IPs:", res["ips"])
    print("URLs:", res["urls"])
    print("Domains:", res["domains"])
    print("CVEs:", res["cves"])
    print("Techniques:", res["mitre_ids"])
