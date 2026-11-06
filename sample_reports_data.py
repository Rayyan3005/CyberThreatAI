"""
Curated sample threat intelligence reports for demo and SOC analyst triage testing.
"""

SAMPLE_REPORTS = {
    "T1566.001 — Malicious Invoice Attachment": {
        "technique_id": "T1566.001",
        "technique_name": "Spearphishing Attachment",
        "title": "Finance Spearphishing via Macro-Enabled XLSM Invoice",
        "text": """On October 14, corporate accounts payable received an unsolicited email spoofing an authorized supplier with the subject line 'URGENT: Outstanding Balance & Invoice Remittance Q3-9941'. 

The email contained a malicious attachment named 'Vendor_Invoice_Payment.xlsm'. Upon opening the attachment, the user was prompted to enable macros. Once executed, an embedded VBA routine invoked an obfuscated PowerShell command that downloaded a secondary loader from hxxp://billing-update-portal[.]top/payload.exe. 

File hash (SHA256): 7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069.
Network logs confirmed outbound beaconing to Command & Control server at 198.51.100.42:8443. The security team quarantined the endpoint and blocked the sender address accounting@fake-supplier[.]org.""",
    },
    "T1566.002 — Microsoft 365 Credential Harvesting Link": {
        "technique_id": "T1566.002",
        "technique_name": "Spearphishing Link",
        "title": "Spoofed Microsoft 365 Password Expiration Alert",
        "text": """SOC Alert: Multiple staff members in legal and human resources received targeted spearphishing messages masquerading as corporate IT Security notifications. 

The email claimed that the recipient's corporate password was scheduled to expire within 2 hours and urged them to click an embedded hyperlink labeled 'Retain Current Credentials'. The deceptive link directed victims to a counterfeit login page hosted at hxxps://login.microsoftonline-portal-verify[.]net/auth/login.php. 

The credential harvesting page replicated company branding and captured entered usernames and corporate passwords in plaintext. Web proxy logs recorded 3 clicks from internal IP addresses heading to malicious external host 203.0.113.88. Two employees reported entering credentials; incident response immediately revoked their active tokens and triggered password resets.""",
    },
    "T1078 — Compromised Admin VPN Credentials": {
        "technique_id": "T1078",
        "technique_name": "Valid Accounts",
        "title": "Unauthorized Corporate VPN Access via Stolen Admin Account",
        "text": """Security Operations Center detected an anomalous remote connection to the enterprise SSL-VPN gateway originating from an unexpected residential IP (192.0.2.145). 

The intruder authenticated using legitimate credentials belonging to domain administrator account 'adm_djenkins'. Investigation revealed that the administrator's password had previously been compromised in an external third-party data breach and reused on the corporate gateway without mandatory multi-factor enforcement. 

Once logged in with this valid account, the adversary accessed internal SharePoint document repositories, queried domain controller configuration objects, and attempted to enumerate active administrative sessions across tier-1 member servers. No malicious binaries were dropped; the intruder relied exclusively on legitimate living-off-the-land utilities and valid account privileges.""",
    },
    "Hybrid / Multi-Indicator — Phishing Attachment & Stolen Credentials": {
        "technique_id": "T1566.001",
        "technique_name": "Spearphishing Attachment",
        "title": "Combined Phishing Delivery with Subsequent Credential Abuse",
        "text": """Threat Intelligence Report: A targeted campaign against defense contractors began with a spearphishing email delivering an attached zip archive named 'Contract_Specifications_2026.zip'. 

The archive contained a weaponized shortcut file that executed a hidden script extracting credentials from local browser storage. The adversary then utilized the stolen credentials to sign in to internal cloud accounts using valid user accounts. 

C2 communication was observed to 185.220.101.5 and 198.51.100.12. Analysts noted the report exhibits indicators for both Spearphishing Attachment (T1566.001) and Valid Accounts (T1078).""",
    },
    "Custom / Blank Workspace": {
        "technique_id": "",
        "technique_name": "",
        "title": "New Custom Investigation",
        "text": "",
    },
}
