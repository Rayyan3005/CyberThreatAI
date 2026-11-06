
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "sample_reports.csv"
DEST = ROOT / "data" / "sample_reports_expanded.csv"

# These are synthetic teaching examples, NOT real incident reports.
new_examples = {
    "T1078": [
        "A fictional exercise describes an intruder signing in to a cloud service with a compromised employee password.",
        "In a simulated incident, an attacker accesses an internal application using credentials stolen from a user.",
        "A training scenario shows an unauthorized person logging in with a valid account after obtaining its password.",
        "A fictional report describes suspicious access to a remote service through a legitimate user account.",
        "During a tabletop exercise, an operator uses a compromised contractor account to reach a company portal.",
        "A simulated investigation finds that a stolen username and password were used to access an employee mailbox.",
        "A fictional scenario describes an attacker authenticating to a VPN with credentials belonging to a real employee.",
        "In a classroom exercise, an intruder reuses exposed account credentials to access a protected resource.",
        "A simulated threat report describes a login to an enterprise service using a compromised administrator account.",
        "A fictional incident describes an unauthorized login using a legitimate account whose credentials were exposed."
    ],
    "T1566.001": [
        "A fictional email campaign delivers a malicious document attachment that the recipient is urged to open.",
        "In a training scenario, a deceptive message attaches an office document containing a harmful payload.",
        "A simulated report describes targeted emails that carry a disguised attachment to selected employees.",
        "A fictional attacker sends a message with an attachment presented as an invoice to trick a recipient.",
        "A classroom exercise describes a spearphishing email that includes a malicious archive as an attachment.",
        "A simulated incident involves a recipient opening an unexpected document attached to a targeted email.",
        "A fictional report describes an email attachment masquerading as a meeting agenda to deceive staff.",
        "In a tabletop scenario, targeted recipients receive an email containing a suspicious attached spreadsheet.",
        "A simulated campaign uses a document attachment in a deceptive email aimed at a finance employee.",
        "A fictional threat report describes targeted phishing messages that rely on a malicious file attachment."
    ],
    "T1566.002": [
        "A fictional phishing message directs a targeted employee to a fraudulent sign-in webpage through a hyperlink.",
        "A training scenario describes an email containing a deceptive URL that leads to a credential-harvesting page.",
        "A simulated incident involves a targeted message persuading a recipient to click a malicious web link.",
        "A fictional attacker sends an email link to a counterfeit company login portal.",
        "A classroom exercise describes spearphishing messages that use URLs instead of file attachments.",
        "A simulated threat report describes a deceptive link in an email that redirects users to a fake authentication page.",
        "A fictional email campaign encourages staff to follow a link to a fraudulent document-sharing website.",
        "In a tabletop exercise, an employee receives a targeted email linking to a spoofed password-reset page.",
        "A simulated report describes a phishing URL sent directly to a specific employee in a deceptive message.",
        "A fictional scenario involves an email hyperlink designed to send a recipient to an attacker-controlled website."
    ]
}

original = pd.read_csv(SOURCE).fillna("")
required = [
    "report_id", "report_text", "technique_id",
    "technique_name", "source_url", "label_evidence"
]
missing = [column for column in required if column not in original.columns]
if missing:
    raise SystemExit(f"Missing required columns: {missing}")

names = {
    "T1078": "Valid Accounts",
    "T1566.001": "Spearphishing Attachment",
    "T1566.002": "Spearphishing Link"
}

rows = []
counter = 1

for technique_id, texts in new_examples.items():
    for text in texts:
        rows.append({
            "report_id": f"SYNNEW{counter:03d}",
            "report_text": text,
            "technique_id": technique_id,
            "technique_name": names[technique_id],
            "source_url": "synthetic://teaching-example",
            "label_evidence": (
                "Synthetic teaching example; not a real incident report."
            )
        })
        counter += 1

additional = pd.DataFrame(rows, columns=required)
combined = pd.concat([original[required], additional], ignore_index=True)

# Remove accidental duplicate text/label pairs.
combined = combined.drop_duplicates(
    subset=["report_text", "technique_id"]
).reset_index(drop=True)

combined.to_csv(DEST, index=False)

print("Expanded synthetic dataset created.")
print(f"Original rows: {len(original)}")
print(f"Additional rows prepared: {len(additional)}")
print(f"Final unique rows: {len(combined)}")
print("\nTechnique counts:")
print(combined["technique_id"].value_counts().to_string())
print(f"\nSaved to: {DEST}")
print("Note: All added examples are synthetic, not real threat reports.")
