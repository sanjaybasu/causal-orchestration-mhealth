#!/usr/bin/env python3
"""
Upload the blinded review workbooks to a private Waymark Drive folder and share each external
reviewer's workbook with that reviewer only (Drive sends the notification email with the link).
The unblinding key is never uploaded.

  python scripts/r1_share_workbooks.py --dir <revision_1/human_review> [--dry-run]
"""

import argparse
import json
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

TOKEN = Path.home() / ".config/slides-comments/token.json"
REVIEWERS = {
    "morgan": ("john.morgan@waymarkcare.com", "John"),
    "moher": ("christian.moher@waymarkcare.com", "Christian"),
}
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
MESSAGE = ("Hi {name}, here is your workbook for the blinded care plan review. The instructions are in my "
           "email from October 16 and on the Instructions sheet. Please edit this file in place and return "
           "it by Monday, November 2. Do not download it to a personal device. Thank you, Sanjay")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    folder = Path(a.dir)
    files = {k: folder / f"care_plan_review_{k}.xlsx" for k in ["basu", *REVIEWERS]}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise SystemExit(f"missing workbooks: {missing}")
    if a.dry_run:
        print("would upload:", [p.name for p in files.values()])
        return

    creds = Credentials.from_authorized_user_file(str(TOKEN))
    if not creds.valid:
        creds.refresh(Request())
    drive = build("drive", "v3", credentials=creds)
    parent = drive.files().create(body={"name": "PDIG care plan review (blinded)",
                                        "mimeType": "application/vnd.google-apps.folder"},
                                  fields="id,webViewLink").execute()
    out = {"folder": parent["webViewLink"]}
    for key, path in files.items():
        f = drive.files().create(body={"name": path.name, "parents": [parent["id"]]},
                                 media_body=MediaFileUpload(str(path), mimetype=XLSX),
                                 fields="id,webViewLink").execute()
        out[key] = f["webViewLink"]
        if key in REVIEWERS:
            email, name = REVIEWERS[key]
            drive.permissions().create(fileId=f["id"], sendNotificationEmail=True,
                                       emailMessage=MESSAGE.format(name=name),
                                       body={"type": "user", "role": "writer", "emailAddress": email}).execute()
    (folder / "drive_links.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
