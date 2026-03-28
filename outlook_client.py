"""
Outlook 365 email client using Microsoft Graph API + MSAL.
Fetches emails and converts them to Documents for indexing.
"""

import json
import logging
import webbrowser
from datetime import datetime, timedelta

import msal
import requests

from config import (
    OUTLOOK_CLIENT_ID,
    OUTLOOK_REDIRECT_URI,
    OUTLOOK_TENANT_ID,
    TOKEN_CACHE_PATH,
)
from document_loader import Document

logger = logging.getLogger(__name__)

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
SCOPES = ["Mail.Read", "Mail.ReadBasic", "User.Read"]


class OutlookClient:
    """Fetches emails from Outlook 365 via Microsoft Graph API."""

    def __init__(self):
        if not OUTLOOK_CLIENT_ID or not OUTLOOK_TENANT_ID:
            raise ValueError(
                "Outlook not configured. Set OUTLOOK_CLIENT_ID and OUTLOOK_TENANT_ID in .env"
            )

        self.cache = msal.SerializableTokenCache()
        if TOKEN_CACHE_PATH.exists():
            self.cache.deserialize(TOKEN_CACHE_PATH.read_text())

        authority = f"https://login.microsoftonline.com/{OUTLOOK_TENANT_ID}"
        self.app = msal.PublicClientApplication(
            OUTLOOK_CLIENT_ID,
            authority=authority,
            token_cache=self.cache,
        )
        self._token = None

    def _save_cache(self):
        if self.cache.has_state_changed:
            TOKEN_CACHE_PATH.write_text(self.cache.serialize())

    def authenticate(self) -> bool:
        """Authenticate with Microsoft. Opens browser for first-time login."""
        accounts = self.app.get_accounts()
        if accounts:
            result = self.app.acquire_token_silent(SCOPES, account=accounts[0])
            if result and "access_token" in result:
                self._token = result["access_token"]
                self._save_cache()
                logger.info("Authenticated via cached token.")
                return True

        flow = self.app.initiate_device_flow(scopes=SCOPES)
        if "user_code" not in flow:
            logger.error(f"Auth flow failed: {flow}")
            return False

        print(f"\n{'='*60}")
        print(f"  To sign in, open: {flow['verification_uri']}")
        print(f"  Enter code: {flow['user_code']}")
        print(f"{'='*60}\n")

        try:
            webbrowser.open(flow["verification_uri"])
        except Exception:
            pass

        result = self.app.acquire_token_by_device_flow(flow)
        if "access_token" in result:
            self._token = result["access_token"]
            self._save_cache()
            logger.info("Authentication successful.")
            return True

        logger.error(f"Auth failed: {result.get('error_description', 'Unknown error')}")
        return False

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._token}"}

    def fetch_emails(
        self,
        folder: str = "inbox",
        days_back: int = 30,
        max_emails: int = 200,
    ) -> list[Document]:
        """Fetch emails and return as Document objects."""
        if not self._token:
            if not self.authenticate():
                return []

        since = (datetime.utcnow() - timedelta(days=days_back)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        documents = []
        url = (
            f"{GRAPH_BASE}/me/mailFolders/{folder}/messages"
            f"?$top=50"
            f"&$filter=receivedDateTime ge {since}"
            f"&$select=subject,from,toRecipients,receivedDateTime,body,bodyPreview"
            f"&$orderby=receivedDateTime desc"
        )

        while url and len(documents) < max_emails:
            resp = requests.get(url, headers=self._headers(), timeout=30)
            if resp.status_code == 401:
                logger.warning("Token expired, re-authenticating...")
                self._token = None
                if not self.authenticate():
                    break
                resp = requests.get(url, headers=self._headers(), timeout=30)

            if resp.status_code != 200:
                logger.error(f"Graph API error {resp.status_code}: {resp.text[:300]}")
                break

            data = resp.json()
            for msg in data.get("value", []):
                doc = self._email_to_document(msg)
                if doc:
                    documents.append(doc)

            url = data.get("@odata.nextLink")

        logger.info(f"Fetched {len(documents)} emails from {folder}")
        return documents[:max_emails]

    def _email_to_document(self, msg: dict) -> Document | None:
        subject = msg.get("subject", "(no subject)")
        from_addr = ""
        if msg.get("from", {}).get("emailAddress"):
            ea = msg["from"]["emailAddress"]
            from_addr = f"{ea.get('name', '')} <{ea.get('address', '')}>"

        to_addrs = []
        for r in msg.get("toRecipients", []):
            ea = r.get("emailAddress", {})
            to_addrs.append(f"{ea.get('name', '')} <{ea.get('address', '')}>")

        date = msg.get("receivedDateTime", "")
        body = msg.get("body", {}).get("content", "")

        if msg.get("body", {}).get("contentType") == "html":
            try:
                from bs4 import BeautifulSoup
                body = BeautifulSoup(body, "html.parser").get_text(separator="\n", strip=True)
            except ImportError:
                body = msg.get("bodyPreview", "")

        if not body or not body.strip():
            return None

        text = f"Subject: {subject}\nFrom: {from_addr}\nTo: {', '.join(to_addrs)}\nDate: {date}\n\n{body}"

        metadata = {
            "filename": f"email_{date[:10]}_{subject[:50]}.eml",
            "filepath": f"outlook://{subject}",
            "extension": ".eml",
            "source": "outlook365",
            "subject": subject,
            "from": from_addr,
            "date": date,
        }

        return Document(text=text.strip(), metadata=metadata)

    def get_folders(self) -> list[dict]:
        """List available mail folders."""
        if not self._token and not self.authenticate():
            return []

        resp = requests.get(
            f"{GRAPH_BASE}/me/mailFolders?$top=50",
            headers=self._headers(),
            timeout=30,
        )
        if resp.status_code != 200:
            return []

        folders = []
        for f in resp.json().get("value", []):
            folders.append({
                "id": f["id"],
                "name": f["displayName"],
                "count": f.get("totalItemCount", 0),
                "unread": f.get("unreadItemCount", 0),
            })
        return folders
