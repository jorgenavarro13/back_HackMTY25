# app/nessie_client.py
import os
import asyncio
import logging
from typing import Optional, Any, Dict, List

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Config desde env
NESSIE_API_KEY = os.getenv("NESSIE_API_KEY", "")
NESSIE_BASE_URL = os.getenv("NESSIE_BASE_URL", "https://api.nessieisreal.com")
NESSIE_FALLBACK_URL = os.getenv("NESSIE_FALLBACK_URL", "")  # opcional

DEFAULT_TIMEOUT = 10.0  # segundos por request
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.0  # segundos (exponencial)


class NessieAPIError(Exception):
    pass


class NessieClient:
    def __init__(
        self,
        api_key: str = NESSIE_API_KEY,
        base_url: str = NESSIE_BASE_URL,
        fallback_url: Optional[str] = NESSIE_FALLBACK_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ):
        if not api_key:
            logger.warning("No NESSIE_API_KEY set in env. Requests will likely fail.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.fallback_url = (fallback_url.rstrip("/") if fallback_url else None)
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=self.timeout, headers=self._default_headers())

    def _default_headers(self) -> Dict[str, str]:
        # Nessie sometimes expects apiKey in query param or header depending on setup.
        # We include it as header; update if your Nessie instance expects query param.
        h = {"Accept": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
            h["x-api-key"] = self.api_key  # safe to include both
        return h

    async def close(self):
        await self._client.aclose()

    async def _request(self, method: str, path: str, params: Optional[Dict] = None) -> Any:
        """
        Tries base_url then fallback_url. Retries each endpoint up to max_retries with backoff.
        Returns parsed JSON on success or raises NessieAPIError.
        """
        urls_to_try = [f"{self.base_url}{path}"]
        if self.fallback_url:
            urls_to_try.append(f"{self.fallback_url}{path}")

        last_exc = None
        for url in urls_to_try:
            for attempt in range(1, self.max_retries + 1):
                try:
                    logger.debug("Nessie request attempt %d to %s (params=%s)", attempt, url, params)
                    resp = await self._client.request(method, url, params=params)
                    if resp.status_code >= 500:
                        # server error -> retry
                        logger.warning("Server error %s on %s: %s", resp.status_code, url, resp.text[:200])
                        raise httpx.HTTPStatusError("Server error", request=resp.request, response=resp)
                    if resp.status_code == 429:
                        # rate limited -> wait and retry (respect Retry-After if present)
                        ra = resp.headers.get("Retry-After")
                        wait = float(ra) if ra else BACKOFF_FACTOR * attempt
                        logger.warning("Rate limited, waiting %s seconds", wait)
                        await asyncio.sleep(wait)
                        continue
                    if resp.status_code >= 400:
                        # client error -> don't retry (likely wrong params)
                        logger.error("Client error %s on %s: %s", resp.status_code, url, resp.text[:300])
                        raise NessieAPIError(f"Client error {resp.status_code}: {resp.text}")
                    # success
                    return resp.json()
                except (httpx.RequestError, httpx.HTTPStatusError) as e:
                    last_exc = e
                    backoff = BACKOFF_FACTOR * (2 ** (attempt - 1))
                    logger.info("Request error (attempt %d/%d) to %s: %s — backing off %ss", attempt, self.max_retries, url, str(e), backoff)
                    await asyncio.sleep(backoff)
                    continue
            # if we exhaust retries for this url, try next (fallback)
            logger.info("Exhausted retries for %s, trying next URL if available", url)

        # If we reach here, no url succeeded
        logger.exception("All endpoints failed. Last exception: %s", last_exc)
        raise NessieAPIError(f"All Nessie endpoints failed. Last error: {last_exc}")

    # ---- Convenience API methods ----
    async def get_accounts_by_customer(self, customer_id: str) -> List[Dict]:
        """
        GET /customers/{customer_id}/accounts  (or similar)
        """
        path = f"/customers/{customer_id}/accounts"
        return await self._request("GET", path)

    async def get_account(self, account_id: str) -> Dict:
        path = f"/accounts/{account_id}"
        return await self._request("GET", path)

    async def get_purchases(self, account_id: str, params: Optional[Dict] = None) -> List[Dict]:
        path = f"/accounts/{account_id}/purchases"
        return await self._request("GET", path, params=params)

    async def get_withdrawals(self, account_id: str, params: Optional[Dict] = None) -> List[Dict]:
        path = f"/accounts/{account_id}/withdrawals"
        return await self._request("GET", path, params=params)

    async def get_transfers(self, account_id: str, params: Optional[Dict] = None) -> List[Dict]:
        path = f"/accounts/{account_id}/transfers"
        return await self._request("GET", path, params=params)

    async def get_deposits(self, account_id: str, params: Optional[Dict] = None) -> List[Dict]:
        path = f"/accounts/{account_id}/deposits"
        return await self._request("GET", path, params=params)

    async def get_bills_by_account(self, account_id: str) -> List[Dict]:
        path = f"/accounts/{account_id}/bills"
        return await self._request("GET", path)

    async def get_customer_bills(self, customer_id: str) -> List[Dict]:
        path = f"/customers/{customer_id}/bills"
        return await self._request("GET", path)

    async def get_merchants(self, params: Optional[Dict] = None) -> List[Dict]:
        path = f"/merchants"
        return await self._request("GET", path, params=params)

    async def get_merchant(self, merchant_id: str) -> Dict:
        path = f"/merchants/{merchant_id}"
        return await self._request("GET", path)

    async def get_atms(self, params: Optional[Dict] = None) -> List[Dict]:
        path = f"/atms"
        return await self._request("GET", path, params=params)

    async def get_branches(self, params: Optional[Dict] = None) -> List[Dict]:
        path = f"/branches"
        return await self._request("GET", path, params=params)

    # Optional helper: fetch all txs for a customer's accounts
    async def get_all_transactions_for_customer(self, customer_id: str) -> List[Dict]:
        """
        Ingest helper: obtiene todas las transacciones (purchases, withdrawals, transfers) para todas las cuentas del cliente.
        """
        accounts = await self.get_accounts_by_customer(customer_id)
        out = []
        # accounts may be a dict or list depending on Ressie instance; normalize
        if isinstance(accounts, dict) and "accounts" in accounts:
            accounts = accounts["accounts"]
        for acct in accounts:
            acct_id = acct.get("id") or acct.get("_id") or acct.get("account_id")
            if not acct_id:
                continue
            purchases = await self.get_purchases(acct_id) or []
            withdrawals = await self.get_withdrawals(acct_id) or []
            transfers = await self.get_transfers(acct_id) or []
            deposits = await self.get_deposits(acct_id) or []
            # annotate type and account_id for normalization later
            for p in purchases:
                p["_txn_type"] = "purchase"
                p["_account_id"] = acct_id
            for w in withdrawals:
                w["_txn_type"] = "withdrawal"
                w["_account_id"] = acct_id
            for t in transfers:
                t["_txn_type"] = "transfer"
                t["_account_id"] = acct_id
            for d in deposits:
                d["_txn_type"] = "deposit"
                d["_account_id"] = acct_id
            out.extend(purchases + withdrawals + transfers + deposits)
        return out


# ---- Ejemplo de uso (async) ----
# from app.nessie_client import NessieClient, NessieAPIError
# client = NessieClient()
# async def demo():
#     try:
#         accs = await client.get_accounts_by_customer("123")
#         print(accs)
#     except NessieAPIError as e:
#         print("Error talking to Nessie:", e)
#     finally:
#         await client.close()
#
# import asyncio
# asyncio.run(demo())
