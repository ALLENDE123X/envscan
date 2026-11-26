"""Fetch a published environment from the Prime Intellect Environments Hub.

Uses the same public endpoint `prime env pull` uses
(GET /api/v1/environmentshub/{owner}/{name}/@{version}) and streams the
package tarball straight into an in memory Package. The archive is never
written to disk, so a hub wide scan needs no temp space. The direct storage
URL (package_url) is preferred over the tracked download URL so scans do not
inflate a publisher's download counter.

Only the standard library is used. An API key is not needed for public envs;
if PRIME_API_KEY is set it is sent, as the prime CLI would.
"""
from __future__ import annotations

import ast
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request

from .package import Package

API = os.environ.get("PRIME_API_BASE_URL", "https://api.primeintellect.ai").rstrip("/")
UA = "envscan/0.1 (static scanner)"
MAX_DOWNLOAD = int(os.environ.get("ENVSCAN_MAX_DOWNLOAD", str(600 * 1024 * 1024)))


class HubError(Exception):
    pass


class _Capped:
    """File like wrapper that stops a download after MAX_DOWNLOAD bytes."""

    def __init__(self, raw, cap):
        self.raw, self.cap, self.n = raw, cap, 0
        self.capped = False

    def read(self, n=-1):
        if self.n >= self.cap:
            self.capped = True
            return b""
        if n is None or n < 0:
            n = self.cap - self.n
        b = self.raw.read(min(n, self.cap - self.n))
        self.n += len(b)
        return b


def _headers():
    h = {"User-Agent": UA, "Accept": "application/json"}
    key = os.environ.get("PRIME_API_KEY")
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def _get(url, timeout=60, tries=5, stream=False, headers=None):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers or _headers())
            resp = urllib.request.urlopen(req, timeout=timeout)
            return resp if stream else resp.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                ra = e.headers.get("Retry-After")
                wait = float(ra) if ra and ra.isdigit() else (2 ** i) + random.random()
                time.sleep(min(wait, 60))
                continue
            raise HubError(f"HTTP {e.code} for {url.split('?')[0]}")
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = e
            time.sleep((2 ** i) + random.random())
    raise HubError(f"giving up on {url.split('?')[0]}: {last}")


def details(env_id: str, version: str = "latest") -> dict:
    owner, name = env_id.split("/", 1)
    q = urllib.parse.quote
    raw = _get(f"{API}/api/v1/environmentshub/{q(owner, safe='')}/{q(name, safe='')}/@{q(version, safe='')}")
    d = json.loads(raw)
    d = d.get("data", d)
    md = d.get("metadata")
    if isinstance(md, str):
        try:
            d["metadata"] = ast.literal_eval(md)
        except Exception:
            try:
                d["metadata"] = json.loads(md)
            except Exception:
                d["metadata"] = {}
    return d


def fetch(env_id: str, version: str = "latest") -> Package:
    d = details(env_id, version)
    url = d.get("package_url") or d.get("tracked_package_url")
    if not url:
        raise HubError("no downloadable package")
    resp = _get(url, timeout=120, stream=True, headers={"User-Agent": UA})
    capped = _Capped(resp, MAX_DOWNLOAD)
    try:
        pkg = Package.from_tar_stream(capped, env_id)
    except Exception as e:
        if capped.capped:
            raise HubError(f"package larger than {MAX_DOWNLOAD} bytes")
        raise HubError(f"archive read failed: {type(e).__name__}: {e}")
    finally:
        try:
            resp.close()
        except Exception:
            pass
    if capped.capped:
        pkg.notes.append(f"download capped at {MAX_DOWNLOAD} bytes; later members not scanned")
    keep = ("id", "version_id", "sha256", "semantic_version", "version", "runtime", "verifiers_requirement",
            "updated_at", "created_at", "owner", "visibility")
    pkg.meta = {k: d.get(k) for k in keep if k in d}
    md = d.get("metadata") if isinstance(d.get("metadata"), dict) else {}
    pkg.meta["metadata"] = {"dependencies": md.get("dependencies", []), "tags": md.get("tags", [])}
    pkg.meta["download_bytes"] = capped.n
    return pkg
