"""Thin clients for each ATS's public job-board API.

Every function returns a list of plain dicts with a common shape:
    {"company": str, "title": str, "location": str, "url": str,
     "posted_at": str|None, "description": str}

`description` is HTML or plain text, whatever the ATS gives us, and is
consumed by filters.jd_stack_mismatch() for the JD-based stack dealbreaker
check. It's fetched from the SAME request as the listing wherever the ATS
supports that (Greenhouse ?content=true, Ashby/Lever include it by
default) — no extra per-job request needed, so this doesn't multiply your
call volume. Workable's widget API may or may not include it depending on
account; if absent, `description` is just "" and the stack check is
skipped for that job (title/location filters still apply).

NOTE ON THIS SANDBOX: outbound HTTP to arbitrary domains (e.g.
boards-api.greenhouse.io) is blocked by this cloud environment's network
allowlist — that's a property of THIS dev sandbox, not of the ATS APIs
themselves (verified working via WebFetch during development). Run this
module on a machine/server with normal internet access — your laptop, a
cron box, a small VM — and it will work as-is.

CONFIDENCE NOTE on fetch_smartrecruiters specifically (2026-08-12): unlike
Greenhouse/Ashby/Workable/Lever, which were each confirmed against a real
live response during development, SmartRecruiters' shape here is built
from their docs (developers.smartrecruiters.com) plus two independent
third-party integrations that describe the same shape (jobspipe.dev,
an Apify scraper) — this sandbox's WebFetch got blocked by
api.smartrecruiters.com's robots.txt, so it's NOT been hit live the way
the others were. Verify it the same way Coveo/Treewalk's slugs got
verified: `python -m app.main --company <slug>` against a real
SmartRecruiters company before trusting it in a real run.
"""
import httpx

from app import filters

USER_AGENT = "job-search-pipeline/0.1 (personal use)"
TIMEOUT = 20.0


def fetch_greenhouse(company_display_name: str, slug: str) -> list[dict]:
    url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"
    resp = httpx.get(
        url, params={"content": "true"}, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT
    )
    resp.raise_for_status()
    data = resp.json()
    jobs = []
    for j in data.get("jobs", []):
        jobs.append({
            "company": company_display_name,
            "title": j.get("title", ""),
            "location": (j.get("location") or {}).get("name", ""),
            "url": j.get("absolute_url", ""),
            "posted_at": j.get("first_published") or j.get("updated_at"),
            "description": j.get("content", ""),  # HTML
        })
    return jobs


def fetch_ashby(company_display_name: str, slug: str) -> list[dict]:
    # Ashby's public posting-API endpoint (no auth needed for public boards).
    url = f"https://api.ashbyhq.com/posting-api/job-board/{slug}"
    resp = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    jobs = []
    for j in data.get("jobs", []):
        loc = j.get("location") or j.get("locationName") or ""
        jobs.append({
            "company": company_display_name,
            "title": j.get("title", ""),
            "location": loc,
            "url": j.get("jobUrl") or j.get("applyUrl", ""),
            "posted_at": j.get("publishedAt"),
            "description": j.get("descriptionPlain") or j.get("descriptionHtml") or "",
        })
    return jobs


def fetch_workable(company_display_name: str, slug: str) -> list[dict]:
    # Workable's public widget API.
    url = f"https://apply.workable.com/api/v1/widget/accounts/{slug}?details=true"
    resp = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    jobs = []
    for j in data.get("jobs", []):
        loc = j.get("location") or {}
        loc_str = ", ".join(filter(None, [loc.get("city"), loc.get("region"), loc.get("country")]))
        if loc.get("workplace") == "remote":
            loc_str = f"Remote ({loc_str})" if loc_str else "Remote"
        jobs.append({
            "company": company_display_name,
            "title": j.get("title", ""),
            "location": loc_str,
            "url": j.get("url") or j.get("shortlink", ""),
            "posted_at": j.get("published_on") or j.get("created_at"),
            # Workable's widget API doesn't reliably include a description
            # field across all accounts — treat missing as unknown, not
            # as "no dealbreaker language", filters.py handles empty safely.
            "description": j.get("description", ""),
        })
    return jobs


def fetch_lever(company_display_name: str, slug: str) -> list[dict]:
    # Lever's public postings API.
    url = f"https://api.lever.co/v0/postings/{slug}?mode=json"
    resp = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    jobs = []
    for j in data:
        cats = j.get("categories", {}) or {}
        loc = cats.get("location", "")
        all_locs = cats.get("allLocations") or []
        if all_locs:
            loc = ", ".join(all_locs)
        jobs.append({
            "company": company_display_name,
            "title": j.get("text", ""),
            "location": loc,
            "url": j.get("hostedUrl", ""),
            "posted_at": j.get("createdAt"),  # epoch millis
            "description": j.get("descriptionPlain") or j.get("description", ""),
        })
    return jobs


def _fetch_smartrecruiters_description(slug: str, posting_id: str) -> str:
    """SmartRecruiters' list endpoint (fetch_smartrecruiters below) doesn't
    include the JD text — only the per-posting detail endpoint does, one
    extra request per job. Best-effort: any failure (private board, 404,
    timeout) just means no description, not a crashed run; filters.py
    treats an empty description as "unknown, don't reject on stack alone"."""
    try:
        resp = httpx.get(
            f"https://api.smartrecruiters.com/v1/companies/{slug}/postings/{posting_id}",
            headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT,
        )
        resp.raise_for_status()
        sections = (resp.json().get("jobAd") or {}).get("sections") or {}
        parts = [
            (sections.get(key) or {}).get("text", "")
            for key in ("jobDescription", "qualifications", "additionalInformation")
        ]
        return "\n\n".join(p for p in parts if p)
    except Exception:
        return ""


def fetch_smartrecruiters(company_display_name: str, slug: str) -> list[dict]:
    # SmartRecruiters' public Posting API — only enabled per-account (not
    # every customer turns it on), same "might just 404" caveat as the
    # other ATSes here. Paginated via limit/offset, 100 postings per page.
    jobs = []
    offset = 0
    limit = 100
    while True:
        url = f"https://api.smartrecruiters.com/v1/companies/{slug}/postings"
        resp = httpx.get(
            url, params={"limit": limit, "offset": offset},
            headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("content", [])
        if not content:
            break

        for p in content:
            loc = p.get("location") or {}
            loc_str = ", ".join(filter(None, [loc.get("city"), loc.get("region"), loc.get("country")]))
            if loc.get("remote"):
                loc_str = f"Remote ({loc_str})" if loc_str else "Remote"

            title = p.get("name", "")
            posting_id = p.get("id", "")
            job_url = p.get("applyUrl") or p.get("ref") or (
                f"https://jobs.smartrecruiters.com/{slug}/{posting_id}" if posting_id else ""
            )
            if not job_url:
                # No applyUrl/ref/id to build any identifier from — skip
                # rather than store url="", which would make dedup.is_new()
                # treat every subsequent url-less posting as a duplicate of
                # the first one (an exact-match dedup key collision).
                continue

            description = ""
            # Only worth the extra per-job request (see
            # _fetch_smartrecruiters_description) for postings that
            # already look like real candidates — same gating idea as
            # aggregator_clients.fetch_adzuna uses for its full-JD fetch,
            # so a company with hundreds of postings doesn't turn into
            # hundreds of extra requests for roles that'd get filtered
            # out on title/location alone anyway.
            if posting_id and filters.title_is_relevant(title) and filters.location_is_allowed(loc_str):
                description = _fetch_smartrecruiters_description(slug, posting_id)

            jobs.append({
                "company": company_display_name,
                "title": title,
                "location": loc_str,
                "url": job_url,
                "posted_at": p.get("releasedDate"),
                "description": description,
            })

        if len(content) < limit:
            break
        offset += limit
    return jobs

# --- Gupy ----------------------------------------------------------------
# Gupy is the dominant ATS in Brazil. Individual career pages
# (<empresa>.gupy.io) don't expose a JSON API, but Gupy's public job portal
# does, and it can be filtered by careerPageName — so one endpoint serves
# both the per-company fetcher below and aggregator_clients.fetch_gupy
# (keyword search across every Gupy company).
#
# Verified live 2026-09-28:
#   - jobName is a substring search on the job TITLE only (not the JD) —
#     "golang" returns ~0, "go" returns hundreds of unrelated titles.
#   - careerPageName must match exactly; partial names return nothing.
#   - limit > 100 returns 400 Bad Request.
#   - pagination.total is NOT reliable (reports 100 when limit=100), so we
#     page until a short page instead of trusting it.
#   - description comes in the listing itself (full JD, no extra request).
#   - some postings omit workplaceType; remote ones usually have no
#     city/state, just country.
GUPY_PORTAL_URL = "https://employability-portal.gupy.io/api/v1/jobs"
GUPY_PAGE_SIZE = 100
GUPY_MAX_PAGES_PER_COMPANY = 20  # safety cap: 2000 postings for one company

_GUPY_CONTRACT_LABELS = {
    "vacancy_type_effective": "CLT",
    "vacancy_legal_entity": "PJ",
    "vacancy_type_temporary": "Temporário",
    "vacancy_type_internship": "Estágio",
    "vacancy_type_apprentice": "Jovem aprendiz",
    "vacancy_type_talent_pool": "Banco de talentos",
    "vacancy_type_associate": "Associado",
    "vacancy_type_autonomous": "Autônomo",
    "vacancy_type_freelancer": "Freelancer",
}


def _gupy_location(j: dict) -> str:
    parts = ", ".join(filter(None, [j.get("city"), j.get("state"), j.get("country")]))
    workplace = j.get("workplaceType") or ("remote" if j.get("isRemoteWork") else "")
    if workplace == "remote":
        return f"Remote ({parts})" if parts else "Remote"
    if workplace == "hybrid":
        return f"Hybrid ({parts})" if parts else "Hybrid"
    return parts


def parse_gupy_job(j: dict, company_display_name: str | None = None) -> dict:
    """Converts one Gupy portal posting to the pipeline's common job shape.
    The contract type (CLT/PJ/...) has no slot in that shape, so it's
    prepended to the description — that way ai_evaluate.py sees it."""
    contract = _GUPY_CONTRACT_LABELS.get(j.get("type", ""), j.get("type") or "")
    description = j.get("description") or ""
    if contract:
        description = f"[Contrato: {contract}]\n\n{description}"
    return {
        "company": company_display_name or j.get("careerPageName") or "Unknown",
        "title": j.get("name", ""),
        "location": _gupy_location(j),
        "url": j.get("jobUrl") or "",
        "posted_at": j.get("publishedDate"),
        "description": description,
    }


def fetch_gupy_postings(params: dict, max_pages: int) -> list[dict]:
    """Raw portal postings for `params` (jobName, careerPageName, state,
    workplaceType, ...), paginated by limit/offset. Shared with
    aggregator_clients.fetch_gupy."""
    postings: list[dict] = []
    for page in range(max_pages):
        resp = httpx.get(
            GUPY_PORTAL_URL,
            params={**params, "limit": GUPY_PAGE_SIZE, "offset": page * GUPY_PAGE_SIZE},
            headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json().get("data") or []
        postings.extend(data)
        if len(data) < GUPY_PAGE_SIZE:
            break
    return postings


def fetch_gupy(company_display_name: str, slug: str) -> list[dict]:
    # For Gupy, `slug` is the company's career page NAME exactly as Gupy
    # shows it (e.g. "Instituto de Pesquisas ELDORADO"), not the subdomain —
    # the portal API can only filter by name. Find it in any of the
    # company's postings on portal.gupy.io, or in the careerPageName field
    # of an aggregator result.
    postings = fetch_gupy_postings({"careerPageName": slug}, GUPY_MAX_PAGES_PER_COMPANY)
    return [
        parse_gupy_job(p, company_display_name)
        for p in postings
        if p.get("careerPageName") == slug and p.get("jobUrl")
    ]

FETCHERS = {
    "greenhouse": fetch_greenhouse,
    "ashby": fetch_ashby,
    "workable": fetch_workable,
    "lever": fetch_lever,
    "smartrecruiters": fetch_smartrecruiters,
    "gupy": fetch_gupy,
}


def fetch_company(company: dict) -> list[dict]:
    fetcher = FETCHERS.get(company["ats"])
    if fetcher is None:
        raise ValueError(f"No fetcher for ATS type: {company['ats']}")
    jobs = fetcher(company["name"], company["slug"])
    for j in jobs:
        j["source"] = company["ats"]
    return jobs
