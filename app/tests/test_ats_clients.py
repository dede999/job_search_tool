"""Sanity test for ats_clients.py's fetch_smartrecruiters — no real network
calls (this dev sandbox blocks outbound httpx to arbitrary domains, and
api.smartrecruiters.com additionally blocked this session's WebFetch via
robots.txt — see the CONFIDENCE NOTE in ats_clients.py). Mocks httpx.get to
verify: (1) basic field parsing incl. remote-location formatting,
(2) pagination across pages via limit/offset, (3) the full-description
fetch is gated to postings that already pass title/location filters —
same idea as aggregator_clients.fetch_adzuna's gating, so a company with
hundreds of postings doesn't turn into hundreds of extra requests,
(4) a failed detail fetch degrades to an empty description, not a crash,
and (5) "smartrecruiters" is wired into FETCHERS/fetch_company. Run with:
python -m pytest app/tests/test_ats_clients.py
"""
from unittest.mock import patch, MagicMock

from app import ats_clients


def _list_resp(content):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"content": content})
    return resp


def _detail_resp(job_description="", qualifications="", additional=""):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    sections = {}
    if job_description:
        sections["jobDescription"] = {"text": job_description}
    if qualifications:
        sections["qualifications"] = {"text": qualifications}
    if additional:
        sections["additionalInformation"] = {"text": additional}
    resp.json = MagicMock(return_value={"jobAd": {"sections": sections}})
    return resp


def _posting(id="1", name="Software Engineer", city="São Paulo", region="SP", country="Brasil", remote=False):
    return {
        "id": id, "name": name,
        "location": {"city": city, "region": region, "country": country, "remote": remote},
        "releasedDate": "2026-08-01T00:00:00.000Z",
    }


def test_basic_parsing_and_remote_location():
    posting = _posting(remote=True)
    with patch("httpx.get", side_effect=[_list_resp([posting]), _detail_resp("Build things with Python.")]):
        jobs = ats_clients.fetch_smartrecruiters("TestCo", "testco")
    assert len(jobs) == 1
    j = jobs[0]
    assert j["company"] == "TestCo"
    assert j["title"] == "Software Engineer"
    assert j["location"] == "Remote (São Paulo, SP, Brasil)"
    assert j["posted_at"] == "2026-08-01T00:00:00.000Z"
    assert j["url"] == "https://jobs.smartrecruiters.com/testco/1"  # fallback construction, no applyUrl/ref given
    assert "Python" in j["description"]
    print("fetch_smartrecruiters: basic parsing + remote location + description fetched — OK")


def test_prefers_applyurl_or_ref_when_present():
    posting = _posting(id="2")
    posting["applyUrl"] = "https://jobs.smartrecruiters.com/TestCo/oa_1234"
    with patch("httpx.get", side_effect=[_list_resp([posting]), _detail_resp("JD text")]):
        jobs = ats_clients.fetch_smartrecruiters("TestCo", "testco")
    assert jobs[0]["url"] == "https://jobs.smartrecruiters.com/TestCo/oa_1234"
    print("fetch_smartrecruiters: prefers applyUrl over constructed URL — OK")


def test_pagination_across_pages():
    page1 = [_posting(id=str(i)) for i in range(100)]  # full page -> expect another request
    page2 = [_posting(id="100")]  # short page -> stop

    # Only postings that pass title/location trigger a detail fetch; all
    # 101 here are "Software Engineer" / São Paulo, SP -> all gated in, so
    # interleave list/detail responses accordingly.
    responses = [_list_resp(page1)]
    responses += [_detail_resp("JD") for _ in range(100)]
    responses += [_list_resp(page2)]
    responses += [_detail_resp("JD")]

    with patch("httpx.get", side_effect=responses) as mock_get:
        jobs = ats_clients.fetch_smartrecruiters("TestCo", "testco")

    assert len(jobs) == 101
    # 2 list calls (full page then short page) + 101 detail calls
    assert mock_get.call_count == 2 + 101
    # second list call used offset=100
    second_list_call = mock_get.call_args_list[101]  # after 100 detail calls following the first list call
    assert second_list_call.kwargs["params"]["offset"] == 100
    print("fetch_smartrecruiters: pagination via limit/offset — OK")


def test_description_fetch_gated_to_promising_postings():
    promising = _posting(id="1", name="Software Engineer", city="São Paulo", region="SP", country="Brasil")
    not_promising_title = _posting(id="2", name="Marketing Operations Manager")
    not_promising_location = _posting(id="3", name="Software Engineer", city="Warsaw", region="", country="Poland")

    with patch("httpx.get", side_effect=[
        _list_resp([promising, not_promising_title, not_promising_location]),
        _detail_resp("Only fetched for the promising one"),
    ]) as mock_get:
        jobs = ats_clients.fetch_smartrecruiters("TestCo", "testco")

    assert mock_get.call_count == 2  # 1 list call + exactly 1 detail call
    by_id = {j["url"].rsplit("/", 1)[-1]: j for j in jobs}
    assert "Only fetched" in by_id["1"]["description"]
    assert by_id["2"]["description"] == ""
    assert by_id["3"]["description"] == ""
    print("fetch_smartrecruiters: full-description fetch gated to promising postings only — OK")


def test_description_fetch_failure_degrades_gracefully():
    posting = _posting()
    with patch("httpx.get", side_effect=[_list_resp([posting]), Exception("500 server error")]):
        jobs = ats_clients.fetch_smartrecruiters("TestCo", "testco")
    assert jobs[0]["description"] == ""
    print("fetch_smartrecruiters: detail-fetch failure -> empty description, no crash — OK")


def test_wired_into_fetchers_and_fetch_company():
    assert ats_clients.FETCHERS["smartrecruiters"] is ats_clients.fetch_smartrecruiters
    with patch("httpx.get", side_effect=[_list_resp([]), ]):
        jobs = ats_clients.fetch_company({"name": "TestCo", "ats": "smartrecruiters", "slug": "testco"})
    assert jobs == []
    print("FETCHERS/fetch_company: smartrecruiters wired in correctly — OK")


if __name__ == "__main__":
    test_basic_parsing_and_remote_location()
    test_prefers_applyurl_or_ref_when_present()
    test_pagination_across_pages()
    test_description_fetch_gated_to_promising_postings()
    test_description_fetch_failure_degrades_gracefully()
    test_wired_into_fetchers_and_fetch_company()
    print("\nAll assertions passed.")