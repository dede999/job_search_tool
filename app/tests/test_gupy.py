"""Tests for the Gupy fetchers (ats_clients.fetch_gupy and
aggregator_clients.fetch_gupy) — no real network calls, httpx.get is
mocked. Verifies: (1) field parsing incl. remote/hybrid location and the
contract type prepended to the description, (2) pagination stops on a
short page instead of trusting pagination.total, (3) the per-company
fetcher drops postings from other career pages, (4) the aggregator strips
max_pages before calling the API, and (5) "gupy" is wired into both
FETCHERS dicts. Run with: python -m pytest app/tests/test_gupy.py
"""
from unittest.mock import patch, MagicMock

from app import aggregator_clients, ats_clients


def _resp(data):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"data": data, "pagination": {"total": 100}})
    return resp


def _posting(id=1, name="Pessoa Desenvolvedora Backend Sênior", page="TestCo",
             workplace="remote", city="", state="", type_="vacancy_legal_entity"):
    return {
        "id": id, "name": name, "careerPageName": page,
        "description": "Go, PostgreSQL e Kafka.",
        "type": type_, "publishedDate": "2026-09-28T14:06:52.955Z",
        "workplaceType": workplace, "city": city, "state": state, "country": "Brasil",
        "jobUrl": f"https://testco.gupy.io/job/{id}",
    }


def test_parsing_remote_and_contract():
    with patch("httpx.get", return_value=_resp([_posting()])):
        jobs = ats_clients.fetch_gupy("TestCo", "TestCo")
    assert len(jobs) == 1
    j = jobs[0]
    assert j["company"] == "TestCo"
    assert j["title"] == "Pessoa Desenvolvedora Backend Sênior"
    assert j["location"] == "Remote (Brasil)"
    assert j["url"] == "https://testco.gupy.io/job/1"
    assert j["description"].startswith("[Contrato: PJ]")
    assert "PostgreSQL" in j["description"]


def test_hybrid_location_and_missing_workplace():
    hybrid = _posting(workplace="hybrid", city="São Paulo", state="São Paulo", type_="vacancy_type_effective")
    onsite = _posting(id=2, workplace=None, city="Curitiba", state="Paraná")
    del onsite["workplaceType"]
    with patch("httpx.get", return_value=_resp([hybrid, onsite])):
        jobs = ats_clients.fetch_gupy("TestCo", "TestCo")
    assert jobs[0]["location"] == "Hybrid (São Paulo, São Paulo, Brasil)"
    assert jobs[0]["description"].startswith("[Contrato: CLT]")
    assert jobs[1]["location"] == "Curitiba, Paraná, Brasil"


def test_pagination_stops_on_short_page():
    page1 = [_posting(id=i) for i in range(100)]
    page2 = [_posting(id=100)]
    get = MagicMock(side_effect=[_resp(page1), _resp(page2)])
    with patch("httpx.get", get):
        jobs = ats_clients.fetch_gupy("TestCo", "TestCo")
    assert len(jobs) == 101
    assert get.call_count == 2
    assert get.call_args_list[1].kwargs["params"]["offset"] == 100


def test_company_fetch_drops_other_career_pages():
    other = _posting(id=2, page="OtherCo")
    with patch("httpx.get", return_value=_resp([_posting(), other])):
        jobs = ats_clients.fetch_gupy("TestCo", "TestCo")
    assert [j["url"] for j in jobs] == ["https://testco.gupy.io/job/1"]


def test_aggregator_strips_max_pages_and_uses_career_page_name():
    get = MagicMock(return_value=_resp([_posting(page="Empresa X")]))
    with patch("httpx.get", get):
        jobs = aggregator_clients.fetch_aggregator(
            {"type": "gupy", "params": {"jobName": "desenvolvedor", "workplaceType": "remote", "max_pages": 3}}
        )
    sent = get.call_args.kwargs["params"]
    assert "max_pages" not in sent
    assert sent["jobName"] == "desenvolvedor" and sent["workplaceType"] == "remote"
    assert jobs[0]["company"] == "Empresa X"
    assert jobs[0]["source"] == "gupy"


def test_wired_into_fetchers():
    assert ats_clients.FETCHERS["gupy"] is ats_clients.fetch_gupy
    assert aggregator_clients.FETCHERS["gupy"] is aggregator_clients.fetch_gupy
 
