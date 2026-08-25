"""Tests for onepager assembly.

Everything here runs against fakes — no Postgres, no LLM. What is being pinned
down is the sourcing contract: which stores get read, what makes a page a draft,
and that curated material is never silently replaced by web research.
"""

import json
from contextlib import contextmanager

import pytest

import onepager


class FakeCursor:
    """Answers the three SELECTs gather_source_material issues, by table name."""

    def __init__(self, rows):
        self._rows = rows
        self._last = None
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        for table in ("portfolio_analyses", "stock_overviews", "thesis_scorecard_data",
                      "stock_onepagers"):
            if table in sql:
                self._last = self._rows.get(table)
                return
        self._last = None

    def fetchone(self):
        return self._last


def make_get_db(rows):
    @contextmanager
    def get_db(commit=False):
        yield (None, FakeCursor(rows))
    return get_db


def parse_analysis_data(row):
    """Stand-in for app_v3._parse_analysis_data."""
    analysis = row.get("analysis") or {}
    if isinstance(analysis, str):
        analysis = json.loads(analysis)
    return {
        "ticker": row.get("ticker"),
        "company": row.get("company"),
        "date_str": "2026-08-01",
        "thesis": analysis.get("thesis", {}),
        "signposts": analysis.get("signposts", []),
        "threats": analysis.get("threats", []),
        "conclusion": analysis.get("conclusion", ""),
    }


THESIS_ROW = {
    "ticker": "DE",
    "company": "Deere & Company",
    "analysis": {
        "thesis": {"pillars": [{"title": "Precision ag", "confidence": "high"}]},
        "signposts": [{"metric": "Engaged acres", "target": "600M by 2030"}],
        "threats": [{"title": "Ag downcycle"}],
        "conclusion": "Great franchise, cyclical timing.",
    },
}

OVERVIEW_ROW = {
    "ticker": "DE",
    "company_name": "Deere & Company",
    "company_overview": "Makes ag and construction equipment.",
    "business_model": "Equipment, financing, parts, software.",
    "business_mix": "PPA ~45-50%, SAT ~20-25%, C&F ~25-30%.",
    "opportunities": "Precision ag, autonomy.",
    "risks": "Commodity cycle.",
    "conclusion": "Quality cyclical.",
}


def make_call_llm(capture=None, payload=None):
    def call_llm(*, messages, system, tier, max_tokens, timeout, **keys):
        if capture is not None:
            capture["system"] = system
            capture["user"] = messages[0]["content"]
            capture["tier"] = tier
        body = payload or {"ticker": "DE", "company": "Deere & Company",
                           "tagline": "The world runs."}
        return {
            "text": json.dumps(body),
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "provider": "anthropic",
            "model": "claude-test",
        }
    return call_llm


def extract_json(text):
    return json.loads(text)


# --------------------------------------------------------------------------
# sourcing
# --------------------------------------------------------------------------

def test_gathers_thesis_and_overview():
    get_db = make_get_db({"portfolio_analyses": THESIS_ROW, "stock_overviews": OVERVIEW_ROW})
    text, facts = onepager.gather_source_material("de", get_db, parse_analysis_data)

    assert facts["ticker"] == "DE"
    assert facts["has_thesis"] and facts["has_overview"]
    assert facts["sources"] == ["thesis", "overview"]
    assert facts["company"] == "Deere & Company"
    assert "EXISTING INVESTMENT THESIS" in text
    assert "COMPANY OVERVIEW" in text
    assert "PPA ~45-50%" in text


def test_missing_stores_are_not_fatal():
    get_db = make_get_db({})
    text, facts = onepager.gather_source_material("XYZ", get_db, parse_analysis_data)
    assert text == ""
    assert facts["sources"] == []
    assert not facts["has_thesis"] and not facts["has_overview"]


# --------------------------------------------------------------------------
# draft flagging — the bit that keeps AI pages out of curated positions
# --------------------------------------------------------------------------

def test_curated_source_is_not_a_draft():
    get_db = make_get_db({"portfolio_analyses": THESIS_ROW})
    data, facts = onepager.build_onepager(
        "DE", get_db=get_db, parse_analysis_data=parse_analysis_data,
        call_llm=make_call_llm(), extract_json=extract_json,
        research_fn=lambda t: "should not be called",
    )
    assert facts["is_draft"] is False
    assert data["meta"]["is_draft"] is False
    assert "research" not in data["meta"]["sources"]


def test_research_only_page_is_marked_draft():
    get_db = make_get_db({})
    data, facts = onepager.build_onepager(
        "NVDA", get_db=get_db, parse_analysis_data=parse_analysis_data,
        call_llm=make_call_llm(), extract_json=extract_json,
        research_fn=lambda t: f"researched notes for {t}",
    )
    assert facts["is_draft"] is True
    assert data["meta"]["is_draft"] is True
    assert "research" in data["meta"]["sources"]


def test_no_source_and_no_research_fn_raises():
    get_db = make_get_db({})
    with pytest.raises(LookupError):
        onepager.build_onepager(
            "XYZ", get_db=get_db, parse_analysis_data=parse_analysis_data,
            call_llm=make_call_llm(), extract_json=extract_json,
            research_fn=None,
        )


def test_research_fn_not_called_when_curated_data_exists():
    get_db = make_get_db({"portfolio_analyses": THESIS_ROW})
    calls = []
    onepager.build_onepager(
        "DE", get_db=get_db, parse_analysis_data=parse_analysis_data,
        call_llm=make_call_llm(), extract_json=extract_json,
        research_fn=lambda t: calls.append(t) or "notes",
    )
    assert calls == []


def test_force_research_augments_but_keeps_curated_primacy():
    get_db = make_get_db({"portfolio_analyses": THESIS_ROW})
    capture = {}
    data, facts = onepager.build_onepager(
        "DE", get_db=get_db, parse_analysis_data=parse_analysis_data,
        call_llm=make_call_llm(capture), extract_json=extract_json,
        research_fn=lambda t: "fresh web notes",
        force_research=True,
    )
    prompt = capture["user"]
    # Curated block must still be present, and ordered ahead of the research.
    assert "EXISTING INVESTMENT THESIS" in prompt
    assert "WEB RESEARCH" in prompt
    assert prompt.index("EXISTING INVESTMENT THESIS") < prompt.index("WEB RESEARCH")
    assert "the curated material wins" in prompt
    # Forcing research on a curated name must not demote it to a draft.
    assert facts["is_draft"] is False


# --------------------------------------------------------------------------
# prompt + output contract
# --------------------------------------------------------------------------

def test_prompt_carries_schema_and_no_invention_rule():
    get_db = make_get_db({"portfolio_analyses": THESIS_ROW})
    capture = {}
    onepager.build_onepager(
        "DE", get_db=get_db, parse_analysis_data=parse_analysis_data,
        call_llm=make_call_llm(capture), extract_json=extract_json,
        research_fn=lambda t: "notes",
    )
    assert "NEVER invent a number" in capture["system"]
    assert "core_question" in capture["user"]
    assert "signposts" in capture["user"]


def test_ticker_is_forced_from_caller_not_model_echo():
    get_db = make_get_db({"portfolio_analyses": THESIS_ROW})
    call_llm = make_call_llm(payload={"ticker": "WRONG", "company": ""})
    data, _ = onepager.build_onepager(
        "de", get_db=get_db, parse_analysis_data=parse_analysis_data,
        call_llm=call_llm, extract_json=extract_json,
        research_fn=lambda t: "notes",
    )
    assert data["ticker"] == "DE"
    # Company falls back to what the DB knows when the model omits it.
    assert data["company"] == "Deere & Company"


def test_non_dict_llm_output_is_rejected():
    get_db = make_get_db({"portfolio_analyses": THESIS_ROW})

    def call_llm(**kwargs):
        return {"text": "[1,2,3]", "provider": "x", "model": "y", "usage": {}}

    with pytest.raises(ValueError):
        onepager.build_onepager(
            "DE", get_db=get_db, parse_analysis_data=parse_analysis_data,
            call_llm=call_llm, extract_json=extract_json,
            research_fn=lambda t: "notes",
        )


def test_long_source_fields_are_truncated_not_dropped():
    huge = {"ticker": "DE", "company": "Deere", "analysis": {"thesis": {"x": "y" * 40000}}}
    get_db = make_get_db({"portfolio_analyses": huge})
    text, _ = onepager.gather_source_material("DE", get_db, parse_analysis_data)
    assert "[truncated]" in text
    assert len(text) < 40000


# --------------------------------------------------------------------------
# depth variants
# --------------------------------------------------------------------------

def test_depth_directive_reaches_the_prompt():
    get_db = make_get_db({"portfolio_analyses": THESIS_ROW})
    capture = {}
    onepager.build_onepager(
        "DE", get_db=get_db, parse_analysis_data=parse_analysis_data,
        call_llm=make_call_llm(capture), extract_json=extract_json,
        research_fn=lambda t: "notes", depth="brief",
    )
    assert "DEPTH: BRIEF" in capture["user"]
    assert "ONE printed page" in capture["user"]


def test_each_depth_sends_a_distinct_directive():
    seen = {}
    for d in ("brief", "standard", "deep"):
        get_db = make_get_db({"portfolio_analyses": THESIS_ROW})
        capture = {}
        onepager.build_onepager(
            "DE", get_db=get_db, parse_analysis_data=parse_analysis_data,
            call_llm=make_call_llm(capture), extract_json=extract_json,
            research_fn=lambda t: "notes", depth=d,
        )
        seen[d] = capture["user"]
    assert len({v for v in seen.values()}) == 3


def test_unknown_depth_falls_back_to_standard():
    get_db = make_get_db({"portfolio_analyses": THESIS_ROW})
    capture = {}
    data, _ = onepager.build_onepager(
        "DE", get_db=get_db, parse_analysis_data=parse_analysis_data,
        call_llm=make_call_llm(capture), extract_json=extract_json,
        research_fn=lambda t: "notes", depth="nonsense",
    )
    assert "DEPTH: STANDARD" in capture["user"]
    # meta records the resolved depth, not the bogus one the caller passed.
    assert data["meta"]["depth"] == "standard"


def test_depth_recorded_in_meta():
    get_db = make_get_db({"portfolio_analyses": THESIS_ROW})
    data, _ = onepager.build_onepager(
        "DE", get_db=get_db, parse_analysis_data=parse_analysis_data,
        call_llm=make_call_llm(), extract_json=extract_json,
        research_fn=lambda t: "notes", depth="deep",
    )
    assert data["meta"]["depth"] == "deep"


def test_brief_is_told_to_drop_summaries():
    """The section summaries are what make the page scroll — Brief must cut them."""
    d = onepager.depth_directive("brief")
    assert "Omit investment_thesis.summary" in d
    assert "company_overview.summary" in d


def test_deep_still_forbids_inventing_figures():
    """Depth must buy reasoning, never fabricated numbers."""
    assert "never invent a figure" in onepager.depth_directive("deep")


# --------------------------------------------------------------------------
# thesis diff — a refresh must never silently overwrite curated judgement
# --------------------------------------------------------------------------

CUR = {
    "thesis": {"pillars": [{"title": "Precision ag", "detail": "a"},
                           {"title": "C&F", "detail": "b"}]},
    "signposts": [{"signpost": "Engaged acres", "target": "600M"}],
    "threats": [{"title": "Ag downcycle", "watch_for": "commodity prices"}],
    "conclusion": "Great franchise, cyclical timing.",
}


def test_identical_thesis_reports_no_changes():
    """A refresh that moved nothing must say so, not show an empty dialog."""
    d = onepager.diff_thesis(CUR, CUR)
    assert d["has_changes"] is False
    assert d["counts"] == {"added": 0, "removed": 0, "changed": 0}


def test_detects_added_removed_and_reworded():
    new = {
        "thesis": {"pillars": [{"title": "Precision ag", "detail": "REWORDED"},
                               {"title": "Autonomy", "detail": "c"}]},
        "signposts": [{"signpost": "Engaged acres", "target": "600M"}],
        "threats": [],
        "conclusion": "Great franchise, cyclical timing.",
    }
    d = onepager.diff_thesis(CUR, new)
    assert [p["title"] for p in d["pillars"]["added"]] == ["Autonomy"]
    assert [p["title"] for p in d["pillars"]["removed"]] == ["C&F"]
    assert d["pillars"]["changed"][0]["label"] == "Precision ag"
    # A row whose body is untouched is neither changed nor dropped.
    assert d["signposts"]["unchanged"] == 1
    assert len(d["threats"]["removed"]) == 1


def test_changed_row_keeps_before_and_after():
    """Approval is only meaningful if the analyst can see both sides."""
    new = {"thesis": {"pillars": [{"title": "Precision ag", "detail": "NEW"}]}}
    d = onepager.diff_thesis(CUR, new)
    ch = d["pillars"]["changed"][0]
    assert ch["before"]["detail"] == "a"
    assert ch["after"]["detail"] == "NEW"


def test_label_matching_is_case_and_space_insensitive():
    new = {"thesis": {"pillars": [{"title": "  precision   AG ", "detail": "a"}]}}
    d = onepager.diff_thesis(CUR, new)
    # Same pillar, same body — must pair up rather than read as add + remove.
    assert not d["pillars"]["added"]
    assert d["pillars"]["unchanged"] == 1


def test_empty_candidate_does_not_read_as_mass_deletion_of_nothing():
    d = onepager.diff_thesis({}, {})
    assert d["has_changes"] is False


def test_conclusion_change_alone_counts_as_a_change():
    new = dict(CUR, conclusion="Materially different view.")
    d = onepager.diff_thesis(CUR, new)
    assert d["conclusion"]["changed"] is True
    assert d["has_changes"] is True
    assert d["counts"]["changed"] == 0     # conclusion is tracked separately


def test_unlabelled_rows_still_pair_up():
    """Signposts sometimes arrive without a title key."""
    cur = {"signposts": [{"metric": "Acres", "target": "600M"}]}
    new = {"signposts": [{"metric": "Acres", "target": "650M"}]}
    d = onepager.diff_thesis(cur, new)
    assert len(d["signposts"]["changed"]) == 1
    assert not d["signposts"]["added"] and not d["signposts"]["removed"]


# --------------------------------------------------------------------------
# change classification — what the analyst must read before approving
# --------------------------------------------------------------------------

DIFF = {
    "pillars": {"added": [{"title": "Oak Street inflects"}], "removed": [],
                "changed": [{"label": "HCB pricing discipline",
                             "before": {"detail": "holds"}, "after": {"detail": "eroding"}}],
                "unchanged": 0},
    "signposts": {"added": [], "removed": [],
                  "changed": [{"label": "HCB MLR", "before": {"v": "89.5%"}, "after": {"v": "91.2%"}}],
                  "unchanged": 0},
    "threats": {"added": [], "removed": [], "changed": [], "unchanged": 0},
    "conclusion": {"changed": False},
}


def _classifier(mapping, capture=None):
    def call_llm(*, messages, system, tier, max_tokens, timeout, **keys):
        if capture is not None:
            capture["system"] = system
            capture["user"] = messages[0]["content"]
        return {"text": json.dumps({"classifications": [
            {"id": k, "layer": v, "why": "because"} for k, v in mapping.items()
        ]}), "provider": "x", "model": "y", "usage": {}}
    return call_llm


def test_classification_groups_and_counts_layers():
    mapping = {
        "pillars:added:Oak Street inflects": "structural",
        "pillars:changed:HCB pricing discipline": "structural_challenge",
        "signposts:changed:HCB MLR": "cyclical",
    }
    out = onepager.classify_diff(DIFF, _classifier(mapping), json.loads)
    assert out["layerCounts"] == {"structural": 1, "cyclical": 1, "structural_challenge": 1}
    # needsReview is what gates a careful read: structural + challenges, not churn.
    assert out["needsReview"] == 2


def test_all_cyclical_run_needs_no_careful_review():
    """The common quarterly refresh should be approvable at a glance."""
    mapping = {k: "cyclical" for k in
               ["pillars:added:Oak Street inflects",
                "pillars:changed:HCB pricing discipline",
                "signposts:changed:HCB MLR"]}
    out = onepager.classify_diff(DIFF, _classifier(mapping), json.loads)
    assert out["needsReview"] == 0


def test_unknown_layer_falls_back_rather_than_dropping_the_change():
    mapping = {"pillars:added:Oak Street inflects": "nonsense"}
    out = onepager.classify_diff(DIFF, _classifier(mapping), json.loads)
    # every change is still accounted for
    assert len(out["layers"]) == 3
    assert out["layers"]["pillars:added:Oak Street inflects"]["layer"] == "cyclical"


def test_classifier_failure_leaves_the_diff_usable():
    """Classification is a review aid; losing it must never block approval."""
    def boom(**kwargs):
        raise RuntimeError("model down")
    out = onepager.classify_diff(DIFF, boom, json.loads)
    assert out is DIFF
    assert "layers" not in out


def test_empty_diff_is_not_sent_to_the_model():
    calls = []
    def call_llm(**kwargs):
        calls.append(1)
        return {"text": "{}", "provider": "x", "model": "y", "usage": {}}
    empty = {"pillars": {"added": [], "removed": [], "changed": [], "unchanged": 3},
             "signposts": {"added": [], "removed": [], "changed": [], "unchanged": 2},
             "threats": {"added": [], "removed": [], "changed": [], "unchanged": 1},
             "conclusion": {"changed": False}}
    onepager.classify_diff(empty, call_llm, json.loads)
    assert calls == []


def test_prompt_carries_the_accumulation_rule():
    """The reason for a third state: slow drift must not read as routine churn."""
    capture = {}
    onepager.classify_diff(DIFF, _classifier({}, capture), json.loads)
    assert "structural_challenge" in capture["system"]
    assert "consecutive" in capture["system"] or "already visible" in capture["system"]
    assert "does not mean frozen" in capture["system"]


def test_ambiguity_resolves_toward_attention():
    assert "prefer the higher-attention class" in onepager.CLASSIFY_SYSTEM


def test_existing_thesis_is_given_as_context_for_direction():
    capture = {}
    onepager.classify_diff(DIFF, _classifier({}, capture), json.loads,
                           current_analysis={"thesis": {"pillars": [{"title": "MLR improves"}]}})
    assert "EXISTING THESIS" in capture["user"]


# --------------------------------------------------------------------------
# selective approval — accept some changes, reject others, edit any
# --------------------------------------------------------------------------

SEL_CUR = {
    "thesis": {"pillars": [{"title": "A", "d": 1}, {"title": "B", "d": 2}]},
    "signposts": [{"signpost": "S1", "v": "old"}],
    "threats": [{"title": "T1"}],
    "conclusion": "old",
}

SEL_DIFF = {
    "pillars": {"added": [{"title": "C"}],
                "removed": [{"title": "B", "d": 2}],
                "changed": [{"label": "A", "before": {"title": "A", "d": 1},
                             "after": {"title": "A", "d": 99}}]},
    "signposts": {"added": [], "removed": [],
                  "changed": [{"label": "S1", "before": {},
                               "after": {"signpost": "S1", "v": "new"}}]},
    "threats": {"added": [], "removed": [{"title": "T1"}], "changed": []},
    "conclusion": {"changed": True, "before": "old", "after": "new"},
}


def test_rejected_changes_leave_the_current_thesis_untouched():
    out, applied = onepager.apply_selected_changes(
        SEL_CUR, SEL_DIFF, ["pillars:added:C", "signposts:changed:S1"])
    titles = [p["title"] for p in out["thesis"]["pillars"]]
    assert titles == ["A", "B", "C"]          # B not removed, C added
    assert out["thesis"]["pillars"][0]["d"] == 1   # A's reword rejected
    assert out["signposts"][0]["v"] == "new"       # this one accepted
    assert len(out["threats"]) == 1                # removal rejected
    assert out["conclusion"] == "old"
    assert applied == 2


def test_accepting_everything_matches_the_candidate_shape():
    ids = ["pillars:added:C", "pillars:removed:B", "pillars:changed:A",
           "signposts:changed:S1", "threats:removed:T1", "conclusion:changed:conclusion"]
    out, applied = onepager.apply_selected_changes(SEL_CUR, SEL_DIFF, ids)
    titles = [p["title"] for p in out["thesis"]["pillars"]]
    assert titles == ["A", "C"] and out["thesis"]["pillars"][0]["d"] == 99
    assert out["threats"] == []
    assert out["conclusion"] == "new"
    assert applied == 6


def test_an_edit_implies_acceptance():
    out, applied = onepager.apply_selected_changes(
        SEL_CUR, SEL_DIFF, [], edits={"pillars:changed:A": {"title": "A", "d": "EDITED"}})
    assert out["thesis"]["pillars"][0]["d"] == "EDITED"
    assert applied == 1


def test_edit_overrides_the_models_wording():
    out, _ = onepager.apply_selected_changes(
        SEL_CUR, SEL_DIFF, ["pillars:changed:A"],
        edits={"pillars:changed:A": {"title": "A", "d": "MINE"}})
    assert out["thesis"]["pillars"][0]["d"] == "MINE"


def test_the_source_analysis_is_never_mutated():
    """Applying must not corrupt the live thesis if anything downstream fails."""
    before = json.dumps(SEL_CUR, sort_keys=True)
    onepager.apply_selected_changes(SEL_CUR, SEL_DIFF,
                                    ["pillars:removed:B", "conclusion:changed:conclusion"])
    assert json.dumps(SEL_CUR, sort_keys=True) == before


def test_accepting_nothing_returns_the_current_thesis():
    out, applied = onepager.apply_selected_changes(SEL_CUR, SEL_DIFF, [])
    assert applied == 0
    assert json.dumps(out, sort_keys=True) == json.dumps(SEL_CUR, sort_keys=True)


def test_changed_row_that_vanished_is_appended_not_dropped():
    """Losing an accepted change silently is worse than re-adding the row."""
    cur = {"thesis": {"pillars": []}}
    diff = {"pillars": {"added": [], "removed": [],
                        "changed": [{"label": "Gone", "before": {}, "after": {"title": "Gone", "d": 5}}]},
            "signposts": {"added": [], "removed": [], "changed": []},
            "threats": {"added": [], "removed": [], "changed": []},
            "conclusion": {"changed": False}}
    out, applied = onepager.apply_selected_changes(cur, diff, ["pillars:changed:Gone"])
    assert applied == 1
    assert out["thesis"]["pillars"][0]["d"] == 5


def test_ids_match_the_ones_classification_assigns():
    """Selections, classifications and edits must key off one identifier."""
    items, _ = onepager.build_classify_prompt(SEL_DIFF)
    ids = {i["id"] for i in items}
    assert "pillars:added:C" in ids
    assert "pillars:changed:A" in ids
    assert "threats:removed:T1" in ids


def test_json_text_edits_are_parsed_into_objects():
    """The editor hands back text; a hand-edited pillar must land as an object."""
    out, _ = onepager.apply_selected_changes(
        SEL_CUR, SEL_DIFF, [], edits={"pillars:changed:A": '{"title": "A", "d": 42}'})
    assert out["thesis"]["pillars"][0] == {"title": "A", "d": 42}


def test_free_text_edits_stay_text():
    out, _ = onepager.apply_selected_changes(
        SEL_CUR, SEL_DIFF, [], edits={"conclusion:changed:conclusion": "my own wording"})
    assert out["conclusion"] == "my own wording"


def test_malformed_json_edit_does_not_break_approval():
    """A broken edit must degrade to text, never raise mid-approval."""
    out, applied = onepager.apply_selected_changes(
        SEL_CUR, SEL_DIFF, [], edits={"pillars:changed:A": '{"title": "A", broken'})
    assert applied == 1
    assert isinstance(out["thesis"]["pillars"][0], str)


def test_removal_takes_one_row_not_every_matching_label():
    """Two pillars can share a title; removing one must not delete both."""
    cur = {"thesis": {"pillars": [{"title": "Dup", "d": 1}, {"title": "Dup", "d": 2}]}}
    diff = {"pillars": {"added": [], "removed": [{"title": "Dup", "d": 1}], "changed": []},
            "signposts": {"added": [], "removed": [], "changed": []},
            "threats": {"added": [], "removed": [], "changed": []},
            "conclusion": {"changed": False}}
    out, applied = onepager.apply_selected_changes(cur, diff, ["pillars:removed:Dup"])
    assert applied == 1
    assert len(out["thesis"]["pillars"]) == 1
