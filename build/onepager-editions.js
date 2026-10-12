import { ResearchAssignment } from './research-assignment';
import * as React from 'react';
import { editionSelection, claimLabel, basisLabel, verifiedEditionPreview, editionReviewState, editionDecision, removedEditionClaims } from './onepager-edition-model.mjs';
export function OnepagerEditions({
  api,
  run,
  assignmentId,
  onOriginal
}) {
  var [pack, setPack] = React.useState(null),
    [rows, setRows] = React.useState([]),
    [selected, setSelected] = React.useState([]),
    [error, setError] = React.useState(''),
    [notice, setNotice] = React.useState(''),
    [busy, setBusy] = React.useState(false),
    [preview, setPreview] = React.useState(null),
    [reviewing, setReviewing] = React.useState(null),
    [ack, setAck] = React.useState(false);
  var alive = React.useRef(true),
    locked = React.useRef(false),
    pending = React.useRef(null),
    decisionAttempt = React.useRef(null);
  var url = api + `/api/research/assignments/${assignmentId}/onepager-editions`;
  async function read(path, body) {
    var r = await fetch(path, {
      ...(body ? {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
      } : {}),
      signal: AbortSignal.timeout(30000)
    });
    var d;
    try {
      d = await r.json();
    } catch {
      throw Error('The response could not be read. Reload saved editions before retrying.');
    }
    if (!r.ok) throw Error(d.error || 'Editions unavailable');
    return d;
  }
  async function load() {
    var [p, c] = await Promise.all([read(url), read(api + `/api/research/stock-analysis-run/${run.id}/corrections`)]);
    if (p.runId !== run.id || p.assignmentId !== assignmentId || p.ticker !== run.ticker || c.runId !== run.id) throw Error('The saved edition belongs to another report. Reopen its assignment.');
    if (alive.current) {
      setPack(p);
      setRows(c.corrections);
    }
  }
  async function reload() {
    if (locked.current) return;
    setPreview(null);
    setReviewing(null);
    setAck(false);
    locked.current = true;
    setBusy(true);
    setError('');
    try {
      await load();
    } catch (e) {
      if (alive.current) setError(e.message);
    } finally {
      locked.current = false;
      if (alive.current) setBusy(false);
    }
  }
  React.useEffect(() => {
    alive.current = true;
    reload();
    return () => {
      alive.current = false;
    };
  }, [url, run.id]);
  async function save() {
    if (locked.current) return;
    setPreview(null);
    locked.current = true;
    setBusy(true);
    setError('');
    setNotice('');
    try {
      var value = editionSelection(pack, rows, selected, '');
      var signature = JSON.stringify(value);
      if (pending.current?.signature !== signature) pending.current = {
        signature,
        body: {
          ...value,
          requestId: crypto.randomUUID()
        }
      };
      var result = await read(url, pending.current.body);
      if (alive.current) {
        pending.current = null;
        setSelected([]);
        setNotice(result.status === 'pending' ? `Pending edition ${result.id} saved. No correction has been accepted or applied.` : `Edition ${result.id} already has a saved ${result.status} decision. Reloaded its retained review.`);
        await load();
      }
    } catch (e) {
      if (alive.current) setError(e.message + ' Inspect saved editions before retrying. An unchanged selection keeps its submission identity.');
    } finally {
      locked.current = false;
      if (alive.current) setBusy(false);
    }
  }
  async function openPreview(id, layout) {
    if (locked.current) return;
    locked.current = true;
    setBusy(true);
    setError('');
    setPreview(null);
    try {
      var value = verifiedEditionPreview(await read(url + `/${encodeURIComponent(id)}/preview?layout=${layout}`), id, assignmentId, run.id, run.ticker);
      if (alive.current) setPreview(value);
    } catch (e) {
      if (alive.current) setError(e.message);
    } finally {
      locked.current = false;
      if (alive.current) setBusy(false);
    }
  }
  async function decide(action) {
    if (locked.current || !reviewing || !ack) return;
    locked.current = true;
    setBusy(true);
    setError('');
    try {
      var body = editionDecision(reviewing, pack, action, '');
      var signature = JSON.stringify(body);
      if (decisionAttempt.current?.signature !== signature) decisionAttempt.current = {
        signature,
        body: {
          ...body,
          requestId: crypto.randomUUID()
        }
      };
      var result = await read(url + `/${reviewing.id}/decision`, decisionAttempt.current.body);
      if (result.decision?.editionId !== reviewing.id || result.decision?.action !== action) throw Error('Decision receipt does not match this edition. Reload saved history.');
      if (alive.current) {
        setNotice(action === 'accept' ? 'Editorial edition accepted. Its version-specific exports are now available. Original research and thesis are retained.' : 'Editorial edition rejected. The current accepted version is retained.');
        setReviewing(null);
        setPreview(null);
        decisionAttempt.current = null;
        await load();
      }
    } catch (e) {
      if (alive.current) setError(e.message + ' Reload history before making another decision; an unchanged retry keeps its identity.');
    } finally {
      locked.current = false;
      if (alive.current) setBusy(false);
    }
  }
  async function exportEdition(id, layout) {
    if (locked.current) return;
    locked.current = true;
    setBusy(true);
    setError('');
    try {
      var response = await fetch(url + `/${encodeURIComponent(id)}/export?layout=${layout}`, {
        signal: AbortSignal.timeout(30000)
      });
      if (!response.ok) {
        var d = await response.json();
        throw Error(d.error || 'Edition export unavailable');
      }
      if (!response.headers.get('Content-Type')?.includes('text/html')) throw Error('Unexpected export format. Reload the edition.');
      var blob = await response.blob(),
        link = document.createElement('a'),
        href = URL.createObjectURL(blob);
      link.href = href;
      link.download = `${run.ticker}-edition-${id}-${layout}.html`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(href), 1000);
      if (alive.current) setNotice('Accepted edition downloaded as a printable HTML document. Open it in your browser and print to PDF if needed.');
    } catch (e) {
      if (alive.current) setError(e.message);
    } finally {
      locked.current = false;
      if (alive.current) setBusy(false);
    }
  }
  function evidence(p) {
    return /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Inspect evidence and provenance"), /*#__PURE__*/React.createElement("p", null, basisLabel(p.basis), " \xB7 ", p.caveat), [p, ...(p.additionalEvidence || [])].map((e, i) => {
      var source = run.sources.find(s => s.id === e.sourceId && s.filename === e.filename && s.originalHash === e.sha256);
      return /*#__PURE__*/React.createElement("div", {
        key: i
      }, /*#__PURE__*/React.createElement("p", null, e.filename, " \xB7 ", e.locator), /*#__PURE__*/React.createElement("blockquote", null, e.excerpt), source ? /*#__PURE__*/React.createElement("button", {
        disabled: busy,
        onClick: () => onOriginal(source)
      }, "Download verified original") : /*#__PURE__*/React.createElement("p", null, "The original no longer matches this report. Inspect the saved source pack."), /*#__PURE__*/React.createElement("small", null, "Original SHA-256 ", e.sha256));
    }), /*#__PURE__*/React.createElement("small", null, "Claim ", p.path));
  }
  function arithmetic(edition) {
    return /*#__PURE__*/React.createElement("section", {
      "aria-label": "Scenario arithmetic review"
    }, /*#__PURE__*/React.createElement("h5", null, "Scenario arithmetic"), /*#__PURE__*/React.createElement("p", null, "Saved calculations only. This check does not reverify originals, judge assumptions or certify valuation quality."), edition.scenarioReview === null ? /*#__PURE__*/React.createElement("p", null, "Baseline changed. Prepare a new edition before checking its calculations.") : !edition.scenarioReview?.length ? /*#__PURE__*/React.createElement("p", null, "No saved scenario calculation inputs available.") : edition.scenarioReview.map(row => /*#__PURE__*/React.createElement("div", {
      key: row.index,
      style: {
        overflowWrap: 'anywhere',
        marginBottom: 12
      }
    }, /*#__PURE__*/React.createElement("strong", null, row.label), /*#__PURE__*/React.createElement("p", {
      role: row.status === 'mismatch' ? 'alert' : undefined
    }, row.message), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Inspect calculation inputs"), [['eps_or_fcf', 'Earnings per share'], ['valuation_multiple', 'P/E and rationale'], ['implied_value', 'Implied value'], ['upside_downside', 'Return and reference price']].map(([key, label]) => /*#__PURE__*/React.createElement("p", {
      key: key
    }, /*#__PURE__*/React.createElement("b", null, label, ":"), " ", row.inputs[key] || 'Not established in saved evidence'))))));
  }
  var selectionError = '';
  if (selected.length && pack) try {
    editionSelection(pack, rows, selected, '');
  } catch (e) {
    selectionError = e.message;
  }
  return /*#__PURE__*/React.createElement("details", {
    className: "sa-prepare"
  }, /*#__PURE__*/React.createElement("summary", null, "One-pager editions \xB7 review history"), /*#__PURE__*/React.createElement("p", null, "Group source-linked corrections into a saved edition for this assignment. Review the complete wording and evidence together, then open a labeled pending preview. Investor sign-in is required to accept or reject an edition. Accepted editions have separate printable exports."), /*#__PURE__*/React.createElement("small", null, "Assignment ", assignmentId, " \xB7 Report ", run.id), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: reload
  }, "Reload editions and corrections"), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error), notice && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, notice), !pack ? /*#__PURE__*/React.createElement("p", null, busy ? 'Loading saved editions…' : 'Saved editions unavailable.') : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Refresh this one-pager from recent sources"), /*#__PURE__*/React.createElement(ResearchAssignment, {
    key: pack.fingerprint + pack.fundamentalsDigest + pack.parentId,
    api: api,
    ticker: run.ticker,
    origin: {
      kind: 'onepager',
      ...Object.fromEntries(['assignmentId', 'runId', 'ticker', 'fingerprint', 'fundamentalsDigest', 'parentId'].map(k => [k, pack[k]]))
    },
    onReport: id => {
      window.location.hash = 'view=stockanalysis&ticker=' + encodeURIComponent(run.ticker) + '&section=stock-analysis&work=analysis%3A' + encodeURIComponent(id);
    }
  })), /*#__PURE__*/React.createElement("h4", null, "Choose proposed changes"), /*#__PURE__*/React.createElement("p", null, "Create corrections in \u201CPropose a correction for review\u201D above, then reload here. Showing up to 50 recent corrections; select at most 20, with one change per claim."), /*#__PURE__*/React.createElement("fieldset", {
    disabled: busy
  }, rows.map(c => /*#__PURE__*/React.createElement("article", {
    key: c.id
  }, /*#__PURE__*/React.createElement("label", {
    className: "sa-check"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: selected.includes(c.id),
    disabled: c.stale || c.fingerprint !== pack.fingerprint || !selected.includes(c.id) && selected.length >= 20,
    onChange: e => setSelected(ids => e.target.checked ? [...ids, c.id] : ids.filter(id => id !== c.id))
  }), /*#__PURE__*/React.createElement("span", null, claimLabel(c.proposal.path), c.stale || c.fingerprint !== pack.fingerprint ? ' · Baseline changed; prepare again' : '')), /*#__PURE__*/React.createElement("div", {
    className: "sa-form-grid"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h5", null, "Original"), /*#__PURE__*/React.createElement("p", null, c.proposal.before === null ? 'New scenario input; no saved value.' : c.proposal.before)), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h5", null, "Proposed \xB7 ", basisLabel(c.proposal.basis)), /*#__PURE__*/React.createElement("p", null, c.proposal.statement))), /*#__PURE__*/React.createElement("p", null, c.proposal.reason), evidence(c.proposal))), !rows.length && /*#__PURE__*/React.createElement("p", null, "No pending corrections have been saved for this report."), selectionError && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, selectionError), /*#__PURE__*/React.createElement("button", {
    disabled: !selected.length || !!selectionError,
    onClick: save
  }, busy ? 'Working…' : `Save pending edition (${selected.length}/20)`)), /*#__PURE__*/React.createElement("h4", null, "Saved editions"), /*#__PURE__*/React.createElement("p", null, "Showing up to ", pack.limit, " most recent editions. Saving an edition alone does not approve it; any recorded decision is shown separately. Original availability and usage restrictions are rechecked when saving; opening this list does not reverify them."), pack.editions.map(e => {
    var review = editionReviewState(e);
    return /*#__PURE__*/React.createElement("article", {
      key: e.id
    }, /*#__PURE__*/React.createElement("h5", null, review.label, " \xB7 ", new Date(e.created_at).toLocaleString()), /*#__PURE__*/React.createElement("small", null, "Edition ", e.id, e.stale ? ' · Baseline or parent changed; new decisions require fresh preparation' : ''), e.decision ? /*#__PURE__*/React.createElement("p", null, "Investor decision recorded ", new Date(e.decision.decidedAt).toLocaleString(), ". Original research remains unchanged. Acceptance records editorial review; it does not certify accuracy or approve a thesis.") : /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("button", {
      disabled: busy || !review.canPreview,
      onClick: () => openPreview(e.id, 'illustrated')
    }, "Preview pending investment map"), /*#__PURE__*/React.createElement("button", {
      disabled: busy || !review.canPreview,
      onClick: () => openPreview(e.id, 'brief')
    }, "Preview pending brief")), e.decision?.action === 'accept' && /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("button", {
      disabled: busy || e.stale,
      onClick: () => exportEdition(e.id, 'illustrated')
    }, "Download accepted investment map"), /*#__PURE__*/React.createElement("button", {
      disabled: busy || e.stale,
      onClick: () => exportEdition(e.id, 'brief')
    }, "Download accepted brief")), review.canPreview && pack.canDecide && /*#__PURE__*/React.createElement("button", {
      disabled: busy,
      onClick: () => {
        setReviewing(e);
        setAck(false);
      }
    }, "Review edition decision"), arithmetic(e), e.record.claims.map(c => /*#__PURE__*/React.createElement("details", {
      key: c.correctionId
    }, /*#__PURE__*/React.createElement("summary", null, claimLabel(c.proposal.path)), /*#__PURE__*/React.createElement("div", {
      className: "sa-form-grid"
    }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h5", null, "Original"), /*#__PURE__*/React.createElement("p", null, c.proposal.before === null ? 'New scenario input; no saved value.' : c.proposal.before)), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h5", null, "Proposed \xB7 ", basisLabel(c.proposal.basis)), /*#__PURE__*/React.createElement("p", null, c.proposal.statement))), /*#__PURE__*/React.createElement("p", null, c.proposal.reason), evidence(c.proposal), /*#__PURE__*/React.createElement("small", null, "Correction ", c.correctionId))));
  }), reviewing && /*#__PURE__*/React.createElement("section", {
    "aria-label": "Review editorial decision"
  }, /*#__PURE__*/React.createElement("h4", null, "Review this exact edition"), /*#__PURE__*/React.createElement("p", null, "Edition ", reviewing.id, ". This complete selection replaces the previous editorial selection; it does not add to it. Original research and thesis stay unchanged."), arithmetic(reviewing), reviewing.record.claims.map(c => /*#__PURE__*/React.createElement("article", {
    key: c.correctionId
  }, /*#__PURE__*/React.createElement("h5", null, claimLabel(c.proposal.path)), /*#__PURE__*/React.createElement("p", null, "Original: ", c.proposal.before === null ? 'New scenario input; no saved value.' : c.proposal.before), /*#__PURE__*/React.createElement("p", null, "Proposed: ", c.proposal.statement), evidence(c.proposal))), /*#__PURE__*/React.createElement("h5", null, "Changes removed from the previous accepted edition"), removedEditionClaims(reviewing, pack).length ? removedEditionClaims(reviewing, pack).map(c => /*#__PURE__*/React.createElement("p", {
    key: c.correctionId
  }, claimLabel(c.proposal.path), ": ", c.proposal.statement, ". This correction will no longer be included; the selected replacement or original wording will be used.")) : /*#__PURE__*/React.createElement("p", null, "None."), /*#__PURE__*/React.createElement("label", {
    className: "sa-check"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    disabled: busy,
    checked: ack,
    onChange: e => setAck(e.target.checked)
  }), "I reviewed the wording, evidence, qualifications and any removed corrections for this exact edition."), /*#__PURE__*/React.createElement("button", {
    disabled: busy || !ack,
    onClick: () => decide('accept')
  }, "Accept this editorial edition"), /*#__PURE__*/React.createElement("button", {
    disabled: busy || !ack,
    onClick: () => decide('reject')
  }, "Reject this edition"), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => setReviewing(null)
  }, "Close review")), preview && /*#__PURE__*/React.createElement("section", {
    "aria-label": "Pending edition preview"
  }, /*#__PURE__*/React.createElement("h4", null, "Pending preview \xB7 not accepted"), /*#__PURE__*/React.createElement("small", null, "Edition ", preview.id, " \xB7 ", preview.layout === 'brief' ? 'Brief' : 'Investment map'), /*#__PURE__*/React.createElement("p", null, preview.caveat), /*#__PURE__*/React.createElement("button", {
    onClick: () => setPreview(null)
  }, "Close pending preview"), /*#__PURE__*/React.createElement("iframe", {
    title: `${run.ticker} pending edition ${preview.id}`,
    sandbox: "",
    srcDoc: preview.html,
    style: {
      width: '100%',
      height: 850,
      border: '1px solid #ccd7de',
      background: 'white'
    }
  })), !pack.editions.length && /*#__PURE__*/React.createElement("p", null, "No saved editions for this assignment.")));
}