import * as React from 'react';
import { editionSelection, claimLabel, basisLabel, verifiedEditionPreview, editionReviewState } from './onepager-edition-model.mjs';
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
    [preview, setPreview] = React.useState(null);
  var alive = React.useRef(true),
    locked = React.useRef(false),
    pending = React.useRef(null);
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
  function evidence(p) {
    var source = run.sources.find(s => s.id === p.sourceId && s.filename === p.filename && s.originalHash === p.sha256);
    return /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Inspect evidence and provenance"), /*#__PURE__*/React.createElement("p", null, basisLabel(p.basis), " \xB7 ", p.filename, " \xB7 ", p.locator), /*#__PURE__*/React.createElement("blockquote", null, p.excerpt), /*#__PURE__*/React.createElement("p", null, p.caveat), source ? /*#__PURE__*/React.createElement("button", {
      disabled: busy,
      onClick: () => onOriginal(source)
    }, "Download verified original") : /*#__PURE__*/React.createElement("p", null, "The original no longer matches this report. Inspect the saved source pack."), /*#__PURE__*/React.createElement("small", null, "Claim ", p.path, " \xB7 Original SHA-256 ", p.sha256));
  }
  var selectionError = '';
  if (selected.length && pack) try {
    editionSelection(pack, rows, selected, '');
  } catch (e) {
    selectionError = e.message;
  }
  return /*#__PURE__*/React.createElement("details", {
    className: "sa-prepare"
  }, /*#__PURE__*/React.createElement("summary", null, "One-pager editions \xB7 review history"), /*#__PURE__*/React.createElement("p", null, "Group source-linked corrections into a saved edition for this assignment. Review the complete wording and evidence together, then open a labeled pending preview. Decision controls and revised exports are not available yet."), /*#__PURE__*/React.createElement("small", null, "Assignment ", assignmentId, " \xB7 Report ", run.id), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: reload
  }, "Reload editions and corrections"), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error), notice && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, notice), !pack ? /*#__PURE__*/React.createElement("p", null, busy ? 'Loading saved editions…' : 'Saved editions unavailable.') : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("h4", null, "Choose proposed changes"), /*#__PURE__*/React.createElement("p", null, "Create corrections in \u201CPropose a correction for review\u201D above, then reload here. Showing up to 50 recent corrections; select at most 20, with one change per claim."), /*#__PURE__*/React.createElement("fieldset", {
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
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h5", null, "Original"), /*#__PURE__*/React.createElement("p", null, c.proposal.before)), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h5", null, "Proposed \xB7 ", basisLabel(c.proposal.basis)), /*#__PURE__*/React.createElement("p", null, c.proposal.statement))), /*#__PURE__*/React.createElement("p", null, c.proposal.reason), evidence(c.proposal))), !rows.length && /*#__PURE__*/React.createElement("p", null, "No pending corrections have been saved for this report."), selectionError && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, selectionError), /*#__PURE__*/React.createElement("button", {
    disabled: !selected.length || !!selectionError,
    onClick: save
  }, busy ? 'Working…' : `Save pending edition (${selected.length}/20)`)), /*#__PURE__*/React.createElement("h4", null, "Saved editions"), /*#__PURE__*/React.createElement("p", null, "Showing up to ", pack.limit, " most recent editions. Saving an edition alone does not approve it; any recorded decision is shown separately. Original availability and usage restrictions are rechecked when saving; opening this list does not reverify them."), pack.editions.map(e => {
    var review = editionReviewState(e);
    return /*#__PURE__*/React.createElement("article", {
      key: e.id
    }, /*#__PURE__*/React.createElement("h5", null, review.label, " \xB7 ", new Date(e.created_at).toLocaleString()), /*#__PURE__*/React.createElement("small", null, "Edition ", e.id, e.stale ? ' · Baseline or parent changed; new decisions require fresh preparation' : ''), e.decision ? /*#__PURE__*/React.createElement("p", null, "Investor decision recorded ", new Date(e.decision.decidedAt).toLocaleString(), ". Original research remains unchanged. Accepted-version exports are not available yet.") : /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("button", {
      disabled: busy || !review.canPreview,
      onClick: () => openPreview(e.id, 'illustrated')
    }, "Preview pending investment map"), /*#__PURE__*/React.createElement("button", {
      disabled: busy || !review.canPreview,
      onClick: () => openPreview(e.id, 'brief')
    }, "Preview pending brief")), e.record.claims.map(c => /*#__PURE__*/React.createElement("details", {
      key: c.correctionId
    }, /*#__PURE__*/React.createElement("summary", null, claimLabel(c.proposal.path)), /*#__PURE__*/React.createElement("div", {
      className: "sa-form-grid"
    }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h5", null, "Original"), /*#__PURE__*/React.createElement("p", null, c.proposal.before)), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h5", null, "Proposed \xB7 ", basisLabel(c.proposal.basis)), /*#__PURE__*/React.createElement("p", null, c.proposal.statement))), /*#__PURE__*/React.createElement("p", null, c.proposal.reason), evidence(c.proposal), /*#__PURE__*/React.createElement("small", null, "Correction ", c.correctionId))));
  }), preview && /*#__PURE__*/React.createElement("section", {
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