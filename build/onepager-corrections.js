import * as React from 'react';
export function OnepagerCorrections({
  api,
  run,
  onOriginal
}) {
  var [pack, setPack] = React.useState(null),
    [error, setError] = React.useState(''),
    [busy, setBusy] = React.useState(false);
  var [path, setPath] = React.useState(''),
    [statement, setStatement] = React.useState(''),
    [basis, setBasis] = React.useState('interpretation'),
    [reason, setReason] = React.useState(''),
    [sourceId, setSource] = React.useState(''),
    [excerpt, setExcerpt] = React.useState(''),
    [locator, setLocator] = React.useState('');
  var [additionalEvidence, setAdditionalEvidence] = React.useState([]);
  var locked = React.useRef(false),
    pending = React.useRef(null),
    alive = React.useRef(true);
  var url = api + `/api/research/stock-analysis-run/${run.id}/corrections`;
  async function call(body) {
    var r = await fetch(url, {
      ...(body ? {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
      } : {}),
      signal: AbortSignal.timeout(30000)
    });
    var d = await r.json();
    if (!r.ok) throw Error(d.error || 'Corrections unavailable');
    return d;
  }
  async function load() {
    try {
      var d = await call();
      if (alive.current) {
        setPack(d);
        setError('');
      }
    } catch (e) {
      if (alive.current) setError(e.message);
    }
  }
  React.useEffect(() => {
    alive.current = true;
    load();
    return () => {
      alive.current = false;
    };
  }, [url]);
  async function submit() {
    if (locked.current) return;
    locked.current = true;
    setBusy(true);
    setError('');
    try {
      var value = {
        fingerprint: pack.fingerprint,
        path,
        statement,
        basis,
        reason,
        sourceId,
        excerpt,
        locator,
        ...(additionalEvidence.length ? {
          additionalEvidence
        } : {})
      };
      var signature = JSON.stringify(value);
      if (pending.current?.signature !== signature) pending.current = {
        signature,
        body: {
          ...value,
          requestId: crypto.randomUUID()
        }
      };
      await call(pending.current.body);
      if (alive.current) {
        pending.current = null;
        setStatement('');
        setReason('');
        setExcerpt('');
        setLocator('');
        setAdditionalEvidence([]);
        await load();
      }
    } catch (e) {
      if (alive.current) setError(e.message + ' Inspect saved corrections before retrying; unchanged form data retains its submission identity.');
    } finally {
      locked.current = false;
      if (alive.current) setBusy(false);
    }
  }
  var selected = pack?.fields.find(f => f.path === path);
  var title = p => p.replace(/^\//, '').split('/').map(x => /^\d+$/.test(x) ? String(Number(x) + 1) : x.replaceAll('_', ' ')).join(' · ');
  return /*#__PURE__*/React.createElement("details", {
    className: "sa-prepare"
  }, /*#__PURE__*/React.createElement("summary", null, "Propose a correction for review"), /*#__PURE__*/React.createElement("p", null, "Save proposed wording separately from this report. The passage is checked against the frozen source and original file. Meaning and completeness still need investor review. These pending records cannot be applied or published here; the current report, one-pager and thesis stay unchanged."), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: load
  }, "Reload saved corrections"), !pack ? /*#__PURE__*/React.createElement("p", null, "Loading saved evidence\u2026") : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("fieldset", {
    disabled: busy
  }, /*#__PURE__*/React.createElement("label", null, "Claim to correct", /*#__PURE__*/React.createElement("select", {
    value: path,
    onChange: e => setPath(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Choose a saved claim"), pack.fields.map(f => /*#__PURE__*/React.createElement("option", {
    key: f.path,
    value: f.path
  }, title(f.path))))), selected && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("h5", null, "Original wording"), /*#__PURE__*/React.createElement("p", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, selected.before || 'Unavailable')), /*#__PURE__*/React.createElement("label", null, "Proposed wording", /*#__PURE__*/React.createElement("textarea", {
    value: statement,
    onChange: e => setStatement(e.target.value),
    maxLength: 2000,
    rows: 4
  })), /*#__PURE__*/React.createElement("label", null, "Claim type", /*#__PURE__*/React.createElement("select", {
    value: basis,
    onChange: e => setBasis(e.target.value)
  }, [['reported_fact', 'Reported fact'], ['management_guidance', 'Management guidance'], ['broker_estimate', 'Broker estimate'], ['interpretation', 'Interpretation'], ['hypothesis', 'Hypothesis']].map(([v, t]) => /*#__PURE__*/React.createElement("option", {
    key: v,
    value: v
  }, t)))), /*#__PURE__*/React.createElement("label", null, "Why this correction is needed", /*#__PURE__*/React.createElement("textarea", {
    value: reason,
    onChange: e => setReason(e.target.value),
    maxLength: 2000,
    rows: 2
  })), /*#__PURE__*/React.createElement("label", null, "Frozen source", /*#__PURE__*/React.createElement("select", {
    value: sourceId,
    onChange: e => setSource(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Choose an original"), pack.sources.map(s => /*#__PURE__*/React.createElement("option", {
    key: s.id,
    value: s.id
  }, s.filename)))), sourceId && /*#__PURE__*/React.createElement("button", {
    type: "button",
    onClick: () => onOriginal(run.sources.find(s => s.id === sourceId))
  }, "Download verified original"), /*#__PURE__*/React.createElement("label", null, "Exact supporting passage", /*#__PURE__*/React.createElement("textarea", {
    value: excerpt,
    onChange: e => setExcerpt(e.target.value),
    maxLength: 4000,
    rows: 4
  })), /*#__PURE__*/React.createElement("label", null, "Page or section locator", /*#__PURE__*/React.createElement("input", {
    value: locator,
    onChange: e => setLocator(e.target.value),
    maxLength: 200
  })), additionalEvidence.map((e, i) => /*#__PURE__*/React.createElement("div", {
    key: i,
    style: {
      borderTop: '1px solid #ddd',
      paddingTop: 12,
      marginTop: 12
    }
  }, /*#__PURE__*/React.createElement("h5", null, "Additional original ", i + 1), /*#__PURE__*/React.createElement("label", null, "Frozen source", /*#__PURE__*/React.createElement("select", {
    value: e.sourceId,
    onChange: event => setAdditionalEvidence(items => items.map((x, j) => j === i ? {
      ...x,
      sourceId: event.target.value
    } : x))
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Choose another original"), pack.sources.map(x => /*#__PURE__*/React.createElement("option", {
    key: x.id,
    value: x.id
  }, x.filename)))), e.sourceId && /*#__PURE__*/React.createElement("button", {
    type: "button",
    onClick: () => onOriginal(run.sources.find(x => x.id === e.sourceId))
  }, "Download verified original"), /*#__PURE__*/React.createElement("label", null, "Exact supporting passage", /*#__PURE__*/React.createElement("textarea", {
    rows: 4,
    maxLength: 4000,
    value: e.excerpt,
    onChange: event => setAdditionalEvidence(items => items.map((x, j) => j === i ? {
      ...x,
      excerpt: event.target.value
    } : x))
  })), /*#__PURE__*/React.createElement("label", null, "Page or section locator", /*#__PURE__*/React.createElement("input", {
    maxLength: 200,
    value: e.locator,
    onChange: event => setAdditionalEvidence(items => items.map((x, j) => j === i ? {
      ...x,
      locator: event.target.value
    } : x))
  })), /*#__PURE__*/React.createElement("button", {
    type: "button",
    onClick: () => setAdditionalEvidence(items => items.filter((_, j) => j !== i))
  }, "Remove additional original"))), /*#__PURE__*/React.createElement("button", {
    type: "button",
    disabled: additionalEvidence.length >= 7,
    onClick: () => setAdditionalEvidence(items => [...items, {
      sourceId: '',
      excerpt: '',
      locator: ''
    }])
  }, "Add another supporting original"), /*#__PURE__*/React.createElement("button", {
    disabled: !path || !statement.trim() || !reason.trim() || !sourceId || !excerpt.trim() || !locator.trim() || additionalEvidence.some(e => !e.sourceId || !e.excerpt.trim() || !e.locator.trim()),
    onClick: submit
  }, busy ? 'Saving…' : 'Save pending correction')), /*#__PURE__*/React.createElement("h4", null, "Pending corrections"), /*#__PURE__*/React.createElement("p", null, "Showing up to ", pack.limit, " most recent records. No correction is accepted automatically."), pack.corrections.length ? pack.corrections.map(c => /*#__PURE__*/React.createElement("article", {
    key: c.id,
    style: {
      overflowWrap: 'anywhere'
    }
  }, /*#__PURE__*/React.createElement("h5", null, title(c.proposal.path)), /*#__PURE__*/React.createElement("small", null, new Date(c.created_at).toLocaleString(), " \xB7 Pending investor review", c.stale ? ' · Baseline changed; prepare again' : ''), /*#__PURE__*/React.createElement("div", {
    className: "sa-form-grid"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h5", null, "Original"), /*#__PURE__*/React.createElement("p", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, c.proposal.before)), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h5", null, "Proposed"), /*#__PURE__*/React.createElement("p", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, c.proposal.statement))), /*#__PURE__*/React.createElement("p", null, c.proposal.reason), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Inspect supporting evidence"), [c.proposal, ...(c.proposal.additionalEvidence || [])].map((e, i) => /*#__PURE__*/React.createElement("div", {
    key: i
  }, /*#__PURE__*/React.createElement("p", null, e.filename, " \xB7 ", e.locator), /*#__PURE__*/React.createElement("blockquote", null, e.excerpt))), /*#__PURE__*/React.createElement("p", null, c.proposal.caveat), /*#__PURE__*/React.createElement("small", null, "Correction ", c.id, " \xB7 ", c.proposal.path)))) : /*#__PURE__*/React.createElement("p", null, "No saved corrections for this report.")));
}