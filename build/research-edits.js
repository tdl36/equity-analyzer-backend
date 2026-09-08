import * as React from 'react';
export function ResearchEdits({
  api,
  ticker,
  instruction,
  analystId,
  onApplied
}) {
  var [data, setData] = React.useState({
      targets: [],
      filenames: [],
      jobs: []
    }),
    [target, setTarget] = React.useState(''),
    [sources, setSources] = React.useState([]),
    [jobId, setJobId] = React.useState(''),
    [selected, setSelected] = React.useState([]);
  var [error, setError] = React.useState(''),
    [loadError, setLoadError] = React.useState(''),
    [busy, setBusy] = React.useState(false),
    [uncertain, setUncertain] = React.useState(false),
    [loading, setLoading] = React.useState(true);
  var alive = React.useRef(true),
    lock = React.useRef(false),
    pending = React.useRef(null);
  var fetchJson = async (path, options = {}) => {
    var c = new AbortController(),
      timer = setTimeout(() => c.abort(), 20000);
    try {
      var r = await fetch(`${api}${path}`, {
        ...options,
        signal: c.signal
      });
      var d = await r.json();
      if (!r.ok) {
        var e = Error(d.error || `Request failed (${r.status})`);
        e.status = r.status;
        throw e;
      }
      return d;
    } finally {
      clearTimeout(timer);
    }
  };
  var refresh = async () => {
    try {
      var d = await fetchJson(`/api/research/edit-targets/${encodeURIComponent(ticker)}`);
      if (alive.current) {
        setData(d);
        setLoadError('');
        setLoading(false);
        if (pending.current && d.jobs.some(j => j.id === pending.current.requestId)) {
          setJobId(pending.current.requestId);
          pending.current = null;
          setUncertain(false);
        }
      }
    } catch (e) {
      if (alive.current) {
        setLoading(false);
        setLoadError(e.message);
      }
    }
  };
  React.useEffect(() => {
    alive.current = true;
    refresh();
    var timer = setInterval(() => {
      if (!document.hidden) refresh();
    }, 5000);
    return () => {
      alive.current = false;
      clearInterval(timer);
    };
  }, [api, ticker]);
  var chosen = data.targets.find(t => `${t.kind}:${t.id}` === target),
    job = data.jobs.find(j => j.id === jobId),
    changes = job?.result?.changes || [];
  var active = data.jobs.some(j => j.targetId === chosen?.id && ['queued', 'running', 'awaiting_approval'].includes(j.status));
  React.useEffect(() => setSelected([]), [jobId]);
  var launch = async () => {
    if (lock.current || !pending.current && (!chosen || !sources.length || !instruction.trim())) return;
    lock.current = true;
    setBusy(true);
    setError('');
    var key = '';
    try {
      key = localStorage.getItem('equity_analyzer_api_key') || '';
    } catch {}
    var body = pending.current || {
      requestId: crypto.randomUUID(),
      targetId: chosen.id,
      kind: chosen.kind,
      filenames: sources,
      instruction,
      analystId,
      apiKey: key
    };
    pending.current = body;
    try {
      var d = await fetchJson(`/api/research/edits/${encodeURIComponent(ticker)}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
      });
      if (alive.current) {
        setJobId(d.jobId);
        pending.current = null;
        setUncertain(false);
        await refresh();
      }
    } catch (e) {
      if (alive.current) {
        setError(e.message);
        if (e.status) pending.current = null;else setUncertain(true);
        await refresh();
      }
    } finally {
      lock.current = false;
      if (alive.current) setBusy(false);
    }
  };
  var decide = async action => {
    if (lock.current || !job) return;
    lock.current = true;
    setBusy(true);
    setError('');
    try {
      await fetchJson(`/api/research/edits/${job.id}/decide`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          action,
          acceptedIds: selected
        })
      });
      await refresh();
    } catch (e) {
      if (alive.current) {
        setError(e.message + ' Check proposal status before retrying.');
        await refresh();
      }
    } finally {
      lock.current = false;
      if (alive.current) setBusy(false);
    }
  };
  return /*#__PURE__*/React.createElement("details", {
    className: "research-edit-panel"
  }, /*#__PURE__*/React.createElement("summary", null, "Turn your instruction into source-backed edits"), /*#__PURE__*/React.createElement("p", null, "Choose the exact saved document and the sources supporting your instruction above. The analyst proposes narrative changes; numeric investment-model fields remain unchanged. A separate review checks proposed wording against source quotations."), (error || loadError) && /*#__PURE__*/React.createElement("p", {
    className: "workspace-error",
    role: "alert"
  }, error || loadError), loading ? /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading saved documents\u2026") : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("fieldset", {
    disabled: busy || uncertain,
    className: "desk-form"
  }, /*#__PURE__*/React.createElement("label", null, "Saved document to revise", /*#__PURE__*/React.createElement("select", {
    value: target,
    onChange: e => setTarget(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Select a saved document"), data.targets.map(t => /*#__PURE__*/React.createElement("option", {
    key: `${t.kind}:${t.id}`,
    value: `${t.kind}:${t.id}`
  }, t.kind === 'note' ? `Note ${t.version} · ${t.status}` : `Investment review · ${t.mode}`, " \xB7 ", String(t.created_at).slice(0, 25))))), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Supporting documents \xB7 ", sources.length, "/10 selected"), /*#__PURE__*/React.createElement("div", {
    className: "research-edit-sources"
  }, data.filenames.map(f => /*#__PURE__*/React.createElement("label", {
    key: f
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: sources.includes(f),
    disabled: !sources.includes(f) && sources.length >= 10,
    onChange: () => setSources(sources.includes(f) ? sources.filter(v => v !== f) : [...sources, f])
  }), /*#__PURE__*/React.createElement("span", null, f)))), !data.filenames.length && /*#__PURE__*/React.createElement("p", null, "Import readable source documents into Charlie before requesting edits.")), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    disabled: !chosen || !sources.length || !instruction.trim() || active,
    onClick: launch
  }, "Prepare edits from my instruction"), active && /*#__PURE__*/React.createElement("p", null, "An active proposal already exists for this document. Select it below to review its status."))), uncertain && /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: launch
  }, "Retry same revision request"), /*#__PURE__*/React.createElement("div", {
    className: "research-chat-controls"
  }, /*#__PURE__*/React.createElement("label", null, "Revision history", /*#__PURE__*/React.createElement("select", {
    value: jobId,
    onChange: e => setJobId(e.target.value),
    disabled: busy
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Select a proposal"), data.jobs.map(j => /*#__PURE__*/React.createElement("option", {
    key: j.id,
    value: j.id
  }, j.kind, " \xB7 ", j.status.replaceAll('_', ' '), " \xB7 ", j.instruction?.slice(0, 75))))), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: refresh
  }, "Refresh proposals")), job && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, /*#__PURE__*/React.createElement("strong", null, job.status.replaceAll('_', ' ')), " \xB7 ", job.instruction, job.error ? ` · ${job.error}` : ''), ['queued', 'running', 'awaiting_approval'].includes(job.status) && /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => decide('dismiss')
  }, "Dismiss proposal"), changes.map(c => /*#__PURE__*/React.createElement("article", {
    className: "amendment-card",
    key: c.id
  }, /*#__PURE__*/React.createElement("label", null, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: job.status === 'applied' ? (job.result.acceptedIds || []).includes(c.id) : selected.includes(c.id),
    disabled: busy || job.status !== 'awaiting_approval' || !c.passageMatched || !c.reviewPassed,
    onChange: () => setSelected(selected.includes(c.id) ? selected.filter(v => v !== c.id) : [...selected, c.id])
  }), /*#__PURE__*/React.createElement("strong", null, c.path.startsWith('blocks.') ? `Note paragraph ${Number(c.path.split('.')[1]) + 1}` : c.path.replaceAll('.', ' › '))), /*#__PURE__*/React.createElement("p", null, c.reason), /*#__PURE__*/React.createElement("div", {
    className: "evidence-compare"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h4", null, "Before"), /*#__PURE__*/React.createElement("p", null, c.before)), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h4", null, "Proposed"), /*#__PURE__*/React.createElement("p", null, c.after))), /*#__PURE__*/React.createElement("p", null, c.passageMatched ? 'Source passage matched.' : 'Source quotation needs review.', " ", c.reviewPassed ? 'Model review passed; your judgment is still required.' : c.reviewIssue || 'Independent review required.'), (c.evidence || []).map((e, i) => /*#__PURE__*/React.createElement("details", {
    key: i
  }, /*#__PURE__*/React.createElement("summary", null, "Source: ", job.result.sources?.find(s => s.id === e.sourceId)?.filename || 'Unavailable'), /*#__PURE__*/React.createElement("blockquote", null, e.excerpt))))), job.status === 'awaiting_approval' && !changes.length && /*#__PURE__*/React.createElement("p", null, "No supported edits were proposed from these documents."), job.status === 'awaiting_approval' && !!changes.length && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, job.kind === 'note' ? 'Applying creates a new note draft. The published note stays unchanged until you accept the draft. Review any inherited charts before publishing.' : 'Applying saves a new review version and regenerates its memo and PDF. The original remains in history.'), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    disabled: busy || !selected.length,
    onClick: () => decide('apply')
  }, "Apply ", selected.length, " selected edits")), job.status === 'applied' && /*#__PURE__*/React.createElement("p", null, job.kind === 'note' ? 'A new note draft is ready in Research Pipeline.' : 'A new investment review version is saved.', " ", onApplied && /*#__PURE__*/React.createElement("button", {
    onClick: () => onApplied({
      kind: job.kind,
      id: job.result.createdId
    })
  }, "Refresh saved research"))));
}