import * as React from 'react';
var blank = () => ({
  decision: '',
  rationale: '',
  revisitWhen: '',
  decisionDate: new Date().toLocaleDateString('en-CA'),
  supersedes: ''
});
export function ResearchDecisions({
  api,
  ticker
}) {
  var [draft, setDraft] = React.useState(blank),
    [rows, setRows] = React.useState([]),
    [revision, setRevision] = React.useState(null),
    [more, setMore] = React.useState(false),
    [busy, setBusy] = React.useState(false),
    [message, setMessage] = React.useState('');
  var alive = React.useRef(true),
    lock = React.useRef(false),
    pending = React.useRef(null);
  var json = async options => {
    var r = await fetch(`${api}/api/research/decisions/${encodeURIComponent(ticker)}`, {
      ...options,
      signal: AbortSignal.timeout(20000)
    });
    var d = await r.json();
    if (!r.ok) {
      var e = Error(d.error || 'Decision log unavailable');
      e.status = r.status;
      throw e;
    }
    return d;
  };
  var load = async () => {
    try {
      var d = await json();
      if (alive.current) {
        setRows(d.decisions);
        setRevision(d.revision);
        setMore(d.hasMore);
      }
    } catch (e) {
      if (alive.current) setMessage(e.message);
    }
  };
  React.useEffect(() => {
    alive.current = true;
    load();
    return () => {
      alive.current = false;
    };
  }, []);
  var change = (key, value) => {
    setDraft(d => ({
      ...d,
      [key]: value
    }));
    pending.current = null;
  };
  var save = async () => {
    if (lock.current || revision === null) return;
    lock.current = true;
    setBusy(true);
    setMessage('Saving decision…');
    var payload = pending.current || {
      ...draft,
      revision,
      requestId: crypto.randomUUID()
    };
    pending.current = payload;
    try {
      await json({
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });
      if (alive.current) {
        pending.current = null;
        setDraft(blank());
        setMessage('Decision recorded. Existing research and portfolio positions were not changed.');
        await load();
      }
    } catch (e) {
      if (e.status === 409) pending.current = null;
      if (alive.current) setMessage(e.message + ' Reload the log if needed; your draft is retained.');
    } finally {
      lock.current = false;
      if (alive.current) setBusy(false);
    }
  };
  return /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Decision log \xB7 ", rows.length, " recent records"), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Record your reasoning and what would make you revisit it. Entries are retained; superseding a record preserves its history. These are research decisions, not trade instructions. Revisit conditions are not yet automatically monitored."), /*#__PURE__*/React.createElement("fieldset", {
    disabled: busy
  }, /*#__PURE__*/React.createElement("label", null, "Decision date", /*#__PURE__*/React.createElement("input", {
    type: "date",
    value: draft.decisionDate,
    onChange: e => change('decisionDate', e.target.value)
  })), [['decision', 'My decision', 2000], ['rationale', 'Why I decided this', 12000], ['revisitWhen', 'Revisit when…', 6000]].map(([key, label, limit]) => /*#__PURE__*/React.createElement("label", {
    key: key
  }, label, /*#__PURE__*/React.createElement("textarea", {
    rows: key === 'decision' ? 2 : 3,
    maxLength: limit,
    value: draft[key],
    onChange: e => change(key, e.target.value)
  }))), /*#__PURE__*/React.createElement("label", null, "Does this replace an earlier decision?", /*#__PURE__*/React.createElement("select", {
    value: draft.supersedes,
    onChange: e => change('supersedes', e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Independent new record"), rows.filter(r => !r.superseded).map(r => /*#__PURE__*/React.createElement("option", {
    key: r.id,
    value: r.id
  }, "#", r.revision, " \xB7 ", r.body.decisionDate, " \xB7 ", r.body.decision.slice(0, 90))))), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    disabled: revision === null || !draft.decision.trim() || !draft.rationale.trim() || !draft.revisitWhen.trim(),
    onClick: save
  }, busy ? 'Saving…' : 'Record decision'), /*#__PURE__*/React.createElement("button", {
    onClick: load
  }, "Reload log \xB7 keep draft")), message && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, message), rows.map(r => /*#__PURE__*/React.createElement("article", {
    key: r.id,
    style: {
      borderTop: '1px solid var(--border, #bbb)',
      padding: '12px 0'
    }
  }, /*#__PURE__*/React.createElement("strong", null, "#", r.revision, " \xB7 ", r.body.decisionDate, " \xB7 ", r.superseded ? 'Superseded' : 'Recorded decision'), /*#__PURE__*/React.createElement("h4", null, r.body.decision), /*#__PURE__*/React.createElement("p", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, r.body.rationale), /*#__PURE__*/React.createElement("p", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, /*#__PURE__*/React.createElement("strong", null, "Revisit when: "), r.body.revisitWhen), r.body.supersedes && /*#__PURE__*/React.createElement("small", null, "Replaces an earlier recorded decision. Original retained."), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "Recorded ", new Date(r.created_at).toLocaleString())))), !rows.length && revision !== null && /*#__PURE__*/React.createElement("p", null, "No decisions recorded for ", ticker, " yet."), more && /*#__PURE__*/React.createElement("p", null, "Showing the latest 50 records. Older records remain stored; shared memory includes the latest 20."));
}