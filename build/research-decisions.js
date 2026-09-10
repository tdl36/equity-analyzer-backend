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
  var [filters, setFilters] = React.useState({
      q: '',
      from: '',
      to: ''
    }),
    [loading, setLoading] = React.useState(false),
    [inspected, setInspected] = React.useState(null),
    [chosen, setChosen] = React.useState(null);
  var applied = React.useRef({}),
    cursor = React.useRef(null),
    sequence = React.useRef(0);
  var alive = React.useRef(true),
    lock = React.useRef(false),
    pending = React.useRef(null);
  var json = async (options, query = '') => {
    var r = await fetch(`${api}/api/research/decisions/${encodeURIComponent(ticker)}${query}`, {
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
  var load = async (append = false, query = applied.current) => {
    var seq = ++sequence.current;
    setLoading(true);
    var params = new URLSearchParams(query);
    if (append && cursor.current) params.set('before', cursor.current);
    try {
      var d = await json(undefined, '?' + params);
      if (alive.current && seq === sequence.current) {
        setRows(previous => append ? [...previous, ...d.decisions.filter(r => !previous.some(p => p.id === r.id))] : d.decisions);
        setRevision(d.revision);
        setMore(d.hasMore);
        cursor.current = d.nextBefore;
        applied.current = query;
      }
    } catch (e) {
      if (alive.current && seq === sequence.current) setMessage(e.message);
    } finally {
      if (alive.current && seq === sequence.current) setLoading(false);
    }
  };
  var inspect = async id => {
    try {
      var d = await json(undefined, '?id=' + encodeURIComponent(id));
      if (alive.current) {
        setInspected(d.decisions[0] || null);
        if (!d.decisions.length) setMessage('Earlier decision not found.');
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
    if (lock.current || loading || revision === null) return;
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
        setChosen(null);
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
  return /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Decision log \xB7 ", rows.length, " loaded records"), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Record your reasoning and what would make you revisit it. Entries are retained; superseding a record preserves its history. These are research decisions, not trade instructions. Revisit conditions are not yet automatically monitored."), /*#__PURE__*/React.createElement("fieldset", {
    disabled: busy || loading
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
    onChange: e => {
      change('supersedes', e.target.value);
      setChosen(rows.find(r => r.id === e.target.value) || null);
    }
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Independent new record"), [...rows, ...(chosen && !rows.some(r => r.id === chosen.id) ? [chosen] : [])].filter(r => !r.superseded).map(r => /*#__PURE__*/React.createElement("option", {
    key: r.id,
    value: r.id
  }, "#", r.revision, " \xB7 ", r.body.decisionDate, " \xB7 ", r.body.decision.slice(0, 90))))), chosen && /*#__PURE__*/React.createElement("p", null, "Replacing #", chosen.revision, ": ", chosen.body.decision), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    disabled: revision === null || !draft.decision.trim() || !draft.rationale.trim() || !draft.revisitWhen.trim(),
    onClick: save
  }, busy ? 'Saving…' : 'Record decision'), /*#__PURE__*/React.createElement("button", {
    onClick: () => load()
  }, "Reload log \xB7 keep draft")), /*#__PURE__*/React.createElement("fieldset", {
    disabled: busy || loading
  }, /*#__PURE__*/React.createElement("legend", null, "Find earlier decisions"), /*#__PURE__*/React.createElement("label", null, "Search decisions and reasoning", /*#__PURE__*/React.createElement("input", {
    maxLength: 200,
    value: filters.q,
    onChange: e => setFilters({
      ...filters,
      q: e.target.value
    }),
    placeholder: "Cash conversion, pricing, management\u2026"
  })), /*#__PURE__*/React.createElement("label", null, "Decision date from", /*#__PURE__*/React.createElement("input", {
    type: "date",
    value: filters.from,
    onChange: e => setFilters({
      ...filters,
      from: e.target.value
    })
  })), /*#__PURE__*/React.createElement("label", null, "Decision date through", /*#__PURE__*/React.createElement("input", {
    type: "date",
    value: filters.to,
    onChange: e => setFilters({
      ...filters,
      to: e.target.value
    })
  })), /*#__PURE__*/React.createElement("button", {
    onClick: () => load(false, filters)
  }, "Search history"), /*#__PURE__*/React.createElement("button", {
    onClick: () => {
      setFilters({
        q: '',
        from: '',
        to: ''
      });
      load(false, {});
    }
  }, "Clear filters")), loading && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading decision history\u2026"), inspected && /*#__PURE__*/React.createElement("aside", {
    style: {
      border: '1px solid var(--border, #bbb)',
      padding: 12
    }
  }, /*#__PURE__*/React.createElement("strong", null, "Earlier record #", inspected.revision, " \xB7 ", inspected.body.decisionDate, " \xB7 ", inspected.superseded ? 'Superseded' : 'Recorded decision'), /*#__PURE__*/React.createElement("h4", null, inspected.body.decision), /*#__PURE__*/React.createElement("p", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, inspected.body.rationale), /*#__PURE__*/React.createElement("p", null, "Revisit when: ", inspected.body.revisitWhen), inspected.body.supersedes && /*#__PURE__*/React.createElement("button", {
    onClick: () => inspect(inspected.body.supersedes)
  }, "Read preceding decision"), /*#__PURE__*/React.createElement("button", {
    onClick: () => setInspected(null)
  }, "Close earlier record")), message && /*#__PURE__*/React.createElement("p", {
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
  }, /*#__PURE__*/React.createElement("strong", null, "Revisit when: "), r.body.revisitWhen), r.body.supersedes && /*#__PURE__*/React.createElement("button", {
    onClick: () => inspect(r.body.supersedes)
  }, "Read the decision this superseded"), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "Recorded ", new Date(r.created_at).toLocaleString())))), !rows.length && revision !== null && /*#__PURE__*/React.createElement("p", null, "No decisions match this view for ", ticker, ". Clear filters to see recent records."), more && /*#__PURE__*/React.createElement("button", {
    disabled: loading || busy,
    onClick: () => load(true)
  }, "Load older matching decisions"), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "History is ordered by when entries were recorded. Date filters use the decision date. Shared AI context still includes only the latest 20 records; searching here does not change an agent\u2019s context."));
}