import { CatalystWatch } from './catalyst-watch';
import { parseTickers } from './research-desk-model.mjs';
import * as React from 'react';
var blank = {
  ticker: '',
  hours: 0,
  lookbackDays: 30,
  kinds: ['transcript', 'broker-report'],
  workflow: 'thesis',
  topic: '',
  instructions: '',
  enabled: true
};
export function CollectionCloud({
  api,
  coverage = []
}) {
  var [data, setData] = React.useState(null),
    [cfg, setCfg] = React.useState(blank),
    [error, setError] = React.useState(''),
    [loadError, setLoadError] = React.useState(''),
    [message, setMessage] = React.useState(''),
    [busy, setBusy] = React.useState(false),
    [uncertain, setUncertain] = React.useState(false);
  var [bulk, setBulk] = React.useState(''),
    [custom, setCustom] = React.useState(false);
  var parsed = parseTickers(bulk);
  var alive = React.useRef(true),
    lock = React.useRef(false),
    pending = React.useRef(null);
  var json = async options => {
    var c = new AbortController(),
      timer = setTimeout(() => c.abort(), 20000);
    try {
      var r = await fetch(`${api}/api/collection/control`, {
        ...options,
        signal: c.signal
      });
      var d = await r.json();
      if (!r.ok) {
        var e = new Error(d.error || `Request failed (${r.status})`);
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
      var value = await json();
      if (alive.current) {
        setData(value);
        setLoadError('');
        if (pending.current && value.commands.some(c => c.id === pending.current.requestId)) {
          pending.current = null;
          setUncertain(false);
          setMessage('Command recorded. Track its Mac acknowledgement below.');
        }
      }
    } catch (e) {
      if (alive.current) setLoadError(e.message);
    }
  };
  React.useEffect(() => {
    alive.current = true;
    refresh();
    var timer = setInterval(() => {
      if (!document.hidden) refresh();
    }, 15000);
    return () => {
      alive.current = false;
      clearInterval(timer);
    };
  }, [api]);
  var send = async (action, payload) => {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    setError('');
    var req = pending.current || {
      requestId: crypto.randomUUID(),
      action,
      payload
    };
    pending.current = req;
    try {
      var d = await json({
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(req)
      });
      if (alive.current) {
        pending.current = null;
        setUncertain(false);
        setMessage(`Command ${d.status}. Collection starts only after Mac processing and browser execution.`);
        await refresh();
      }
    } catch (e) {
      if (alive.current) {
        setError(e.message);
        if (e.status) {
          pending.current = null;
          setUncertain(false);
        } else setUncertain(true);
        await refresh();
      }
    } finally {
      lock.current = false;
      if (alive.current) setBusy(false);
    }
  };
  var snapshot = data?.snapshot,
    policies = snapshot?.policies || [];
  var updated = data?.updatedAt ? new Date(data.updatedAt.replace(' ', 'T') + 'Z') : null;
  var stale = !updated || !Number.isFinite(updated.getTime()) || Date.now() - updated.getTime() > 180000;
  return /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel collection-controls"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "CLOUD CONTROLS / MAC COLLECTION"), /*#__PURE__*/React.createElement("h2", null, "Refresh your coverage from anywhere."), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Commands are stored in Charlie and picked up by your Mac agent. AlphaSense downloads still require this Mac awake, Codex running and Chrome signed in. The browser worker checks managed requests every 15 minutes."), /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, updated ? `Mac last reported ${updated.toLocaleString()}.` : 'No Mac collection report yet.', " ", stale ? 'No recent report; commands will wait for the Mac agent.' : 'Mac collection bridge is reporting.'), (error || loadError) && /*#__PURE__*/React.createElement("p", {
    role: "alert",
    className: "workspace-error"
  }, error || loadError), message && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, message), /*#__PURE__*/React.createElement("fieldset", {
    className: "desk-form",
    disabled: busy || uncertain
  }, /*#__PURE__*/React.createElement("div", {
    className: "desk-filter"
  }, /*#__PURE__*/React.createElement("label", null, "Ticker", /*#__PURE__*/React.createElement("input", {
    value: cfg.ticker,
    maxLength: 20,
    onChange: e => setCfg({
      ...cfg,
      ticker: e.target.value.toUpperCase().trim()
    }),
    placeholder: "MDT"
  })), /*#__PURE__*/React.createElement("label", null, "Frequency", /*#__PURE__*/React.createElement("select", {
    value: custom ? 'custom' : cfg.hours,
    onChange: e => {
      setCustom(e.target.value === 'custom');
      setCfg({
        ...cfg,
        hours: e.target.value === 'custom' ? cfg.hours || 168 : Number(e.target.value)
      });
    }
  }, [[0, 'Manual'], [1, 'Hourly'], [4, 'Every 4 hours'], [12, 'Every 12 hours'], [24, 'Daily'], [168, 'Weekly'], [336, 'Biweekly (14 days)'], [720, 'Monthly (30 days)'], ['custom', 'Custom interval']].map(([v, l]) => /*#__PURE__*/React.createElement("option", {
    key: v,
    value: v
  }, l)))), custom && /*#__PURE__*/React.createElement("label", null, "Custom interval (hours)", /*#__PURE__*/React.createElement("input", {
    type: "number",
    min: "1",
    max: "8760",
    step: "1",
    value: cfg.hours || 1,
    onChange: e => setCfg({
      ...cfg,
      hours: Number(e.target.value)
    })
  })), /*#__PURE__*/React.createElement("label", null, "Initial lookback days", /*#__PURE__*/React.createElement("input", {
    type: "number",
    min: "1",
    max: "365",
    value: cfg.lookbackDays,
    onChange: e => setCfg({
      ...cfg,
      lookbackDays: Number(e.target.value)
    })
  }))), /*#__PURE__*/React.createElement("label", null, "Destination and workflow", /*#__PURE__*/React.createElement("select", {
    value: cfg.workflow,
    onChange: e => setCfg({
      ...cfg,
      workflow: e.target.value
    })
  }, /*#__PURE__*/React.createElement("option", {
    value: "thesis"
  }, "STOCKS \u2192 thesis proposal intake"), /*#__PURE__*/React.createElement("option", {
    value: "note"
  }, "STOCKS \u2192 draft note and existing intake"), /*#__PURE__*/React.createElement("option", {
    value: "recap"
  }, "CATALYSTS \u2192 event recap"))), cfg.workflow === 'recap' && /*#__PURE__*/React.createElement("label", null, "Event folder", /*#__PURE__*/React.createElement("input", {
    value: cfg.topic,
    maxLength: 160,
    placeholder: "MDT F1Q27 Earnings",
    onChange: e => setCfg({
      ...cfg,
      topic: e.target.value
    })
  })), /*#__PURE__*/React.createElement("div", {
    className: "desk-filter collection-checks"
  }, [['transcript', 'Event transcripts'], ['broker-report', 'Broker reports'], ['press-release', 'Press releases']].map(([k, l]) => /*#__PURE__*/React.createElement("label", {
    key: k
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: cfg.kinds.includes(k),
    onChange: () => setCfg({
      ...cfg,
      kinds: cfg.kinds.includes(k) ? cfg.kinds.filter(v => v !== k) : [...cfg.kinds, k]
    })
  }), l)), /*#__PURE__*/React.createElement("label", null, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: cfg.enabled,
    onChange: e => setCfg({
      ...cfg,
      enabled: e.target.checked
    })
  }), "Policy enabled")), /*#__PURE__*/React.createElement("label", null, "Collection instructions", /*#__PURE__*/React.createElement("textarea", {
    rows: 3,
    maxLength: 3000,
    value: cfg.instructions,
    onChange: e => setCfg({
      ...cfg,
      instructions: e.target.value
    }),
    placeholder: "Prioritize earnings transcripts and material guidance changes\u2026"
  })), /*#__PURE__*/React.createElement("label", null, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: !!cfg.createFolder,
    onChange: e => setCfg({
      ...cfg,
      createFolder: e.target.checked
    })
  }), "Create the ticker folder if it does not exist"), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    disabled: !cfg.ticker || !cfg.kinds.length,
    onClick: () => send('save', cfg)
  }, "Save policy on Mac"), /*#__PURE__*/React.createElement("button", {
    disabled: !cfg.ticker || !cfg.kinds.length,
    onClick: () => send('save_trigger', cfg)
  }, "Save & refresh now")), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Configure a coverage group"), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Apply the frequency, source types and workflow above to multiple tickers. Existing policies for these tickers will be replaced. The Mac validates every destination before saving the group; one invalid destination rejects the whole group."), /*#__PURE__*/React.createElement("fieldset", {
    disabled: busy || uncertain,
    className: "desk-form"
  }, /*#__PURE__*/React.createElement("label", null, "Tickers", /*#__PURE__*/React.createElement("textarea", {
    rows: 3,
    value: bulk,
    onChange: e => setBulk(e.target.value),
    placeholder: "MDT, ABT, DE"
  })), /*#__PURE__*/React.createElement("button", {
    onClick: () => setBulk([...new Set([...coverage.map(a => a.ticker), ...policies.map(p => p.ticker)].filter(Boolean))].join(', '))
  }, "Load covered companies"), /*#__PURE__*/React.createElement("p", null, parsed.tickers.length, "/100 tickers \xB7 ", cfg.hours ? `Every ${cfg.hours} hours` : 'Manual refresh', " \xB7 ", cfg.kinds.join(', '), " \xB7 ", cfg.workflow === 'recap' ? `CATALYSTS / ticker / ${cfg.topic}` : 'STOCKS / ticker'), parsed.invalid.length > 0 && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, "Invalid tickers: ", parsed.invalid.join(', ')), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    disabled: !parsed.tickers.length || parsed.tickers.length > 100 || !!parsed.invalid.length,
    onClick: () => send('save_batch', {
      policies: parsed.tickers.map(ticker => ({
        ...cfg,
        ticker
      }))
    })
  }, "Save ", parsed.tickers.length, " ticker policies on Mac"))), uncertain && /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => send()
  }, "Retry same command"), /*#__PURE__*/React.createElement("button", {
    onClick: refresh,
    disabled: busy
  }, "Check status"), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "The existing ticker folder must exist in iCloud. A recap event subfolder may be created. Source restrictions and duplicate checks remain enforced. Research generation uses configured API credits; drafts still require review."), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Coverage setup gaps"), /*#__PURE__*/React.createElement("p", null, "Covered companies without a managed policy: ", coverage.filter(a => !policies.some(p => p.ticker === a.ticker)).map(a => a.ticker).join(', ') || 'None'), /*#__PURE__*/React.createElement("p", null, "Mac-reported STOCKS folders: ", snapshot?.folders?.stocks?.join(', ') || 'Not yet reported'), /*#__PURE__*/React.createElement("p", null, "Mac-reported CATALYSTS folders: ", snapshot?.folders?.catalysts?.join(', ') || 'Not yet reported')), /*#__PURE__*/React.createElement(CatalystWatch, {
    api: api,
    policies: policies
  }), /*#__PURE__*/React.createElement("h3", null, "Policies reported by Mac \xB7 ", policies.length), policies.map(p => /*#__PURE__*/React.createElement("div", {
    className: "desk-row",
    key: p.ticker
  }, /*#__PURE__*/React.createElement("button", {
    disabled: busy || uncertain,
    onClick: () => {
      setCfg({
        ...p
      });
      setCustom(![0, 1, 4, 12, 24, 168, 336, 720].includes(p.hours));
    }
  }, /*#__PURE__*/React.createElement("strong", null, p.ticker), /*#__PURE__*/React.createElement("span", null, p.enabled ? p.hours ? `Every ${p.hours} hours` : 'Manual' : 'Paused', " \xB7 ", p.workflow === 'recap' ? `CATALYSTS / ${p.topic}` : 'STOCKS')), /*#__PURE__*/React.createElement("small", null, "Last verified: ", p.lastSuccess || 'Never', " \xB7 Next due: ", p.nextDue ? new Date(p.nextDue * 1000).toLocaleString() : 'Manual / paused'), /*#__PURE__*/React.createElement("button", {
    disabled: busy || uncertain,
    onClick: () => send('save', {
      ...p,
      enabled: !p.enabled
    })
  }, p.enabled ? 'Pause schedule' : 'Resume schedule'), /*#__PURE__*/React.createElement("button", {
    disabled: busy || uncertain,
    onClick: () => send('trigger', {
      ticker: p.ticker
    })
  }, "Refresh now"))), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Cloud commands \xB7 ", data?.commands?.length || 0), (data?.commands || []).map(c => /*#__PURE__*/React.createElement("p", {
    key: c.id
  }, c.ticker || 'Coverage group', " \xB7 ", c.input?.action, " \xB7 ", c.status === 'applied' ? 'Applied on Mac' : c.status, c.error ? ` · ${c.error}` : ''))), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Browser collection requests \xB7 ", snapshot?.requests?.length || 0), (snapshot?.requests || []).map(r => /*#__PURE__*/React.createElement("p", {
    key: r.id
  }, r.ticker, " \xB7 ", r.status, " \xB7 ", r.issue || (r.result?.newDocuments != null ? `${r.result.newDocuments} new eligible documents` : 'No completed result reported'), ['needs_auth', 'attention'].includes(r.status) && /*#__PURE__*/React.createElement("button", {
    disabled: busy || uncertain,
    onClick: () => send('retry', {
      ticker: r.ticker,
      refreshRequestId: r.id
    })
  }, r.status === 'needs_auth' ? 'Signed in on Mac — resume' : 'Retry after resolving'), !['complete', 'cancelled'].includes(r.status) && /*#__PURE__*/React.createElement("button", {
    disabled: busy || uncertain,
    onClick: () => send('cancel', {
      ticker: r.ticker,
      refreshRequestId: r.id
    })
  }, "Cancel refresh")))), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Cancellation stops future work when acknowledged on Mac; an in-progress browser download may finish. Complete sign-in directly in AlphaSense before resuming an authentication-paused request."));
}