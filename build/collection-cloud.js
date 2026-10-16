function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
import { commandRecoveryKey, readCommand, retainCommand, clearCommand, commandReceipt, exactCommand } from './collection-command-recovery.mjs';
import { CollectionSynthesisStatus } from './collection-synthesis';
import { CollectionRequests } from './collection-requests';
import { parseTimestamp, workId } from './workspace-model.mjs';
import { CatalystWatch } from './catalyst-watch';
import { parseTickers } from './research-desk-model.mjs';
import * as React from 'react';
import { collectionHealth } from './collection-health.mjs';
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
export function CollectionCloud(props) {
  return /*#__PURE__*/React.createElement(CollectionCloudSession, _extends({
    key: props.api
  }, props));
}
function CollectionCloudSession({
  api,
  coverage = [],
  initialTicker = '',
  routeWork = '',
  onWorkChange
}) {
  var recoveryKey = commandRecoveryKey(api);
  var [restored] = React.useState(() => {
    try {
      return {
        body: readCommand(sessionStorage, recoveryKey),
        error: ''
      };
    } catch (e) {
      return {
        body: null,
        error: e.message
      };
    }
  });
  var [recoveryIssue, setRecoveryIssue] = React.useState(restored.error);
  var [data, setData] = React.useState(null),
    [cfg, setCfg] = React.useState(blank),
    [error, setError] = React.useState(''),
    [loadError, setLoadError] = React.useState(''),
    [message, setMessage] = React.useState(''),
    [busy, setBusy] = React.useState(false),
    [uncertain, setUncertain] = React.useState(!!restored.body || !!restored.error);
  var [bulk, setBulk] = React.useState(''),
    [custom, setCustom] = React.useState(false);
  var [query, setQuery] = React.useState(initialTicker),
    [policyOpen, setPolicyOpen] = React.useState(false);
  var policyEditor = React.useRef(null);
  var parsed = parseTickers(bulk);
  var alive = React.useRef(true),
    lock = React.useRef(false),
    pending = React.useRef(restored.body),
    readSerial = React.useRef(0);
  var json = async (options, query = '') => {
    var c = new AbortController(),
      timer = setTimeout(() => c.abort(), 20000);
    try {
      var r = await fetch(`${api}/api/collection/control${query}`, {
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
    var serial = ++readSerial.current,
      current = () => alive.current && serial === readSerial.current;
    try {
      var value = await json();
      if (!current()) return;
      setData(value);
      setLoadError('');
      var body;
      try {
        body = readCommand(sessionStorage, recoveryKey);
        setRecoveryIssue('');
      } catch (e) {
        setRecoveryIssue(e.message);
        setUncertain(true);
        return;
      }
      pending.current = body;
      setUncertain(!!body);
      if (body) {
        var receipt = exactCommand(await json(undefined, '?requestId=' + encodeURIComponent(body.requestId)), body);
        if (!current()) return;
        if (receipt) {
          clearCommand(sessionStorage, recoveryKey, body);
          pending.current = null;
          setUncertain(false);
          setMessage(`Command ${receipt.status === 'applied' ? 'applied on Mac' : receipt.status}. Its exact saved receipt was recovered; browser collection may still be pending.${typeof receipt.error === 'string' && receipt.error ? ' ' + receipt.error : ''}`);
        }
      }
    } catch (e) {
      if (current()) setLoadError(e.message);
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
      readSerial.current++;
      clearInterval(timer);
    };
  }, [api]);
  var send = async (action, payload) => {
    if (lock.current || recoveryIssue) return;
    lock.current = true;
    setBusy(true);
    setError('');
    var req = pending.current || {
      requestId: crypto.randomUUID(),
      action,
      payload
    };
    try {
      retainCommand(sessionStorage, recoveryKey, req);
      pending.current = req;
      setUncertain(true);
      var d = commandReceipt(await json({
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(req)
      }), req);
      if (alive.current) {
        clearCommand(sessionStorage, recoveryKey, req);
        pending.current = null;
        setUncertain(false);
        setMessage(`Command ${d.status}. Collection starts only after Mac processing and browser execution.`);
        await refresh();
      }
    } catch (e) {
      if (alive.current) {
        setError(e.message);
        if (e.status === 400 || e.status === 409) {
          try {
            clearCommand(sessionStorage, recoveryKey, req);
            pending.current = null;
            setUncertain(false);
          } catch (clearError) {
            setRecoveryIssue(clearError.message);
          }
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
  var health = collectionHealth(snapshot);
  var updated = parseTimestamp(data?.updatedAt);
  var stale = !updated || Date.now() - updated.getTime() < 0 || Date.now() - updated.getTime() > 180000;
  var visiblePolicies = policies.filter(p => !query.trim() || p.ticker.toUpperCase().includes(query.trim().toUpperCase()));
  return /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel collection-controls"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "CLOUD CONTROLS / MAC COLLECTION"), /*#__PURE__*/React.createElement("h2", null, "Collection & coverage."), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Commands are stored in Charlie and picked up by your Mac agent. AlphaSense downloads still require this Mac awake, Codex running and Chrome signed in. The scheduled browser worker picks up managed requests."), /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, updated ? `Mac last reported ${updated.toLocaleString()}.` : 'No Mac collection report yet.', " ", stale ? 'No recent report; commands will wait for the Mac agent.' : 'Mac collection bridge is reporting.'), /*#__PURE__*/React.createElement("div", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("h3", null, "Automation readiness"), /*#__PURE__*/React.createElement("p", null, health.scheduled, " scheduled tickers \xB7 ", health.verified, "/", health.total, " policies with a successful refresh recorded \xB7 ", health.pending, " open requests \xB7 ", health.needsAttention, " need attention"), health.overdue > 0 && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, health.overdue, " queued requests have waited over 30 minutes; oldest ", health.oldestMinutes, " minutes. A connected Mac is not proof that browser collection is progressing. Inspect the requests below and the browser worker."), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Counts reflect the latest Mac snapshot. A successful refresh is delivery evidence, not a guarantee of complete source coverage or research accuracy.")), recoveryIssue && /*#__PURE__*/React.createElement("p", {
    role: "alert",
    className: "workspace-error"
  }, recoveryIssue, " No new collection commands will be sent."), uncertain && pending.current && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "An unresolved collection command is retained in this browser tab. Check status to find its exact receipt, or retry the same command. New commands are blocked until it is resolved."), (error || loadError) && /*#__PURE__*/React.createElement("p", {
    role: "alert",
    className: "workspace-error"
  }, error || loadError), message && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, message), /*#__PURE__*/React.createElement("a", {
    className: "workspace-primary",
    href: "#view=desk&section=ready",
    style: {
      display: 'inline-block',
      marginBottom: 16
    }
  }, "Ready to process \u2014 choose synthesis jobs \u2192"), /*#__PURE__*/React.createElement(CollectionSynthesisStatus, {
    api: api
  }), /*#__PURE__*/React.createElement(CollectionRequests, {
    focusId: workId(routeWork, 'collection'),
    onClearFocus: () => onWorkChange?.(''),
    requests: snapshot?.requests || [],
    query: query,
    onQuery: setQuery,
    busy: busy || uncertain,
    onAction: send,
    available: !!snapshot
  }), /*#__PURE__*/React.createElement("details", {
    ref: policyEditor,
    open: policyOpen,
    onToggle: e => setPolicyOpen(e.currentTarget.open)
  }, /*#__PURE__*/React.createElement("summary", null, "Create or edit a collection policy"), /*#__PURE__*/React.createElement("p", null, "Choose a reported policy below to edit its existing settings, or configure a new one. Nothing changes until you save."), /*#__PURE__*/React.createElement("fieldset", {
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
  }, [['transcript', 'Event transcripts'], ['broker-report', 'Broker reports'], ['press-release', 'Press releases'], ['presentation', 'Presentations']].map(([k, l]) => /*#__PURE__*/React.createElement("label", {
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
  }, "Save ", parsed.tickers.length, " ticker policies on Mac")))), uncertain && pending.current && /*#__PURE__*/React.createElement("button", {
    disabled: busy || !!recoveryIssue,
    onClick: () => send()
  }, "Retry same command"), /*#__PURE__*/React.createElement("button", {
    onClick: refresh,
    disabled: busy
  }, "Check status"), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "The existing ticker folder must exist in iCloud. A recap event subfolder may be created. Source restrictions and duplicate checks remain enforced. Research generation uses configured API credits; drafts still require review."), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Coverage setup gaps"), /*#__PURE__*/React.createElement("p", null, "Covered companies without a managed policy: ", coverage.filter(a => !policies.some(p => p.ticker === a.ticker)).map(a => a.ticker).join(', ') || 'None'), /*#__PURE__*/React.createElement("p", null, "Mac-reported STOCKS folders: ", snapshot?.folders?.stocks?.join(', ') || 'Not yet reported'), /*#__PURE__*/React.createElement("p", null, "Mac-reported CATALYSTS folders: ", snapshot?.folders?.catalysts?.join(', ') || 'Not yet reported')), /*#__PURE__*/React.createElement(CatalystWatch, {
    api: api,
    policies: policies
  }), /*#__PURE__*/React.createElement("h3", null, "Matching policies \xB7 ", visiblePolicies.length, " / ", policies.length), visiblePolicies.map(p => /*#__PURE__*/React.createElement("div", {
    className: "desk-row",
    key: p.ticker
  }, /*#__PURE__*/React.createElement("button", {
    disabled: busy || uncertain,
    onClick: () => {
      setCfg({
        ...p
      });
      setPolicyOpen(true);
      requestAnimationFrame(() => policyEditor.current?.scrollIntoView({
        block: 'nearest',
        behavior: 'smooth'
      }));
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
  }, c.ticker || 'Coverage group', " \xB7 ", c.input?.action, " \xB7 ", c.status === 'applied' ? 'Applied on Mac' : c.status, c.error ? ` · ${c.error}` : ''))));
}