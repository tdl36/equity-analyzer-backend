import * as React from 'react';
import { parseTickers } from './research-desk-model.mjs';
export function CatalystWatch({
  api,
  policies
}) {
  var [data, setData] = React.useState(null),
    [error, setError] = React.useState(''),
    [message, setMessage] = React.useState(''),
    [busy, setBusy] = React.useState(false);
  var [enabled, setEnabled] = React.useState(false),
    [automatic, setAutomatic] = React.useState(false),
    [tickers, setTickers] = React.useState(''),
    [limit, setLimit] = React.useState(10);
  var dirty = React.useRef(false),
    lock = React.useRef(false),
    alive = React.useRef(true);
  var json = async options => {
    var r = await fetch(`${api}/api/research/catalyst-watch`, {
      ...options,
      signal: AbortSignal.timeout(20000)
    });
    var d = await r.json();
    if (!r.ok) throw Error(d.error || `Catalyst detection unavailable (${r.status})`);
    return d;
  };
  var refresh = async () => {
    try {
      var d = await json();
      if (!alive.current) return;
      setData(d);
      setError('');
      if (!dirty.current) {
        setEnabled(d.config.enabled);
        setAutomatic(d.config.automatic);
        setTickers(d.config.tickers.join(', '));
        setLimit(d.config.dailyLimit);
      }
    } catch (e) {
      if (alive.current) setError(e.message);
    }
  };
  React.useEffect(() => {
    alive.current = true;
    refresh();
    var timer = setInterval(() => {
      if (!document.hidden) refresh();
    }, 30000);
    return () => {
      alive.current = false;
      clearInterval(timer);
    };
  }, [api]);
  var parsed = parseTickers(tickers);
  var edit = fn => {
    dirty.current = true;
    fn();
  };
  var save = async () => {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    setMessage('');
    try {
      var d = await json({
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          enabled,
          automatic,
          tickers: parsed.tickers,
          dailyLimit: limit
        })
      });
      if (alive.current) {
        setData(d);
        dirty.current = false;
        setMessage('Catalyst watch settings saved. New signals appear below as the Mac checks each company.');
      }
    } catch (e) {
      if (alive.current) setMessage(`Save not confirmed: ${e.message}. Refresh status before retrying.`);
    } finally {
      lock.current = false;
      if (alive.current) setBusy(false);
    }
  };
  return /*#__PURE__*/React.createElement("details", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("summary", null, "Event-driven research \xB7 catalyst watch"), /*#__PURE__*/React.createElement("h3", null, "Investigate new developments as they appear."), /*#__PURE__*/React.createElement("p", null, "Company-news headlines are screened for clinical results, regulatory decisions, guidance and corporate transactions. Matches are potential catalysts, not verified investment conclusions. Primary-source confirmation happens in the subsequent research."), error && /*#__PURE__*/React.createElement("p", {
    className: "workspace-error",
    role: "alert"
  }, error), data && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, data.scope), !data.hasNewsKey && /*#__PURE__*/React.createElement("p", {
    className: "workspace-error"
  }, "Set the Finnhub key in Settings to enable news detection."), data.config.lastIssue && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, data.config.lastIssue), /*#__PURE__*/React.createElement("fieldset", {
    disabled: busy,
    className: "desk-form"
  }, /*#__PURE__*/React.createElement("label", null, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: enabled,
    onChange: e => edit(() => setEnabled(e.target.checked))
  }), "Monitor new events"), /*#__PURE__*/React.createElement("label", null, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: automatic,
    onChange: e => edit(() => setAutomatic(e.target.checked))
  }), "Automatically queue AlphaSense collection and an analyst recap draft"), /*#__PURE__*/React.createElement("label", null, "Watched tickers", /*#__PURE__*/React.createElement("textarea", {
    rows: 3,
    value: tickers,
    onChange: e => edit(() => setTickers(e.target.value))
  })), /*#__PURE__*/React.createElement("button", {
    onClick: () => edit(() => setTickers(policies.filter(p => p.enabled).map(p => p.ticker).join(', ')))
  }, "Use active collection coverage"), /*#__PURE__*/React.createElement("label", null, "Maximum automatic event requests per UTC day", /*#__PURE__*/React.createElement("input", {
    type: "number",
    min: "1",
    max: "30",
    value: limit,
    onChange: e => edit(() => setLimit(Number(e.target.value)))
  })), /*#__PURE__*/React.createElement("p", null, parsed.tickers.length, " watched companies \xB7 ", data.config.used || 0, " event requests counted on ", data.config.day || 'no day yet', ". Signals above the limit remain visible for manual research; they are not silently backfilled."), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    disabled: !parsed.tickers.length || !!parsed.invalid.length || parsed.tickers.length > 100,
    onClick: save
  }, "Save catalyst watch")), /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, message), /*#__PURE__*/React.createElement("button", {
    onClick: refresh
  }, "Refresh detection status"), /*#__PURE__*/React.createElement("p", null, "Last lookup: ", data.config.lastTicker || 'None', " \xB7 ", data.config.lastCheck ? new Date(data.config.lastCheck * 1000).toLocaleString() : 'Not checked yet'), (data.events || []).map(e => /*#__PURE__*/React.createElement("article", {
    className: "amendment-card",
    key: e.id
  }, /*#__PURE__*/React.createElement("strong", null, e.ticker, " \xB7 ", e.status.replaceAll('_', ' ')), /*#__PURE__*/React.createElement("p", null, e.input?.title), /*#__PURE__*/React.createElement("p", null, e.input?.reason), typeof e.input?.url === 'string' && e.input.url.startsWith('https://') && /*#__PURE__*/React.createElement("a", {
    href: e.input.url,
    target: "_blank",
    rel: "noopener noreferrer"
  }, "Read original news \u2192"), /*#__PURE__*/React.createElement("p", null, e.result?.reason || `Collection command: ${e.result?.commandId || 'Not queued'}`))), !data.events?.length && /*#__PURE__*/React.createElement("p", null, "No new qualifying signals have been recorded since monitoring was enabled.")));
}