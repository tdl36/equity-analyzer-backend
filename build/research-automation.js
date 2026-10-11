import * as React from 'react';
var {
  useState,
  useEffect
} = React;
export function ResearchAutomationControl({
  api
}) {
  var [data, setData] = useState(null),
    [error, setError] = useState(''),
    [busy, setBusy] = useState(false);
  var mounted = React.useRef(false),
    mutating = React.useRef(false),
    serial = React.useRef(0);
  useEffect(() => {
    var live = true,
      pending = false;
    var c = new AbortController();
    mounted.current = true;
    var update = async () => {
      if (pending || mutating.current) return;
      pending = true;
      var generation = serial.current;
      try {
        var r = await fetch(`${api}/api/research/automation`, {
          signal: AbortSignal.any([c.signal, AbortSignal.timeout(20000)])
        });
        if (!r.ok) throw new Error(`Automation status unavailable (${r.status})`);
        var d = await r.json();
        if (live && generation === serial.current) {
          setData(d);
          setError('');
        }
      } catch (e) {
        if (live && generation === serial.current) setError(e.message);
      } finally {
        pending = false;
      }
    };
    update();
    var timer = setInterval(update, 30000);
    return () => {
      live = false;
      mounted.current = false;
      c.abort();
      clearInterval(timer);
    };
  }, [api]);
  var updatePolicy = async (enabled, dailyLimit) => {
    if (mutating.current) return;
    mutating.current = true;
    serial.current++;
    setBusy(true);
    setError('');
    try {
      var r = await fetch(`${api}/api/research/automation`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          enabled,
          dailyLimit
        }),
        signal: AbortSignal.timeout(20000)
      });
      var d = await r.json();
      if (!r.ok) throw new Error(d.error || 'Could not update automation');
      if (mounted.current) setData(d);
    } catch (e) {
      if (mounted.current) setError('Settings could not be confirmed. Check the next refreshed status before changing them again. ' + e.message);
    } finally {
      mutating.current = false;
      if (mounted.current) setBusy(false);
    }
  };
  return /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel thesis-amendments"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "AUTOMATIC SOURCE INTAKE"), /*#__PURE__*/React.createElement("h3", null, "New files \u2192 thesis proposals"), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Watches new supported documents in STOCKS ticker roots and new uploads to Charlie. Imports needed files through your local agent, then prepares proposals for companies with saved theses. Applying edits remains your decision."), error && /*#__PURE__*/React.createElement("p", {
    className: "workspace-error",
    role: "alert"
  }, error), data ? /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "evidence-summary"
  }, /*#__PURE__*/React.createElement("strong", null, data.enabled ? 'Enabled' : 'Paused'), /*#__PURE__*/React.createElement("span", null, data.usedToday, " / ", data.dailyLimit, " daily comparisons reserved (UTC)"), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => updatePolicy(!data.enabled, data.dailyLimit)
  }, busy ? 'Saving…' : data.enabled ? 'Pause automation' : 'Enable automation')), /*#__PURE__*/React.createElement("label", null, "Daily comparison limit", /*#__PURE__*/React.createElement("select", {
    "aria-label": "Daily comparison limit",
    value: data.dailyLimit,
    disabled: busy,
    onChange: e => updatePolicy(data.enabled, Number(e.target.value))
  }, [1, 2, 3, 4, 5].map(n => /*#__PURE__*/React.createElement("option", {
    key: n,
    value: n
  }, n, " comparisons per UTC day")))), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Changing the limit can allow pending comparisons to run within your existing budget. Each comparison may use multiple model calls."), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Only new filenames are detected; replacing an existing file does not trigger another run. Existing documents become the starting inventory when enabled. One unresolved proposal per company; pending files wait for your review."), !data.manifestInitialized && /*#__PURE__*/React.createElement("p", {
    className: "workspace-notice"
  }, "Waiting for the local agent\u2019s first inventory to establish the iCloud baseline."), !data.hasServerKey && /*#__PURE__*/React.createElement("p", {
    className: "workspace-notice"
  }, "A server-side model key is required. Browser-only API keys cannot run unattended jobs."), data.lastIssue && /*#__PURE__*/React.createElement("p", {
    className: "workspace-notice"
  }, data.lastIssue), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Recent intake (", data.events.length, " most recent files)"), data.events.map(e => /*#__PURE__*/React.createElement("article", {
    className: "evidence-document",
    key: e.id
  }, /*#__PURE__*/React.createElement("strong", null, e.ticker, " \xB7 ", e.input?.filename), /*#__PURE__*/React.createElement("small", null, e.status === 'submitted' ? 'Comparison job submitted' : e.status, e.error ? ` · ${e.error}` : ''), e.waitReason && /*#__PURE__*/React.createElement("p", null, e.waitReason))))) : /*#__PURE__*/React.createElement("p", null, "Loading automation status\u2026"));
}