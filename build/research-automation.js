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
  useEffect(() => {
    var live = true;
    var update = async () => {
      try {
        var r = await fetch(`${api}/api/research/automation`);
        if (!r.ok) throw new Error(`Automation status unavailable (${r.status})`);
        var d = await r.json();
        if (live) setData(d);
      } catch (e) {
        if (live) setError(e.message);
      }
    };
    update();
    var timer = setInterval(update, 30000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [api]);
  var toggle = async () => {
    setBusy(true);
    setError('');
    try {
      var r = await fetch(`${api}/api/research/automation`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          enabled: !data.enabled
        })
      });
      var d = await r.json();
      if (!r.ok) throw new Error(d.error || 'Could not update automation');
      setData(d);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
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
    onClick: toggle
  }, busy ? 'Saving…' : data.enabled ? 'Pause automation' : 'Enable automation')), /*#__PURE__*/React.createElement("p", {
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
  }, /*#__PURE__*/React.createElement("strong", null, e.ticker, " \xB7 ", e.input?.filename), /*#__PURE__*/React.createElement("small", null, e.status === 'submitted' ? 'Comparison job submitted' : e.status, e.error ? ` · ${e.error}` : ''))))) : /*#__PURE__*/React.createElement("p", null, "Loading automation status\u2026"));
}