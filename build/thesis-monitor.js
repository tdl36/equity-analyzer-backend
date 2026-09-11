import * as React from 'react';
export function ThesisMonitor({
  api,
  ticker,
  disabled
}) {
  var [data, setData] = React.useState(null),
    [error, setError] = React.useState(''),
    [busy, setBusy] = React.useState(false);
  var live = React.useRef(true),
    sequence = React.useRef(0),
    lock = React.useRef(false);
  async function load() {
    var n = ++sequence.current;
    try {
      var r = await fetch(`${api}/api/research/investment-case/${ticker}/monitor`, {
        signal: AbortSignal.timeout(20000)
      });
      var d = await r.json();
      if (!r.ok) throw Error(d.error || 'Monitoring unavailable');
      if (live.current && n === sequence.current) {
        setData(d);
        setError('');
      }
    } catch (e) {
      if (live.current && n === sequence.current) setError(e.message);
    }
  }
  React.useEffect(() => {
    live.current = true;
    load();
    var timer = setInterval(load, 20000);
    return () => {
      live.current = false;
      clearInterval(timer);
    };
  }, [api, ticker]);
  async function save(enabled, resetPending = false) {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    try {
      var r = await fetch(`${api}/api/research/investment-case/${ticker}/monitor`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          enabled,
          revision: data.revision,
          resetPending
        }),
        signal: AbortSignal.timeout(20000)
      });
      var d = await r.json();
      if (!r.ok) throw Error(d.error || 'Could not save monitoring');
      if (live.current) {
        setData(d);
        setError('');
      }
    } catch (e) {
      if (live.current) setError(e.message + ' Reload status before retrying.');
    } finally {
      lock.current = false;
      if (live.current) setBusy(false);
    }
  }
  return /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel thesis-monitor"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "NEW EVIDENCE \u2192 THESIS REVIEW"), /*#__PURE__*/React.createElement("h3", null, "Monitor newly imported sources."), /*#__PURE__*/React.createElement("p", null, "Once enabled, the connected Mac agent checks for new or changed originals in Charlie and queues an evidence comparison. Your saved underweight conditions inform that review. Changes still require your acceptance."), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "The first enrollment records existing files as the baseline. It does not download documents or backfill old research. Up to 10 new sources are compared per batch; later batches wait for the current proposal to be closed. Uses configured server model credits. While paused, new files remain eligible when resumed."), data && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("strong", null, data.enabled ? 'Monitoring enabled' : 'Monitoring paused'), " \xB7 ", data.baselineSources || 0, " known source files"), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    disabled: disabled || busy,
    onClick: () => save(!data.enabled)
  }, busy ? 'Saving…' : data.enabled ? 'Pause monitoring' : 'Enable prospective monitoring'), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: load
  }, "Refresh monitoring status"), data.checked_at && /*#__PURE__*/React.createElement("p", null, "Last check: ", new Date(data.checked_at).toLocaleString()), data.last_result?.message && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, data.last_result.state, ": ", data.last_result.message), data.last_result?.filenames && /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Sources submitted in latest batch"), data.last_result.filenames.map(n => /*#__PURE__*/React.createElement("p", {
    key: n
  }, n)), /*#__PURE__*/React.createElement("p", null, "Proposal: ", data.last_result.proposalId)), data.pending && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, "A submission reservation is retained for retry. No source is marked processed until a proposal receipt is confirmed."), /*#__PURE__*/React.createElement("button", {
    disabled: disabled || busy,
    onClick: () => save(data.enabled, true)
  }, "Recheck reservation against latest baseline"))), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error));
}