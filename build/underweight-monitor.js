import * as React from 'react';
export function UnderweightMonitor({
  api,
  onOpen,
  disabled
}) {
  var [data, setData] = React.useState(null),
    [error, setError] = React.useState(''),
    [filter, setFilter] = React.useState('open');
  React.useEffect(() => {
    var live = true;
    async function load() {
      try {
        var r = await fetch(api + '/api/research/underweights', {
          signal: AbortSignal.timeout(20000)
        });
        var d = await r.json();
        if (!r.ok) throw Error(d.error || 'Underweights unavailable');
        if (live) {
          setData(d);
          setError('');
        }
      } catch (e) {
        if (live) setError(e.message);
      }
    }
    load();
    var timer = setInterval(load, 60000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [api]);
  var rows = (data?.records || []).filter(r => filter === 'all' || r.body.status === 'open');
  return /*#__PURE__*/React.createElement("details", {
    className: "workspace-panel underweight-monitor",
    open: true
  }, /*#__PURE__*/React.createElement("summary", null, "Underweight monitor \xB7 ", data?.records.length ?? '…', " recorded reviews"), /*#__PURE__*/React.createElement("p", null, "Which nonownership decisions deserve another look? Weights and condition assessments below are analyst-recorded, not live portfolio feeds."), /*#__PURE__*/React.createElement("label", null, "Show", /*#__PURE__*/React.createElement("select", {
    value: filter,
    onChange: e => setFilter(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: "open"
  }, "Open reviews"), /*#__PURE__*/React.createElement("option", {
    value: "all"
  }, "All latest reviews"))), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error), data && !rows.length && /*#__PURE__*/React.createElement("p", null, "No matching underweight reviews. Open a company below, then choose Decisions & underweights to establish one."), /*#__PURE__*/React.createElement("div", {
    className: "underweight-grid"
  }, rows.map(r => {
    var conditions = r.body.reviewConditions || [],
      met = conditions.filter(c => c.state === 'met' || c.state === 'partly_met').length;
    return /*#__PURE__*/React.createElement("article", {
      key: r.id
    }, /*#__PURE__*/React.createElement("p", {
      className: "workspace-eyebrow"
    }, r.ticker, " \xB7 ", r.body.reason.replaceAll('_', ' ')), /*#__PURE__*/React.createElement("h3", null, r.body.title), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("strong", null, r.body.activeWeightPct, "% active weight"), " \xB7 ", r.body.mandate), /*#__PURE__*/React.createElement("p", null, "Versus ", r.body.benchmark, " \xB7 inputs ", r.body.asOf, (Date.now() - Date.parse(r.body.asOf)) / 86400000 > 30 ? ' · Older than 30 days' : ''), /*#__PURE__*/React.createElement("p", null, r.body.rationale), /*#__PURE__*/React.createElement("p", null, conditions.length ? `${met} of ${conditions.length} conditions assessed as met / partly met` : 'No structured reconsideration conditions recorded'), /*#__PURE__*/React.createElement("p", null, "Next review ", r.body.dueDate, " \xB7 ", r.body.owner), /*#__PURE__*/React.createElement("p", null, "Decision: ", (r.body.reviewDecision || 'pending').replaceAll('_', ' ')), /*#__PURE__*/React.createElement("button", {
      disabled: disabled,
      onClick: () => onOpen(r.ticker)
    }, "Review ", r.ticker));
  })), data?.limited && /*#__PURE__*/React.createElement("p", null, "Showing the first 200 latest reviews by due date. This is not an exhaustive portfolio inventory."));
}