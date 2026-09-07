import * as React from 'react';
import { prioritizedResearch } from './research-priorities.mjs';
export function PortfolioPriorities({
  api,
  analyses,
  activities,
  onCompany,
  onPlan
}) {
  var [profile, setProfile] = React.useState(null),
    [positions, setPositions] = React.useState([]),
    [asOf, setAsOf] = React.useState(new Date().toISOString().slice(0, 10)),
    [error, setError] = React.useState(''),
    [busy, setBusy] = React.useState(false),
    [loaded, setLoaded] = React.useState(false);
  var alive = React.useRef(true),
    lock = React.useRef(false);
  React.useEffect(() => {
    alive.current = true;
    var c = new AbortController(),
      timer = setTimeout(() => c.abort(), 20000);
    fetch(`${api}/api/research/priorities`, {
      signal: c.signal
    }).then(async r => {
      var d = await r.json();
      if (!r.ok) throw Error(d.error || 'Could not load portfolio priorities');
      if (alive.current) {
        setProfile(d.profile);
        setPositions(d.profile?.positions || []);
        if (d.profile?.asOf) setAsOf(d.profile.asOf);
        setLoaded(true);
      }
    }).catch(e => {
      if (alive.current) setError(e.message);
    }).finally(() => clearTimeout(timer));
    return () => {
      alive.current = false;
      c.abort();
    };
  }, [api]);
  var save = async () => {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    setError('');
    try {
      var r = await fetch(`${api}/api/research/priorities`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          asOf,
          positions: positions.map(p => ({
            ...p,
            weightPct: p.weightPct === '' ? null : Number(p.weightPct)
          }))
        }),
        signal: AbortSignal.timeout(20000)
      });
      var d = await r.json();
      if (!r.ok) throw Error(d.error || 'Save failed');
      if (alive.current) setProfile(d.profile);
    } catch (e) {
      if (alive.current) setError(e.message + ' Refresh this view to check saved settings before editing again.');
    } finally {
      lock.current = false;
      if (alive.current) setBusy(false);
    }
  };
  var result = prioritizedResearch(analyses, activities, profile);
  return /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "PORTFOLIO / RESEARCH ATTENTION"), /*#__PURE__*/React.createElement("h2", null, "Put research effort where it matters."), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "A transparent research queue using user-reported exposure, thesis age and pending or failed analyst activities. This score is a workflow heuristic, not a return forecast or position-sizing recommendation."), error && /*#__PURE__*/React.createElement("p", {
    className: "workspace-error",
    role: "alert"
  }, error), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Configure portfolio weights \xB7 ", profile?.positions?.length || 0, " positions"), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Enter signed portfolio weights: positive for long, negative for short. These are not connected to a broker or independently verified. Weights older than 30 days are excluded from ordering."), /*#__PURE__*/React.createElement("fieldset", {
    className: "desk-form",
    disabled: busy || !loaded
  }, /*#__PURE__*/React.createElement("label", null, "Holdings as of", /*#__PURE__*/React.createElement("input", {
    type: "date",
    value: asOf,
    onChange: e => setAsOf(e.target.value)
  })), positions.map((p, i) => /*#__PURE__*/React.createElement("div", {
    className: "desk-filter",
    key: i
  }, /*#__PURE__*/React.createElement("label", null, "Ticker", /*#__PURE__*/React.createElement("input", {
    value: p.ticker,
    maxLength: 20,
    onChange: e => setPositions(positions.map((p, j) => j === i ? {
      ...p,
      ticker: e.target.value.toUpperCase().trim()
    } : p))
  })), /*#__PURE__*/React.createElement("label", null, "Signed weight %", /*#__PURE__*/React.createElement("input", {
    type: "number",
    min: "-100",
    max: "100",
    step: "any",
    value: p.weightPct,
    onChange: e => setPositions(positions.map((p, j) => j === i ? {
      ...p,
      weightPct: e.target.value
    } : p))
  })), /*#__PURE__*/React.createElement("button", {
    onClick: () => setPositions(positions.filter((_, j) => j !== i))
  }, "Remove"))), /*#__PURE__*/React.createElement("button", {
    disabled: positions.length >= 200,
    onClick: () => setPositions([...positions, {
      ticker: '',
      weightPct: ''
    }])
  }, "Add position"), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    onClick: save
  }, "Save weights"))), /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, !profile ? 'No holdings profile. Ordering uses research age and activity counts.' : result.fresh ? `Using reported weights as of ${profile.asOf}.` : `Holdings dated ${profile.asOf} are stale; weights are excluded.`), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Score = 10 \xD7 absolute weight (%) + thesis age (capped at 180 days) \xF7 6 + 5 \xD7 activity count (capped at 5). Missing thesis dates receive the 180-day age value. No research is started automatically."), result.rows.map(r => /*#__PURE__*/React.createElement("div", {
    className: "desk-row",
    key: r.ticker
  }, /*#__PURE__*/React.createElement("button", {
    onClick: () => onCompany(r.ticker, 'portfolio')
  }, /*#__PURE__*/React.createElement("strong", null, r.ticker), /*#__PURE__*/React.createElement("span", null, r.reasons.join(' · '))), /*#__PURE__*/React.createElement("small", null, "Priority ", r.score.toFixed(1)), /*#__PURE__*/React.createElement("button", {
    onClick: () => onPlan([r.ticker])
  }, "Plan research \u2192"))));
}