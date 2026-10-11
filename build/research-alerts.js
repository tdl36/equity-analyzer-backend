import * as React from 'react';
import { researchAlerts, incompleteResearchAlerts } from './research-alert-model.mjs';
import { activityDestination } from './activity-model.mjs';
export function ResearchAlerts({
  api,
  onNavigate
}) {
  var [data, setData] = React.useState(null),
    [dismissed, setDismissed] = React.useState({}),
    [unavailable, setUnavailable] = React.useState(false);
  React.useEffect(() => {
    var alive = true,
      pending = false;
    var controller = new AbortController();
    var load = async () => {
      if (pending) return;
      pending = true;
      try {
        var r = await fetch(api + '/api/activity', {
          signal: AbortSignal.any([controller.signal, AbortSignal.timeout(15000)])
        });
        if (!r.ok) throw Error();
        var d = await r.json();
        if (alive) {
          setData(d);
          setUnavailable(incompleteResearchAlerts(d));
        }
      } catch {
        if (alive) setUnavailable(true);
      } finally {
        pending = false;
      }
    };
    load();
    var timer = setInterval(load, 30000);
    var visible = () => {
      if (!document.hidden) load();
    };
    document.addEventListener('visibilitychange', visible);
    return () => {
      alive = false;
      controller.abort();
      clearInterval(timer);
      document.removeEventListener('visibilitychange', visible);
    };
  }, [api]);
  var alerts = researchAlerts(data, dismissed),
    row = alerts[0];
  if (!row && !unavailable) return null;
  return /*#__PURE__*/React.createElement(ResearchAlertBanner, {
    row: row,
    count: alerts.length,
    unavailable: unavailable,
    onNavigate: onNavigate,
    onDismiss: () => setDismissed(d => ({
      ...d,
      [row.alertKey]: true
    }))
  });
}
export function ResearchAlertBanner({
  row,
  count = 1,
  unavailable = false,
  onNavigate,
  onDismiss
}) {
  return /*#__PURE__*/React.createElement("aside", {
    role: "alert",
    className: "research-global-alert"
  }, /*#__PURE__*/React.createElement("style", null, `.research-global-alert{flex-shrink:0;background:#fff2da;color:#171510;border-bottom:1px solid #d3b77d;padding:12px 24px;display:flex;align-items:center;gap:16px;font:15px/1.4 Calibri,sans-serif}.research-global-alert p{margin:0;flex:1;min-width:0;overflow-wrap:anywhere}.research-global-alert button{font:inherit;color:#171510;background:#fff;border:1px solid #a58d61;border-radius:5px;padding:7px 12px;flex-shrink:0}@media(max-width:650px){.research-global-alert{padding:10px 14px;flex-wrap:wrap;gap:8px}.research-global-alert p{flex-basis:100%}}`), /*#__PURE__*/React.createElement("p", null, row ? /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("strong", null, row.ticker, " research needs attention."), " ", row.error || row.step || 'Open the saved issue before retrying.', count > 1 && ` ${count - 1} other companies also need attention.`) : /*#__PURE__*/React.createElement("strong", null, "Research alerts may be incomplete."), " ", unavailable && 'Some statuses may be missing or out of date. Check Activity and the company workspace.'), /*#__PURE__*/React.createElement("button", {
    onClick: () => {
      if (row) {
        var dest = activityDestination(row);
        onNavigate(dest.view, row.ticker, dest.section, dest.work);
      } else onNavigate('activity');
    }
  }, row ? 'Open saved work' : 'Open Activity'), row && /*#__PURE__*/React.createElement("button", {
    "aria-label": `Dismiss ${row.ticker} research alert for this session`,
    onClick: onDismiss
  }, "Dismiss"));
}