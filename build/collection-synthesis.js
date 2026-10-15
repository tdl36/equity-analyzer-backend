import * as React from 'react';
import { synthesisRows } from './collection-synthesis-model.mjs';
export function CollectionSynthesisStatus({
  api
}) {
  var [data, setData] = React.useState({}),
    [checked, setChecked] = React.useState(null);
  React.useEffect(() => {
    var live = true,
      pending = false;
    var controller = new AbortController();
    setData({});
    setChecked(null);
    var refresh = async () => {
      if (pending) return;
      pending = true;
      var results = await Promise.allSettled(['/api/catalysts/auto-mode', '/api/research/automation', '/api/analysts'].map(async path => {
        var r = await fetch(`${api}${path}`, {
          signal: AbortSignal.any([controller.signal, AbortSignal.timeout(20000)])
        });
        if (!r.ok) throw new Error('Status unavailable');
        return r.json();
      }));
      if (live) {
        setData(Object.fromEntries(['catalysts', 'intake', 'analysts'].map((key, i) => [key, results[i].status === 'fulfilled' ? results[i].value : null])));
        setChecked(new Date());
      }
      pending = false;
    };
    refresh();
    var timer = setInterval(() => {
      if (!document.hidden) refresh();
    }, 30000);
    return () => {
      live = false;
      controller.abort();
      clearInterval(timer);
    };
  }, [api]);
  return /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel",
    "aria-label": "Automatic synthesis settings"
  }, /*#__PURE__*/React.createElement("h3", null, "Automatic synthesis"), /*#__PURE__*/React.createElement("p", null, "Collection and importing can continue with these switches off. During backfill, keep all three off to avoid automatic generation from new documents."), synthesisRows(data).map(row => /*#__PURE__*/React.createElement("p", {
    key: row.label
  }, /*#__PURE__*/React.createElement("strong", null, row.label, ": ", row.status), ' · ', /*#__PURE__*/React.createElement("a", {
    href: row.href
  }, row.control))), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, checked ? `Last checked ${checked.toLocaleTimeString()}.` : 'Checking saved settings…', " These are separate controls. Pausing them does not cancel work already running or change outputs explicitly requested in an assignment. Open the linked settings to change them."));
}