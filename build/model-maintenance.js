import React from 'react';
export function ModelMaintenance({
  apiUrl
}) {
  var [data, setData] = React.useState(null);
  var [error, setError] = React.useState('');
  var load = React.useCallback(async () => {
    setError('');
    try {
      var response = await fetch(`${apiUrl}/api/models/maintenance`);
      if (!response.ok) throw new Error('Model status could not be loaded.');
      setData(await response.json());
    } catch (e) {
      setError(e.message);
    }
  }, [apiUrl]);
  React.useEffect(() => {
    load();
  }, [load]);
  return /*#__PURE__*/React.createElement(ModelPolicyView, {
    data: data,
    error: error,
    load: load
  });
}
export function ModelPolicyView({
  data,
  error = '',
  load = () => {}
}) {
  return /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel max-w-3xl space-y-4",
    style: {
      overflowWrap: 'anywhere',
      color: 'var(--ink, #202018)'
    }
  }, /*#__PURE__*/React.createElement("div", {
    className: "flex justify-between gap-3"
  }, /*#__PURE__*/React.createElement("h2", null, "AI model maintenance"), /*#__PURE__*/React.createElement("button", {
    onClick: load,
    className: "text-sm underline"
  }, "Refresh status")), /*#__PURE__*/React.createElement("p", {
    className: "text-sm text-slate-300"
  }, "Charlie checks for model releases and retirements through scheduled maintenance. Eligible changes are tested and deployed together; saved model selections are preserved. Retired selections stop with an actionable error."), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error), !data && !error && /*#__PURE__*/React.createElement("p", null, "Loading model policy\u2026"), data && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", {
    className: "text-sm"
  }, "Last maintenance attempt: ", /*#__PURE__*/React.createElement("strong", null, data.lastCheck?.checkedAt || data.verifiedAt)), data.lastCheck && /*#__PURE__*/React.createElement("p", {
    className: "text-sm",
    role: "status"
  }, data.lastCheck.status, ": ", data.lastCheck.summary), /*#__PURE__*/React.createElement("p", {
    className: "text-sm text-slate-400"
  }, data.policy), /*#__PURE__*/React.createElement("div", {
    className: "space-y-2"
  }, Object.entries(data.roles).map(([role, model]) => /*#__PURE__*/React.createElement("div", {
    key: role,
    className: "border-b border-white/10 pb-2 flex flex-wrap justify-between gap-2 text-sm"
  }, /*#__PURE__*/React.createElement("span", null, role.replaceAll('_', ' ')), /*#__PURE__*/React.createElement("span", null, model)))), !!data.notices.length && /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h3", {
    className: "font-semibold mb-2"
  }, "Retirement notices"), data.notices.map(n => /*#__PURE__*/React.createElement("p", {
    className: "text-sm mb-2",
    key: n.model
  }, n.model, " \xB7 ", n.date, " \xB7 ", n.message))), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", {
    className: "cursor-pointer"
  }, "Verified changes and sources"), /*#__PURE__*/React.createElement("div", {
    className: "space-y-2 mt-3 text-sm"
  }, [...data.history].reverse().map((h, i) => /*#__PURE__*/React.createElement("p", {
    key: i
  }, h.date, ": ", h.summary)), data.sources.map(url => /*#__PURE__*/React.createElement("p", {
    key: url
  }, /*#__PURE__*/React.createElement("a", {
    className: "underline",
    href: url,
    target: "_blank",
    rel: "noreferrer"
  }, url))))), /*#__PURE__*/React.createElement("p", {
    className: "text-xs text-slate-400"
  }, data.maintenanceDependency), /*#__PURE__*/React.createElement("p", {
    className: "text-xs text-slate-400"
  }, "Registry ", data.revision, ". Prices are estimates, not invoices. Unknown prices are excluded from totals and flagged in API usage.")));
}