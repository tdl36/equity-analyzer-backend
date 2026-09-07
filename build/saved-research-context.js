import * as React from 'react';
var {
  useState
} = React;
var date = v => v ? String(v).slice(0, 10) : 'Date unavailable';
// Render stored structured research as text, never as executable HTML.
function ResearchText({
  value,
  depth = 0
}) {
  if (value == null || value === '') return null;
  if (typeof value !== 'object') return /*#__PURE__*/React.createElement("p", {
    style: {
      whiteSpace: 'pre-wrap',
      overflowWrap: 'anywhere'
    }
  }, String(value));
  if (depth > 5) return /*#__PURE__*/React.createElement("p", null, "Open the company research to inspect additional detail.");
  if (Array.isArray(value)) return /*#__PURE__*/React.createElement("div", null, value.map((v, i) => /*#__PURE__*/React.createElement(ResearchText, {
    key: i,
    value: v,
    depth: depth + 1
  })));
  var heading = ['title', 'pillar', 'metric', 'threat'].find(k => typeof value[k] === 'string' && value[k]);
  var entries = Object.entries(value).filter(([key, v]) => key !== heading && !key.startsWith('_') && v != null && v !== '');
  var order = ['summary', 'description', 'target', 'triggerPoints', 'pillars', 'confidence', 'sources'];
  entries.sort(([a], [b]) => (order.includes(a) ? order.indexOf(a) : 5) - (order.includes(b) ? order.indexOf(b) : 5));
  return /*#__PURE__*/React.createElement("article", {
    className: "evidence-saved-field"
  }, heading && /*#__PURE__*/React.createElement("h4", {
    className: "evidence-pillar-title"
  }, value[heading]), entries.map(([key, v]) => key === 'sources' ? /*#__PURE__*/React.createElement("details", {
    key: key
  }, /*#__PURE__*/React.createElement("summary", null, "Saved source references \xB7 not independently checked"), /*#__PURE__*/React.createElement(ResearchText, {
    value: v,
    depth: depth + 1
  })) : /*#__PURE__*/React.createElement("section", {
    key: key
  }, !['summary', 'description'].includes(key) && /*#__PURE__*/React.createElement("h4", null, key.replaceAll('_', ' ').replace(/([a-z])([A-Z])/g, '$1 $2')), /*#__PURE__*/React.createElement(ResearchText, {
    value: v,
    depth: depth + 1
  }))));
}
export function SavedResearchContext({
  data,
  ticker,
  onCompany
}) {
  var thesis = data?.savedThesis,
    documents = data?.documents;
  var [all, setAll] = useState(false);
  var uploaded = documents?.uploaded || [],
    local = documents?.local || [];
  var items = [...uploaded.map(d => ({
    ...d,
    location: 'Saved in Charlie'
  })), ...local.map(d => ({
    ...d,
    location: `iCloud inventory · ${d.folder === 'main' ? 'STOCKS' : d.folder || 'Folder unspecified'}`
  }))];
  return /*#__PURE__*/React.createElement("div", {
    className: "evidence-layout evidence-saved-context"
  }, /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel evidence-main"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "SAVED COMPANY RESEARCH"), /*#__PURE__*/React.createElement("h3", null, thesis ? `${ticker} · Saved investment thesis` : `${ticker} · Research sources`), thesis && /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, thesis.company, " \xB7 Updated ", date(thesis.updatedAt))), !data.current && /*#__PURE__*/React.createElement("div", {
    className: "workspace-notice"
  }, /*#__PURE__*/React.createElement("strong", null, thesis ? 'Your saved thesis is available.' : 'No saved thesis found.'), /*#__PURE__*/React.createElement("p", null, "No investment-review comparison baseline exists yet. Available documents below are an inventory, not verified support for individual claims.")), thesis ? /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(ResearchText, {
    value: thesis.thesis
  }), [['Signposts', thesis.signposts], ['Threats', thesis.threats], ['Conclusion', thesis.conclusion]].map(([label, value]) => value && /*#__PURE__*/React.createElement("details", {
    key: label
  }, /*#__PURE__*/React.createElement("summary", null, label), /*#__PURE__*/React.createElement(ResearchText, {
    value: value
  })))) : /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "You can inspect available sources and open the company workspace without generating a report."), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    onClick: () => onCompany(ticker, 'portfolio')
  }, "Open ", ticker, " research \u2192")), /*#__PURE__*/React.createElement("aside", {
    className: "workspace-panel evidence-sidebar"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "AVAILABLE SOURCE DOCUMENTS"), /*#__PURE__*/React.createElement("h3", null, uploaded.length, " saved in Charlie \xB7 ", local.length, " in iCloud inventory"), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Locations are listed separately; the same document may appear in both. Presence here does not mean a review has read or verified it."), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, documents?.localUpdatedAt ? `iCloud inventory reported ${String(documents.localUpdatedAt).replace('T', ' ')}. This is the agent’s last report, not a live folder scan.` : 'The local agent has not reported an iCloud inventory to this server yet. This does not mean your folders are empty.')), !items.length && /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "No document entries are currently available in Charlie\u2019s inventories."), (all ? items : items.slice(0, 20)).map((d, i) => /*#__PURE__*/React.createElement("article", {
    className: "evidence-document",
    key: `${d.location}-${d.filename}-${i}`
  }, /*#__PURE__*/React.createElement("strong", null, d.filename || 'Unnamed document'), /*#__PURE__*/React.createElement("small", null, d.location, d.addedAt ? ` · Added ${date(d.addedAt)}` : ''))), items.length > 20 && /*#__PURE__*/React.createElement("button", {
    onClick: () => setAll(v => !v)
  }, all ? 'Show first 20' : `Show all ${items.length} inventory entries`)));
}