import * as React from 'react';
import { claimLabel } from './onepager-edition-model.mjs';
function sections(a) {
  return [{
    key: 'summary',
    title: 'Company and investment case',
    text: a.thesis.summary
  }, ...a.thesis.pillars.map((p, i) => ({
    key: 'pillar-' + i,
    title: p.title,
    text: p.description
  })), ...a.signposts.map((p, i) => ({
    key: 'signpost-' + i,
    title: 'Watch: ' + p.metric,
    text: [p.target, p.timeframe].filter(Boolean).join('\n')
  })), ...a.threats.map((p, i) => ({
    key: 'risk-' + i,
    title: 'Risk: ' + p.threat,
    text: p.triggerPoints
  })), {
    key: 'conclusion',
    title: 'Conclusion',
    text: a.conclusion || ''
  }];
}
export function PackageDraftPreview({
  analysis,
  original,
  claims,
  words
}) {
  var before = new Map(sections(original).map(f => [f.key, f]));
  var rows = sections(analysis);
  var changes = rows.filter(f => JSON.stringify(before.get(f.key)) !== JSON.stringify(f)).length + claims.filter(c => c.proposal.statement !== words[c.correctionId]).length;
  return /*#__PURE__*/React.createElement("section", {
    "aria-label": "Unsaved package preview",
    className: "pe-preview"
  }, /*#__PURE__*/React.createElement("style", null, `.pe-preview{border:1px solid #bbb;padding:18px;margin:16px 0;background:#fff}.pe-preview-grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}.pe-preview p{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.6}.pe-preview h4{font-size:20px;border-bottom:2px solid #000;padding-bottom:8px}.pe-preview h5{font-size:16px;margin:12px 0 6px}.pe-preview details{margin-top:8px}.pe-preview .pe-change{font-weight:bold;font-size:12px}@media(max-width:800px){.pe-preview-grid{grid-template-columns:1fr}.pe-preview{padding:12px}}`), /*#__PURE__*/React.createElement("h3", null, "Read your draft before saving"), /*#__PURE__*/React.createElement("p", null, changes, " changed sections. This is an unsaved wording preview, not an approved document or a print-layout proof. Evidence references are retained, not revalidated against your edits."), /*#__PURE__*/React.createElement("div", {
    className: "pe-preview-grid"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h4", null, "Stock summary & proposed thesis"), rows.map(f => {
    var old = before.get(f.key),
      changed = JSON.stringify(old) !== JSON.stringify(f);
    return /*#__PURE__*/React.createElement("article", {
      key: f.key
    }, /*#__PURE__*/React.createElement("h5", null, f.title), changed && /*#__PURE__*/React.createElement("span", {
      className: "pe-change"
    }, "Changed in this draft"), /*#__PURE__*/React.createElement("p", null, f.text || 'No wording entered.'), changed && /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Compare saved wording"), /*#__PURE__*/React.createElement("strong", null, old.title), /*#__PURE__*/React.createElement("p", null, old.text)));
  })), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h4", null, "One-pager wording"), claims.map(c => {
    var p = c.proposal,
      changed = p.statement !== words[c.correctionId];
    return /*#__PURE__*/React.createElement("article", {
      key: c.correctionId
    }, /*#__PURE__*/React.createElement("h5", null, claimLabel(p.path)), changed && /*#__PURE__*/React.createElement("span", {
      className: "pe-change"
    }, "Changed in this draft"), /*#__PURE__*/React.createElement("p", null, words[c.correctionId] || 'No wording entered.'), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Saved wording & attached evidence"), /*#__PURE__*/React.createElement("p", null, p.statement), [p, ...(p.additionalEvidence || [])].map((e, i) => /*#__PURE__*/React.createElement("div", {
      key: i
    }, /*#__PURE__*/React.createElement("small", null, e.locator), /*#__PURE__*/React.createElement("blockquote", null, e.excerpt)))));
  }))));
}