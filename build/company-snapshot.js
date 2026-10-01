import * as React from 'react';
import { companySnapshot, compareSnapshots, snapshotDocument } from './company-snapshot-model.mjs';
export function CompanySnapshot({
  ticker,
  versions,
  revision,
  disabled,
  onEdit,
  onUpdate
}) {
  var [selected, setSelected] = React.useState(''),
    [baseline, setBaseline] = React.useState('');
  var version = versions.find(v => String(v.revision) === (selected || String(revision)));
  var snapshot = companySnapshot(ticker, version);
  var earlier = versions.find(v => String(v.revision) === baseline);
  var changes = earlier ? compareSnapshots(companySnapshot(ticker, earlier), snapshot) : [];
  var download = () => {
    var url = URL.createObjectURL(new Blob([snapshotDocument(snapshot)], {
      type: 'text/html;charset=utf-8'
    }));
    var a = document.createElement('a');
    a.href = url;
    a.download = `${ticker}-investment-map-r${snapshot.revision}.html`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return /*#__PURE__*/React.createElement("section", {
    "aria-label": "Company snapshot",
    className: "company-snapshot",
    style: {
      fontFamily: 'Calibri, sans-serif',
      color: '#000'
    }
  }, /*#__PURE__*/React.createElement("style", null, `.company-snapshot{background:#fff;padding:20px;border-radius:12px}.company-snapshot *{color:#000!important;overflow-wrap:anywhere}.company-snapshot select,.company-snapshot button,.company-snapshot .workspace-panel{background:#f6f6f2!important;color:#000!important;max-width:100%}.company-snapshot select{width:100%}.company-snapshot label{min-width:0;max-width:100%}.company-snapshot .snapshot-actions{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0;align-items:end}.company-snapshot .snapshot-actions label{flex:1 1 240px}.company-snapshot button{border:1px solid #aaa;padding:10px 14px;border-radius:6px}.company-snapshot button:disabled{opacity:.5}.company-snapshot h3{font-size:28px;margin:12px 0}.company-snapshot header{margin-bottom:20px}.company-snapshot>p{margin:16px 0}.company-snapshot>h4{margin-top:24px}.company-snapshot article,.company-snapshot aside{margin-bottom:16px}.company-snapshot h4{font-weight:700;margin-bottom:12px}.company-snapshot h5{font-weight:700;margin-top:16px}`), /*#__PURE__*/React.createElement("header", null, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "SAVED RESEARCH / COMPANY SNAPSHOT"), /*#__PURE__*/React.createElement("h3", null, "The case, at a glance."), /*#__PURE__*/React.createElement("p", null, "A concise view of your saved working assumptions. Opening this view does not launch research.")), !version ? /*#__PURE__*/React.createElement("div", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("h4", null, revision ? 'Saved revision unavailable' : 'Start with your investment case'), /*#__PURE__*/React.createElement("p", null, revision ? 'Reload the company to refresh its saved history.' : 'Create a case or load your existing thesis as a draft, then review and save it.'), /*#__PURE__*/React.createElement("button", {
    disabled: disabled,
    onClick: onEdit
  }, "Open current thesis")) : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "snapshot-actions"
  }, /*#__PURE__*/React.createElement("label", null, "Saved version", /*#__PURE__*/React.createElement("select", {
    "aria-label": "Saved version",
    value: selected || String(revision),
    onChange: e => {
      setSelected(e.target.value);
      setBaseline('');
    }
  }, versions.map(v => /*#__PURE__*/React.createElement("option", {
    key: v.revision,
    value: v.revision
  }, "Revision ", v.revision, " \xB7 ", v.created_at ? new Date(v.created_at).toLocaleString() : 'Date unavailable · refresh history')))), /*#__PURE__*/React.createElement("button", {
    onClick: download
  }, "Download investment map"), /*#__PURE__*/React.createElement("button", {
    disabled: disabled,
    onClick: onEdit
  }, "Edit current thesis"), /*#__PURE__*/React.createElement("button", {
    disabled: disabled,
    onClick: onUpdate
  }, "Update thesis \xB7 review new evidence")), /*#__PURE__*/React.createElement("p", null, "Viewing revision ", snapshot.revision, snapshot.revision !== revision ? ' · historical snapshot' : '', ". Updates assess the latest saved case, revision ", revision, ", after you select sources. Downloaded maps retain the version shown here."), disabled && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Unsaved edits or a save in progress: this Snapshot continues to show saved content. Save before preparing an update."), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit,minmax(min(100%,280px),1fr))',
      gap: 16
    }
  }, [['Investment thesis', snapshot.thesis], ['Market expectations · investor recorded', snapshot.marketBaseline], ['Where my view differs', snapshot.variantView], ['What would change my mind', snapshot.changeConditions]].map(([label, text]) => /*#__PURE__*/React.createElement("article", {
    className: "workspace-panel",
    key: label
  }, /*#__PURE__*/React.createElement("h4", null, label), /*#__PURE__*/React.createElement("p", {
    style: {
      whiteSpace: 'pre-wrap',
      overflowWrap: 'anywhere'
    }
  }, text || 'Not recorded')))), /*#__PURE__*/React.createElement("h4", null, "Assumptions \u2192 evidence \u2192 next test"), /*#__PURE__*/React.createElement("p", null, "Classification is investor recorded. Saved excerpts establish provenance, not truth or currentness."), snapshot.assumptions.map(a => /*#__PURE__*/React.createElement("article", {
    className: "workspace-panel",
    key: a.id
  }, /*#__PURE__*/React.createElement("h4", null, a.claim), /*#__PURE__*/React.createElement("p", null, String(a.evidenceType || 'interpretation').replaceAll('_', ' ')), /*#__PURE__*/React.createElement("dl", null, [['Supports', a.support], ['Challenges', a.contrary], ['Next test', a.nextTest], ['Manual source reference · unverified', a.sourceReference]].map(([label, text]) => /*#__PURE__*/React.createElement(React.Fragment, {
    key: label
  }, /*#__PURE__*/React.createElement("dt", null, /*#__PURE__*/React.createElement("strong", null, label)), /*#__PURE__*/React.createElement("dd", {
    style: {
      whiteSpace: 'pre-wrap',
      overflowWrap: 'anywhere',
      margin: '4px 0 16px'
    }
  }, text || 'Not recorded')))), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, a.evidence.length ? `${a.evidence.length} current saved excerpt link(s)` : 'No current saved excerpt links'), a.evidence.map((e, i) => /*#__PURE__*/React.createElement("blockquote", {
    key: i
  }, /*#__PURE__*/React.createElement("strong", null, e.filename || 'Source name unavailable'), /*#__PURE__*/React.createElement("p", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, e.excerpt), /*#__PURE__*/React.createElement("small", null, e.provenance)))))), /*#__PURE__*/React.createElement("aside", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("h4", null, "Coverage still to build"), /*#__PURE__*/React.createElement("p", null, snapshot.gaps.length ? snapshot.gaps.join(' · ') : 'All tracked case fields have content. This does not establish completeness or verification.'), /*#__PURE__*/React.createElement("p", null, "Full business, financial and 22-section research coverage is not assessed by this saved-case Snapshot.")), /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("h4", null, "What changed between saved cases?"), /*#__PURE__*/React.createElement("label", null, "Compare with an earlier revision", /*#__PURE__*/React.createElement("select", {
    value: baseline,
    onChange: e => setBaseline(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Choose a baseline"), versions.filter(v => v.revision < snapshot.revision).map(v => /*#__PURE__*/React.createElement("option", {
    key: v.revision,
    value: v.revision
  }, "Revision ", v.revision)))), earlier && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, "Revision ", earlier.revision, " \u2192 ", snapshot.revision, ". Wording changes only; this does not infer why the investment case changed."), !changes.length && /*#__PURE__*/React.createElement("p", null, "No thesis or assumption wording changed."), changes.map((c, i) => /*#__PURE__*/React.createElement("article", {
    key: i
  }, /*#__PURE__*/React.createElement("h5", null, c.label), /*#__PURE__*/React.createElement("p", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, "Before: ", c.before || 'Not recorded'), /*#__PURE__*/React.createElement("p", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, "After: ", c.after || 'Not recorded')))))));
}