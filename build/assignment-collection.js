import * as React from 'react';
import { assignmentCollection } from './assignment-collection-model.mjs';
export function AssignmentCollection({
  job
}) {
  var c = assignmentCollection(job);
  if (!c) return null;
  return /*#__PURE__*/React.createElement("section", {
    "aria-label": "Assignment collection status",
    style: {
      padding: '12px 0',
      overflowWrap: 'anywhere'
    }
  }, /*#__PURE__*/React.createElement("strong", null, c.label), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "Last saved Mac report: ", c.reportedAt, ". ", c.fresh ? 'This is the latest saved report, not a live browser observation.' : 'Report is stale or its time is unavailable. Refresh before interpreting current browser progress.')), c.issue && /*#__PURE__*/React.createElement("p", {
    role: "status",
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, /*#__PURE__*/React.createElement("strong", null, "Reported collection issue: "), c.issue), /*#__PURE__*/React.createElement("p", null, c.verified, " verified originals recorded by this assignment. Collection progress does not establish complete coverage or evidence incorporated into the accepted thesis."), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Reported searches and handoffs"), c.counts.map(r => /*#__PURE__*/React.createElement("p", {
    key: r.label
  }, r.label, ": ", r.count)), c.tasks.length ? /*#__PURE__*/React.createElement("ul", null, c.tasks.map((t, i) => /*#__PURE__*/React.createElement("li", {
    key: i
  }, t.kind.replaceAll('-', ' '), " \xB7 ", t.status.replaceAll('_', ' '), t.expected !== null ? ` · ${t.expected} selected` : ''))) : /*#__PURE__*/React.createElement("p", null, "No search-progress receipt available."), /*#__PURE__*/React.createElement("p", null, "Staging, iCloud handoff and existing-copy counts are not verified cloud-import receipts or completed research.")), c.href && /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("a", {
    href: c.href
  }, "Open this collection in Operations")), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "Collection uses the existing Operations worker. It needs the Mac awake and online, the agent and Codex/browser available, and AlphaSense signed in. Resolve the reported issue there; this view starts no collection or research.")));
}