import * as React from 'react';
var labels = {
  'transcript': 'Transcripts',
  'broker-report': 'Broker reports',
  'press-release': 'Press releases',
  'presentation': 'Presentations',
  unknown: 'Type not recorded'
};
export function ProcessingCoverage({
  coverage: c
}) {
  if (!c || c.version !== 1) return /*#__PURE__*/React.createElement("p", null, "Evidence coverage summary unavailable for this saved proposal. Inspect its frozen sources; no thesis-incorporation assessment is available.");
  var d = c.publication,
    r = c.thesisReferences;
  return /*#__PURE__*/React.createElement("section", {
    "aria-label": "Selected evidence coverage",
    style: {
      margin: '12px 0',
      padding: 12,
      border: '1px solid var(--border-color,#d8dedd)',
      overflowWrap: 'anywhere'
    }
  }, /*#__PURE__*/React.createElement("strong", null, "Evidence coverage at proposal save"), /*#__PURE__*/React.createElement("p", null, "Requested collection window: ", c.collectionWindow.since || 'unknown', " \u2013 ", c.collectionWindow.until || 'unknown', ". This does not certify that every relevant source was collected."), /*#__PURE__*/React.createElement("p", null, c.total, " selected originals \xB7 ", c.roles.collection, " from collection \xB7 ", c.roles.supplemental, " supplemental history \xB7 ", c.roles.unknown, " with role unrecorded."), /*#__PURE__*/React.createElement("p", null, d.known ? `Recorded publication dates: ${d.earliest} – ${d.latest}.` : 'No publication dates recorded.', " ", d.unknown, " of ", c.total, " dates unknown. This range is not continuous or exhaustive evidence coverage."), /*#__PURE__*/React.createElement("p", null, d.beforeWindow, " dated before the collection window \xB7 ", d.withinWindow, " within it \xB7 ", d.afterWindow, " after it. Older sources can supply historical context; a discovery window does not limit a longitudinal thesis review."), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Source mix and thesis references"), /*#__PURE__*/React.createElement("ul", null, Object.entries(c.kinds).filter(([, n]) => n > 0).map(([kind, n]) => /*#__PURE__*/React.createElement("li", {
    key: kind
  }, labels[kind] || kind, ": ", n))), /*#__PURE__*/React.createElement("p", null, r.referenced, " exact-byte references found in the saved thesis register \xB7 ", r.noReference, " without a matching reference \xB7 ", r.unknown, " not checked in this saved proposal."), /*#__PURE__*/React.createElement("p", null, "A reference does not prove that evidence was incorporated or remains valid. No matching reference does not prove a source is new or unused. Accepted-thesis evidence coverage and freshness require investor review.")), c.metadataRecorded < c.total && /*#__PURE__*/React.createElement("p", null, "Metadata was recorded for ", c.metadataRecorded, " of ", c.total, " selected originals. Missing fields remain unknown; current library metadata has not been substituted."));
}