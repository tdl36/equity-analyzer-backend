import * as React from 'react';
import { recoveryEvidence } from './research-recovery-model.mjs';
export function ResearchRecovery({
  run
}) {
  var evidence = recoveryEvidence(run);
  if (!['attention', 'cancelled'].includes(run.status)) return null;
  return /*#__PURE__*/React.createElement("aside", {
    className: "research-recovery",
    "aria-label": "Research recovery evidence"
  }, /*#__PURE__*/React.createElement("style", null, `.research-recovery{margin:20px 0;padding:18px 20px;background:#f6f7f3;border:1px solid #cbd2ca;border-left:4px solid #85968b;border-radius:6px;font-family:Calibri,sans-serif;color:#000}.research-recovery .rr-eyebrow{font-size:12px;letter-spacing:.1em;text-transform:uppercase;margin:0 0 8px}.research-recovery .rr-title{font-size:19px;font-weight:700;margin:0 0 8px}.research-recovery .rr-counts{display:flex;flex-wrap:wrap;gap:8px 24px;padding:10px 0;border-top:1px solid #d9dfd7}.research-recovery details{margin-top:8px}.research-recovery summary{cursor:pointer}.research-recovery li{margin:12px 0;overflow-wrap:anywhere}.research-recovery code{font:13px Calibri,sans-serif;overflow-wrap:anywhere}@media(max-width:600px){.research-recovery{padding:14px}.research-recovery .rr-counts{display:flex;flex-direction:column;gap:4px}}`), /*#__PURE__*/React.createElement("p", {
    className: "rr-eyebrow"
  }, "Recovery evidence"), /*#__PURE__*/React.createElement("p", {
    className: "rr-title"
  }, evidence.title), /*#__PURE__*/React.createElement("p", null, evidence.next), /*#__PURE__*/React.createElement("div", {
    className: "rr-counts"
  }, /*#__PURE__*/React.createElement("span", null, evidence.savedCount, " saved responses"), /*#__PURE__*/React.createElement("span", null, evidence.previousAttempts, " archived responses"), evidence.stage && /*#__PURE__*/React.createElement("span", null, "Stopped stage: ", evidence.stage)), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Inspect saved response records"), /*#__PURE__*/React.createElement("p", null, "These records confirm preservation only. They do not establish source validation, investor approval or provider billing. Response text stays private."), !evidence.receipts.length ? /*#__PURE__*/React.createElement("p", null, "No detailed response records are available for this run. Older jobs may predate response preservation.") : /*#__PURE__*/React.createElement("ul", null, evidence.receipts.map(r => /*#__PURE__*/React.createElement("li", {
    key: r.stage
  }, /*#__PURE__*/React.createElement("strong", null, r.stage), " \xB7 ", Number.isInteger(r.characters) ? r.characters.toLocaleString() + ' characters' : 'Text size unavailable', " \xB7 Stop reason: ", r.stopReason || 'unavailable', r.sha256 && /*#__PURE__*/React.createElement("div", null, "Response fingerprint: ", /*#__PURE__*/React.createElement("code", null, r.sha256)))))));
}