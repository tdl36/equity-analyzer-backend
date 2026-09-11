import * as React from 'react';
var fields = {
  support: 'Supporting evidence',
  contrary: 'Risks & contrary evidence',
  nextTest: 'Next test'
};
var ready = c => c.passageMatched && c.reviewPassed;
export function ProposalReview({
  job,
  body,
  revision,
  disabled,
  onCommit,
  onDebate,
  onRepair
}) {
  var changes = job.result?.changes || [];
  var [repairBusy, setRepairBusy] = React.useState(false),
    [repairNotice, setRepairNotice] = React.useState('');
  React.useEffect(() => {
    if (['queued', 'running'].includes(job.status)) setRepairNotice('');
  }, [job.status]);
  var repairLock = React.useRef(false);
  var requestRepair = async () => {
    if (repairLock.current) return;
    repairLock.current = true;
    setRepairBusy(true);
    setRepairNotice('Submitting revision request…');
    try {
      var result = await onRepair(repairCount);
      setRepairNotice(result?.ok ? 'Revision queued. Charlie is re-reading the originals; results refresh automatically.' : result?.error || 'The request could not be confirmed. Reload the results before retrying.');
    } catch (e) {
      setRepairNotice(e.message || 'Revision request failed. Please retry.');
    } finally {
      repairLock.current = false;
      setRepairBusy(false);
    }
  };
  var repairCount = job.result?.repairHistory?.length || 0;
  var needsRepair = changes.filter(c => !ready(c)).length;
  var [filter, setFilter] = React.useState('ready'),
    [selectedId, setSelectedId] = React.useState(null),
    [assumptionId, setAssumptionId] = React.useState(''),
    [field, setField] = React.useState('support');
  var available = changes.filter(c => filter === 'ready' ? ready(c) : !ready(c));
  var selected = available.find(c => c.id === selectedId) || available[0];
  var assumption = body.assumptions.find(a => a.id === (selected?.assumptionId || assumptionId));
  var effectiveField = selected?.field || field;
  var applied = !!selected && !!assumption && assumption[effectiveField] === selected.after;
  var canAccept = selected && ready(selected) && ['awaiting_approval', 'applied'].includes(job.status) && assumption && !applied;
  var title = c => body.assumptions.find(a => a.id === c.assumptionId)?.claim || 'Research change';
  return /*#__PURE__*/React.createElement("div", {
    className: "proposal-review"
  }, /*#__PURE__*/React.createElement("div", {
    className: "proposal-review-filters",
    "aria-label": "Filter proposed changes"
  }, [['ready', 'Ready to review', changes.filter(ready).length], ['flagged', 'Charlie needs to revise', changes.filter(c => !ready(c)).length]].map(([id, label, count]) => /*#__PURE__*/React.createElement("button", {
    key: id,
    "aria-pressed": filter === id,
    onClick: () => {
      setFilter(id);
      setSelectedId(null);
    }
  }, label, /*#__PURE__*/React.createElement("strong", null, count)))), /*#__PURE__*/React.createElement("p", {
    className: "proposal-review-intro"
  }, filter === 'ready' ? 'These changes passed source-passage and model checks. Read each proposal before accepting it.' : 'Charlie could not substantiate these drafts. You do not need to verify them manually. Ask Charlie to re-read the originals and revise; your saved case stays unchanged.'), !available.length ? /*#__PURE__*/React.createElement("p", {
    className: "case-evidence-banner"
  }, filter === 'ready' ? 'No changes are ready for acceptance. Open Charlie needs to revise to request a supported revision.' : 'No drafts need revision.') : /*#__PURE__*/React.createElement("div", {
    className: "proposal-review-layout"
  }, /*#__PURE__*/React.createElement("nav", {
    className: "proposal-change-list",
    "aria-label": "Proposed changes"
  }, available.map((c, i) => /*#__PURE__*/React.createElement("button", {
    key: c.id,
    "aria-pressed": selected?.id === c.id,
    onClick: () => {
      setSelectedId(c.id);
      setAssumptionId('');
      setField('support');
    }
  }, /*#__PURE__*/React.createElement("small", null, "CHANGE ", changes.indexOf(c) + 1, " \xB7 ", fields[c.field] || 'Research wording'), /*#__PURE__*/React.createElement("span", null, title(c)), /*#__PURE__*/React.createElement("em", null, body.assumptions.some(a => a.id === c.assumptionId && a[c.field] === c.after) ? 'Already reflected in case' : ready(c) ? 'Open for review →' : 'Unsupported draft →')))), selected && /*#__PURE__*/React.createElement("article", {
    className: "proposal-change-reader",
    "aria-label": "Selected change"
  }, /*#__PURE__*/React.createElement("header", null, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "CHANGE ", changes.indexOf(selected) + 1, " / ", fields[effectiveField] || 'PROPOSED WORDING'), /*#__PURE__*/React.createElement("h3", null, title(selected)), /*#__PURE__*/React.createElement("span", {
    className: "proposal-status"
  }, applied ? 'Already reflected in current case' : ready(selected) ? 'Ready for your review' : 'Draft failed Charlie’s evidence checks')), !ready(selected) && /*#__PURE__*/React.createElement("section", {
    className: "proposal-verification"
  }, /*#__PURE__*/React.createElement("h4", null, "Charlie needs to fix its draft"), /*#__PURE__*/React.createElement("p", null, selected.repairOutcome === 'no_supported_change' ? 'After re-reading the originals, Charlie could not produce a supported replacement. Your current wording has been retained.' : !selected.passageMatched ? 'Charlie could not locate its quoted passage in the saved original. This draft is not ready for an investment decision.' : 'The quotation matched, but the reviewer found claims it does not adequately support.'), ['queued', 'running'].includes(job.status) ? /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Charlie is re-reading the originals and checking revised drafts. This view refreshes automatically; your saved case is unchanged.") : onRepair && job.status === 'awaiting_approval' && repairCount < 2 ? /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    disabled: disabled || repairBusy,
    onClick: requestRepair
  }, repairBusy ? 'Submitting…' : `Re-read sources & revise ${needsRepair} draft${needsRepair === 1 ? '' : 's'}`), /*#__PURE__*/React.createElement("p", {
    className: "proposal-footnote"
  }, "Charlie will shorten unsupported drafts, check quotations and run an independent review. Other proposals are preserved. Uses model credits; nothing is accepted automatically.")) : /*#__PURE__*/React.createElement("p", null, "Close this review with your conclusion below, then assess additional source documents if needed."), repairCount > 0 && /*#__PURE__*/React.createElement("p", null, repairCount, " revision attempt", repairCount === 1 ? '' : 's', " recorded. If evidence is still insufficient, you can leave your thesis unchanged using \u201CFinish this review\u201D below."), job.error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, "Revision could not complete: ", job.error), repairNotice && !job.error && /*#__PURE__*/React.createElement("p", {
    role: "status",
    "aria-live": "polite"
  }, repairNotice), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "What failed the checks?"), /*#__PURE__*/React.createElement("p", null, selected.reviewIssue || 'The supporting quotation did not match the saved original.'))), ready(selected) ? /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("section", null, /*#__PURE__*/React.createElement("h4", null, "What changes and why"), /*#__PURE__*/React.createElement("p", null, selected.reason || 'No rationale recorded.')), /*#__PURE__*/React.createElement("section", {
    className: "proposal-wording"
  }, /*#__PURE__*/React.createElement("h4", null, "Proposed wording for your case"), /*#__PURE__*/React.createElement("p", null, selected.after))) : /*#__PURE__*/React.createElement("details", {
    className: "proposal-wording"
  }, /*#__PURE__*/React.createElement("summary", null, "Rejected draft \xB7 not applied"), /*#__PURE__*/React.createElement("h4", null, "Charlie\u2019s original rationale \xB7 unverified"), /*#__PURE__*/React.createElement("p", null, selected.reason), /*#__PURE__*/React.createElement("h4", null, "Rejected replacement"), /*#__PURE__*/React.createElement("p", null, selected.after)), /*#__PURE__*/React.createElement("details", {
    className: "proposal-before"
  }, /*#__PURE__*/React.createElement("summary", null, "Current wording \xB7 unchanged until you accept"), /*#__PURE__*/React.createElement("p", null, assumption?.[effectiveField] || selected.before || 'Not recorded')), /*#__PURE__*/React.createElement("details", {
    className: "proposal-source"
  }, /*#__PURE__*/React.createElement("summary", null, "Read supporting source passages \xB7 ", selected.evidence?.length || 0), selected.evidence?.map((e, i) => /*#__PURE__*/React.createElement("section", {
    key: i
  }, /*#__PURE__*/React.createElement("h4", null, job.result.sources?.find(s => s.id === e.sourceId)?.filename || 'Source reference unavailable'), /*#__PURE__*/React.createElement("small", null, e.status === 'passage_matched' ? 'Passage matched at generation' : 'Unmatched quotation — do not rely on this as source text'), /*#__PURE__*/React.createElement("blockquote", null, e.excerpt || 'No excerpt available')))), /*#__PURE__*/React.createElement("div", {
    className: "proposal-actions"
  }, /*#__PURE__*/React.createElement("button", {
    onClick: () => onDebate({
      prompt: 'Challenge this proposed interpretation. Separate what the source actually says from inference, address verification findings, and identify missing evidence. Do not claim any edits were applied.',
      content: JSON.stringify({
        caseRevision: revision,
        investmentCase: body,
        proposalId: job.id,
        change: selected,
        sources: job.result.sources
      })
    })
  }, "Discuss with Charlie"), ready(selected) && !selected.assumptionId && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("label", null, "Investment assumption", /*#__PURE__*/React.createElement("select", {
    value: assumptionId,
    onChange: e => setAssumptionId(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Choose an assumption"), body.assumptions.map(a => /*#__PURE__*/React.createElement("option", {
    key: a.id,
    value: a.id
  }, a.claim)))), /*#__PURE__*/React.createElement("label", null, "Destination", /*#__PURE__*/React.createElement("select", {
    value: field,
    onChange: e => setField(e.target.value)
  }, Object.entries(fields).map(([k, v]) => /*#__PURE__*/React.createElement("option", {
    key: k,
    value: k
  }, v))))), ready(selected) ? /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    disabled: disabled || !canAccept,
    onClick: () => onCommit({
      mode: 'source_change',
      sourceChange: {
        jobId: job.id,
        changeId: selected.id,
        assumptionId: assumption.id,
        field: effectiveField
      }
    })
  }, applied ? 'Already reflected in case' : 'Accept this change') : /*#__PURE__*/React.createElement("p", null, "This draft cannot be accepted. Use the revision action above, or finish this review with no thesis change.")), /*#__PURE__*/React.createElement("p", {
    className: "proposal-footnote"
  }, "Accepting saves a new case revision. Discussion and opening a card do not change your research."))));
}