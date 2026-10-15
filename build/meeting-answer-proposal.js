function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
import { linkedProposals, proposalLink, proposalIntentKey, readProposalIntent, reserveProposalIntent, proposalReceipt, clearProposalIntent } from './meeting-answer-proposal-model.mjs';
import * as React from 'react';
export function MeetingAnswerProposal(props) {
  return /*#__PURE__*/React.createElement(ProposalHandoff, _extends({
    key: props.api + '|' + props.ticker + '|' + props.record.id
  }, props));
}
function ProposalHandoff({
  api,
  ticker,
  record,
  disabled = false
}) {
  var [data, setData] = React.useState({
      jobs: [],
      activeProposal: null,
      truncated: false
    }),
    [busy, setBusy] = React.useState(false),
    [message, setMessage] = React.useState(''),
    [error, setError] = React.useState(''),
    [loaded, setLoaded] = React.useState(false);
  var [pending, setPending] = React.useState(null),
    [receipt, setReceipt] = React.useState(null);
  var lock = React.useRef(false),
    alive = React.useRef(false),
    sequence = React.useRef(0);
  var key = proposalIntentKey(api, ticker, record.id);
  var refresh = async () => {
    if (lock.current) return;
    var seq = ++sequence.current;
    try {
      var saved = readProposalIntent(sessionStorage, key, record.id);
      if (alive.current) setPending(saved);
      if (saved) {
        var _r = await fetch(`${api}/api/research/amendments/${encodeURIComponent(ticker)}?requestId=${encodeURIComponent(saved.requestId)}`, {
          signal: AbortSignal.timeout(20000)
        });
        if (!_r.ok) throw Error('Exact proposal status unavailable. No submission was retried.');
        var found = proposalReceipt(await _r.json(), ticker, saved);
        if (!alive.current || seq !== sequence.current) return;
        if (found) {
          clearProposalIntent(sessionStorage, key, record.id, saved);
          setPending(null);
          setReceipt(found);
          setMessage('Saved proposal found. No additional comparison was started.');
        }
      }
      var r = await fetch(`${api}/api/research/amendments/${encodeURIComponent(ticker)}?workId=${encodeURIComponent(record.id)}`, {
        signal: AbortSignal.timeout(20000)
      });
      if (!r.ok) throw Error('Proposal status unavailable. Previously loaded links remain available.');
      var d = linkedProposals(await r.json(), ticker, record.id);
      if (alive.current && seq === sequence.current) {
        setData(d);
        setError('');
        setLoaded(true);
      }
    } catch (e) {
      if (alive.current && seq === sequence.current) setError(e.message || 'Browser storage is unavailable. Proposal submission is paused.');
    }
  };
  React.useEffect(() => {
    alive.current = true;
    refresh();
    var timer = setInterval(() => {
      if (!document.hidden) refresh();
    }, 15000);
    return () => {
      alive.current = false;
      sequence.current++;
      clearInterval(timer);
    };
  }, [api, ticker, record.id]);
  var submit = async () => {
    if (lock.current) return;
    lock.current = true;
    sequence.current++;
    setBusy(true);
    setMessage('');
    try {
      var payload = reserveProposalIntent(sessionStorage, key, record.id, {
        requestId: crypto.randomUUID(),
        workEvidence: {
          id: record.id,
          revision: record.revision
        },
        instructions: ''
      });
      setPending(payload);
      var r = await fetch(`${api}/api/research/amendments/${encodeURIComponent(ticker)}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(20000)
      });
      var d = await r.json();
      if (!r.ok) throw Error(d.error || 'Proposal submission not confirmed.');
      if (d.jobId !== payload.requestId) throw Error('Submission receipt did not match the saved request.');
      if (alive.current) setMessage('Submission received. Checking the exact saved proposal; no thesis edits were applied.');
    } catch (e) {
      if (alive.current) setMessage(`${e.message} Check proposal status before retrying; any retry keeps the original saved request.`);
    } finally {
      lock.current = false;
      if (alive.current) {
        await refresh();
        setBusy(false);
      }
    }
  };
  var linked = data.jobs,
    active = data.activeProposal;
  return /*#__PURE__*/React.createElement("details", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("summary", null, "Recorded answer \u2192 thesis proposal"), /*#__PURE__*/React.createElement("p", null, "Compare this saved follow-up (revision ", record.revision, ") with the current thesis. Charlie will retain the meeting, question, recorded answer and your separate interpretation, then prepare proposed edits for review."), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Uses model credits only when you submit. Recorded notes are not independently verified transcripts. Changed answers or case baselines require an updated follow-up. The saved thesis stays unchanged until you explicitly accept edits."), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error), pending && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Unresolved submission for saved follow-up revision ", pending.workEvidence.revision, pending.workEvidence.revision !== record.revision ? ' (different from the current revision)' : '', ". Reloading only checks its status. An explicit retry reuses this exact request; it does not use newer answers."), receipt && /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("a", {
    href: proposalLink(ticker, receipt.id)
  }, "Review recovered proposal"), " \xB7 ", receipt.status.replaceAll('_', ' ')), !loaded && !error && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading linked proposals\u2026"), loaded && !error && !linked.length && /*#__PURE__*/React.createElement("p", null, "No saved thesis proposal is linked to this follow-up. Other company proposals are separate."), linked.map(j => /*#__PURE__*/React.createElement("p", {
    key: j.id
  }, "Follow-up r", j.work_evidence.revision, j.work_evidence.revision !== record.revision ? ' · Different saved revision' : '', " \xB7 ", j.status.replaceAll('_', ' '), j.error ? ` · ${j.error}` : '', " \xB7 ", /*#__PURE__*/React.createElement("a", {
    href: proposalLink(ticker, j.id)
  }, "Review this saved proposal"))), data.truncated && /*#__PURE__*/React.createElement("p", null, "Showing the 30 most recent proposals for this follow-up. Older proposal links remain valid; this is not a complete history."), active && /*#__PURE__*/React.createElement("p", null, "A thesis proposal already exists for this company. ", /*#__PURE__*/React.createElement("a", {
    href: proposalLink(ticker, active.id)
  }, "Review the active company proposal"), " before starting another. It may relate to different evidence."), /*#__PURE__*/React.createElement("button", {
    disabled: disabled || busy || !!error || active || !loaded,
    onClick: submit
  }, busy ? 'Submitting…' : pending ? 'Retry original saved request' : 'Prepare thesis proposal from saved answer'), ' ', /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: refresh
  }, "Check proposal status"), ' ', /*#__PURE__*/React.createElement("a", {
    href: `#view=desk&section=evidence&ticker=${encodeURIComponent(ticker)}`
  }, "All company proposals \u2192"), message && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, message));
}