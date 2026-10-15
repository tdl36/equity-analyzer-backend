function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
import { linkedProposals, proposalLink } from './meeting-answer-proposal-model.mjs';
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
  var lock = React.useRef(false),
    pending = React.useRef(null),
    alive = React.useRef(false),
    sequence = React.useRef(0);
  var refresh = async () => {
    var seq = ++sequence.current;
    try {
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
      if (alive.current && seq === sequence.current) setError(e.message);
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
    setBusy(true);
    setMessage('');
    var payload = pending.current || {
      requestId: crypto.randomUUID(),
      workEvidence: {
        id: record.id,
        revision: record.revision
      },
      instructions: ''
    };
    pending.current = payload;
    try {
      var r = await fetch(`${api}/api/research/amendments/${ticker}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(20000)
      });
      var d = await r.json();
      if (!r.ok) {
        if (r.status < 500) pending.current = null;
        throw Error(d.error || 'Proposal submission not confirmed.');
      }
      pending.current = null;
      if (alive.current) setMessage('Thesis comparison queued. Review the proposed wording in Evidence & changes; no thesis edits were applied.');
    } catch (e) {
      if (alive.current) setMessage(`${e.message}${pending.current ? ' Use Check status first; submitting again reuses the same request identity.' : ''}`);
    } finally {
      if (alive.current) {
        await refresh();
        setBusy(false);
      }
      lock.current = false;
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
  }, error), !loaded && !error && /*#__PURE__*/React.createElement("p", {
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
  }, busy ? 'Submitting…' : pending.current ? 'Reconcile same proposal request' : 'Prepare thesis proposal from saved answer'), ' ', /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: refresh
  }, "Check proposal status"), ' ', /*#__PURE__*/React.createElement("a", {
    href: `#view=desk&section=evidence&ticker=${encodeURIComponent(ticker)}`
  }, "All company proposals \u2192"), message && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, message));
}