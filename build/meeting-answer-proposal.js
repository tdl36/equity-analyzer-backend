import * as React from 'react';
export function MeetingAnswerProposal({
  api,
  ticker,
  record,
  disabled = false
}) {
  var [jobs, setJobs] = React.useState([]),
    [busy, setBusy] = React.useState(false),
    [message, setMessage] = React.useState(''),
    [error, setError] = React.useState(''),
    [loaded, setLoaded] = React.useState(false);
  var lock = React.useRef(false),
    pending = React.useRef(null),
    alive = React.useRef(false);
  var refresh = async () => {
    try {
      var r = await fetch(`${api}/api/research/amendments/${ticker}`, {
        signal: AbortSignal.timeout(20000)
      });
      if (!r.ok) throw Error('Proposal status unavailable.');
      var d = await r.json();
      if (alive.current) {
        setJobs(d.jobs || []);
        setError('');
        setLoaded(true);
      }
    } catch (e) {
      if (alive.current) setError(e.message);
    }
  };
  React.useEffect(() => {
    alive.current = true;
    refresh();
    var timer = setInterval(refresh, 15000);
    return () => {
      alive.current = false;
      clearInterval(timer);
    };
  }, [api, ticker]);
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
  var linked = jobs.filter(j => j.work_evidence?.id === record.id);
  var active = jobs.some(j => ['queued', 'running', 'awaiting_approval'].includes(j.status));
  return /*#__PURE__*/React.createElement("details", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("summary", null, "Recorded answer \u2192 thesis proposal"), /*#__PURE__*/React.createElement("p", null, "Compare this saved follow-up (revision ", record.revision, ") with the current thesis. Charlie will retain the meeting, question, recorded answer and your separate interpretation, then prepare proposed edits for review."), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Uses model credits only when you submit. Recorded notes are not independently verified transcripts. Changed answers or case baselines require an updated follow-up. The saved thesis stays unchanged until you explicitly accept edits."), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error), linked.map(j => /*#__PURE__*/React.createElement("p", {
    key: j.id
  }, "Follow-up r", j.work_evidence.revision, " \xB7 ", j.status.replaceAll('_', ' '), j.error ? ` · ${j.error}` : '')), active && /*#__PURE__*/React.createElement("p", null, "A thesis proposal already exists for this company. Review or dismiss it before starting another."), /*#__PURE__*/React.createElement("button", {
    disabled: disabled || busy || !!error || active || !loaded,
    onClick: submit
  }, busy ? 'Submitting…' : pending.current ? 'Reconcile same proposal request' : 'Prepare thesis proposal from saved answer'), ' ', /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: refresh
  }, "Check proposal status"), ' ', /*#__PURE__*/React.createElement("a", {
    href: `#view=desk&section=evidence&ticker=${encodeURIComponent(ticker)}`
  }, "Review thesis proposals \u2192"), message && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, message));
}