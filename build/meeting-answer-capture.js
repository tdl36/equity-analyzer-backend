import * as React from 'react';
import { checkedAnswer, answerPayload, answerReceipt } from './meeting-answer-capture-model.mjs';
export function MeetingAnswerCapture({
  api,
  meetingId,
  question,
  answers,
  onSaved
}) {
  var matches = answers.filter(a => a.question === question && a.meeting_id === meetingId);
  var [selected, setSelected] = React.useState('');
  var answer = matches.length === 1 ? matches[0] : matches.find(a => String(a.id) === selected);
  return /*#__PURE__*/React.createElement("div", null, matches.length > 1 && /*#__PURE__*/React.createElement("label", null, "Several saved records use this question. Choose the record to inspect; these records do not identify a question-pack version.", /*#__PURE__*/React.createElement("select", {
    value: selected,
    onChange: e => setSelected(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Choose a saved answer record"), matches.map(a => /*#__PURE__*/React.createElement("option", {
    key: a.id,
    value: a.id
  }, "Record ", a.id, " \xB7 ", a.status, a.response_notes ? ' · has saved notes' : '')))), answer ? /*#__PURE__*/React.createElement(AnswerEditor, {
    key: api + '|' + meetingId + '|' + answer.id,
    api: api,
    initial: answer,
    onSaved: onSaved
  }) : /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, matches.length ? 'Select the exact saved record above before recording an answer.' : 'Answer record unavailable for this question.'));
}
export function AnswerEditor({
  api,
  initial,
  onSaved
}) {
  var key = 'charlie-meeting-answer-v1:' + JSON.stringify([api, initial.meeting_id, initial.id]);
  var [base, setBase] = React.useState(initial),
    [text, setText] = React.useState(initial.response_notes || ''),
    [pending, setPending] = React.useState(null),
    [remote, setRemote] = React.useState(null),
    [message, setMessage] = React.useState(''),
    [error, setError] = React.useState(''),
    [busy, setBusy] = React.useState(false),
    [ready, setReady] = React.useState(false);
  var alive = React.useRef(false),
    lock = React.useRef(false);
  var persist = (b, t, p) => {
    var raw = JSON.stringify({
      base: b,
      text: t,
      pending: p
    });
    sessionStorage.setItem(key, raw);
    if (sessionStorage.getItem(key) !== raw) throw Error('Browser recovery could not be saved. No answer was sent.');
  };
  React.useEffect(() => {
    alive.current = true;
    try {
      var raw = sessionStorage.getItem(key);
      if (raw) {
        var d = JSON.parse(raw);
        checkedAnswer(d.base, initial.meeting_id, initial.id, initial.question);
        if (typeof d.text !== 'string' || d.text.length > 10000 || d.pending !== null && JSON.stringify(d.pending) !== JSON.stringify(answerPayload(d.base, d.text))) throw Error('Saved answer recovery is inconsistent.');
        setBase(d.base);
        setText(d.text);
        setPending(d.pending);
        setMessage('Recovered your local answer draft. Check the saved answer before changing its baseline.');
      }
      setReady(true);
    } catch (e) {
      setError(e.message + ' Local recovery must be resolved before saving.');
    }
    return () => {
      alive.current = false;
    };
  }, [key]);
  var dirty = text !== (base.response_notes || '') || !!pending;
  React.useEffect(() => {
    var warn = e => {
      if (dirty) {
        e.preventDefault();
        e.returnValue = '';
      }
    };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [dirty]);
  var confirm = row => {
    sessionStorage.removeItem(key);
    if (sessionStorage.getItem(key) !== null) throw Error('Answer saved, but local recovery could not be cleared. Check the saved answer again.');
    setBase(row);
    onSaved?.(row);
    setPending(null);
    setRemote(null);
    setError('');
    setMessage('Exact answer saved to meeting history. Research and thesis approval remain separate.');
  };
  var save = async () => {
    if (lock.current || !ready || remote) return;
    lock.current = true;
    setBusy(true);
    setError('');
    var payload = pending || answerPayload(base, text);
    try {
      persist(base, text, payload);
      setPending(payload);
      var r = await fetch(`${api}/api/mp/past-questions/${base.id}/note`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(20000)
      });
      var d = await r.json();
      if (!alive.current) return;
      if (r.status === 409) {
        setRemote(checkedAnswer(d.answer, base.meeting_id, base.id, base.question));
        throw Error('Another saved answer differs from your baseline. Your draft is retained; review it below.');
      }
      if (!r.ok) throw Error(d.error || 'Save not confirmed. Check the saved answer before retrying.');
      confirm(answerReceipt(d, base, payload));
    } catch (e) {
      if (alive.current) setError(e.message);
    } finally {
      lock.current = false;
      if (alive.current) setBusy(false);
    }
  };
  var check = async () => {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    setError('');
    try {
      var r = await fetch(`${api}/api/mp/meetings/${base.meeting_id}/answers/${base.id}`, {
        signal: AbortSignal.timeout(20000)
      });
      if (!r.ok) throw Error('Saved answer could not be read. Your draft is retained.');
      var d = await r.json();
      var row = checkedAnswer(d.answer, base.meeting_id, base.id, base.question);
      if (!alive.current) return;
      if (pending && row.response_notes === pending.responseNotes && row.status === pending.status) confirm(row);else if (row.response_notes !== base.response_notes || row.status !== base.status) {
        setRemote(row);
        setMessage('Review the current saved answer below before replacing it.');
      } else setMessage(pending ? 'No matching saved answer found. Retry keeps the same text and baseline.' : 'Saved answer still matches your baseline.');
    } catch (e) {
      if (alive.current) setError(e.message);
    } finally {
      lock.current = false;
      if (alive.current) setBusy(false);
    }
  };
  return /*#__PURE__*/React.createElement("div", {
    style: {
      overflowWrap: 'anywhere'
    }
  }, /*#__PURE__*/React.createElement("p", null, "Saved record ", base.id, " \xB7 ", base.status), /*#__PURE__*/React.createElement("label", null, "Management answer", /*#__PURE__*/React.createElement("textarea", {
    className: "w-full",
    value: text,
    maxLength: 10000,
    disabled: busy || !!pending || !ready,
    onChange: e => {
      var t = e.target.value;
      setText(t);
      setMessage('Draft retained in this browser session.');
      try {
        persist(base, t, null);
        setError('');
      } catch (x) {
        setError(x.message);
      }
    },
    placeholder: "Capture what management actually said\u2026"
  })), /*#__PURE__*/React.createElement("div", {
    className: "workspace-actions"
  }, /*#__PURE__*/React.createElement("button", {
    disabled: busy || !ready || !text.trim() || !!remote,
    onClick: save
  }, busy ? 'Checking…' : pending ? 'Retry same answer save' : 'Save answer'), /*#__PURE__*/React.createElement("button", {
    disabled: busy || !ready,
    onClick: check
  }, "Check saved answer")), remote && /*#__PURE__*/React.createElement("section", {
    "aria-label": "Changed saved answer"
  }, /*#__PURE__*/React.createElement("h4", null, "Current saved answer \xB7 ", remote.status), /*#__PURE__*/React.createElement("p", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, remote.response_notes || 'No answer text saved.'), /*#__PURE__*/React.createElement("p", null, "Your draft above is unchanged. Review both before saving a replacement."), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => {
      try {
        persist(remote, text, null);
        setBase(remote);
        setPending(null);
        setRemote(null);
        setError('');
        setMessage('Reviewed baseline selected. Your draft is unchanged; Save answer explicitly replaces that baseline.');
      } catch (e) {
        setError(e.message);
      }
    }
  }, "Use this reviewed answer as baseline")), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error), /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, message));
}