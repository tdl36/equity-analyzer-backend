import * as React from 'react';
export function MeetingSession({
  api,
  meetingId,
  questionSet
}) {
  var [quality, setQuality] = React.useState(null);
  var [open, setOpen] = React.useState(false),
    [high, setHigh] = React.useState(true),
    [answers, setAnswers] = React.useState([]),
    [versions, setVersions] = React.useState([]),
    [error, setError] = React.useState(''),
    [history, setHistory] = React.useState(false);
  React.useEffect(() => {
    if (!open && !history) return;
    var active = true;
    Promise.all(['answers', 'versions', 'quality'].map(async part => {
      var r = await fetch(`${api}/api/mp/meetings/${meetingId}/${part}`, {
        signal: AbortSignal.timeout(20000)
      });
      if (!r.ok) throw Error('Meeting context could not be loaded.');
      return r.json();
    })).then(([a, v, q]) => {
      if (active) {
        setQuality(q);
        setAnswers(a.answers);
        setVersions(v.versions);
        setError('');
      }
    }).catch(e => {
      if (active) setError(e.message);
    });
    return () => {
      active = false;
    };
  }, [api, meetingId, questionSet.id, open, history]);
  return /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel mb-3"
  }, /*#__PURE__*/React.createElement("div", {
    className: "desk-filter"
  }, /*#__PURE__*/React.createElement("button", {
    onClick: () => setOpen(!open)
  }, open ? 'Close meeting mode' : 'Open meeting mode'), /*#__PURE__*/React.createElement("button", {
    onClick: () => setHistory(!history)
  }, history ? 'Hide version history' : 'Compare saved versions')), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error), quality && (open || history) && /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, quality.questions, " questions \xB7 ", quality.highPriority, " must-ask \xB7 ", quality.withVerifiedPassages, " with verified passage records \xB7 ", quality.citedSources, "/", quality.availableSources, " sources cited \xB7 ", quality.duplicateQuestions, " exact duplicates. ", quality.scope), quality?.numericPremisesToReview?.length > 0 && (open || history) && /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Numeric premises to review \xB7 ", quality.numericPremisesToReview.length, " questions"), /*#__PURE__*/React.createElement("p", null, "These values were not found verbatim in their attached passages. They may be valid periods, requested targets or calculations; check the original before treating them as established facts."), quality.numericPremisesToReview.map(r => /*#__PURE__*/React.createElement("p", {
    key: r.question
  }, "Question ", r.question, ": ", r.values.join(', ')))), history && /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", null, "Previous versions remain unchanged. Review revisions before using them in a meeting."), versions.map(v => /*#__PURE__*/React.createElement("details", {
    key: v.id
  }, /*#__PURE__*/React.createElement("summary", null, "Version ", v.version, " \xB7 ", v.topics.reduce((n, t) => n + (t.questions || []).length, 0), " questions", v.id === questionSet.id ? ' · Current' : ''), v.topics.map((t, i) => /*#__PURE__*/React.createElement("div", {
    key: i
  }, /*#__PURE__*/React.createElement("h4", null, t.topic), /*#__PURE__*/React.createElement("ol", null, t.questions.map((q, j) => /*#__PURE__*/React.createElement("li", {
    key: j
  }, q.question)))))))), open && /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h3", null, "Meeting mode \xB7 v", questionSet.version), /*#__PURE__*/React.createElement("label", null, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: high,
    onChange: e => setHigh(e.target.checked)
  }), " Must-ask questions only"), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Capture actual management answers here. Saving an answer marks the question answered; generating a question does not."), questionSet.topics.map((t, i) => /*#__PURE__*/React.createElement("section", {
    key: i
  }, /*#__PURE__*/React.createElement("h4", null, t.topic), t.questions.filter(q => !high || q.priority === 'high').map((q, j) => /*#__PURE__*/React.createElement("article", {
    className: "workspace-panel",
    key: q.question
  }, /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("strong", null, q.question)), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Evidence and follow-up"), /*#__PURE__*/React.createElement("p", null, q.context), /*#__PURE__*/React.createElement("p", null, q.source), q.supporting_quotes?.map((c, k) => /*#__PURE__*/React.createElement("blockquote", {
    key: k
  }, c.quote, /*#__PURE__*/React.createElement("cite", null, " \u2014 ", c.filename))), /*#__PURE__*/React.createElement("p", null, q.follow_up_angle)), /*#__PURE__*/React.createElement(AnswerCapture, {
    api: api,
    answer: answers.find(a => a.question === q.question)
  })))))));
}
function AnswerCapture({
  api,
  answer
}) {
  var [note, setNote] = React.useState(answer?.response_notes || ''),
    [message, setMessage] = React.useState(''),
    [busy, setBusy] = React.useState(false);
  React.useEffect(() => {
    setNote(answer?.response_notes || '');
    setMessage('');
  }, [answer?.id]);
  if (!answer) return /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Answer record unavailable for this question.");
  return /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("label", null, "Management answer", /*#__PURE__*/React.createElement("textarea", {
    className: "w-full",
    value: note,
    maxLength: 10000,
    onChange: e => setNote(e.target.value),
    placeholder: "Capture what management actually said\u2026"
  })), /*#__PURE__*/React.createElement("button", {
    disabled: busy || !note.trim(),
    onClick: async () => {
      setBusy(true);
      try {
        var r = await fetch(api + '/api/mp/past-questions/' + answer.id + '/note', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            responseNotes: note.trim(),
            status: 'answered'
          }),
          signal: AbortSignal.timeout(20000)
        });
        if (!r.ok) throw Error('Answer could not be saved. Your text is still here.');
        setMessage('Answer saved to meeting history.');
      } catch (e) {
        setMessage(e.message);
      } finally {
        setBusy(false);
      }
    }
  }, busy ? 'Saving…' : 'Save answer'), /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, message));
}