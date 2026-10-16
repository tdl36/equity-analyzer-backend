function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
import { MeetingAnswerCapture } from './meeting-answer-capture';
import { originLink } from './research-meeting-model.mjs';
import * as React from 'react';
import { checkedMeetingContext } from './meeting-context-model.mjs';
function useContextRead(api, meetingId, questionSetId, part, enabled) {
  var [state, setState] = React.useState({
      data: null,
      error: '',
      loading: false
    }),
    [attempt, setAttempt] = React.useState(0);
  var sequence = React.useRef(0);
  React.useEffect(() => {
    if (!enabled) return;
    var active = true;
    var serial = ++sequence.current;
    setState(previous => ({
      ...previous,
      error: '',
      loading: true
    }));
    (async () => {
      try {
        var suffix = part === 'quality' ? `?questionSetId=${questionSetId}` : '';
        var r = await fetch(`${api}/api/mp/meetings/${meetingId}/${part}${suffix}`, {
          signal: AbortSignal.timeout(20000)
        });
        if (!r.ok) throw Error('Saved ' + part + ' could not be loaded.');
        var data = checkedMeetingContext(part, await r.json(), meetingId, questionSetId);
        if (active && serial === sequence.current) setState({
          data,
          error: '',
          loading: false
        });
      } catch (e) {
        if (active && serial === sequence.current) setState(previous => ({
          ...previous,
          error: e.message,
          loading: false
        }));
      }
    })();
    return () => {
      active = false;
    };
  }, [api, meetingId, questionSetId, part, enabled, attempt]);
  return {
    ...state,
    retry: () => setAttempt(n => n + 1),
    update: fn => {
      sequence.current++;
      setState(previous => ({
        ...previous,
        data: fn(previous.data),
        loading: false
      }));
    }
  };
}
function ContextStatus({
  name,
  read
}) {
  return /*#__PURE__*/React.createElement("div", null, read.loading && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading ", name, "\u2026"), read.error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, name, ": ", read.error, " ", read.data ? 'Previously loaded records remain visible. ' : '', /*#__PURE__*/React.createElement("button", {
    disabled: read.loading,
    onClick: read.retry
  }, "Retry ", name)));
}
export function MeetingSession(props) {
  return /*#__PURE__*/React.createElement(MeetingSessionView, _extends({
    key: props.api + '|' + props.meetingId + '|' + props.questionSet.id
  }, props));
}
function MeetingSessionView({
  api,
  meetingId,
  questionSet
}) {
  var [open, setOpen] = React.useState(false),
    [high, setHigh] = React.useState(true),
    [history, setHistory] = React.useState(false);
  var answerRead = useContextRead(api, meetingId, questionSet.id, 'answers', open);
  var versionRead = useContextRead(api, meetingId, questionSet.id, 'versions', history);
  var qualityRead = useContextRead(api, meetingId, questionSet.id, 'quality', open || history);
  var quality = qualityRead.data,
    answers = answerRead.data?.answers || [],
    versions = versionRead.data?.versions || [];
  return /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel mb-3"
  }, /*#__PURE__*/React.createElement("div", {
    className: "desk-filter"
  }, /*#__PURE__*/React.createElement("button", {
    onClick: () => setOpen(!open)
  }, open ? 'Close meeting mode' : 'Open meeting mode'), /*#__PURE__*/React.createElement("button", {
    onClick: () => setHistory(!history)
  }, history ? 'Hide version history' : 'Compare saved versions')), open && /*#__PURE__*/React.createElement(ContextStatus, {
    name: "saved answers",
    read: answerRead
  }), " ", history && /*#__PURE__*/React.createElement(ContextStatus, {
    name: "version history",
    read: versionRead
  }), " ", (open || history) && /*#__PURE__*/React.createElement(ContextStatus, {
    name: "quality checks",
    read: qualityRead
  }), quality && (open || history) && /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, quality.questions, " saved items \xB7 ", quality.highPriority, " must-ask \xB7 ", quality.withVerifiedPassages, " with verified passage records \xB7 ", quality.citedSources, "/", quality.availableSources, " sources cited \xB7 ", quality.duplicateQuestions, " exact duplicates. ", quality.scope), quality?.nonQuestionItems?.length > 0 && (open || history) && /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Review items ", quality.nonQuestionItems.join(', '), ": they may be coverage notes rather than management questions. Ask Charlie to revise these before using them in a meeting."), quality?.numericPremisesToReview?.length > 0 && (open || history) && /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Numeric premises to review \xB7 ", quality.numericPremisesToReview.length, " questions"), /*#__PURE__*/React.createElement("p", null, "These values were not found verbatim in their attached passages. They may be valid periods, requested targets or calculations; check the original before treating them as established facts."), quality.numericPremisesToReview.map(r => /*#__PURE__*/React.createElement("p", {
    key: r.question
  }, "Question ", r.question, ": ", r.values.join(', ')))), history && /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", null, "Previous versions remain unchanged. Review revisions before using them in a meeting."), versionRead.data?.truncated && /*#__PURE__*/React.createElement("p", null, "Showing the newest 30 saved versions. Older versions remain saved; this list is incomplete."), versionRead.data && !versions.length && /*#__PURE__*/React.createElement("p", null, "No saved versions were returned for this meeting."), versions.map(v => /*#__PURE__*/React.createElement("details", {
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
  }, /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("strong", null, q.question)), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Evidence and follow-up"), /*#__PURE__*/React.createElement("p", null, q.context), /*#__PURE__*/React.createElement("p", null, q.source), originLink(q.research_origin) && /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("a", {
    href: originLink(q.research_origin)
  }, "Open the saved ", q.research_origin.ticker, " research report"), " \xB7 ", q.research_origin.path), q.supporting_quotes?.map((c, k) => /*#__PURE__*/React.createElement("blockquote", {
    key: k
  }, c.quote, /*#__PURE__*/React.createElement("cite", null, " \u2014 ", c.filename))), /*#__PURE__*/React.createElement("p", null, q.follow_up_angle)), /*#__PURE__*/React.createElement(MeetingAnswerCapture, {
    api: api,
    meetingId: meetingId,
    question: q.question,
    answers: answers,
    onSaved: row => answerRead.update(previous => ({
      ...previous,
      answers: (previous?.answers || []).map(a => a.id === row.id ? row : a)
    }))
  })))))));
}