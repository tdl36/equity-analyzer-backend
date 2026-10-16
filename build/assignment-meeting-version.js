import * as React from 'react';
import { originLink } from './research-meeting-model.mjs';
export function AssignmentMeetingVersion({
  api,
  assignmentId,
  meetingId,
  ticker,
  version
}) {
  var [open, setOpen] = React.useState(false),
    [data, setData] = React.useState(null),
    [error, setError] = React.useState(''),
    [attempt, setAttempt] = React.useState(0);
  React.useEffect(() => {
    setData(null);
    setError('');
    if (!open) return;
    var active = true;
    (async () => {
      try {
        var r = await fetch(`${api}/api/research/assignments/${assignmentId}/meeting-versions/${version.id}`, {
          signal: AbortSignal.timeout(20000)
        });
        if (!r.ok) throw Error('This saved version could not be loaded. Retry to read the same version.');
        var d = await r.json(),
          q = d.questionSet;
        if (d.assignmentId !== assignmentId || q?.id !== version.id || q.meeting_id !== meetingId || q.ticker !== ticker || q.version !== version.version || !Array.isArray(q.topics)) throw Error('Saved version identity could not be verified. No other version has been substituted.');
        if (active) setData(q);
      } catch (e) {
        if (active) setError(e.message);
      }
    })();
    return () => {
      active = false;
    };
  }, [open, attempt, api, assignmentId, meetingId, ticker, version.id, version.version]);
  return /*#__PURE__*/React.createElement("article", {
    style: {
      overflowWrap: 'anywhere'
    }
  }, /*#__PURE__*/React.createElement("button", {
    "aria-expanded": open,
    onClick: () => setOpen(!open)
  }, open ? 'Close' : 'Inspect', " saved version ", version.version), open && /*#__PURE__*/React.createElement("section", {
    "aria-label": `Saved meeting version ${version.version}`
  }, /*#__PURE__*/React.createElement("p", null, "Read-only saved questions and evidence. This does not generate research, accept a thesis change, or record a management answer."), error ? /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error, " ", /*#__PURE__*/React.createElement("button", {
    onClick: () => setAttempt(n => n + 1)
  }, "Retry saved version ", version.version)) : !data ? /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading saved version ", version.version, "\u2026") : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("h4", null, ticker, " \xB7 Version ", data.version, " \xB7 ", data.status), /*#__PURE__*/React.createElement("p", null, new Date(data.created_at).toLocaleString(), " \xB7 ", data.generation_model), data.topics.length === 0 && /*#__PURE__*/React.createElement("p", null, "No questions were saved in this version."), data.topics.map((topic, i) => /*#__PURE__*/React.createElement("section", {
    key: i
  }, /*#__PURE__*/React.createElement("h5", null, topic.topic), /*#__PURE__*/React.createElement("p", null, topic.description), /*#__PURE__*/React.createElement("ol", null, (topic.questions || []).map((q, j) => /*#__PURE__*/React.createElement("li", {
    key: j
  }, /*#__PURE__*/React.createElement("strong", null, q.question), q.priority && /*#__PURE__*/React.createElement("p", null, q.priority, " priority"), /*#__PURE__*/React.createElement("p", null, q.context), q.source && /*#__PURE__*/React.createElement("p", null, "Source: ", q.source), q.source_filenames?.length > 0 && /*#__PURE__*/React.createElement("p", null, "Saved source references: ", q.source_filenames.join(', ')), q.supporting_quotes?.length > 0 ? q.supporting_quotes.map((c, k) => /*#__PURE__*/React.createElement("blockquote", {
    key: k
  }, /*#__PURE__*/React.createElement("p", null, c.quote), /*#__PURE__*/React.createElement("cite", null, c.filename))) : /*#__PURE__*/React.createElement("p", null, "No supporting passages were saved with this question. Check the original sources before relying on its premise."), q.source_support && /*#__PURE__*/React.createElement("p", null, q.source_support), originLink(q.research_origin) && /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("a", {
    href: originLink(q.research_origin)
  }, "Open saved research report")), q.follow_up_angle && /*#__PURE__*/React.createElement("p", null, "Follow-up: ", q.follow_up_angle)))))))));
}