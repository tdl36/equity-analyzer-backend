import * as React from 'react';
import { checkedAssignmentMeeting } from './assignment-review-model.mjs';
import { originLink } from './research-meeting-model.mjs';
export function AssignmentMeetingOutput({
  api,
  job,
  doc
}) {
  var identity = JSON.stringify([api, job.id, job.ticker, job.result.reportId, job.result.meetingId, job.result.meetingQuestionSetId]);
  return /*#__PURE__*/React.createElement(SavedMeeting, {
    key: identity,
    api: api,
    job: job,
    doc: doc
  });
}
function SavedMeeting({
  api,
  job,
  doc
}) {
  var [data, setData] = React.useState(null),
    [error, setError] = React.useState(''),
    [attempt, retry] = React.useState(0);
  React.useEffect(() => {
    var live = true;
    var controller = new AbortController();
    setData(null);
    setError('');
    (async () => {
      try {
        var r = await fetch(`${api}/api/research/assignments/${encodeURIComponent(job.id)}/saved-meeting`, {
          signal: AbortSignal.any([controller.signal, AbortSignal.timeout(20000)])
        });
        if (!r.ok) throw Error('This assignment’s exact meeting version could not be opened. No newer version has been substituted.');
        var saved = checkedAssignmentMeeting(await r.json(), job);
        if (live) setData(saved);
      } catch (e) {
        if (live) setError(e.message);
      }
    })();
    return () => {
      live = false;
      controller.abort();
    };
  }, [attempt]);
  return /*#__PURE__*/React.createElement("section", {
    className: "review-evidence",
    "aria-label": "Delivered meeting questions"
  }, /*#__PURE__*/React.createElement("h3", null, "Meeting questions from this assignment"), /*#__PURE__*/React.createElement("p", null, "This is the saved version delivered from the shared research pack. The current meeting may contain later revisions. Opening either view starts no generation."), error ? /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error, " ", /*#__PURE__*/React.createElement("button", {
    onClick: () => retry(n => n + 1)
  }, "Retry delivered version")) : !data ? /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading the delivered meeting version\u2026") : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("strong", null, job.ticker, " \xB7 Delivered version ", data.version), " \xB7 ", data.status, " \xB7 ", new Date(data.created_at).toLocaleString()), !data.topics.length && /*#__PURE__*/React.createElement("p", null, "No questions were saved in this version."), data.topics.map((topic, i) => /*#__PURE__*/React.createElement("section", {
    key: i
  }, /*#__PURE__*/React.createElement("h4", null, topic.topic), topic.description && /*#__PURE__*/React.createElement("p", null, topic.description), /*#__PURE__*/React.createElement("ol", null, (topic.questions || []).map((q, j) => /*#__PURE__*/React.createElement("li", {
    key: j
  }, /*#__PURE__*/React.createElement("strong", null, q.question), q.priority && /*#__PURE__*/React.createElement("p", null, q.priority, " priority"), /*#__PURE__*/React.createElement("p", null, q.context), q.source && /*#__PURE__*/React.createElement("p", null, "Source: ", q.source), q.source_filenames?.length > 0 && /*#__PURE__*/React.createElement("p", null, "Saved source references: ", q.source_filenames.join(', ')), q.supporting_quotes?.length > 0 ? q.supporting_quotes.map((c, k) => /*#__PURE__*/React.createElement("blockquote", {
    key: k
  }, /*#__PURE__*/React.createElement("p", null, c.quote), /*#__PURE__*/React.createElement("cite", null, c.filename))) : /*#__PURE__*/React.createElement("p", null, "No supporting passages were saved with this question. Check the original sources before relying on its premise."), q.source_support && /*#__PURE__*/React.createElement("p", null, q.source_support), originLink(q.research_origin) && /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("a", {
    style: {
      color: "inherit"
    },
    href: originLink(q.research_origin)
  }, "Open saved research report")), q.follow_up_angle && /*#__PURE__*/React.createElement("p", null, "Follow-up: ", q.follow_up_angle))))))), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("a", {
    className: "review-link",
    href: doc.href
  }, "Open current meeting workspace")), /*#__PURE__*/React.createElement("p", null, "Use the current workspace to record actual answers and follow-up. This saved view does not accept research or change the thesis."));
}