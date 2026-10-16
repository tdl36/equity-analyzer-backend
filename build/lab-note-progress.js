function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
import * as React from 'react';
import { checkedNoteProgress, batchOutcome } from './lab-note-progress-model.mjs';
export function LabNoteProgress(props) {
  return /*#__PURE__*/React.createElement(Inspection, _extends({
    key: JSON.stringify([props.api, props.ticker, props.id])
  }, props));
}
function Inspection({
  api,
  ticker,
  id,
  read,
  onOpen,
  onRecover,
  disabled,
  describeReason
}) {
  var [data, setData] = React.useState(null),
    [error, setError] = React.useState(''),
    [busy, setBusy] = React.useState(false),
    [attempt, refresh] = React.useState(0);
  React.useEffect(() => {
    var live = true,
      pending = false;
    var controller = new AbortController();
    var load = async () => {
      if (pending) return;
      pending = true;
      setBusy(true);
      try {
        var d = checkedNoteProgress(await read(api, `/api/pipeline-lab/note-job/${encodeURIComponent(id)}/inspection?ticker=${encodeURIComponent(ticker)}`, {
          signal: controller.signal
        }), id, ticker);
        if (live) {
          setData(d);
          setError('');
        }
      } catch (e) {
        if (live) setError('Saved progress could not be verified. Retained information may be out of date; retry this read before recovery.');
      } finally {
        pending = false;
        if (live) setBusy(false);
      }
    };
    load();
    var timer = setInterval(load, 10000);
    return () => {
      live = false;
      controller.abort();
      clearInterval(timer);
    };
  }, [api, ticker, id, attempt]);
  return /*#__PURE__*/React.createElement("section", {
    "aria-label": "Exact full-note progress",
    style: {
      border: '1px solid #8885',
      padding: 16,
      margin: '16px 0',
      overflowWrap: 'anywhere'
    }
  }, /*#__PURE__*/React.createElement("h4", null, ticker, " \xB7 Selected full-note progress"), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "Saved note ", id)), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => refresh(n => n + 1)
  }, "Refresh selected progress"), busy && !data && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading this note\u2019s saved progress\u2026"), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error), data && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("strong", null, data.status === 'draft' ? 'Draft saved' : data.status), " \xB7 ", data.modelKey, " \xB7 updated ", new Date(data.updatedAt).toLocaleString()), data.reason && /*#__PURE__*/React.createElement("p", null, describeReason?.(data.reason) || 'Inspect the saved outcome before continuing.'), /*#__PURE__*/React.createElement("ol", null, Array.from({
    length: data.batchCount
  }, (_, i) => {
    var call = data.batches.find(b => b.batch === i + 1);
    return /*#__PURE__*/React.createElement("li", {
      key: i
    }, "Batch ", i + 1, ": ", batchOutcome(call?.status));
  })), /*#__PURE__*/React.createElement("p", null, "A dispatch record does not confirm a charge or provider completion. Saved replies do not establish an accepted research note. This read does not start or resume generation."), /*#__PURE__*/React.createElement("div", {
    className: "workspace-actions"
  }, data.revision ? /*#__PURE__*/React.createElement("button", {
    disabled: disabled || busy || !!error,
    onClick: () => onOpen(id, data.revision)
  }, "Open selected draft revision ", data.revision) : /*#__PURE__*/React.createElement("p", null, "No draft revision is saved for this job."), data.acceptedRevision && /*#__PURE__*/React.createElement("button", {
    disabled: disabled || busy || !!error,
    onClick: () => onOpen(id, data.acceptedRevision)
  }, "Open selected accepted revision ", data.acceptedRevision), data.status === 'attention' && data.reason !== 'parent_stopped' && /*#__PURE__*/React.createElement("button", {
    disabled: disabled || busy || !!error,
    onClick: async () => {
      await onRecover(id);
      refresh(n => n + 1);
    }
  }, "Reconcile selected saved progress")), data.status === 'attention' && data.reason !== 'parent_stopped' && /*#__PURE__*/React.createElement("p", null, "Reconciliation reuses saved replies where possible. Safe unfinished batches may resume under the original authorization and spend credits; uncertain calls are not repeated.")));
}