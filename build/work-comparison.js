import * as React from 'react';
import { latestWorkForComparison, workDifferences } from './work-comparison.mjs';
// Read-only: never replace the editor, advance its baseline or submit work.
export function WorkComparison({
  api,
  ticker,
  record,
  draft,
  baseline,
  onOpen,
  replaceDisabled = false
}) {
  var [state, setState] = React.useState(null),
    alive = React.useRef(true),
    sequence = React.useRef(0);
  React.useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      ++sequence.current;
    };
  }, []);
  var load = async () => {
    var seq = ++sequence.current;
    setState({
      loading: true
    });
    try {
      var response = await fetch(api + '/api/research/workbench/' + encodeURIComponent(ticker), {
        cache: 'no-store',
        signal: AbortSignal.timeout(20000)
      });
      if (!response.ok) throw Error('Saved work is unavailable. Retry the comparison; your draft is unchanged.');
      var saved = latestWorkForComparison(await response.json(), record, ticker, draft.kind);
      if (alive.current && seq === sequence.current) setState({
        saved,
        checked: new Date().toLocaleTimeString()
      });
    } catch (e) {
      if (alive.current && seq === sequence.current) setState({
        error: e.message
      });
    }
  };
  var rows = state?.saved ? workDifferences(draft, state.saved.draft) : [];
  return /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel",
    "aria-label": "Saved work comparison"
  }, /*#__PURE__*/React.createElement("button", {
    disabled: state?.loading,
    onClick: load
  }, state ? 'Refresh saved work comparison' : 'Compare draft with latest saved work'), state && /*#__PURE__*/React.createElement("button", {
    onClick: () => {
      ++sequence.current;
      setState(null);
    }
  }, "Close comparison"), state?.loading && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Reading saved work\u2026 Your draft stays unchanged."), state?.error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, state.error), state?.saved && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("h3", null, "Draft versus saved work"), /*#__PURE__*/React.createElement("p", null, "Your draft is based on work revision ", record.revision, ", case revision ", baseline, ". Latest saved work at ", state.checked, ": revision ", state.saved.revision, ", case revision ", state.saved.caseRevision, "."), state.saved.revision > record.revision && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, "A newer work revision is saved. Your editing baseline has not changed. Review the differences before deciding what to keep."), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Read-only comparison of editable fields. Derived calculations and saved evidence snapshots remain in the saved record and version history. This read does not establish that sources or the investment case are current. Refresh to check for subsequent saves."), !rows.length ? /*#__PURE__*/React.createElement("p", null, "No editable field differences. Revision and case baselines above still apply.") : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, rows.length, " differing fields. Unchanged fields are omitted."), rows.map(r => /*#__PURE__*/React.createElement("div", {
    key: r.key,
    className: "workspace-panel",
    style: {
      overflowWrap: 'anywhere'
    }
  }, /*#__PURE__*/React.createElement("h4", null, r.label), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 260px), 1fr))',
      gap: '1rem'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      minWidth: 0
    }
  }, /*#__PURE__*/React.createElement("strong", null, "Your draft"), /*#__PURE__*/React.createElement("p", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, r.local)), /*#__PURE__*/React.createElement("div", {
    style: {
      minWidth: 0
    }
  }, /*#__PURE__*/React.createElement("strong", null, "Saved revision ", state.saved.revision), /*#__PURE__*/React.createElement("p", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, r.saved)))))), /*#__PURE__*/React.createElement("p", null, "Keep editing your draft, or open the compared saved version. Replacing unfinished edits requires confirmation. Nothing is merged or saved by this comparison."), /*#__PURE__*/React.createElement("button", {
    disabled: replaceDisabled,
    onClick: () => onOpen({
      id: state.saved.id,
      revision: state.saved.revision
    }, state.saved.draft)
  }, "Open compared saved revision ", state.saved.revision)));
}