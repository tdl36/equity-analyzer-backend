import * as React from 'react';
export function AssignmentWorkspace({
  api,
  id,
  onNavigate,
  onSection
}) {
  var [open, setOpen] = React.useState(false),
    [data, setData] = React.useState(null),
    [error, setError] = React.useState('');
  React.useEffect(() => {
    if (!open) return;
    var active = true;
    var load = async () => {
      try {
        var r = await fetch(api + '/api/research/assignments/' + id, {
          signal: AbortSignal.timeout(20000)
        });
        var d = await r.json();
        if (!r.ok) throw Error(d.error || 'Assignment unavailable');
        if (active) {
          setData(d);
          setError('');
        }
      } catch (e) {
        if (active) setError(e.message);
      }
    };
    load();
    var timer = setInterval(load, 10000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [open, id, api]);
  return /*#__PURE__*/React.createElement("div", {
    className: "assignment-workspace"
  }, /*#__PURE__*/React.createElement("button", {
    onClick: () => setOpen(!open)
  }, open ? 'Close assignment workspace' : 'Inspect assignment →'), open && /*#__PURE__*/React.createElement("section", {
    "aria-label": "Assignment workspace"
  }, error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error), data && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("h3", null, data.command.ticker, " \xB7 Assignment workspace"), /*#__PURE__*/React.createElement("details", {
    className: "assignment-brief"
  }, /*#__PURE__*/React.createElement("summary", null, "Full assignment and focus areas"), /*#__PURE__*/React.createElement("p", null, data.assignment.instruction)), /*#__PURE__*/React.createElement("p", null, data.assignment.since, " through ", data.assignment.until), /*#__PURE__*/React.createElement("ol", null, /*#__PURE__*/React.createElement("li", null, "Command: ", data.command.status), /*#__PURE__*/React.createElement("li", null, "Collection: ", data.collection?.status || 'awaiting Mac snapshot'), /*#__PURE__*/React.createElement("li", null, "Preparation: ", data.preparation?.status || 'not started', " \xB7 ", data.preparation?.stage || 'waiting for verified sources')), (data.collection?.issue || data.preparation?.error || data.command.error) && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, data.collection?.issue || data.preparation?.error || data.command.error), data.collection?.sourceProgress && /*#__PURE__*/React.createElement("details", {
    open: true
  }, /*#__PURE__*/React.createElement("summary", null, "Collection progress"), /*#__PURE__*/React.createElement("p", null, (data.collection.sourceProgress.documents.handed_off || 0) + (data.collection.sourceProgress.documents.duplicate || 0), " originals saved to iCloud \xB7 ", data.collection.sourceProgress.documents.staged || 0, " awaiting handoff \xB7 ", data.collection.sourceProgress.documents.held || 0, " held"), data.collection.sourceProgress.tasks.map(t => /*#__PURE__*/React.createElement("p", {
    key: t.kind
  }, t.kind.replaceAll('-', ' '), " \xB7 ", t.status.replaceAll('_', ' '), t.expected != null ? ` · ${t.expected} selected` : '')), /*#__PURE__*/React.createElement("p", null, "Meeting originals appear below after collection and the analyst brief finish.")), data.selection && /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Selected sources and decisions"), data.selection.candidates.map(c => /*#__PURE__*/React.createElement("p", {
    key: c.url
  }, /*#__PURE__*/React.createElement("strong", null, c.decision), " \xB7 ", c.publisher, " \xB7 ", c.title, /*#__PURE__*/React.createElement("br", null), c.reason))), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, data.documents.length, " meeting originals \xB7 ", data.preparation?.cachedSources || 0, " analyses reused"), data.documents.map(d => /*#__PURE__*/React.createElement("p", {
    key: d.filename
  }, d.filename))), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, data.versions.length, " saved meeting versions"), data.versions.map(v => /*#__PURE__*/React.createElement("p", {
    key: v.id
  }, "Version ", v.version, " \xB7 ", v.status, " \xB7 ", new Date(v.created_at).toLocaleString(), " \xB7 ", v.generation_model))), data.assignment.sourcePolicy && /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Frozen source preferences"), /*#__PURE__*/React.createElement("p", null, "Mode: ", data.assignment.sourcePolicy.mode), data.assignment.sourcePolicy.rules.map((r, i) => /*#__PURE__*/React.createElement("p", {
    key: i
  }, r.name, " \xB7 ", r.analyst || 'all analysts', " \xB7 ", r.ticker || r.subsector || 'all companies', " \xB7 ", r.disposition))), data.meetingId && /*#__PURE__*/React.createElement("button", {
    onClick: () => {
      window.dispatchEvent(new CustomEvent('charlie-open-meeting', {
        detail: {
          id: Number(data.meetingId)
        }
      }));
      onNavigate('meetingprep');
    }
  }, "Open questions and supporting passages \u2192"), /*#__PURE__*/React.createElement("button", {
    onClick: () => onSection('collection')
  }, "Collection controls"), /*#__PURE__*/React.createElement("button", {
    onClick: () => onNavigate('analysts')
  }, "Open analyst inbox"))));
}