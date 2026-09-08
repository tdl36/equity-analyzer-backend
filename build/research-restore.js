import * as React from 'react';
export function ResearchRestore({
  api,
  record,
  onClose,
  onRestored
}) {
  var [preview, setPreview] = React.useState(null),
    [error, setError] = React.useState(''),
    [busy, setBusy] = React.useState(false),
    [result, setResult] = React.useState(null),
    [uncertain, setUncertain] = React.useState(false);
  var pending = React.useRef(null),
    locked = React.useRef(false);
  var json = async (path, options = {}) => {
    var r = await fetch(`${api}${path}`, {
      ...options,
      signal: AbortSignal.timeout(30000)
    });
    var d = await r.json();
    if (!r.ok) {
      var e = Error(d.error || `Request failed (${r.status})`);
      e.status = r.status;
      throw e;
    }
    return d;
  };
  React.useEffect(() => {
    var alive = true;
    setPreview(null);
    setError('');
    setResult(null);
    pending.current = null;
    setUncertain(false);
    json(`/api/research/restoration-preview/${record.kind}/${encodeURIComponent(record.recordId)}`).then(d => {
      if (alive) setPreview(d);
    }).catch(e => {
      if (alive) setError(e.message);
    });
    return () => {
      alive = false;
    };
  }, [api, record.kind, record.recordId]);
  var restore = async () => {
    if (locked.current || !preview) return;
    locked.current = true;
    setBusy(true);
    setError('');
    var d = preview;
    var body = pending.current || {
      requestId: crypto.randomUUID(),
      kind: d.kind,
      sourceId: d.source.id,
      sourceHash: d.source.hash,
      latestId: d.latest.id,
      latestHash: d.latest.hash
    };
    pending.current = body;
    try {
      var saved = await json('/api/research/restorations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
      });
      setResult(saved);
      pending.current = null;
      setUncertain(false);
      onRestored?.();
    } catch (e) {
      setError(e.message);
      if (e.status) {
        pending.current = null;
        setUncertain(false);
      } else setUncertain(true);
    } finally {
      locked.current = false;
      setBusy(false);
    }
  };
  return /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel research-restoration",
    "aria-label": "Restore a historical research version"
  }, /*#__PURE__*/React.createElement("div", {
    className: "workspace-section-heading"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "VERSION RESTORATION"), /*#__PURE__*/React.createElement("h3", null, "Review before restoring.")), /*#__PURE__*/React.createElement("button", {
    disabled: busy || uncertain,
    onClick: onClose
  }, "Close preview")), /*#__PURE__*/React.createElement("p", null, "Restoration creates a new saved version and preserves every original. Notes return as drafts. Investment reviews retain historical assumptions and require a fresh quality review; restoration does not refresh prices, evidence or estimates."), error && /*#__PURE__*/React.createElement("p", {
    role: "alert",
    className: "workspace-error"
  }, error), !preview && !error && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading the saved versions\u2026"), preview && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "evidence-compare"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h4", null, "Selected historical version"), /*#__PURE__*/React.createElement("p", null, preview.source.createdAt, " \xB7 ", preview.source.status), /*#__PURE__*/React.createElement("pre", null, preview.source.markdown || 'No saved narrative preview.'), preview.source.previewTruncated && /*#__PURE__*/React.createElement("p", null, "Preview limited to the first 120,000 characters; restoration uses the complete saved version.")), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h4", null, "Latest saved version"), /*#__PURE__*/React.createElement("p", null, preview.latest.createdAt, " \xB7 ", preview.latest.status), /*#__PURE__*/React.createElement("pre", null, preview.latest.markdown || 'No saved narrative preview.'), preview.latest.previewTruncated && /*#__PURE__*/React.createElement("p", null, "Preview limited to the first 120,000 characters."))), result ? /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, result.kind === 'note' ? 'A restored draft is ready in Research Pipeline.' : 'A historical review version has been saved with a fresh-review requirement.', " The original versions remain available.") : /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    disabled: busy || preview.source.id === preview.latest.id,
    onClick: restore
  }, busy ? 'Restoring…' : uncertain ? 'Retry same restoration request' : preview.source.id === preview.latest.id ? 'Already the latest saved version' : preview.kind === 'note' ? 'Restore as a new note draft' : 'Restore as a historical review version')));
}