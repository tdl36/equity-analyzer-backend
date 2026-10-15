import { OriginalDownload } from './original-download';
import * as React from 'react';
import { labDocuments, visibleLabDocuments, groupLabDocuments, labSourceDate, taskSourceGuide } from './pipeline-lab-model.mjs';
export function ProcessingSourceLibrary({
  api,
  ticker,
  call,
  path,
  selected,
  onChange,
  purpose,
  remaining
}) {
  var [documents, setDocuments] = React.useState([]),
    [loaded, setLoaded] = React.useState(false),
    [busy, setBusy] = React.useState(false),
    [error, setError] = React.useState(''),
    [cursor, setCursor] = React.useState(null),
    [query, setQuery] = React.useState(''),
    [type, setType] = React.useState('All');
  var lock = React.useRef(false),
    live = React.useRef(true);
  React.useEffect(() => {
    live.current = true;
    return () => {
      live.current = false;
    };
  }, []);
  var load = async (after = null) => {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    setError('');
    try {
      var data = await call(path + (after ? '?after=' + encodeURIComponent(after) : ''));
      if (!Array.isArray(data.documents) || data.documents.some(r => !/^thesis_[1-9][0-9]{0,18}$/.test(r.id) || typeof r.eligible !== 'boolean') || data.nextCursor !== null && !/^[1-9][0-9]{0,18}$/.test(data.nextCursor)) throw Error('Source inventory response is incomplete. Reload the library.');
      var merged = after ? [...documents, ...data.documents] : data.documents;
      labDocuments(merged, []); // Reject ambiguous identities before showing choices.
      if (live.current) {
        setDocuments(merged);
        setCursor(data.nextCursor);
        setLoaded(true);
      }
    } catch (e) {
      if (live.current) setError(e.message);
    } finally {
      lock.current = false;
      if (live.current) setBusy(false);
    }
  };
  var rows = labDocuments(documents, []),
    shown = visibleLabDocuments(rows, query, type, 'date');
  var missing = selected.filter(id => !documents.some(d => d.id === 'thesis_' + id));
  return /*#__PURE__*/React.createElement("section", {
    "aria-label": "Supplemental source library",
    style: {
      margin: '16px 0',
      padding: 12,
      border: '1px solid var(--border-color,#d8dedd)'
    }
  }, /*#__PURE__*/React.createElement("strong", null, purpose === 'event_note' ? 'Add historical context for this event' : 'Add foundational and historical evidence'), /*#__PURE__*/React.createElement("p", null, "Stored originals for this company outside this collection. Older sources remain eligible. Metadata suggests relevance; review the original before selecting. Files are checked byte-for-byte when the proposal is saved and approved."), !loaded ? /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => load()
  }, "Browse wider source library") : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, selected.length, " supplemental originals selected \xB7 ", documents.length, " library entries loaded", cursor ? ' · more available' : '', ". No library documents are selected automatically."), /*#__PURE__*/React.createElement("div", {
    className: "sa-form-grid"
  }, /*#__PURE__*/React.createElement("label", null, "Find library sources", /*#__PURE__*/React.createElement("input", {
    "aria-label": "Find library sources",
    value: query,
    onChange: e => setQuery(e.target.value),
    placeholder: "Filename, publisher or author"
  })), /*#__PURE__*/React.createElement("label", null, "Source type", /*#__PURE__*/React.createElement("select", {
    "aria-label": "Library source type",
    value: type,
    onChange: e => setType(e.target.value)
  }, ['All', ...new Set(rows.map(r => r.type))].map(t => /*#__PURE__*/React.createElement("option", {
    key: t
  }, t))))), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "Newest recorded publication dates first; unknown dates last. Filtering covers loaded entries only. A filename-based type suggestion is labeled below.")), /*#__PURE__*/React.createElement("div", {
    style: {
      maxHeight: 360,
      overflow: 'auto'
    }
  }, groupLabDocuments(shown).map(group => /*#__PURE__*/React.createElement("section", {
    key: group.label
  }, /*#__PURE__*/React.createElement("h5", null, group.label), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, taskSourceGuide(group.label, purpose === 'event_note' ? 'note' : 'thesis'))), group.rows.map(r => {
    var id = r.id.slice(7),
      checked = selected.includes(id);
    return /*#__PURE__*/React.createElement("div", {
      key: r.id
    }, /*#__PURE__*/React.createElement("label", {
      className: "sa-check",
      style: {
        display: 'flex',
        alignItems: 'start',
        gap: 8,
        padding: '8px 0'
      }
    }, /*#__PURE__*/React.createElement("input", {
      type: "checkbox",
      checked: checked,
      disabled: !r.eligible || !checked && selected.length >= remaining,
      onChange: e => onChange(e.target.checked ? [...selected, id] : selected.filter(v => v !== id))
    }), /*#__PURE__*/React.createElement("span", null, r.filename, /*#__PURE__*/React.createElement("small", {
      style: {
        display: 'block'
      }
    }, "Published: ", labSourceDate(r.publishedDate), r.publisher ? ' · ' + r.publisher : '', " \xB7 ", r.typeRecorded ? 'Recorded type' : 'Suggested type from filename', !r.eligible ? ' · Excluded or reference-only' : ''))), r.eligible && /*#__PURE__*/React.createElement(OriginalDownload, {
      url: api + '/api/pipeline-lab/original/' + encodeURIComponent(ticker) + '/' + r.id,
      filename: r.filename,
      sha256: r.recordedSha256 || undefined
    }));
  }))), !shown.length && /*#__PURE__*/React.createElement("p", null, "No matching entries in the loaded library.")), cursor && /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => load(cursor)
  }, "Load more library sources")), busy && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading source metadata\u2026"), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error, " ", /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => load(loaded ? cursor : null)
  }, "Retry library read")), missing.length > 0 && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, missing.length, " selected identities are outside the currently loaded entries. Review them before saving. ", /*#__PURE__*/React.createElement("button", {
    onClick: () => onChange(selected.filter(id => !missing.includes(id)))
  }, "Remove unseen selections")));
}