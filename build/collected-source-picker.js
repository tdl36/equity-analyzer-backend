import { OriginalDownload } from './original-download';
import * as React from 'react';
import { collectedSources, duplicateSelections } from './collected-source-model.mjs';
import { visibleLabDocuments, groupLabDocuments, labSourceDate, taskSourceGuide } from './pipeline-lab-model.mjs';
export function CollectedSourcePicker({
  api,
  call,
  job,
  selected,
  onChange,
  purpose,
  remaining,
  onReady
}) {
  var [rows, setRows] = React.useState([]),
    [error, setError] = React.useState(''),
    [busy, setBusy] = React.useState(false),
    [loaded, setLoaded] = React.useState(false),
    [query, setQuery] = React.useState(''),
    [type, setType] = React.useState('All'),
    [onlySelected, setOnlySelected] = React.useState(false);
  var live = React.useRef(false),
    lock = React.useRef(false);
  var load = async () => {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    onReady(false);
    try {
      var result = collectedSources(await call('/api/research/assignments/' + job.id + '/processing-sources'), job);
      if (live.current) {
        setRows(result);
        setLoaded(true);
        setError('');
        onReady(true);
      }
    } catch (e) {
      if (live.current) setError(e.message);
    } finally {
      lock.current = false;
      if (live.current) setBusy(false);
    }
  };
  React.useEffect(() => {
    live.current = true;
    load();
    return () => {
      live.current = false;
    };
  }, [job.id, api]);
  var duplicates = duplicateSelections(rows, selected),
    shown = visibleLabDocuments(rows.filter(r => !onlySelected || selected.includes(r.filename)), query, type, 'date');
  var hidden = selected.filter(name => !shown.some(r => r.filename === name)).length;
  return /*#__PURE__*/React.createElement("section", {
    "aria-label": "Collected evidence",
    style: {
      margin: '16px 0',
      padding: 12,
      border: '1px solid var(--border-color,#d8dedd)',
      overflowWrap: 'anywhere'
    }
  }, /*#__PURE__*/React.createElement("strong", null, "Choose collected evidence"), /*#__PURE__*/React.createElement("p", null, selected.length, " selected from ", job.result.sources.length, " collection receipts \xB7 no sources selected automatically."), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "Newest recorded publication dates first; unknown dates last. Metadata helps organize evidence and is not a content review. Downloads use your signed-in session and must match the collection receipt. Its bytes and permissions are checked against the collection receipt when saving and approving.")), /*#__PURE__*/React.createElement("div", {
    className: "sa-form-grid"
  }, /*#__PURE__*/React.createElement("label", null, "Find collected sources", /*#__PURE__*/React.createElement("input", {
    "aria-label": "Find collected sources",
    value: query,
    onChange: e => setQuery(e.target.value),
    placeholder: "Filename, publisher or author"
  })), /*#__PURE__*/React.createElement("label", null, "Collected source type", /*#__PURE__*/React.createElement("select", {
    "aria-label": "Collected source type",
    value: type,
    onChange: e => setType(e.target.value)
  }, ['All', ...new Set(rows.map(r => r.type))].map(t => /*#__PURE__*/React.createElement("option", {
    key: t
  }, t))))), /*#__PURE__*/React.createElement("label", {
    className: "sa-check"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: onlySelected,
    onChange: e => setOnlySelected(e.target.checked)
  }), "Show selected collected sources only"), hidden > 0 && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, hidden, " selections hidden by these filters. Filtering never changes your selections."), /*#__PURE__*/React.createElement("div", {
    style: {
      maxHeight: 380,
      overflow: 'auto'
    }
  }, groupLabDocuments(shown).map(group => /*#__PURE__*/React.createElement("section", {
    key: group.label
  }, /*#__PURE__*/React.createElement("h5", null, group.label), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, taskSourceGuide(group.label, purpose === 'event_note' ? 'note' : 'thesis'))), group.rows.map(r => {
    var checked = selected.includes(r.filename),
      duplicate = duplicates.has(r.filename);
    return /*#__PURE__*/React.createElement("div", {
      key: r.filename,
      style: {
        padding: '10px 0',
        borderTop: '1px solid var(--border-color,#d8dedd)'
      }
    }, /*#__PURE__*/React.createElement("label", {
      style: {
        display: 'flex',
        alignItems: 'start',
        gap: 8
      }
    }, /*#__PURE__*/React.createElement("input", {
      type: "checkbox",
      "aria-label": r.filename,
      checked: checked,
      disabled: !checked && (busy || !!error || r.availability !== 'available' || duplicate || selected.length >= remaining),
      onChange: e => onChange(e.target.checked ? [...selected, r.filename] : selected.filter(n => n !== r.filename))
    }), /*#__PURE__*/React.createElement("span", null, r.filename)), /*#__PURE__*/React.createElement("small", {
      style: {
        display: 'block',
        marginLeft: 24
      }
    }, "Published: ", labSourceDate(r.publishedDate), r.publisher ? ' · ' + r.publisher : '', r.author ? ' · ' + r.author : '', " \xB7 ", r.typeRecorded ? 'Recorded document type' : 'Type suggested from filename', r.availability !== 'available' ? ' · ' + {
      missing: 'Stored original missing',
      ambiguous: 'Stored identity ambiguous',
      restricted: 'Excluded or reference-only'
    }[r.availability] : '', duplicate ? ' · Same receipt bytes as a selected source; keep one copy' : ''), r.id && r.availability === 'available' && /*#__PURE__*/React.createElement(OriginalDownload, {
      url: api + '/api/pipeline-lab/original/' + encodeURIComponent(job.ticker) + '/' + r.id,
      filename: r.filename,
      sha256: r.receiptSha256
    }));
  }))), loaded && !shown.length && /*#__PURE__*/React.createElement("p", null, "No collected sources match these filters.")), busy && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading collected source metadata\u2026"), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error, " Your saved selections are retained."), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: load
  }, "Refresh collected source metadata"));
}