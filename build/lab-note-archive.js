import * as React from 'react';
import { noteArchivePage, appendNoteArchive } from './lab-note-archive-model.mjs';
export function LabNoteArchive({
  api,
  ticker,
  read,
  onOpen,
  disabled,
  describeReason
}) {
  var [open, setOpen] = React.useState(false),
    [state, setState] = React.useState({
      rows: [],
      cursor: null,
      loaded: false,
      busy: false,
      error: ''
    });
  var serial = React.useRef(0),
    locked = React.useRef(false);
  var load = async (more = false) => {
    if (locked.current) return;
    locked.current = true;
    var token = ++serial.current;
    setState(s => ({
      ...s,
      busy: true,
      error: ''
    }));
    try {
      var page = noteArchivePage(await read(api, `/api/pipeline-lab/note-jobs/${ticker}${more && state.cursor ? '?cursor=' + encodeURIComponent(state.cursor) : ''}`), ticker);
      if (token === serial.current) setState(s => ({
        rows: more ? appendNoteArchive(s.rows, page.rows) : page.rows,
        cursor: page.cursor,
        loaded: true,
        busy: false,
        error: ''
      }));
    } catch (e) {
      if (token === serial.current) setState(s => ({
        ...s,
        busy: false,
        error: 'Saved note archive could not be loaded. Your loaded entries remain available; retry this read.'
      }));
    } finally {
      if (token === serial.current) locked.current = false;
    }
  };
  React.useEffect(() => {
    if (open) load();
    return () => {
      serial.current++;
      locked.current = false;
    };
  }, [open, api, ticker]);
  return /*#__PURE__*/React.createElement("section", {
    "aria-label": "Saved full-note archive",
    style: {
      margin: '20px 0',
      overflowWrap: 'anywhere'
    }
  }, /*#__PURE__*/React.createElement("button", {
    "aria-expanded": open,
    onClick: () => setOpen(v => !v)
  }, open ? 'Close full-note archive' : 'Find older full notes'), open && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("h4", null, ticker, " \xB7 Saved full-note archive"), /*#__PURE__*/React.createElement("p", null, "Browse all saved dates. Status reflects the last archive read. Opening a revision starts no generation and does not accept research."), /*#__PURE__*/React.createElement("button", {
    disabled: state.busy,
    onClick: () => load()
  }, "Refresh archive from newest"), state.busy && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading saved notes\u2026"), state.error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, state.error), state.loaded && !state.rows.length && /*#__PURE__*/React.createElement("p", null, "No saved full-note jobs for this company."), state.rows.map(row => /*#__PURE__*/React.createElement("article", {
    key: row.noteId,
    style: {
      borderTop: '1px solid #8884',
      padding: '12px 0'
    }
  }, /*#__PURE__*/React.createElement("strong", null, row.title), /*#__PURE__*/React.createElement("p", null, new Date(row.createdAt).toLocaleString(), " \xB7 ", row.status === 'draft' ? 'Draft saved' : row.status), row.reason && /*#__PURE__*/React.createElement("p", null, describeReason?.(row.reason) || 'Inspect saved progress before retrying; an uncertain paid outcome must not be repeated.'), row.acceptedRevision && /*#__PURE__*/React.createElement("p", null, "Investor accepted revision ", row.acceptedRevision, row.revision > row.acceptedRevision ? ' · newer edits need review' : ''), /*#__PURE__*/React.createElement("div", {
    className: "workspace-actions"
  }, row.revision && /*#__PURE__*/React.createElement("button", {
    disabled: disabled,
    onClick: () => onOpen(row.noteId, row.revision)
  }, "Open saved revision ", row.revision), row.acceptedRevision && row.acceptedRevision !== row.revision && /*#__PURE__*/React.createElement("button", {
    disabled: disabled,
    onClick: () => onOpen(row.noteId, row.acceptedRevision)
  }, "Open accepted revision ", row.acceptedRevision)), !row.revision && /*#__PURE__*/React.createElement("p", null, "No saved draft revision is available for this job."))), state.cursor && /*#__PURE__*/React.createElement("button", {
    disabled: state.busy,
    onClick: () => load(true)
  }, state.error ? 'Retry older full notes' : 'Load older full notes'), state.error && !state.cursor && /*#__PURE__*/React.createElement("button", {
    disabled: state.busy,
    onClick: () => load()
  }, "Retry note archive")));
}