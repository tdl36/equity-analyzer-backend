function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
import * as React from 'react';
import { checkedNote, checkedNoteHistory, confirmedNoteEdit } from './pipeline-lab-note-model.mjs';
import { LabNoteProgress } from './lab-note-progress';
import { LabNoteArchive } from './lab-note-archive';
import NoteProvenance from './lab-note-provenance';
import { readEditBuffer, writeEditBuffer } from './pipeline-lab-edit-buffer.mjs';
var reasons = {
  parent_stopped: "The parent assignment was stopped. No further note batches will start. An already dispatched call may finish; saved replies and drafts are retained.",
  provider_outcome_unknown: 'Provider outcome is uncertain. Do not start a replacement: inspect the saved call before spending again.',
  preflight_failed: 'Credentials, budget, input size or token-count service need attention. No new generation was dispatched by this preflight.',
  worker_interrupted: 'The worker stopped or lost its lease. Reconcile saved progress before continuing.',
  source_check_failed: 'A frozen original is missing, changed or cannot be represented. Restore the exact original or review a new selection.',
  saved_reply_needs_review: 'The reply was saved but could not be interpreted safely. It will not be regenerated automatically.',
  intermediate_reply_truncated: 'A saved batch reached its output limit. Further generation stopped.',
  merge_context_exceeds_bound: 'The merged note exceeds its input bound. Further generation stopped.'
};
export default function LabNote(props) {
  return /*#__PURE__*/React.createElement(LabNoteWorkspace, _extends({
    key: JSON.stringify([props.api, props.ticker, props.focusId || ''])
  }, props));
}
function LabNoteWorkspace({
  api,
  ticker,
  selection,
  read,
  renderHtml,
  focusId = '',
  onUnsavedChange
}) {
  var [jobs, setJobs] = React.useState([]),
    [draft, setDraft] = React.useState(null),
    [history, setHistory] = React.useState([]),
    [nextBefore, setNextBefore] = React.useState(null),
    [error, setError] = React.useState(''),
    [actionBusy, setBusy] = React.useState(false),
    [text, setText] = React.useState(''),
    [title, setTitle] = React.useState(''),
    [editing, setEditing] = React.useState(false),
    [allowPdfText, setAllowPdfText] = React.useState(false),
    [pending, setPending] = React.useState(false),
    [pendingEdit, setPendingEdit] = React.useState(false);
  var [localRecovery, setLocalRecovery] = React.useState(null),
    [bufferError, setBufferError] = React.useState(''),
    [compareEdits, setCompareEdits] = React.useState(false);
  var [selectedJob, setSelectedJob] = React.useState(focusId),
    progressPanel = React.useRef(null);
  var [noteLoading, setNoteLoading] = React.useState(false),
    [noteTarget, setNoteTarget] = React.useState(null),
    [historyLoading, setHistoryLoading] = React.useState(false),
    [historyError, setHistoryError] = React.useState('');
  var busy = actionBusy || noteLoading,
    alive = React.useRef(true),
    noteRequest = React.useRef(0),
    historyRequest = React.useRef(0);
  React.useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      noteRequest.current++;
      historyRequest.current++;
    };
  }, []);
  var current = token => alive.current && token === noteRequest.current;
  var bufferKey = id => key + ':unsaved:' + id;
  var change = (nextTitle, nextText) => {
    setTitle(nextTitle);
    setText(nextText);
    try {
      if (nextTitle === draft.title && nextText === draft.markdown) sessionStorage.removeItem(bufferKey(draft.noteId));else writeEditBuffer(sessionStorage, bufferKey(draft.noteId), {
        revision: draft.revision,
        title: nextTitle,
        markdown: nextText
      });
      setBufferError('');
    } catch {
      setBufferError('Local recovery could not be saved. Keep this screen open and save a new draft revision before leaving.');
    }
  };
  var unsaved = !!draft && editing && (text !== draft.markdown || title !== draft.title);
  React.useEffect(() => {
    onUnsavedChange?.(unsaved);
    return () => onUnsavedChange?.(false);
  }, [unsaved, onUnsavedChange]);
  React.useEffect(() => {
    if (!unsaved) return;
    var warn = e => {
      e.preventDefault();
      e.returnValue = '';
    };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [unsaved]);
  var review = React.useRef(null);
  var lock = React.useRef(false),
    key = 'charlie-lab-full-note:' + api + ':' + ticker;
  var load = () => read(api, `/api/pipeline-lab/note-jobs/${ticker}`).then(d => {
    if (alive.current) setJobs(d.jobs);
  });
  React.useEffect(() => {
    var live = true;
    var refresh = () => read(api, `/api/pipeline-lab/note-jobs/${ticker}`).then(d => {
      if (live) setJobs(d.jobs);
    }).catch(e => {
      if (live) setError(e.message);
    });
    refresh();
    var timer = setInterval(refresh, 10000);
    try {
      setPending(!!sessionStorage.getItem(key));
    } catch {
      setError('Request storage is unavailable. Generation is disabled until browser storage works.');
    }
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [api, ticker]);
  var action = async fn => {
    if (lock.current || !alive.current) return;
    lock.current = true;
    setBusy(true);
    setError('');
    try {
      await fn();
    } catch (e) {
      if (alive.current) setError(e.message);
    } finally {
      lock.current = false;
      if (alive.current) setBusy(false);
    }
  };
  var post = (path, body) => read(api, path, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body)
  });
  var loadHistory = async (id, token, before = null) => {
    var historyToken = ++historyRequest.current;
    setHistoryLoading(true);
    setHistoryError('');
    try {
      var h = checkedNoteHistory(await read(api, `/api/pipeline-lab/notes/${encodeURIComponent(id)}/history?limit=50${before ? '&before=' + before : ''}`), id, ticker, before);
      if (current(token) && historyToken === historyRequest.current) {
        setHistory(previous => before ? [...previous, ...h.versions] : h.versions);
        setNextBefore(h.nextBefore);
      }
    } catch (e) {
      if (current(token) && historyToken === historyRequest.current) setHistoryError(e.message);
    } finally {
      if (current(token) && historyToken === historyRequest.current) setHistoryLoading(false);
    }
  };
  var displayNote = (d, token) => {
    if (!current(token)) return;
    setText(d.markdown);
    setTitle(d.title);
    setEditing(false);
    setDraft(d);
    setHistory([]);
    setNextBefore(null);
    setHistoryError('');
    // History is supplementary; it cannot hide or replace an exact saved draft.
    void loadHistory(d.noteId, token);
    requestAnimationFrame(() => {
      if (current(token)) review.current?.scrollIntoView({
        block: 'start'
      });
    });
  };
  var open = async (id, revision) => {
    if (!alive.current) return;
    var token = ++noteRequest.current;
    historyRequest.current++;
    setNoteTarget({
      id,
      revision
    });
    setNoteLoading(true);
    setError('');
    setHistoryLoading(false);
    try {
      var d = checkedNote(await read(api, `/api/pipeline-lab/notes/${encodeURIComponent(id)}${revision !== undefined ? '?revision=' + revision : ''}`), id, ticker, revision);
      displayNote(d, token);
    } catch (e) {
      if (current(token)) setError(e.status === 404 ? 'The requested note revision is unavailable. Inspect saved job progress separately; no other note was substituted.' : e.message);
    } finally {
      if (current(token)) setNoteLoading(false);
    }
  };
  React.useEffect(() => {
    if (focusId) void open(focusId);
  }, [focusId]);
  var older = () => {
    if (!historyLoading && draft) void loadHistory(draft.noteId, noteRequest.current, nextBefore);
  };
  React.useEffect(() => {
    setPendingEdit(false);
    setLocalRecovery(null);
    setCompareEdits(false);
    setBufferError('');
    if (!draft) return;
    try {
      var raw = sessionStorage.getItem(key + ':edit:' + draft.noteId);
      if (!raw) {
        try {
          var buffer = readEditBuffer(sessionStorage, bufferKey(draft.noteId));
          if (buffer) {
            if (buffer.revision === draft.revision) {
              setTitle(buffer.title);
              setText(buffer.markdown);
              setEditing(true);
            } else setLocalRecovery(buffer);
          }
        } catch (e) {
          setLocalRecovery({
            unreadable: true
          });
          setBufferError(e.message);
        }
        return;
      }
      var body = JSON.parse(raw);
      if (!body || typeof body.eventId !== 'string' || !Number.isInteger(body.expectedRevision) || typeof body.title !== 'string' || typeof body.markdown !== 'string') throw Error('Pending edit is unreadable. Preserve this browser session for recovery.');
      setPendingEdit(true);
      setTitle(body.title);
      setText(body.markdown);
      setEditing(true);
    } catch (e) {
      setPendingEdit(true);
      setError(e.message);
    }
  }, [key, draft]);
  var start = () => action(async () => {
    var body = JSON.parse(sessionStorage.getItem(key) || 'null');
    if (!body) {
      if (!selection) throw Error('Save and check note sources first.');
      body = {
        requestId: crypto.randomUUID(),
        revision: selection.revision,
        digest: selection.digest,
        allowPdfText
      };
      sessionStorage.setItem(key, JSON.stringify(body));
    }
    if (sessionStorage.getItem(key) !== JSON.stringify(body)) throw Error('Could not preserve the submission identity. Nothing was sent.');
    setPending(true);
    try {
      await post(`/api/pipeline-lab/note-jobs/${ticker}`, body);
      sessionStorage.removeItem(key);
      setPending(false);
      await load();
    } catch (e) {
      if ([400, 409].includes(e.status)) {
        sessionStorage.removeItem(key);
        setPending(false);
      }
      throw e;
    }
  });
  var save = () => action(async () => {
    var id = draft.noteId,
      token = noteRequest.current,
      editKey = key + ':edit:' + id;
    var body = JSON.parse(sessionStorage.getItem(editKey) || 'null');
    if (!body) {
      body = {
        eventId: crypto.randomUUID(),
        expectedRevision: draft.revision,
        title,
        markdown: text
      };
      sessionStorage.setItem(editKey, JSON.stringify(body));
    }
    if (sessionStorage.getItem(editKey) !== JSON.stringify(body)) throw Error('Could not preserve the edit identity. Nothing was sent.');
    setPendingEdit(true);
    var saved;
    try {
      saved = await confirmedNoteEdit({
        post,
        read: path => read(api, path),
        id,
        ticker,
        body
      });
    } catch (e) {
      if (current(token) && e.editRejected) {
        // Preserve the original baseline, especially when recovery opened a newer note.
        var recovery = {
          revision: body.expectedRevision,
          title: body.title,
          markdown: body.markdown
        };
        writeEditBuffer(sessionStorage, bufferKey(id), recovery);
        sessionStorage.removeItem(editKey);
        setPendingEdit(false);
        setEditing(false);
        setLocalRecovery(recovery);
        setCompareEdits(false);
      }
      throw e;
    }
    if (!current(token)) return;
    // A verified immutable version now holds this exact text centrally. A failed
    // read, mismatched receipt or navigation never clears the frozen submission.
    sessionStorage.removeItem(bufferKey(id));
    sessionStorage.removeItem(editKey);
    setPendingEdit(false);
    setNoteTarget({
      id,
      revision: saved.revision
    });
    displayNote(saved, token);
    await load();
  });
  var download = format => action(async () => {
    var r = await fetch(api + `/api/pipeline-lab/notes/${draft.noteId}/export?revision=${draft.revision}&format=${format}`);
    if (!r.ok) {
      var d = await r.json();
      throw Error(d.error || 'Export unavailable.');
    }
    var url = URL.createObjectURL(await r.blob());
    var a = document.createElement('a');
    a.href = url;
    a.download = `${ticker}-Lab-note-v${draft.revision}.${format}`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  });
  var prose = value => renderHtml ? /*#__PURE__*/React.createElement("div", {
    dangerouslySetInnerHTML: {
      __html: renderHtml(value)
    }
  }) : /*#__PURE__*/React.createElement("div", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, value);
  return /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel",
    style: {
      marginTop: 24
    },
    "aria-label": "Full Lab research notes"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "FULL RESEARCH NOTE \xB7 VERSIONED DRAFT"), /*#__PURE__*/React.createElement("h3", null, "Write, review and retain your research"), /*#__PURE__*/React.createElement("p", null, "Uses your checked note selection and a frozen copy of the existing thesis. The approved research model may make up to eight batch calls, capped at 32,000 output tokens each. Central credentials and the monthly budget apply. Generation spends credits; acceptance is a separate investor action."), /*#__PURE__*/React.createElement("label", null, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: allowPdfText,
    disabled: busy || pending,
    onChange: e => setAllowPdfText(e.target.checked)
  }), " Allow text extraction for PDFs too large to attach (images and layout may be lost)"), /*#__PURE__*/React.createElement("div", {
    className: "workspace-actions",
    style: {
      marginTop: 12
    }
  }, /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    disabled: busy || !selection && !pending,
    onClick: start
  }, pending ? 'Reconcile same submission' : 'Generate full note draft'), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => action(load)
  }, "Refresh saved notes")), !selection && !pending && /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "Choose Note sources, save your settings and check originals to enable generation. Previous notes remain available below.")), noteLoading && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Opening the exact saved note\u2026"), error && /*#__PURE__*/React.createElement("p", {
    role: "alert",
    className: "workspace-error"
  }, error, noteTarget && /*#__PURE__*/React.createElement("button", {
    disabled: busy || pendingEdit,
    onClick: () => {
      if (!unsaved || window.confirm('Leave these edits in browser-session recovery and retry opening the saved note?')) action(() => open(noteTarget.id, noteTarget.revision));
    }
  }, "Retry saved note")), selectedJob && /*#__PURE__*/React.createElement("div", {
    ref: progressPanel
  }, /*#__PURE__*/React.createElement(LabNoteProgress, {
    api: api,
    ticker: ticker,
    id: selectedJob,
    read: read,
    disabled: busy || pendingEdit,
    describeReason: reason => reasons[reason],
    onOpen: (id, revision) => {
      if (!unsaved || window.confirm('Leave these edits in browser-session recovery and open this saved revision?')) action(() => open(id, revision));
    },
    onRecover: id => action(async () => {
      await post(`/api/pipeline-lab/note-job/${id}/recover`, {});
      await load();
    })
  })), jobs.map(job => /*#__PURE__*/React.createElement("article", {
    key: job.noteId,
    style: {
      borderTop: '1px solid #8884',
      padding: '16px 0',
      overflowWrap: 'anywhere'
    }
  }, /*#__PURE__*/React.createElement("strong", null, new Date(job.createdAt).toLocaleString(), " \xB7 ", job.status === 'draft' ? 'Draft saved' : job.status), job.acceptedRevision && /*#__PURE__*/React.createElement("p", null, "Investor accepted revision ", job.acceptedRevision, job.revision > job.acceptedRevision ? ' · newer edits need review' : ''), job.reason && /*#__PURE__*/React.createElement("p", null, reasons[job.reason] || job.reason), /*#__PURE__*/React.createElement("div", {
    className: "workspace-actions"
  }, job.revision && /*#__PURE__*/React.createElement("button", {
    disabled: busy || pendingEdit,
    onClick: () => {
      if (!unsaved || window.confirm('Leave these edits in browser-session recovery and open this saved revision?')) action(() => open(job.noteId));
    }
  }, "Review latest revision ", job.revision), job.acceptedRevision && /*#__PURE__*/React.createElement("button", {
    disabled: busy || pendingEdit,
    onClick: () => {
      if (!unsaved || window.confirm('Leave these edits in browser-session recovery and open this saved revision?')) action(() => open(job.noteId, job.acceptedRevision));
    }
  }, "Open accepted revision"), job.status === 'attention' && job.reason !== 'parent_stopped' && /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => action(async () => {
      await post(`/api/pipeline-lab/note-job/${job.noteId}/recover`, {});
      await load();
    })
  }, "Reconcile saved progress")))), !jobs.length && /*#__PURE__*/React.createElement("p", null, "No recent full-note jobs in this list."), /*#__PURE__*/React.createElement(LabNoteArchive, {
    onInspect: id => {
      setSelectedJob(id);
      requestAnimationFrame(() => progressPanel.current?.scrollIntoView({
        block: 'start'
      }));
    },
    api: api,
    ticker: ticker,
    read: read,
    describeReason: reason => reasons[reason],
    disabled: busy || pendingEdit,
    onOpen: (id, revision) => {
      if (!unsaved || window.confirm('Leave these edits in browser-session recovery and open this saved revision?')) action(() => open(id, revision));
    }
  }), draft && /*#__PURE__*/React.createElement("div", {
    ref: review,
    style: {
      borderTop: '2px solid #8885',
      paddingTop: 20
    }
  }, /*#__PURE__*/React.createElement("h3", null, draft.title), /*#__PURE__*/React.createElement("p", null, "Revision ", draft.revision, " \xB7 ", draft.acceptance === undefined ? 'Acceptance status unavailable' : draft.acceptance?.revision === draft.revision ? 'Investor accepted · ' + new Date(draft.acceptance.acceptedAt).toLocaleString() : 'Draft — investor review required'), /*#__PURE__*/React.createElement("div", {
    className: "workspace-actions"
  }, /*#__PURE__*/React.createElement("select", {
    "aria-label": "Note revision",
    value: draft.revision,
    disabled: busy || editing || pendingEdit,
    onChange: e => action(() => open(draft.noteId, Number(e.target.value)))
  }, !history.some(h => h.revision === draft.revision) && /*#__PURE__*/React.createElement("option", {
    value: draft.revision
  }, "Revision ", draft.revision, " \xB7 current view"), history.map(h => /*#__PURE__*/React.createElement("option", {
    key: h.revision,
    value: h.revision
  }, "Revision ", h.revision, " \xB7 ", new Date(h.createdAt).toLocaleString()))), nextBefore && /*#__PURE__*/React.createElement("button", {
    disabled: busy || historyLoading || editing || pendingEdit,
    onClick: older
  }, "Load older revisions"), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => download('docx')
  }, "Export Word"), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => download('md')
  }, "Export Markdown"), /*#__PURE__*/React.createElement("button", {
    disabled: busy || pendingEdit || !!localRecovery,
    onClick: () => {
      if (unsaved && !window.confirm('Discard unsaved note edits?')) return;
      try {
        sessionStorage.removeItem(bufferKey(draft.noteId));
        setBufferError('');
      } catch {
        setBufferError('Local recovery could not be cleared. Keep this screen open.');
        return;
      }
      setText(draft.markdown);
      setTitle(draft.title);
      setEditing(!editing);
    }
  }, editing ? 'Discard unsaved edits' : 'Edit draft')), historyLoading && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading revision history; the exact saved note remains available."), historyError && /*#__PURE__*/React.createElement("div", {
    role: "alert"
  }, /*#__PURE__*/React.createElement("p", null, "Revision history unavailable: ", historyError), /*#__PURE__*/React.createElement("button", {
    disabled: busy || historyLoading,
    onClick: () => loadHistory(draft.noteId, noteRequest.current, nextBefore)
  }, "Retry revision history")), /*#__PURE__*/React.createElement("details", {
    open: true
  }, /*#__PURE__*/React.createElement("summary", null, "Quality and source checks"), /*#__PURE__*/React.createElement("ul", null, draft.qualityIssues.map((q, i) => /*#__PURE__*/React.createElement("li", {
    key: i
  }, q)))), unsaved && !pendingEdit && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Unsaved edits are kept in this browser tab\u2019s session when storage is available. They are not saved in Charlie or synced to other devices. Save a new draft revision to retain them centrally."), bufferError && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, bufferError), localRecovery && /*#__PURE__*/React.createElement("div", {
    role: "status"
  }, /*#__PURE__*/React.createElement("p", null, localRecovery.unreadable ? 'A local edit recovery copy is unreadable.' : 'Local edits belong to revision ' + localRecovery.revision + '. They have not been applied to this saved revision. Open their original revision to review or copy them; saving still checks the server baseline.'), !localRecovery.unreadable && /*#__PURE__*/React.createElement("button", {
    disabled: busy || pendingEdit,
    onClick: () => action(() => open(draft.noteId, localRecovery.revision))
  }, "Open original revision with local edits"), /*#__PURE__*/React.createElement("button", {
    disabled: busy || pendingEdit,
    onClick: () => {
      if (window.confirm('Permanently discard this browser-session recovery copy?')) {
        try {
          sessionStorage.removeItem(bufferKey(draft.noteId));
          setLocalRecovery(null);
          setCompareEdits(false);
          setBufferError('');
        } catch {
          setBufferError('Local recovery could not be cleared.');
        }
      }
    }
  }, "Discard local recovery copy")), (unsaved || localRecovery && !localRecovery.unreadable) && !pendingEdit && /*#__PURE__*/React.createElement("div", {
    style: {
      margin: '16px 0'
    }
  }, /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    "aria-expanded": compareEdits,
    onClick: () => setCompareEdits(v => !v)
  }, compareEdits ? 'Hide edit comparison' : 'Compare local edits with saved revision'), compareEdits && /*#__PURE__*/React.createElement("section", {
    "aria-label": "Note edit comparison"
  }, /*#__PURE__*/React.createElement("h4", null, "Review before changing the saved note"), /*#__PURE__*/React.createElement("p", null, "Showing the saved revision currently open and your local copy. This does not merge, save, accept or change the edit baseline. Other saved revisions may exist."), /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 320px), 1fr))',
      gap: 16
    }
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h4", null, "Saved revision ", draft.revision), /*#__PURE__*/React.createElement("label", null, "Saved title", /*#__PURE__*/React.createElement("input", {
    "aria-label": "Comparison saved title",
    readOnly: true,
    value: draft.title,
    style: {
      width: '100%'
    }
  })), /*#__PURE__*/React.createElement("label", null, "Saved text", /*#__PURE__*/React.createElement("textarea", {
    "aria-label": "Comparison saved text",
    readOnly: true,
    value: draft.markdown,
    style: {
      width: '100%',
      minHeight: 280,
      fontFamily: 'Calibri, sans-serif'
    }
  }))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h4", null, "Local edits based on revision ", localRecovery?.revision || draft.revision), /*#__PURE__*/React.createElement("label", null, "Local title", /*#__PURE__*/React.createElement("input", {
    "aria-label": "Comparison local title",
    readOnly: true,
    value: localRecovery?.title ?? title,
    style: {
      width: '100%'
    }
  })), /*#__PURE__*/React.createElement("label", null, "Local text", /*#__PURE__*/React.createElement("textarea", {
    "aria-label": "Comparison local text",
    readOnly: true,
    value: localRecovery?.markdown ?? text,
    style: {
      width: '100%',
      minHeight: 280,
      fontFamily: 'Calibri, sans-serif'
    }
  })))), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "Copy any text you need before explicitly discarding a recovery copy. Saving an older baseline still requires resolving the server revision conflict; this comparison never overwrites newer work.")))), pendingEdit && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "A previous edit has no confirmed response. The submitted text is preserved below. Reconcile that same edit before changing it; this does not generate research."), pendingEdit && !editing && /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: save
  }, "Reconcile same edit"), /*#__PURE__*/React.createElement(NoteProvenance, {
    value: draft.researchProvenance
  }), editing ? /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("label", null, "Title", /*#__PURE__*/React.createElement("input", {
    "aria-label": "Note title",
    readOnly: pendingEdit || busy,
    style: {
      width: '100%'
    },
    value: title,
    onChange: e => change(e.target.value, text)
  })), /*#__PURE__*/React.createElement("label", null, "Note text (Markdown)", /*#__PURE__*/React.createElement("textarea", {
    "aria-label": "Note text",
    readOnly: pendingEdit || busy,
    style: {
      width: '100%',
      minHeight: 400,
      fontFamily: 'Calibri, sans-serif'
    },
    value: text,
    onChange: e => change(title, e.target.value)
  })), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: save
  }, pendingEdit ? 'Reconcile same edit' : 'Save new draft revision')) : /*#__PURE__*/React.createElement("div", {
    className: "lab-note-prose",
    style: {
      background: '#fff',
      color: '#000',
      fontFamily: 'Calibri, sans-serif',
      padding: 24,
      overflowWrap: 'anywhere'
    }
  }, prose(draft.markdown)), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Source appendix and frozen originals"), prose(draft.generationEvidence?.sourcesMarkdown || 'No appendix supplied.'), /*#__PURE__*/React.createElement("ul", null, draft.sourcePack.map(s => /*#__PURE__*/React.createElement("li", {
    key: s.reference,
    style: {
      overflowWrap: 'anywhere'
    }
  }, s.filename, " \xB7 ", s.reference, " \xB7 ", s.bytes, " bytes", /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("small", null, "SHA-256 ", s.sha256))))), (draft.generationEvidence?.chartSpecs || []).map((c, i) => /*#__PURE__*/React.createElement("details", {
    key: i
  }, /*#__PURE__*/React.createElement("summary", null, c.period, " \xB7 ", c.kind, " segment figures"), /*#__PURE__*/React.createElement("p", null, "Model-produced values; verify units and source figures before use."), /*#__PURE__*/React.createElement("table", null, /*#__PURE__*/React.createElement("thead", null, /*#__PURE__*/React.createElement("tr", null, /*#__PURE__*/React.createElement("th", null, "Segment"), /*#__PURE__*/React.createElement("th", null, "Value"))), /*#__PURE__*/React.createElement("tbody", null, c.data.map(r => /*#__PURE__*/React.createElement("tr", {
    key: r.segment
  }, /*#__PURE__*/React.createElement("td", null, r.segment), /*#__PURE__*/React.createElement("td", null, r.value.toLocaleString()))))))), /*#__PURE__*/React.createElement("p", null, "Acceptance records this exact Lab revision. It does not change the current thesis or original Pipeline notes. Source caveats remain in the history and exports."), /*#__PURE__*/React.createElement("button", {
    disabled: busy || editing || pendingEdit || draft.acceptance === undefined || draft.acceptance?.revision === draft.revision,
    onClick: () => {
      if (window.confirm(`Accept revision ${draft.revision} as your reviewed Lab research note?`)) action(async () => {
        await post(`/api/pipeline-lab/notes/${draft.noteId}/accept`, {
          revision: draft.revision
        });
        await open(draft.noteId, draft.revision);
        await load();
      });
    }
  }, "Accept reviewed revision")));
}