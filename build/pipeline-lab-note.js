import * as React from 'react';
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
export default function LabNote({
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
    [busy, setBusy] = React.useState(false),
    [text, setText] = React.useState(''),
    [title, setTitle] = React.useState(''),
    [editing, setEditing] = React.useState(false),
    [allowPdfText, setAllowPdfText] = React.useState(false),
    [pending, setPending] = React.useState(false),
    [pendingEdit, setPendingEdit] = React.useState(false);
  var [localRecovery, setLocalRecovery] = React.useState(null),
    [bufferError, setBufferError] = React.useState('');
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
  var load = () => read(api, `/api/pipeline-lab/note-jobs/${ticker}`).then(d => setJobs(d.jobs));
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
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    setError('');
    try {
      await fn();
    } catch (e) {
      setError(e.message);
    } finally {
      lock.current = false;
      setBusy(false);
    }
  };
  var post = (path, body) => read(api, path, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body)
  });
  var open = async (id, revision) => {
    var d = await read(api, `/api/pipeline-lab/notes/${id}${revision ? '?revision=' + revision : ''}`);
    if (d.ticker !== ticker) throw Error('This note belongs to a different stock. Open its original company link.');
    setText(d.markdown);
    setTitle(d.title);
    setEditing(false);
    setDraft(d);
    if (draft?.noteId !== id || !revision) {
      var h = await read(api, `/api/pipeline-lab/notes/${id}/history?limit=50`);
      setHistory(h.versions);
      setNextBefore(h.nextBefore || null);
    }
  };
  React.useEffect(() => {
    if (!focusId) return;
    var live = true;
    setDraft(null);
    setError('');
    var inspect = async () => {
      try {
        var d = await read(api, `/api/pipeline-lab/notes/${encodeURIComponent(focusId)}`);
        if (d.ticker !== ticker) throw Error('This note belongs to a different stock. Open its original company link.');
        var h = await read(api, `/api/pipeline-lab/notes/${encodeURIComponent(focusId)}/history?limit=50`);
        if (live) {
          setText(d.markdown);
          setTitle(d.title);
          setHistory(h.versions);
          setNextBefore(h.nextBefore || null);
          setEditing(false);
          setDraft(d);
          requestAnimationFrame(() => review.current?.scrollIntoView({
            block: 'start'
          }));
        }
      } catch (e) {
        if (live) setError(e.status === 404 ? 'The linked note has no saved draft yet, or is unavailable. Check its assignment progress; no other note was substituted.' : e.message);
      }
    };
    inspect();
    return () => {
      live = false;
    };
  }, [api, ticker, focusId]);
  var older = () => action(async () => {
    var h = await read(api, `/api/pipeline-lab/notes/${draft.noteId}/history?limit=50&before=${nextBefore}`);
    setHistory(previous => [...new Map([...previous, ...h.versions].map(v => [v.revision, v])).values()].sort((a, b) => b.revision - a.revision));
    setNextBefore(h.nextBefore || null);
  });
  React.useEffect(() => {
    setPendingEdit(false);
    setLocalRecovery(null);
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
    var editKey = key + ':edit:' + draft.noteId;
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
    try {
      await post(`/api/pipeline-lab/notes/${draft.noteId}/revisions`, body);
      sessionStorage.removeItem(editKey);
      sessionStorage.removeItem(bufferKey(draft.noteId));
      setPendingEdit(false);
      await open(draft.noteId);
      await load();
    } catch (e) {
      if ([400, 409].includes(e.status)) {
        sessionStorage.removeItem(editKey);
        setPendingEdit(false);
      }
      throw e;
    }
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
  }, "Refresh saved notes")), !selection && !pending && /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "Choose Note sources, save your settings and check originals to enable generation. Previous notes remain available below.")), error && /*#__PURE__*/React.createElement("p", {
    role: "alert",
    className: "workspace-error"
  }, error), jobs.map(job => /*#__PURE__*/React.createElement("article", {
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
  }, "Reconcile saved progress")))), !jobs.length && /*#__PURE__*/React.createElement("p", null, "No recent full-note jobs in this list."), draft && /*#__PURE__*/React.createElement("div", {
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
    disabled: busy || editing || pendingEdit,
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
  }, editing ? 'Discard unsaved edits' : 'Edit draft')), /*#__PURE__*/React.createElement("details", {
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
          setBufferError('');
        } catch {
          setBufferError('Local recovery could not be cleared.');
        }
      }
    }
  }, "Discard local recovery copy")), pendingEdit && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "A previous edit has no confirmed response. The submitted text is preserved below. Reconcile that same edit before changing it; this does not generate research."), pendingEdit && !editing && /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: save
  }, "Reconcile same edit"), editing ? /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("label", null, "Title", /*#__PURE__*/React.createElement("input", {
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
    disabled: busy || editing || pendingEdit || draft.acceptance?.revision === draft.revision,
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