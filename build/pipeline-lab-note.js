import * as React from 'react';
var reasons = {
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
  renderHtml
}) {
  var [jobs, setJobs] = React.useState([]),
    [draft, setDraft] = React.useState(null),
    [history, setHistory] = React.useState([]),
    [error, setError] = React.useState(''),
    [busy, setBusy] = React.useState(false),
    [text, setText] = React.useState(''),
    [title, setTitle] = React.useState(''),
    [editing, setEditing] = React.useState(false),
    [allowPdfText, setAllowPdfText] = React.useState(false),
    [pending, setPending] = React.useState(false);
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
    setDraft(d);
    setText(d.markdown);
    setTitle(d.title);
    setEditing(false);
    var h = await read(api, `/api/pipeline-lab/notes/${id}/history?limit=50`);
    setHistory(h.versions);
  };
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
    try {
      await post(`/api/pipeline-lab/notes/${draft.noteId}/revisions`, body);
      sessionStorage.removeItem(editKey);
      await open(draft.noteId);
      await load();
    } catch (e) {
      if ([400, 409].includes(e.status)) sessionStorage.removeItem(editKey);
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
    disabled: busy,
    onClick: () => action(() => open(job.noteId))
  }, "Review latest revision ", job.revision), job.acceptedRevision && /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => action(() => open(job.noteId, job.acceptedRevision))
  }, "Open accepted revision"), job.status === 'attention' && /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => action(async () => {
      await post(`/api/pipeline-lab/note-job/${job.noteId}/recover`, {});
      await load();
    })
  }, "Reconcile saved progress")))), !jobs.length && /*#__PURE__*/React.createElement("p", null, "No saved full Lab notes yet."), draft && /*#__PURE__*/React.createElement("div", {
    style: {
      borderTop: '2px solid #8885',
      paddingTop: 20
    }
  }, /*#__PURE__*/React.createElement("h3", null, draft.title), /*#__PURE__*/React.createElement("p", null, "Revision ", draft.revision, " \xB7 ", jobs.find(j => j.noteId === draft.noteId)?.acceptedRevision === draft.revision ? 'Investor accepted' : 'Draft — investor review required'), /*#__PURE__*/React.createElement("div", {
    className: "workspace-actions"
  }, /*#__PURE__*/React.createElement("select", {
    "aria-label": "Note revision",
    value: draft.revision,
    disabled: busy || editing,
    onChange: e => action(() => open(draft.noteId, Number(e.target.value)))
  }, history.map(h => /*#__PURE__*/React.createElement("option", {
    key: h.revision,
    value: h.revision
  }, "Revision ", h.revision, " \xB7 ", new Date(h.createdAt).toLocaleString()))), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => download('docx')
  }, "Export Word"), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => download('md')
  }, "Export Markdown"), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => setEditing(!editing)
  }, editing ? 'Discard unsaved edits' : 'Edit draft')), /*#__PURE__*/React.createElement("details", {
    open: true
  }, /*#__PURE__*/React.createElement("summary", null, "Quality and source checks"), /*#__PURE__*/React.createElement("ul", null, draft.qualityIssues.map((q, i) => /*#__PURE__*/React.createElement("li", {
    key: i
  }, q)))), editing ? /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("label", null, "Title", /*#__PURE__*/React.createElement("input", {
    style: {
      width: '100%'
    },
    value: title,
    onChange: e => setTitle(e.target.value)
  })), /*#__PURE__*/React.createElement("label", null, "Note text (Markdown)", /*#__PURE__*/React.createElement("textarea", {
    style: {
      width: '100%',
      minHeight: 400,
      fontFamily: 'Calibri, sans-serif'
    },
    value: text,
    onChange: e => setText(e.target.value)
  })), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: save
  }, "Save new draft revision")) : /*#__PURE__*/React.createElement("div", {
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
    disabled: busy || editing || jobs.find(j => j.noteId === draft.noteId)?.acceptedRevision === draft.revision,
    onClick: () => {
      if (window.confirm(`Accept revision ${draft.revision} as your reviewed Lab research note?`)) action(async () => {
        await post(`/api/pipeline-lab/notes/${draft.noteId}/accept`, {
          revision: draft.revision
        });
        await load();
      });
    }
  }, "Accept reviewed revision")));
}