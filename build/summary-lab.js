import React, { useEffect, useState } from 'react';
import { labDocument, emailDocument } from './summary-lab-format.mjs';
var SECTIONS = [['brief', 'Brief', 'brief'], ['takeaways', 'Key Takeaways', 'summary'], ['meeting', 'Meeting Summary', 'meeting_summary'], ['questions', 'Follow-up Questions', 'questions'], ['assessment', 'Overall Assessment', 'assessment']];
export function SummaryLab({
  api,
  getKey,
  renderHtml,
  pickFromICloud
}) {
  var [sharing, setSharing] = useState(false),
    [edits, setEdits] = useState({}),
    [sending, setSending] = useState(false);
  var [sources, setSources] = useState([]),
    [runs, setRuns] = useState([]),
    [id, setId] = useState(''),
    [row, setRow] = useState(null),
    [sid, setSid] = useState(''),
    [source, setSource] = useState(''),
    [title, setTitle] = useState(''),
    [focus, setFocus] = useState(''),
    [section, setSection] = useState('brief'),
    [compare, setCompare] = useState(false),
    [error, setError] = useState(''),
    [busy, setBusy] = useState(false),
    [feedback, setFeedback] = useState(''),
    [notice, setNotice] = useState(''),
    [importing, setImporting] = useState(false),
    [importStatus, setImportStatus] = useState('');
  async function req(path = '', body) {
    var r = await fetch(`${api}/api/summary-lab${path}`, {
      ...(body ? {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
      } : {}),
      signal: AbortSignal.timeout(30000)
    });
    var d = await r.json();
    if (!r.ok) throw Error(d.error || 'Request failed');
    return d;
  }
  useEffect(() => {
    var active = true;
    async function load() {
      try {
        var [a, b] = await Promise.all([req('/sources'), req()]);
        if (active) {
          setSources(a.sources);
          setRuns(b.experiments);
        }
      } catch (e) {
        if (active) setError(e.message);
      }
    }
    load();
    var t = setInterval(load, 12000);
    return () => {
      active = false;
      clearInterval(t);
    };
  }, [api]);
  useEffect(() => {
    if (!id) {
      setRow(null);
      return;
    }
    var active = true;
    setRow(null);
    async function load() {
      try {
        var d = await req('/' + id);
        if (active) setRow(d);
      } catch (e) {
        if (active) setError(e.message);
      }
    }
    load();
    var t = setInterval(load, 6000);
    return () => {
      active = false;
      clearInterval(t);
    };
  }, [id, api]);
  useEffect(() => {
    setFeedback(row?.feedback || '');
    setSharing(false);
    setEdits({});
  }, [row?.id]);
  async function start() {
    setBusy(true);
    setError('');
    setNotice('');
    try {
      var d = await req('', {
        summaryId: sid || undefined,
        source,
        title: title.trim() || 'Untitled experiment',
        focus,
        apiKey: getKey()
      });
      setId(d.id);
      setRuns((await req()).experiments);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  async function importCloud() {
    setImporting(true);
    setError('');
    setImportStatus('Choose documents from iCloud…');
    try {
      var files = await pickFromICloud({
        mode: 'bytes',
        title: 'Summary Lab · choose iCloud documents'
      });
      if (!files?.length) {
        setImportStatus('');
        return;
      }
      var records = [];
      for (var i = 0; i < files.length; i++) {
        var file = files[i];
        setImportStatus(`Reading ${i + 1} of ${files.length}: ${file.name}`);
        if (!/\.(pdf|docx|txt|md|csv|png|jpe?g|webp)$/i.test(file.name)) throw Error(`${file.name}: choose PDF, DOCX, text or image documents. For audio, select its saved Summary transcript.`);
        var form = new FormData();
        form.append('files', file);
        form.append('apiKey', getKey() || '');
        var response = await fetch(`${api}/api/extract-summary-text`, {
          method: 'POST',
          body: form,
          signal: AbortSignal.timeout(120000)
        });
        var data = await response.json();
        if (!response.ok || !data.text?.trim() || /^\[(Error processing|Unsupported file type|Image file:)/.test(data.text.trim())) throw Error(`${file.name}: ${data.error || 'No readable text was extracted. Try a text-searchable PDF or paste the text.'}`);
        records.push(`=== SOURCE: ${file.name} ===\n${data.text}`);
      }
      setSid('');
      setSource(records.join('\n\n'));
      if (!title.trim()) setTitle(files[0].name.replace(/\.[^.]+$/, '').slice(0, 300));
      setImportStatus(`Imported ${files.length} document${files.length === 1 ? '' : 's'}. Review the source text below, then generate.`);
    } catch (e) {
      setError(`${e.message} No imported documents were applied; your previous source is preserved.`);
      setImportStatus('Import did not finish.');
    } finally {
      setImporting(false);
    }
  }
  async function retry() {
    setBusy(true);
    setError('');
    try {
      await req('/' + id + '/retry', {
        apiKey: getKey()
      });
      setRow(await req('/' + id));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  var state = row?.state || {},
    text = (sharing ? edits[section] : state.sections?.[section]) || '',
    baseline = SECTIONS.find(s => s[0] === section)?.[2];
  function openEmail() {
    setEdits({
      ...state.sections
    });
    setSharing(true);
    setCompare(false);
    setNotice('Review and edit each section before sending. Edits affect this email only and are discarded when you close the preview.');
  }
  async function sendEmail() {
    setSending(true);
    setError('');
    try {
      var creds = JSON.parse(localStorage.getItem('emailCredentials') || '{}');
      if (!creds.email) throw Error('Set your recipient email and Gmail credentials in Settings first.');
      var response = await fetch(`${api}/api/email-summary-section`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          email: creds.email,
          subject: `Summary Lab: ${row.title}`,
          title: row.title,
          section: 'summary_lab',
          content: emailDocument(row.title, SECTIONS.map(([k, l]) => [l, edits[k] || ''])),
          smtpConfig: {
            use_gmail: creds.useGmail,
            gmail_user: creds.gmailUser,
            gmail_app_password: creds.gmailPassword,
            from_email: creds.gmailUser
          }
        })
      });
      var data = await response.json().catch(() => ({}));
      if (!response.ok) throw Error(data.error || 'Email delivery could not be confirmed. Check your inbox before retrying.');
      setNotice(`All five sections emailed to ${creds.email}.`);
    } catch (e) {
      setError(e.message);
    } finally {
      setSending(false);
    }
  }
  async function copy(all = false) {
    try {
      var items = all ? SECTIONS.map(([k, l]) => [l, (sharing ? edits[k] : state.sections?.[k]) || '']) : [[SECTIONS.find(s => s[0] === section)?.[1] || '', text]];
      var html = emailDocument(row.title, items);
      var doc = new DOMParser().parseFromString(html, 'text/html');
      doc.querySelectorAll('p,h1,h2,h3,li,blockquote').forEach(el => el.append('\n'));
      var plain = doc.body.textContent;
      if (window.ClipboardItem && navigator.clipboard.write) await navigator.clipboard.write([new ClipboardItem({
        'text/html': new Blob([html], {
          type: 'text/html'
        }),
        'text/plain': new Blob([plain], {
          type: 'text/plain'
        })
      })]);else await navigator.clipboard.writeText(plain);
      setNotice('Formatted notes copied.');
    } catch {
      setError('Clipboard unavailable. Select the note text to copy.');
    }
  }
  function download() {
    var url = URL.createObjectURL(new Blob([JSON.stringify(row, null, 2)], {
      type: 'application/json'
    }));
    var a = document.createElement('a');
    a.href = url;
    a.download = `Summary-Lab-${id}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return /*#__PURE__*/React.createElement("main", {
    className: "summary-lab"
  }, /*#__PURE__*/React.createElement("style", null, `.summary-lab{box-sizing:border-box;--lab-border:rgba(153,142,119,.35);max-width:1500px;margin:0 auto;height:100%;min-height:0;overflow-y:auto;overflow-x:hidden;overscroll-behavior-y:contain;padding:32px;color:var(--text-primary,#e9e2d3);width:100%;min-width:0}.summary-lab *{box-sizing:border-box}.summary-lab h1,.summary-lab h2{font-family:Georgia,serif;line-height:1.2}.summary-lab h1{font-size:38px;margin:8px 0 14px}.summary-lab h2{font-size:24px;margin:0 0 18px}.summary-lab p{line-height:1.65}.summary-lab .muted{opacity:.72;font-size:13px}.summary-lab .eyebrow{color:#c9a857;letter-spacing:.13em;text-transform:uppercase;font-size:11px}.summary-lab .layout{display:grid;grid-template-columns:300px minmax(0,1fr);gap:24px;margin-top:28px}.summary-lab .panel{border:1px solid var(--lab-border);border-radius:14px;padding:24px;background:rgba(127,115,89,.045);min-width:0}.summary-lab label{display:block;font-size:13px;margin:16px 0 6px}.summary-lab input,.summary-lab select,.summary-lab textarea{width:100%;padding:11px;border:1px solid var(--lab-border);border-radius:7px;background:var(--bg-secondary,#211e18);color:inherit;font:inherit;min-width:0}.summary-lab select option{background:#211e18;color:#eee}.summary-lab button{padding:10px 14px;min-height:44px;border:1px solid var(--lab-border);border-radius:7px;font:inherit;cursor:pointer;background:transparent;color:inherit}.summary-lab button:focus-visible,.summary-lab input:focus-visible,.summary-lab textarea:focus-visible,.summary-lab select:focus-visible{outline:2px solid #c9a857;outline-offset:3px}.summary-lab button:disabled{opacity:.45;cursor:default}.summary-lab button.primary,.summary-lab button[aria-pressed=true]{background:#c9a857;color:#18150f}.summary-lab .controls{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}.summary-lab .experiment{display:block;width:100%;text-align:left;margin:10px 0;overflow-wrap:anywhere}.summary-lab .reader{font-family:Calibri,Carlito,Arial,sans-serif;font-size:11pt;line-height:1.55;overflow-wrap:anywhere;max-width:90ch;background:#fff;color:#242424;padding:28px;border:1px solid #dedbd4;border-radius:4px}.summary-lab .reader h1,.summary-lab .reader h2,.summary-lab .reader h3,.summary-lab .reader h4{font:700 11pt/1.5 Calibri,Carlito,Arial,sans-serif;margin:20px 0 8px}.summary-lab .reader p{margin:0 0 12px;line-height:1.55}.summary-lab .reader ul,.summary-lab .reader ol{padding-left:23px;margin:10px 0 16px;list-style-position:outside}.summary-lab .reader ul{list-style-type:disc}.summary-lab .reader ol{list-style-type:decimal}.summary-lab .reader li{margin:6px 0}.summary-lab .reader blockquote{border-left:3px solid #b9af94;padding-left:14px;margin:14px 0}.summary-lab .reader hr{border:0;border-top:1px solid #ddd;margin:20px 0}.summary-lab .email-editor{font:11pt/1.5 Calibri,Carlito,Arial,sans-serif;min-height:240px}.summary-lab .reader strong{font-weight:700}.summary-lab .pair{display:grid;gap:24px;grid-template-columns:repeat(2,minmax(0,1fr))}.summary-lab .status{padding:14px;border-left:3px solid #c9a857;background:rgba(201,168,87,.08);margin:16px 0;overflow-wrap:anywhere}.summary-lab details{border-top:1px solid var(--lab-border);padding:16px 0;margin-top:16px}.summary-lab summary{cursor:pointer;min-height:32px}.summary-lab pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;line-height:1.7}.summary-lab .original{overflow-wrap:anywhere;line-height:1.8}.summary-lab .original table{display:block;overflow:auto;max-width:100%}@media(max-width:900px){.summary-lab{padding:18px 18px 112px}.summary-lab .layout,.summary-lab .pair{grid-template-columns:1fr}.summary-lab h1{font-size:30px}.summary-lab .panel{padding:18px}}`), /*#__PURE__*/React.createElement("div", {
    className: "eyebrow"
  }, "Charlie / Research experiments"), /*#__PURE__*/React.createElement("h1", null, "Summary Lab"), /*#__PURE__*/React.createElement("p", null, "Read thoroughly. Preserve what was said. Separate what it means."), /*#__PURE__*/React.createElement("p", {
    className: "muted"
  }, "An independent trial with five familiar sections. Original summaries and automatic workflows are unchanged."), error && /*#__PURE__*/React.createElement("div", {
    role: "alert",
    className: "status"
  }, error, /*#__PURE__*/React.createElement("button", {
    onClick: () => setError(''),
    "aria-label": "Dismiss error"
  }, "Dismiss")), /*#__PURE__*/React.createElement("div", {
    className: "layout"
  }, /*#__PURE__*/React.createElement("aside", null, /*#__PURE__*/React.createElement("section", {
    className: "panel"
  }, /*#__PURE__*/React.createElement("h2", null, "New experiment"), /*#__PURE__*/React.createElement("button", {
    disabled: busy || importing,
    onClick: importCloud
  }, importing ? 'Importing…' : 'Browse iCloud documents'), /*#__PURE__*/React.createElement("p", {
    className: "muted"
  }, "STOCKS and CATALYSTS \xB7 the same connected folders as Summary"), importStatus && /*#__PURE__*/React.createElement("p", {
    role: "status",
    className: "muted"
  }, importStatus), /*#__PURE__*/React.createElement("label", {
    htmlFor: "lab-source"
  }, "Saved transcript or document"), /*#__PURE__*/React.createElement("select", {
    id: "lab-source",
    value: sid,
    onChange: e => setSid(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Paste source text instead"), sources.map(s => /*#__PURE__*/React.createElement("option", {
    key: s.id,
    value: s.id
  }, s.title))), !sid && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("label", {
    htmlFor: "lab-text"
  }, "Complete source text"), /*#__PURE__*/React.createElement("textarea", {
    id: "lab-text",
    rows: 8,
    value: source,
    onChange: e => setSource(e.target.value),
    placeholder: "Paste a transcript or extracted document text\u2026"
  }), /*#__PURE__*/React.createElement("p", {
    className: "muted"
  }, source.length.toLocaleString(), " characters \xB7 no silent character cutoff")), /*#__PURE__*/React.createElement("label", {
    htmlFor: "lab-title"
  }, "Experiment name"), /*#__PURE__*/React.createElement("input", {
    id: "lab-title",
    value: title,
    maxLength: 300,
    onChange: e => setTitle(e.target.value),
    placeholder: "MMM conference \xB7 first trial"
  }), /*#__PURE__*/React.createElement("label", {
    htmlFor: "lab-focus"
  }, "Optional emphasis"), /*#__PURE__*/React.createElement("textarea", {
    id: "lab-focus",
    rows: 3,
    maxLength: 4000,
    value: focus,
    onChange: e => setFocus(e.target.value),
    placeholder: "Preserve the segment detail and management\u2019s margin explanation."
  }), /*#__PURE__*/React.createElement("p", {
    className: "muted"
  }, "Uses model credits. Thorough review makes several passes over the source; long transcripts take longer. Experiments remain saved when you leave."), /*#__PURE__*/React.createElement("button", {
    className: "primary",
    disabled: busy || importing || !sid && !source.trim(),
    onClick: start
  }, busy ? 'Starting…' : 'Generate all five sections')), /*#__PURE__*/React.createElement("section", {
    className: "panel",
    style: {
      marginTop: 20
    }
  }, /*#__PURE__*/React.createElement("h2", null, "Experiments"), !runs.length && /*#__PURE__*/React.createElement("p", {
    className: "muted"
  }, "Your experiments will appear here."), runs.map(r => /*#__PURE__*/React.createElement("button", {
    className: "experiment",
    "aria-pressed": id === r.id,
    key: r.id,
    onClick: () => {
      setId(r.id);
      setNotice('');
    }
  }, r.title, /*#__PURE__*/React.createElement("div", {
    className: "muted"
  }, r.status === 'complete' ? 'Ready to compare' : r.status, " \xB7 ", new Date(r.created_at).toLocaleDateString()))))), /*#__PURE__*/React.createElement("section", {
    className: "panel"
  }, !row ? /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "eyebrow"
  }, "Independent source review"), /*#__PURE__*/React.createElement("h2", {
    style: {
      marginTop: 12
    }
  }, id ? 'Loading experiment…' : 'Your next research note starts here'), /*#__PURE__*/React.createElement("p", null, "Choose an existing Summary to compare against a frozen copy of its original output, or paste a new source. One action generates Brief, Key Takeaways, Meeting Summary, Follow-up Questions and Assessment."), /*#__PURE__*/React.createElement("p", {
    className: "muted"
  }, "This version reviews saved text. It cannot re-listen to audio or verify OCR against page images; material ambiguities remain visible.")) : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "eyebrow"
  }, row.version, " \xB7 ", row.model), /*#__PURE__*/React.createElement("h2", {
    style: {
      marginTop: 12
    }
  }, row.title), /*#__PURE__*/React.createElement("div", {
    className: "status",
    role: "status"
  }, row.error || state.progress || 'Queued', /*#__PURE__*/React.createElement("div", {
    className: "muted"
  }, Object.keys(state.parts || {}).length, " / ", state.totalParts || '—', " source parts reviewed \xB7 ", row.status)), (row.status === 'failed' || row.status !== 'complete' && Date.now() - new Date(row.updated_at).getTime() > 180000) && /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: retry
  }, "Resume saved experiment"), /*#__PURE__*/React.createElement("div", {
    className: "controls",
    "aria-label": "Experimental sections"
  }, SECTIONS.map(([k, l]) => /*#__PURE__*/React.createElement("button", {
    key: k,
    "aria-pressed": section === k,
    onClick: () => setSection(k)
  }, l))), /*#__PURE__*/React.createElement("div", {
    className: "controls"
  }, /*#__PURE__*/React.createElement("button", {
    disabled: !Object.keys(row.baseline || {}).length,
    "aria-pressed": compare,
    onClick: () => setCompare(!compare)
  }, "Compare with original"), /*#__PURE__*/React.createElement("button", {
    disabled: !text,
    onClick: () => copy()
  }, "Copy section"), /*#__PURE__*/React.createElement("button", {
    disabled: row.status !== 'complete',
    onClick: () => copy(true)
  }, "Copy all"), /*#__PURE__*/React.createElement("button", {
    disabled: row.status !== 'complete' || sending,
    onClick: openEmail
  }, "Email all sections"), /*#__PURE__*/React.createElement("button", {
    onClick: download
  }, "Download experiment")), notice && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, notice), row.status !== 'complete' && text && /*#__PURE__*/React.createElement("p", {
    className: "muted"
  }, "Draft in progress. Source checks and revisions may still change this text."), sharing && /*#__PURE__*/React.createElement("section", {
    className: "status"
  }, /*#__PURE__*/React.createElement("strong", null, "Email preview \xB7 all five sections"), /*#__PURE__*/React.createElement("p", null, "Review the formatted note below. Switch section tabs to edit each section. Source references and qualifications are retained; private reviewer notes and the full transcript are excluded."), /*#__PURE__*/React.createElement("label", {
    htmlFor: "lab-email-edit"
  }, "Edit ", SECTIONS.find(s => s[0] === section)?.[1], " for this email"), /*#__PURE__*/React.createElement("textarea", {
    className: "email-editor",
    id: "lab-email-edit",
    value: text,
    onChange: e => setEdits({
      ...edits,
      [section]: e.target.value
    })
  }), /*#__PURE__*/React.createElement("div", {
    className: "controls"
  }, /*#__PURE__*/React.createElement("button", {
    className: "primary",
    disabled: sending,
    onClick: sendEmail
  }, sending ? 'Sending…' : 'Send all sections to myself'), /*#__PURE__*/React.createElement("button", {
    disabled: sending,
    onClick: () => {
      setSharing(false);
      setNotice('');
    }
  }, "Close email preview"))), /*#__PURE__*/React.createElement("div", {
    className: compare ? 'pair' : ''
  }, compare && /*#__PURE__*/React.createElement("article", null, /*#__PURE__*/React.createElement("h3", null, "Original \xB7 frozen at experiment start"), /*#__PURE__*/React.createElement("div", {
    className: "original",
    dangerouslySetInnerHTML: {
      __html: renderHtml(row.baseline?.[baseline] || '<p>No original section saved.</p>')
    }
  })), /*#__PURE__*/React.createElement("article", null, /*#__PURE__*/React.createElement("h3", null, sharing ? 'Email preview' : 'Research note'), /*#__PURE__*/React.createElement("div", {
    className: "reader",
    dangerouslySetInnerHTML: {
      __html: labDocument(text || 'This section will appear after source review and generation.')
    }
  }))), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Reviewer notes and limitations"), /*#__PURE__*/React.createElement("pre", null, state.finalReview || 'Final cross-section review has not finished.'), /*#__PURE__*/React.createElement("p", {
    className: "muted"
  }, "Model review is not independent factual verification. Exact supporting passages are checked against saved source text. The reviewed record may still contain omissions or interpretation errors."), state.hierarchicalSynthesis && /*#__PURE__*/React.createElement("p", null, "Long-source synthesis used consolidated records; complete part records remain below."), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Initial checks for this section"), /*#__PURE__*/React.createElement("pre", null, state.checks?.[section] || 'Not yet available.'))), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Source record and supporting passages \xB7 ", Object.keys(state.parts || {}).length, " parts"), Object.values(state.parts || {}).map(p => /*#__PURE__*/React.createElement("details", {
    key: p.id
  }, /*#__PURE__*/React.createElement("summary", null, p.id, " \xB7 characters ", p.start, "\u2013", p.end), /*#__PURE__*/React.createElement("pre", null, p.record), /*#__PURE__*/React.createElement("h4", null, "Exact original passages"), p.passages.map((v, i) => /*#__PURE__*/React.createElement("blockquote", {
    key: i,
    className: "reader"
  }, v)), /*#__PURE__*/React.createElement("h4", null, "Ambiguities / proposed corrections"), /*#__PURE__*/React.createElement("pre", null, JSON.stringify(p.issues, null, 2)))), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Immutable original source"), /*#__PURE__*/React.createElement("pre", null, row.source))), /*#__PURE__*/React.createElement("label", {
    htmlFor: "lab-feedback"
  }, "Your evaluation"), /*#__PURE__*/React.createElement("textarea", {
    id: "lab-feedback",
    rows: 4,
    value: feedback,
    onChange: e => setFeedback(e.target.value),
    placeholder: "What improved? What was lost, overstated, or harder to read?"
  }), /*#__PURE__*/React.createElement("button", {
    style: {
      marginTop: 10
    },
    onClick: async () => {
      try {
        await req('/' + id + '/feedback', {
          feedback
        });
        setNotice('Evaluation saved.');
      } catch (e) {
        setError(e.message);
      }
    }
  }, "Save evaluation")))));
}