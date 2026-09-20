import React, { useEffect, useRef, useState } from 'react';
import { documentHtml, emailDocument, labFanoutPlan, youtubeLanguagePayload } from './summary-lab-format.mjs';
var ENGLISH_SECTIONS = [['brief', 'Executive Brief', 'brief'], ['takeaways', 'Key Takeaways', 'summary'], ['meeting', 'Meeting Summary', 'meeting_summary'], ['questions', 'Follow-up Questions', 'questions'], ['assessment', 'Overall Assessment', 'assessment']];
var KOREAN_SECTION = ['korean', 'Korean Interpretation · 한국어 핵심 정리', 'korean_takeaways'];
var sectionsForMode = mode => mode === 'korean_only' ? [KOREAN_SECTION] : mode === 'korean_bilingual' ? [...ENGLISH_SECTIONS, KOREAN_SECTION] : ENGLISH_SECTIONS;
var INTAKES = [['saved', 'Saved Summary'], ['document', 'Document'], ['audio', 'Audio'], ['youtube', 'YouTube'], ['paste', 'Paste text']];
var PENDING_KEY = 'charlie_summary_lab_pending_intake';
export function SummaryLab({
  api,
  getKey,
  getGeminiKey,
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
    [compare, setCompare] = useState(false),
    [error, setError] = useState(''),
    [busy, setBusy] = useState(false),
    [feedback, setFeedback] = useState(''),
    [notice, setNotice] = useState(''),
    [importing, setImporting] = useState(false),
    [importStatus, setImportStatus] = useState('');
  var [intake, setIntake] = useState('saved'),
    [audioFile, setAudioFile] = useState(null),
    [youtubeUrl, setYoutubeUrl] = useState(''),
    [youtubeTicker, setYoutubeTicker] = useState(''),
    [youtubeKorean, setYoutubeKorean] = useState(false),
    [youtubeKoreanOnly, setYoutubeKoreanOnly] = useState(false),
    [ingest, setIngest] = useState(null);
  var [expanded, setExpanded] = useState({
    brief: true,
    takeaways: true,
    meeting: false,
    questions: false,
    assessment: false,
    korean: true
  });
  var audioInput = useRef(null),
    monitoring = useRef('');
  var visibleSections = sectionsForMode(row?.state?.outputMode || 'english');
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
  async function refreshLists() {
    var [a, b] = await Promise.all([req('/sources'), req()]);
    setSources(a.sources);
    setRuns(b.experiments);
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

  // sourceJobId is set only when a completed transcription triggers this run.
  // The pending job lives in localStorage, so every tab and every reload
  // resumes it; without the reference each one starts its own paid experiment.
  async function startLab(summaryId, labTitle, outputMode = 'english', labFocus = focus, sourceJobId) {
    var d = await req('', {
      summaryId: summaryId || undefined,
      source: summaryId ? undefined : source,
      title: (labTitle || title).trim() || 'Untitled experiment',
      focus: labFocus,
      outputMode,
      sourceJobId,
      apiKey: getKey()
    });
    setId(d.id);
    await refreshLists();
    return d;
  }
  async function start() {
    setBusy(true);
    setError('');
    setNotice('');
    try {
      await startLab(sid || undefined, title, 'english', focus);
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
        if (!/\.(pdf|docx|txt|md|csv|png|jpe?g|webp)$/i.test(file.name)) throw Error(`${file.name}: choose PDF, DOCX, text or image documents.`);
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
      setImportStatus(`Imported ${files.length} document${files.length === 1 ? '' : 's'}. Ready to generate.`);
    } catch (e) {
      setError(`${e.message} No imported documents were applied; your previous source is preserved.`);
      setImportStatus('Import did not finish.');
    } finally {
      setImporting(false);
    }
  }
  async function chooseAudioFromCloud() {
    try {
      var files = await pickFromICloud({
        mode: 'bytes',
        title: 'Summary Lab · choose one audio file'
      });
      var file = files?.[0];
      if (!file) return;
      if (!/\.(mp3|mp4|mpeg|mpga|m4a|wav|webm|ogg|flac)$/i.test(file.name)) throw Error('Choose an MP3, M4A, WAV, MP4, MPEG, WebM, OGG or FLAC audio file.');
      setAudioFile(file);
      if (!title.trim()) setTitle(file.name.replace(/\.[^.]+$/, '').slice(0, 300));
    } catch (e) {
      setError(e.message);
    }
  }
  async function monitorJob(jobId, label, jobTitle = title, jobFocus = focus, outputMode = 'english') {
    if (!jobId || monitoring.current === jobId) return;
    monitoring.current = jobId;
    try {
      for (var i = 0; i < 1080; i++) {
        var response = await fetch(`${api}/api/transcribe-audio/${encodeURIComponent(jobId)}`, {
          signal: AbortSignal.timeout(20000)
        });
        var data = await response.json();
        if (!response.ok) throw Error(data.error || 'Processing status is unavailable.');
        var phase = data.status || 'processing';
        setIngest({
          jobId,
          label,
          phase,
          progress: data.progress || ''
        });
        localStorage.setItem(PENDING_KEY, JSON.stringify({
          jobId,
          label,
          title: jobTitle,
          focus: jobFocus,
          outputMode
        }));
        if (['complete', 'done'].includes(phase)) {
          var plan = labFanoutPlan(data);
          if (plan.error) throw Error(plan.error);
          localStorage.removeItem(PENDING_KEY);
          setBusy(true);
          if (plan.adoptId) {
            setId(plan.adoptId);
            await refreshLists();
            setNotice(`${label} was transcribed and saved. Charlie already started its Summary Lab experiment for this recording, so it is shown here instead of starting a second run.`);
          } else {
            setNotice(`${label} was transcribed and saved. Improved analysis is now running.`);
            await startLab(plan.summaryId, jobTitle || label, outputMode, jobFocus, jobId);
          }
          setIngest(null);
          setBusy(false);
          return;
        }
        if (['failed', 'error'].includes(phase)) {
          localStorage.removeItem(PENDING_KEY);
          setIngest(null);
          throw Error(data.error || `${label} processing failed.`);
        }
        await new Promise(resolve => setTimeout(resolve, 5000));
      }
      throw Error(`${label} is still processing after 90 minutes. Its saved job is preserved; reopen Summary Lab to resume checking.`);
    } catch (e) {
      setError(e.message);
      setBusy(false);
    } finally {
      monitoring.current = '';
    }
  }
  useEffect(() => {
    try {
      var pending = JSON.parse(localStorage.getItem(PENDING_KEY) || 'null');
      if (pending?.jobId) {
        if (pending.title) setTitle(pending.title);
        if (pending.focus) setFocus(pending.focus);
        setIngest({
          ...pending,
          phase: 'checking',
          progress: 'Reconnecting to saved job…'
        });
        monitorJob(pending.jobId, pending.label || 'Source', pending.title || '', pending.focus || '', pending.outputMode || 'english');
      }
    } catch {}
  }, [api]);
  async function processAudio() {
    if (!audioFile) return;
    setBusy(true);
    setError('');
    setNotice('');
    try {
      var form = new FormData();
      form.append('file', audioFile);
      form.append('detailLevel', 'standard');
      form.append('apiKey', getKey() || '');
      form.append('geminiApiKey', getGeminiKey?.() || '');
      form.append('origin', 'summary-lab');
      var response = await fetch(`${api}/api/auto-process-audio`, {
        method: 'POST',
        body: form,
        signal: AbortSignal.timeout(30 * 60 * 1000)
      });
      var data = await response.json();
      if (!response.ok) throw Error(data.error || 'Audio upload failed.');
      var label = audioFile.name,
        jobTitle = title || label;
      localStorage.setItem(PENDING_KEY, JSON.stringify({
        jobId: data.jobId,
        label,
        title: jobTitle,
        focus,
        outputMode: 'english'
      }));
      setBusy(false);
      await monitorJob(data.jobId, label, jobTitle, focus, 'english');
    } catch (e) {
      setError(e.message);
      setBusy(false);
    }
  }
  async function processYoutube() {
    setBusy(true);
    setError('');
    setNotice('');
    try {
      var language = youtubeLanguagePayload(youtubeKorean, youtubeKoreanOnly);
      var response = await fetch(`${api}/api/youtube-summarize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          url: youtubeUrl.trim(),
          ticker: youtubeTicker.trim().toUpperCase(),
          generateKorean: language.generateKorean,
          koreanOnly: language.koreanOnly,
          apiKey: getKey()
        }),
        signal: AbortSignal.timeout(30000)
      });
      var data = await response.json();
      if (!response.ok) throw Error(data.error || 'YouTube processing could not start.');
      var label = data.title || 'YouTube video',
        jobTitle = title || label;
      if (!title.trim()) setTitle(label);
      localStorage.setItem(PENDING_KEY, JSON.stringify({
        jobId: data.jobId,
        label,
        title: jobTitle,
        focus,
        outputMode: language.outputMode
      }));
      setBusy(false);
      await monitorJob(data.jobId, label, jobTitle, focus, language.outputMode);
    } catch (e) {
      setError(e.message);
      setBusy(false);
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
  async function stopRun() {
    setBusy(true);
    setError('');
    try {
      await req('/' + id + '/stop', {});
      setNotice('Stopping at the next checkpoint. Completed stages stay saved and you can resume later.');
      setRow(await req('/' + id));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  var state = row?.state || {};
  function openEmail() {
    setEdits({
      ...state.sections
    });
    setSharing(true);
    setCompare(false);
    setExpanded(Object.fromEntries(visibleSections.map(([key]) => [key, true])));
    setNotice('Review and edit each section before sending. Edits affect this email only.');
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
          content: emailDocument(row.title, visibleSections.map(([k, l]) => [l, edits[k] || '']), renderHtml),
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
      setNotice(`${visibleSections.length === 1 ? 'The section' : `All ${visibleSections.length} sections`} emailed to ${creds.email}.`);
    } catch (e) {
      setError(e.message);
    } finally {
      setSending(false);
    }
  }
  async function copy(all = false, key = 'brief') {
    try {
      var items = all ? visibleSections.map(([k, l]) => [l, (sharing ? edits[k] : state.sections?.[k]) || '']) : [[visibleSections.find(s => s[0] === key)?.[1] || '', (sharing ? edits[key] : state.sections?.[key]) || '']];
      var html = emailDocument(row.title, items, renderHtml);
      var doc = new DOMParser().parseFromString(html, 'text/html');
      doc.querySelectorAll('p,h1,h2,h3,li,blockquote').forEach(el => el.append('\n'));
      var plain = doc.body.textContent || '';
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
  var canGenerate = intake === 'saved' && !!sid || (intake === 'paste' || intake === 'document') && !!source.trim();
  return /*#__PURE__*/React.createElement("main", {
    className: "summary-lab"
  }, /*#__PURE__*/React.createElement("style", null, `.summary-lab{box-sizing:border-box;--lab-border:rgba(153,142,119,.35);max-width:1500px;margin:0 auto;height:100%;min-height:0;overflow-y:auto;overflow-x:hidden;overscroll-behavior-y:contain;padding:32px;color:var(--text-primary,#e9e2d3);width:100%;min-width:0}.summary-lab *{box-sizing:border-box}.summary-lab h1,.summary-lab h2{font-family:Georgia,serif;line-height:1.2}.summary-lab h1{font-size:38px;margin:8px 0 14px}.summary-lab h2{font-size:24px;margin:0 0 18px}.summary-lab p{line-height:1.65}.summary-lab .muted{opacity:.72;font-size:13px}.summary-lab .eyebrow{color:#c9a857;letter-spacing:.13em;text-transform:uppercase;font-size:11px}.summary-lab .layout{display:grid;grid-template-columns:330px minmax(0,1fr);gap:24px;margin-top:28px}.summary-lab .panel{border:1px solid var(--lab-border);border-radius:14px;padding:24px;background:rgba(127,115,89,.045);min-width:0}.summary-lab label{display:block;font-size:13px;margin:16px 0 6px}.summary-lab input,.summary-lab select,.summary-lab textarea{width:100%;padding:11px;border:1px solid var(--lab-border);border-radius:7px;background:var(--bg-secondary,#211e18);color:inherit;font:inherit;min-width:0}.summary-lab .check-row{display:flex;align-items:flex-start;gap:9px;margin:14px 0 0;line-height:1.45;cursor:pointer}.summary-lab .check-row.nested{margin:9px 0 0 26px}.summary-lab .check-row input[type=checkbox]{width:17px;height:17px;min-width:17px;margin:1px 0 0;padding:0;accent-color:#c9a857}.summary-lab select option{background:#211e18;color:#eee}.summary-lab button{padding:10px 14px;min-height:44px;border:1px solid var(--lab-border);border-radius:7px;font:inherit;cursor:pointer;background:transparent;color:inherit}.summary-lab button:focus-visible,.summary-lab input:focus-visible,.summary-lab textarea:focus-visible,.summary-lab select:focus-visible{outline:2px solid #c9a857;outline-offset:3px}.summary-lab button:disabled{opacity:.45;cursor:default}.summary-lab button.primary,.summary-lab button[aria-pressed=true]{background:#c9a857;color:#18150f}.summary-lab .controls{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}.summary-lab .intake-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;margin-bottom:16px}.summary-lab .intake-grid button{padding:8px;min-height:38px;font-size:12px}.summary-lab .dropzone{padding:18px;border:1px dashed var(--lab-border);border-radius:10px;text-align:center;background:rgba(201,168,87,.035)}.summary-lab .experiment{display:block;width:100%;text-align:left;margin:10px 0;overflow-wrap:anywhere}.summary-lab .reader{font-family:Calibri,Carlito,Arial,sans-serif;font-size:11pt;line-height:1.55;overflow-wrap:anywhere;max-width:94ch;background:#fff;color:#242424;padding:28px;border:1px solid #dedbd4;border-radius:4px}.summary-lab .reader h1,.summary-lab .reader h2,.summary-lab .reader h3,.summary-lab .reader h4{font:700 11pt/1.5 Calibri,Carlito,Arial,sans-serif;margin:20px 0 8px}.summary-lab .reader p{margin:0 0 12px;line-height:1.55}.summary-lab .reader ul,.summary-lab .reader ol{padding-left:23px;margin:10px 0 16px}.summary-lab .reader li{margin:6px 0}.summary-lab .reader blockquote{border-left:3px solid #b9af94;padding-left:14px;margin:14px 0}.summary-lab .reader table{display:block;max-width:100%;overflow:auto;border-collapse:collapse}.summary-lab .reader th,.summary-lab .reader td{border:1px solid #ddd;padding:7px 9px;text-align:left}.summary-lab .email-editor{font:11pt/1.5 Calibri,Carlito,Arial,sans-serif;min-height:220px}.summary-lab .pair{display:grid;gap:24px;grid-template-columns:repeat(2,minmax(0,1fr))}.summary-lab .status{padding:14px;border-left:3px solid #c9a857;background:rgba(201,168,87,.08);margin:16px 0;overflow-wrap:anywhere}.summary-lab details.lab-section{border:1px solid var(--lab-border);border-radius:10px;margin:12px 0;padding:0;overflow:hidden}.summary-lab details.lab-section>summary{list-style:none;cursor:pointer;padding:16px 18px;display:flex;justify-content:space-between;align-items:center;gap:12px;background:rgba(127,115,89,.04)}.summary-lab details.lab-section>summary::-webkit-details-marker{display:none}.summary-lab .section-body{padding:18px}.summary-lab .chevron{display:inline-block;transition:transform .18s ease}.summary-lab details[open] .chevron{transform:rotate(90deg)}.summary-lab details.audit{border-top:1px solid var(--lab-border);padding:16px 0;margin-top:16px}.summary-lab pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;line-height:1.7}.summary-lab .original{overflow-wrap:anywhere;line-height:1.8}.summary-lab .original table{display:block;overflow:auto;max-width:100%}@media(max-width:900px){.summary-lab{padding:18px 18px 112px}.summary-lab .layout,.summary-lab .pair{grid-template-columns:1fr}.summary-lab h1{font-size:30px}.summary-lab .panel{padding:18px}.summary-lab .reader{padding:20px}.summary-lab .intake-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}`), /*#__PURE__*/React.createElement("div", {
    className: "eyebrow"
  }, "Charlie / Research experiments"), /*#__PURE__*/React.createElement("h1", null, "Summary Lab"), /*#__PURE__*/React.createElement("p", null, "Read thoroughly. Preserve what was said. Separate what it means."), /*#__PURE__*/React.createElement("p", {
    className: "muted"
  }, "An independent trial with five familiar sections and optional Korean interpretation. Original summaries and automatic workflows remain unchanged."), error && /*#__PURE__*/React.createElement("div", {
    role: "alert",
    className: "status"
  }, error, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("button", {
    onClick: () => setError(''),
    "aria-label": "Dismiss error"
  }, "Dismiss"))), ingest && /*#__PURE__*/React.createElement("div", {
    className: "status",
    role: "status"
  }, /*#__PURE__*/React.createElement("strong", null, ingest.label), /*#__PURE__*/React.createElement("div", null, ingest.progress || ingest.phase), /*#__PURE__*/React.createElement("p", {
    className: "muted"
  }, "You can leave this page. Charlie keeps the saved job and Summary Lab reconnects when you return.")), /*#__PURE__*/React.createElement("div", {
    className: "layout"
  }, /*#__PURE__*/React.createElement("aside", null, /*#__PURE__*/React.createElement("section", {
    className: "panel"
  }, /*#__PURE__*/React.createElement("h2", null, "Add a source"), /*#__PURE__*/React.createElement("div", {
    className: "intake-grid",
    role: "group",
    "aria-label": "Source type"
  }, INTAKES.map(([key, label]) => /*#__PURE__*/React.createElement("button", {
    key: key,
    "aria-pressed": intake === key,
    onClick: () => {
      setIntake(key);
      setError('');
    }
  }, label))), intake === 'saved' && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("label", {
    htmlFor: "lab-source"
  }, "Saved transcript or document"), /*#__PURE__*/React.createElement("select", {
    id: "lab-source",
    value: sid,
    onChange: e => setSid(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Choose a saved Summary\u2026"), sources.map(s => /*#__PURE__*/React.createElement("option", {
    key: s.id,
    value: s.id
  }, s.title)))), intake === 'document' && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "dropzone"
  }, /*#__PURE__*/React.createElement("strong", null, "PDF, Word, text or image"), /*#__PURE__*/React.createElement("p", {
    className: "muted"
  }, "Choose one or several documents from the same connected iCloud sources."), /*#__PURE__*/React.createElement("button", {
    disabled: busy || importing,
    onClick: importCloud
  }, importing ? 'Importing…' : 'Browse iCloud documents')), importStatus && /*#__PURE__*/React.createElement("p", {
    role: "status",
    className: "muted"
  }, importStatus), source && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("label", {
    htmlFor: "lab-document-text"
  }, "Extracted source text"), /*#__PURE__*/React.createElement("textarea", {
    id: "lab-document-text",
    rows: 6,
    value: source,
    onChange: e => setSource(e.target.value)
  }), /*#__PURE__*/React.createElement("p", {
    className: "muted"
  }, source.length.toLocaleString(), " characters \xB7 no silent cutoff"))), intake === 'paste' && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("label", {
    htmlFor: "lab-text"
  }, "Complete source text"), /*#__PURE__*/React.createElement("textarea", {
    id: "lab-text",
    rows: 9,
    value: source,
    onChange: e => setSource(e.target.value),
    placeholder: "Paste a transcript or document text\u2026"
  }), /*#__PURE__*/React.createElement("p", {
    className: "muted"
  }, source.length.toLocaleString(), " characters \xB7 no silent cutoff")), intake === 'audio' && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("input", {
    ref: audioInput,
    type: "file",
    hidden: true,
    accept: ".mp3,.mp4,.mpeg,.mpga,.m4a,.wav,.webm,.ogg,.flac",
    onChange: e => {
      var file = e.target.files?.[0];
      if (file) {
        setAudioFile(file);
        if (!title.trim()) setTitle(file.name.replace(/\.[^.]+$/, '').slice(0, 300));
      }
    }
  }), /*#__PURE__*/React.createElement("div", {
    className: "dropzone"
  }, /*#__PURE__*/React.createElement("strong", null, audioFile?.name || 'Audio recording'), /*#__PURE__*/React.createElement("p", {
    className: "muted"
  }, "MP3, M4A, WAV, MP4, MPEG, WebM, OGG or FLAC. The full transcript is saved before improved analysis begins."), /*#__PURE__*/React.createElement("div", {
    className: "controls"
  }, /*#__PURE__*/React.createElement("button", {
    onClick: () => audioInput.current?.click()
  }, "Choose file"), /*#__PURE__*/React.createElement("button", {
    onClick: chooseAudioFromCloud
  }, "Browse iCloud")), audioFile && /*#__PURE__*/React.createElement("p", {
    className: "muted"
  }, (audioFile.size / 1024 / 1024).toFixed(1), " MB")), !getGeminiKey?.() && /*#__PURE__*/React.createElement("p", {
    className: "muted"
  }, "Audio transcription requires the Gemini key configured in Settings.")), intake === 'youtube' && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("label", {
    htmlFor: "lab-youtube"
  }, "YouTube link"), /*#__PURE__*/React.createElement("input", {
    id: "lab-youtube",
    type: "url",
    value: youtubeUrl,
    onChange: e => setYoutubeUrl(e.target.value),
    placeholder: "https://www.youtube.com/watch?v=\u2026"
  }), /*#__PURE__*/React.createElement("label", {
    htmlFor: "lab-youtube-ticker"
  }, "Ticker \xB7 optional"), /*#__PURE__*/React.createElement("input", {
    id: "lab-youtube-ticker",
    value: youtubeTicker,
    maxLength: 8,
    onChange: e => setYoutubeTicker(e.target.value.toUpperCase()),
    placeholder: "ABT"
  }), /*#__PURE__*/React.createElement("label", {
    className: "check-row"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: youtubeKorean,
    onChange: e => {
      setYoutubeKorean(e.target.checked);
      if (!e.target.checked) setYoutubeKoreanOnly(false);
    }
  }), /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("strong", null, "\uD55C\uAD6D\uC5B4 \uD575\uC2EC \uC815\uB9AC\uB3C4 \uD568\uAED8 \uC0DD\uC131"), /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    className: "muted"
  }, "English analysis plus a source-reviewed Korean interpretation."))), youtubeKorean && /*#__PURE__*/React.createElement("label", {
    className: "check-row nested"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: youtubeKoreanOnly,
    onChange: e => setYoutubeKoreanOnly(e.target.checked)
  }), /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("strong", null, "\uD55C\uAD6D\uC5B4\uB9CC \uC0DD\uC131"), /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("span", {
    className: "muted"
  }, "Skip the five English sections and generate only the Korean interpretation."))), /*#__PURE__*/React.createElement("p", {
    className: "muted"
  }, "Charlie retrieves the available transcript through your connected Mac, saves it, then starts the improved analysis in the selected language.")), /*#__PURE__*/React.createElement("label", {
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
  }, "Thorough review makes several passes over the complete source. Long transcripts take longer and remain saved if you leave."), intake === 'audio' ? /*#__PURE__*/React.createElement("button", {
    className: "primary",
    disabled: busy || !audioFile || !getGeminiKey?.(),
    onClick: processAudio
  }, busy ? 'Starting…' : 'Transcribe and analyze') : intake === 'youtube' ? /*#__PURE__*/React.createElement("button", {
    className: "primary",
    disabled: busy || !youtubeUrl.trim(),
    onClick: processYoutube
  }, busy ? 'Starting…' : 'Fetch transcript and analyze') : /*#__PURE__*/React.createElement("button", {
    className: "primary",
    disabled: busy || importing || !canGenerate,
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
  }, r.automatic ? 'Auto from SUMMARIES · ' : '', r.status === 'complete' ? 'Ready to review' : r.status === 'cancelled' ? 'Stopped' : r.status, " \xB7 ", new Date(r.created_at).toLocaleDateString()))))), /*#__PURE__*/React.createElement("section", {
    className: "panel"
  }, !row ? /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "eyebrow"
  }, "Independent source review"), /*#__PURE__*/React.createElement("h2", {
    style: {
      marginTop: 12
    }
  }, id ? 'Loading experiment…' : 'Your next research note starts here'), /*#__PURE__*/React.createElement("p", null, "Add a saved Summary, document, recording, YouTube link or pasted transcript. Charlie generates Executive Brief, Key Takeaways, Meeting Summary, Follow-up Questions and Overall Assessment."), /*#__PURE__*/React.createElement("p", {
    className: "muted"
  }, "Audio and YouTube are transcribed first. Source ambiguities remain visible for review.")) : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
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
  }, Object.keys(state.parts || {}).length, " / ", state.totalParts || '—', " source parts reviewed \xB7 ", row.status)), (row.status === 'failed' || row.status === 'cancelled' || row.status !== 'complete' && Date.now() - new Date(row.updated_at).getTime() > 180000) && /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: retry
  }, "Resume saved experiment"), (row.status === 'queued' || row.status === 'running') && !row.cancel_requested && /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: stopRun
  }, "Stop this experiment"), row.cancel_requested && row.status !== 'cancelled' && /*#__PURE__*/React.createElement("p", {
    className: "muted",
    role: "status"
  }, "Stopping at the next checkpoint\u2026"), /*#__PURE__*/React.createElement("div", {
    className: "controls"
  }, /*#__PURE__*/React.createElement("button", {
    disabled: !Object.keys(row.baseline || {}).length,
    "aria-pressed": compare,
    onClick: () => setCompare(!compare)
  }, "Compare with original"), /*#__PURE__*/React.createElement("button", {
    disabled: row.status !== 'complete',
    onClick: () => copy(true)
  }, "Copy all"), /*#__PURE__*/React.createElement("button", {
    disabled: row.status !== 'complete' || sending,
    onClick: openEmail
  }, "Email all sections"), /*#__PURE__*/React.createElement("button", {
    onClick: download
  }, "Download experiment")), /*#__PURE__*/React.createElement("div", {
    className: "controls"
  }, /*#__PURE__*/React.createElement("button", {
    onClick: () => setExpanded(Object.fromEntries(visibleSections.map(([key]) => [key, true])))
  }, "Expand all"), /*#__PURE__*/React.createElement("button", {
    onClick: () => setExpanded(Object.fromEntries(visibleSections.map(([key]) => [key, false])))
  }, "Collapse all")), notice && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, notice), sharing && /*#__PURE__*/React.createElement("section", {
    className: "status"
  }, /*#__PURE__*/React.createElement("strong", null, "Email preview \xB7 ", visibleSections.length === 1 ? 'Korean interpretation' : `all ${visibleSections.length} sections`), /*#__PURE__*/React.createElement("p", null, "Review any section below before sending. Edits affect this email only; audit notes and the full transcript are excluded."), /*#__PURE__*/React.createElement("div", {
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
  }, "Close email preview"))), visibleSections.map(([key, label, baseline]) => {
    var value = (sharing ? edits[key] : state.sections?.[key]) || '';
    return /*#__PURE__*/React.createElement("details", {
      key: key,
      className: "lab-section",
      open: !!expanded[key],
      onToggle: e => {
        var open = e.currentTarget.open;
        setExpanded(current => ({
          ...current,
          [key]: open
        }));
      }
    }, /*#__PURE__*/React.createElement("summary", null, /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("span", {
      className: "chevron",
      "aria-hidden": "true"
    }, "\u203A"), " ", /*#__PURE__*/React.createElement("strong", null, label)), /*#__PURE__*/React.createElement("span", {
      className: "muted"
    }, value ? 'Ready' : 'Waiting')), /*#__PURE__*/React.createElement("div", {
      className: "section-body"
    }, /*#__PURE__*/React.createElement("div", {
      className: "controls"
    }, /*#__PURE__*/React.createElement("button", {
      disabled: !value,
      onClick: () => copy(false, key)
    }, "Copy section")), row.status !== 'complete' && value && /*#__PURE__*/React.createElement("p", {
      className: "muted"
    }, "Draft in progress. Source checks may still revise this section."), sharing ? /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("label", {
      htmlFor: `lab-edit-${key}`
    }, "Edit for this email"), /*#__PURE__*/React.createElement("textarea", {
      className: "email-editor",
      id: `lab-edit-${key}`,
      value: value,
      onChange: e => setEdits({
        ...edits,
        [key]: e.target.value
      })
    }), /*#__PURE__*/React.createElement("div", {
      className: "reader",
      dangerouslySetInnerHTML: {
        __html: documentHtml(value || 'This section is not available.', renderHtml)
      }
    })) : /*#__PURE__*/React.createElement("div", {
      className: compare ? 'pair' : ''
    }, compare && /*#__PURE__*/React.createElement("article", null, /*#__PURE__*/React.createElement("h3", null, "Original \xB7 frozen at experiment start"), /*#__PURE__*/React.createElement("div", {
      className: "original",
      dangerouslySetInnerHTML: {
        __html: documentHtml(row.baseline?.[baseline] || 'No original section saved.', renderHtml)
      }
    })), /*#__PURE__*/React.createElement("article", null, compare && /*#__PURE__*/React.createElement("h3", null, "Improved"), /*#__PURE__*/React.createElement("div", {
      className: "reader",
      dangerouslySetInnerHTML: {
        __html: documentHtml(value || 'This section will appear after source review and generation.', renderHtml)
      }
    })))));
  }), /*#__PURE__*/React.createElement("details", {
    className: "audit"
  }, /*#__PURE__*/React.createElement("summary", null, "Reviewer notes and limitations"), /*#__PURE__*/React.createElement("pre", null, state.finalReview || 'Final cross-section review has not finished.'), /*#__PURE__*/React.createElement("p", {
    className: "muted"
  }, "Model review is not independent factual verification. Exact supporting passages are checked against saved source text."), state.hierarchicalSynthesis && /*#__PURE__*/React.createElement("p", null, "Long-source synthesis used consolidated records; complete part records remain below.")), /*#__PURE__*/React.createElement("details", {
    className: "audit"
  }, /*#__PURE__*/React.createElement("summary", null, "Source record and supporting passages \xB7 ", Object.keys(state.parts || {}).length, " parts"), Object.values(state.parts || {}).map(p => /*#__PURE__*/React.createElement("details", {
    className: "audit",
    key: p.id
  }, /*#__PURE__*/React.createElement("summary", null, p.id, " \xB7 characters ", p.start, "\u2013", p.end), /*#__PURE__*/React.createElement("pre", null, p.record), /*#__PURE__*/React.createElement("h4", null, "Exact original passages"), p.passages.map((v, i) => /*#__PURE__*/React.createElement("blockquote", {
    key: i,
    className: "reader"
  }, v)), /*#__PURE__*/React.createElement("h4", null, "Ambiguities / proposed corrections"), /*#__PURE__*/React.createElement("pre", null, JSON.stringify(p.issues, null, 2)))), /*#__PURE__*/React.createElement("details", {
    className: "audit"
  }, /*#__PURE__*/React.createElement("summary", null, "Immutable original source"), /*#__PURE__*/React.createElement("pre", null, row.source))), /*#__PURE__*/React.createElement("label", {
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