import { parseTimestamp } from './workspace-model.mjs';
import * as React from 'react';
var {
  useState,
  useEffect,
  useRef
} = React;
export function ThesisAmendments({
  api,
  ticker,
  context,
  onApplied
}) {
  var [instructions, setInstructions] = useState(context?.bridge?.instructions || '');
  var [jobs, setJobs] = useState([]),
    [selected, setSelected] = useState([]),
    [accepted, setAccepted] = useState([]);
  var [error, setError] = useState(''),
    [busy, setBusy] = useState(false),
    [loaded, setLoaded] = useState(false);
  var alive = useRef(true),
    lock = useRef(false),
    requestId = useRef(null),
    sequence = useRef(0);
  var documents = context?.documents?.uploaded || [];
  var thesisDate = parseTimestamp(context?.savedThesis?.updatedAt);
  var isNew = d => thesisDate && parseTimestamp(d.addedAt) > thesisDate;
  var newCount = documents.filter(isNew).length;
  var job = jobs.find(j => ['queued', 'running', 'awaiting_approval'].includes(j.status)) || jobs[0];
  var changes = job?.result?.changes || [];
  var fieldLabel = path => {
    if (path === 'thesis.summary') return 'Thesis summary';
    if (path === 'conclusion') return 'Conclusion';
    var parts = path.split('.'),
      i = Number(parts[0] === 'thesis' ? parts[2] : parts[1]);
    if (parts[0] === 'thesis') return context?.savedThesis?.thesis?.pillars?.[i]?.title || `Pillar ${i + 1}`;
    return `${parts[0] === 'signposts' ? 'Signpost' : 'Risk'} ${i + 1}`;
  };
  var fetchJson = async (path, options = {}) => {
    var controller = new AbortController(),
      timer = setTimeout(() => controller.abort(), 30000);
    try {
      var r = await fetch(`${api}${path}`, {
        ...options,
        signal: controller.signal
      });
      var d = await r.json();
      if (!r.ok) throw new Error(d.error || `Request failed (${r.status})`);
      return d;
    } finally {
      clearTimeout(timer);
    }
  };
  var refresh = async () => {
    var id = ++sequence.current;
    try {
      var d = await fetchJson(`/api/research/amendments/${encodeURIComponent(ticker)}`);
      if (alive.current && id === sequence.current) {
        setJobs(d.jobs || []);
        setLoaded(true);
      }
    } catch (e) {
      if (alive.current && id === sequence.current) {
        setError(e.name === 'AbortError' ? 'The request timed out. Refresh to check status.' : e.message);
        setLoaded(true);
      }
    }
  };
  useEffect(() => {
    alive.current = true;
    refresh();
    var timer = setInterval(() => {
      if (!document.hidden) refresh();
    }, 5000);
    return () => {
      alive.current = false;
      sequence.current++;
      clearInterval(timer);
    };
  }, [api, ticker]);
  useEffect(() => {
    setAccepted([]);
    if (job && !['queued', 'running', 'awaiting_approval'].includes(job.status)) requestId.current = null;
  }, [job?.id, job?.status]);
  var mutate = async (path, body) => {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    setError('');
    try {
      var d = await fetchJson(path, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
      });
      if (alive.current) {
        await refresh();
        if (['applied', 'reverted'].includes(d.status)) onApplied();
      }
    } catch (e) {
      if (alive.current) {
        setError(e.message + ' Check the proposal status before retrying.');
        await refresh();
      }
    } finally {
      lock.current = false;
      if (alive.current) setBusy(false);
    }
  };
  var launch = () => {
    if (!requestId.current) requestId.current = crypto.randomUUID();
    var key = '';
    try {
      key = localStorage.getItem('equity_analyzer_api_key') || '';
    } catch {}
    return mutate(`/api/research/amendments/${encodeURIComponent(ticker)}`, {
      filenames: selected,
      requestId: requestId.current,
      apiKey: key,
      instructions,
      ...(context?.bridge ? {
        commandId: context.bridge.commandId,
        commandRevision: context.bridge.revision
      } : {})
    });
  };
  var resume = () => {
    var key = '';
    try {
      key = localStorage.getItem('equity_analyzer_api_key') || '';
    } catch {}
    return mutate(`/api/research/amendment/${job.id}/resume`, {
      apiKey: key
    });
  };
  var active = job && ['queued', 'running', 'awaiting_approval'].includes(job.status);
  return /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel thesis-amendments"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "NEW EVIDENCE \u2192 PROPOSED THESIS EDITS"), /*#__PURE__*/React.createElement("h3", null, "What should change?"), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Compare selected sources with your saved thesis. Charlie drafts targeted edits and checks their supporting quotations. Your thesis changes only when you apply selected edits."), context?.bridge && /*#__PURE__*/React.createElement("div", {
    className: "workspace-notice"
  }, /*#__PURE__*/React.createElement("strong", null, "Sources from ", context.bridge.topic), /*#__PURE__*/React.createElement("p", null, context.bridge.ready.length, " source(s) match the recap input hashes. Choose up to 10 below. This comparison uses original documents, not the recap as evidence."), context.bridge.blocked.length > 0 && /*#__PURE__*/React.createElement("details", {
    open: true
  }, /*#__PURE__*/React.createElement("summary", null, context.bridge.blocked.length, " source(s) need attention"), context.bridge.blocked.map((d, i) => /*#__PURE__*/React.createElement("p", {
    key: i
  }, /*#__PURE__*/React.createElement("strong", null, d.filename), " \xB7 ", d.reason)))), error && /*#__PURE__*/React.createElement("div", {
    className: "workspace-error",
    role: "alert"
  }, error, /*#__PURE__*/React.createElement("button", {
    onClick: () => {
      setError('');
      refresh();
    }
  }, "Refresh status")), !loaded ? /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading proposals\u2026") : /*#__PURE__*/React.createElement(React.Fragment, null, context?.bridge && job && job.command_id !== context.bridge.commandId && /*#__PURE__*/React.createElement("p", {
    className: "workspace-notice"
  }, "The proposal below belongs to another comparison for ", ticker, ". Review or dismiss any active proposal before preparing one from this command."), active ? /*#__PURE__*/React.createElement("div", {
    className: "workspace-notice"
  }, /*#__PURE__*/React.createElement("strong", null, job.status === 'awaiting_approval' ? 'Proposal ready for your review' : 'Comparing sources…'), job.status !== 'awaiting_approval' && /*#__PURE__*/React.createElement("p", null, job.recoverable === 'true' ? `Interrupted running proposals can recover when the Mac heartbeat and server research key are available. Recovery attempts: ${job.recovery_attempts || 0}/2. Queued jobs and provider errors need separate attention.` : 'The proposal is saved as a job. If a server restart interrupts it, dismiss it and prepare a new comparison.', " No thesis edits have been applied."), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => mutate(`/api/research/amendment/${job.id}/decide`, {
      action: 'dismiss'
    })
  }, "Dismiss proposal")) : /*#__PURE__*/React.createElement(React.Fragment, null, job?.status === 'failed' && job.result?.checkpoint && /*#__PURE__*/React.createElement("div", {
    className: "workspace-notice"
  }, /*#__PURE__*/React.createElement("p", null, "A completed proposal stage is saved. Resume checks the original inputs and reuses saved work; an unfinished model call may need to run again. Up to two resume attempts."), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: resume
  }, "Resume saved proposal work \u2192")), job && /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer",
    role: "status"
  }, "Last proposal: ", job.status.replaceAll('_', ' '), job.error ? ` · ${job.error}` : ''), !context?.savedThesis ? /*#__PURE__*/React.createElement("p", null, "Save an investment thesis before preparing edits.") : !documents.length ? /*#__PURE__*/React.createElement("p", null, "Import source documents into Charlie first. Files listed only in the iCloud inventory must be imported before this comparison can read them.") : /*#__PURE__*/React.createElement("details", {
    open: true
  }, /*#__PURE__*/React.createElement("summary", null, "Select documents to compare (", selected.length, "/10)"), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, newCount, " document(s) were added to Charlie after the saved thesis update. This is upload timing, not publication date or proof they contain new information."), /*#__PURE__*/React.createElement("div", {
    className: "amendment-documents"
  }, documents.map(d => /*#__PURE__*/React.createElement("label", {
    key: d.filename
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: selected.includes(d.filename),
    disabled: busy || !selected.includes(d.filename) && selected.length >= 10,
    onChange: () => {
      requestId.current = null;
      setSelected(old => old.includes(d.filename) ? old.filter(n => n !== d.filename) : [...old, d.filename]);
    }
  }), /*#__PURE__*/React.createElement("span", null, d.filename, isNew(d) && /*#__PURE__*/React.createElement("small", null, "Added after thesis update"))))), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Uses your configured research model and API credits. Up to 10 saved documents per comparison; scanned or oversized sources may need preparation first."), /*#__PURE__*/React.createElement("label", {
    className: "earnings-search"
  }, "Instructions for the thesis analyst", /*#__PURE__*/React.createElement("textarea", {
    rows: 3,
    maxLength: 6000,
    value: instructions,
    onChange: e => {
      setInstructions(e.target.value);
      requestId.current = null;
    },
    placeholder: "Correct the guidance period, challenge a pillar, or update a specific conclusion\u2026"
  })), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    disabled: busy || !selected.length,
    onClick: launch
  }, busy ? 'Submitting…' : 'Find material changes →'))), ['awaiting_approval', 'applied'].includes(job?.status) && /*#__PURE__*/React.createElement(React.Fragment, null, !changes.length ? /*#__PURE__*/React.createElement("p", {
    className: "workspace-notice"
  }, "No material edits were proposed from these sources. This is not a guarantee that the thesis is complete or correct.") : changes.map(c => /*#__PURE__*/React.createElement("article", {
    className: "amendment-card",
    key: c.id
  }, /*#__PURE__*/React.createElement("label", null, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: job.status === 'applied' ? (job.result.acceptedIds || []).includes(c.id) : accepted.includes(c.id),
    disabled: busy || job.status !== 'awaiting_approval' || !c.passageMatched || !c.reviewPassed,
    onChange: () => setAccepted(old => old.includes(c.id) ? old.filter(id => id !== c.id) : [...old, c.id])
  }), /*#__PURE__*/React.createElement("strong", null, fieldLabel(c.path), job.status === 'applied' ? (job.result.acceptedIds || []).includes(c.id) ? ' · Applied' : ' · Not applied' : '')), /*#__PURE__*/React.createElement("p", null, c.reason), /*#__PURE__*/React.createElement("div", {
    className: "evidence-compare"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h4", null, "Saved text"), /*#__PURE__*/React.createElement("p", null, c.before)), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h4", null, "Proposed text"), /*#__PURE__*/React.createElement("p", null, c.after))), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, c.passageMatched ? 'Source quotation matched.' : 'Blocked: source quotation did not match.', " ", c.reviewPassed ? 'Independent check passed; analyst judgment required.' : 'Blocked: independent review needs revision.', " ", c.reviewIssue), c.evidence.map((e, i) => /*#__PURE__*/React.createElement("details", {
    key: i
  }, /*#__PURE__*/React.createElement("summary", null, "Inspect supporting quotation"), /*#__PURE__*/React.createElement("strong", null, job.result.sources?.find(s => s.id === e.sourceId)?.filename || 'Unknown source'), /*#__PURE__*/React.createElement("blockquote", null, e.excerpt))))), job.status === 'applied' && /*#__PURE__*/React.createElement("button", {
    className: "workspace-secondary",
    disabled: busy,
    onClick: () => mutate(`/api/research/amendment/${job.id}/decide`, {
      action: 'revert'
    })
  }, "Restore thesis from before these edits"), changes.length > 0 && job.status === 'awaiting_approval' && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Applying saves only checked fields, preserves the original thesis in the proposal history, and rejects the proposal if your saved thesis has changed since comparison."), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    disabled: busy || !accepted.length,
    onClick: () => mutate(`/api/research/amendment/${job.id}/decide`, {
      action: 'apply',
      acceptedIds: accepted
    })
  }, "Apply ", accepted.length, " selected edit", accepted.length === 1 ? '' : 's')))));
}