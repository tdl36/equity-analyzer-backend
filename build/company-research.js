import * as React from 'react';
import { researchProgress, researchDocument, researchCaseDraft } from './company-research-model.mjs';
export function CompanyResearch({
  api,
  ticker,
  revision,
  disabled,
  onUpdate,
  onDraft
}) {
  var [inventory, setInventory] = React.useState(null),
    [error, setError] = React.useState(''),
    [notice, setNotice] = React.useState('');
  var [selected, setSelected] = React.useState(''),
    [run, setRun] = React.useState(null),
    [files, setFiles] = React.useState([]),
    [confirmed, setConfirmed] = React.useState(false),
    [ack, setAck] = React.useState(false),
    [busy, setBusy] = React.useState(false),
    [view, setView] = React.useState('snapshot'),
    [refresh, setRefresh] = React.useState(0);
  var pending = React.useRef(null),
    mutation = React.useRef(false),
    alive = React.useRef(true);
  React.useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  var json = async (path, options = {}) => {
    var res = await fetch(api + path, {
      ...options,
      signal: AbortSignal.timeout(30000)
    });
    var data = await res.json();
    if (!res.ok) throw Error(data.error || 'Research unavailable');
    return data;
  };
  React.useEffect(() => {
    var live = true,
      inflight = false;
    var load = async () => {
      if (inflight) return;
      inflight = true;
      try {
        var d = await json('/api/research/company/' + encodeURIComponent(ticker));
        if (live) {
          setInventory(d);
          setError('');
          setSelected(old => old || d.runs[0]?.id || '');
        }
      } catch (e) {
        if (live) setError(e.message);
      } finally {
        inflight = false;
      }
    };
    load();
    var timer = setInterval(load, 15000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [api, ticker, refresh]);
  React.useEffect(() => {
    var live = true,
      inflight = false;
    setRun(null);
    setAck(false);
    if (!selected) return;
    var load = async () => {
      if (inflight) return;
      inflight = true;
      try {
        var d = await json('/api/research/company-run/' + encodeURIComponent(selected));
        if (live) setRun(d);
      } catch (e) {
        if (live) setNotice(e.message);
      } finally {
        inflight = false;
      }
    };
    load();
    var timer = setInterval(load, 5000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [api, selected, refresh]);
  var key = () => {
    try {
      return localStorage.getItem('equity_analyzer_api_key') || '';
    } catch {
      return '';
    }
  };
  var mutate = async (path, payload, success) => {
    if (mutation.current) return;
    mutation.current = true;
    setBusy(true);
    setNotice('Submitting…');
    try {
      var d = await json(path, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          ...payload,
          apiKey: key()
        })
      });
      if (alive.current) {
        success?.(d);
        setRefresh(x => x + 1);
        setNotice('Request saved. Progress and partial work appear below.');
      }
    } catch (e) {
      if (alive.current) setNotice(e.message + ' Check saved runs before retrying; unchanged submissions reuse the same request.');
    } finally {
      mutation.current = false;
      if (alive.current) setBusy(false);
    }
  };
  var start = () => {
    var input = {
      filenames: [...files].sort(),
      revision,
      confirmed
    };
    var signature = JSON.stringify(input);
    if (pending.current?.signature !== signature) pending.current = {
      signature,
      payload: {
        ...input,
        requestId: crypto.randomUUID()
      }
    };
    mutate('/api/research/company/' + ticker, pending.current.payload, d => {
      setSelected(d.id);
      pending.current = null;
    });
  };
  var stats = researchProgress(run),
    sourceById = new Map((run?.sources || []).map(s => [s.id, s]));
  var download = compact => {
    var url = URL.createObjectURL(new Blob([researchDocument(run, compact)], {
      type: 'text/html;charset=utf-8'
    }));
    var a = document.createElement('a');
    a.href = url;
    a.download = `${ticker}-research-${run.id}${compact ? '-map' : ''}.html`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  var active = inventory?.runs.some(r => ['queued', 'running', 'attention'].includes(r.status));
  return /*#__PURE__*/React.createElement("section", {
    className: "company-research",
    "aria-label": "Source-backed company research"
  }, /*#__PURE__*/React.createElement("style", null, `.company-research{font-family:Calibri,sans-serif;background:#fff;color:#000;padding:24px;border-radius:12px}.company-research *{color:#000!important;overflow-wrap:anywhere}.company-research button,.company-research select{background:#f4f4ef!important;border:1px solid #aaa;border-radius:6px;padding:10px;max-width:100%}.company-research button:disabled{opacity:.5}.company-research h3{font-size:26px;margin:12px 0}.company-research h4{font-size:20px;font-weight:bold;margin:16px 0}.company-research p{margin:12px 0;white-space:pre-wrap}.company-research label{display:block;margin:12px 0}.company-research select{width:100%}.company-research article{padding:18px;border:1px solid #ccc;border-radius:8px;margin:16px 0}.company-research blockquote{padding:12px;border-left:3px solid #aaa}.company-research nav{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0}.company-research input{margin-right:10px}.company-research summary{cursor:pointer;font-weight:700;margin:16px 0}`), /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, ticker, " / SOURCE-BACKED RESEARCH"), /*#__PURE__*/React.createElement("h3", null, "Build the research. Preserve the evidence."), /*#__PURE__*/React.createElement("p", null, "Deep Research uses selected originals already stored in Charlie. Each revision freezes its sources and case baseline, covers 22 sections, and reviews generated claims. It does not search the web or collect new documents."), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error, " ", /*#__PURE__*/React.createElement("button", {
    onClick: () => setRefresh(x => x + 1)
  }, "Reload research")), /*#__PURE__*/React.createElement("details", {
    open: !inventory?.runs.length
  }, /*#__PURE__*/React.createElement("summary", null, "Prepare a new Deep Research revision"), /*#__PURE__*/React.createElement("p", null, "Baseline: saved case R", revision, !revision ? ' · initiation, no saved case yet' : '', ". Confirm the issuer in the originals; filing under a ticker alone does not establish identity."), /*#__PURE__*/React.createElement("fieldset", {
    disabled: disabled || busy || !!error || !inventory || active
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      maxHeight: 300,
      overflowY: 'auto'
    }
  }, inventory?.documents.map(d => /*#__PURE__*/React.createElement("label", {
    key: d.filename
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: files.includes(d.filename),
    disabled: !d.eligible || !files.includes(d.filename) && files.length >= 8,
    onChange: e => {
      setFiles(old => e.target.checked ? [...old, d.filename] : old.filter(n => n !== d.filename));
      setConfirmed(false);
    }
  }), d.filename, !d.eligible ? ' · restricted for AI use' : ''))), !inventory?.documents.length && /*#__PURE__*/React.createElement("p", null, "No stored originals found. Import permitted sources through Charlie\u2019s collection workflow."), /*#__PURE__*/React.createElement("p", null, files.length, " / 8 sources selected \xB7 combined readable text limit: 160,000 characters. Missing coverage is recorded explicitly."), /*#__PURE__*/React.createElement("label", null, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: confirmed,
    onChange: e => setConfirmed(e.target.checked)
  }), "I checked that these originals concern ", ticker, " and permit AI research."), /*#__PURE__*/React.createElement("button", {
    disabled: !files.length || !confirmed || files.some(f => !inventory?.documents.some(d => d.filename === f && d.eligible)),
    onClick: start
  }, "Start Deep Research"), /*#__PURE__*/React.createElement("p", null, "Uses model credits: up to six research calls and six review calls, plus explicitly acknowledged retries. Monthly budget checks apply before each call. No automatic thesis changes.")), disabled && /*#__PURE__*/React.createElement("p", null, "Save your case edits before starting research."), active && /*#__PURE__*/React.createElement("p", null, "An unfinished research revision already exists. Inspect it below, then resume or stop it before starting another.")), /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, notice), /*#__PURE__*/React.createElement("label", null, "Saved research revision", /*#__PURE__*/React.createElement("select", {
    "aria-label": "Saved research revision",
    value: selected,
    onChange: e => {
      setSelected(e.target.value);
      setNotice('');
    }
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Choose research"), inventory?.runs.map(r => /*#__PURE__*/React.createElement("option", {
    key: r.id,
    value: r.id
  }, new Date(r.created_at).toLocaleString(), " \xB7 ", r.status, " \xB7 case R", r.baseline.revision)))), run && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("strong", null, run.status.replaceAll('_', ' ')), " \xB7 ", stats.sections, "/22 sections saved \xB7 ", stats.stages, "/12 stages \xB7 ", stats.supported, " claims passed excerpt and model checks \xB7 ", stats.unresolved, " unresolved"), /*#__PURE__*/React.createElement("p", null, "Research ID ", run.id, " \xB7 ", run.model, " \xB7 case baseline R", run.baseline.revision), run.error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, run.error), run.baselineStale && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, "Your case has changed to R", run.currentRevision, ". This research retains baseline R", run.baseline.revision, "; review the differences before using it."), run.status !== 'complete' && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, "After an interruption, resume uses saved stages. A worker still processing a call will reject a second worker. Stopping takes effect after the current call; its usage may still be charged."), run.state?.inFlight && /*#__PURE__*/React.createElement("label", null, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: ack,
    onChange: e => setAck(e.target.checked)
  }), "I checked provider usage and accept that retrying the interrupted call may incur another charge."), /*#__PURE__*/React.createElement("nav", null, /*#__PURE__*/React.createElement("button", {
    disabled: busy || !!run.state?.inFlight && !ack,
    onClick: () => mutate('/api/research/company-run/' + run.id + '/resume', {
      acknowledgeRetry: ack
    })
  }, "Resume saved work"), /*#__PURE__*/React.createElement("button", {
    disabled: busy || run.cancel_requested,
    onClick: () => mutate('/api/research/company-run/' + run.id + '/stop', {})
  }, run.cancel_requested ? 'Stop requested' : 'Stop research'))), /*#__PURE__*/React.createElement("nav", {
    "aria-label": "Research views"
  }, [['snapshot', 'Snapshot'], ['full', 'Full research'], ['sources', 'Sources & baseline']].map(([id, label]) => /*#__PURE__*/React.createElement("button", {
    key: id,
    "aria-pressed": view === id,
    onClick: () => setView(id)
  }, label)), /*#__PURE__*/React.createElement("button", {
    disabled: !stats.sections,
    onClick: () => download(false)
  }, "Download research"), /*#__PURE__*/React.createElement("button", {
    disabled: run.status !== 'complete',
    onClick: () => download(true)
  }, "Download investment map"), /*#__PURE__*/React.createElement("button", {
    disabled: disabled || !revision || run.status !== 'complete' || run.baselineStale,
    onClick: () => onUpdate({
      id: run.id,
      filenames: run.input.filenames
    })
  }, "Update thesis \xB7 select evidence"), !revision && run.status === 'complete' && !run.baseline.revision && /*#__PURE__*/React.createElement("button", {
    disabled: disabled || !stats.supported,
    onClick: () => onDraft(researchCaseDraft(run, () => crypto.randomUUID()))
  }, "Prepare initial thesis draft \xB7 review before saving")), view === 'sources' ? /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("h4", null, "Frozen originals"), run.sources.map(s => /*#__PURE__*/React.createElement("article", {
    key: s.id
  }, /*#__PURE__*/React.createElement("strong", null, s.filename), /*#__PURE__*/React.createElement("p", null, "Original hash: ", s.originalHash), /*#__PURE__*/React.createElement("p", null, "Extraction hash: ", s.extractionHash), /*#__PURE__*/React.createElement("p", null, "Observed URL: ", s.sourceUrl || 'Not recorded'))), /*#__PURE__*/React.createElement("h4", null, "Case baseline R", run.baseline.revision), /*#__PURE__*/React.createElement("p", null, run.baseline.body.thesis || 'No thesis recorded at submission.'), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Full frozen case"), /*#__PURE__*/React.createElement("pre", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, JSON.stringify(run.baseline.body, null, 2)))) : /*#__PURE__*/React.createElement(React.Fragment, null, (run.state?.sections || []).filter(s => view === 'full' || ['summary', 'business', 'variant', 'risks', 'monitor', 'synthesis'].includes(s.id)).map(s => /*#__PURE__*/React.createElement("article", {
    key: s.id
  }, /*#__PURE__*/React.createElement("h4", null, s.title), !s.claims.length && /*#__PURE__*/React.createElement("p", null, "No supported conclusion generated from this source pack."), s.claims.map(c => /*#__PURE__*/React.createElement("section", {
    key: c.id
  }, /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("strong", null, c.basis.replaceAll('_', ' '), " \xB7 ", c.review === 'supported' && c.passageMatched ? 'Passage matched + model reviewed' : 'Needs review')), /*#__PURE__*/React.createElement("p", null, c.statement), /*#__PURE__*/React.createElement("p", null, c.reviewReason || 'Review pending'), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Supporting excerpts \xB7 ", c.evidence.length), c.evidence.map((e, i) => /*#__PURE__*/React.createElement("blockquote", {
    key: i
  }, /*#__PURE__*/React.createElement("strong", null, sourceById.get(e.sourceId)?.filename || 'Unknown source', " \xB7 ", e.matched ? 'Passage matched' : 'Unmatched'), /*#__PURE__*/React.createElement("p", null, e.excerpt)))))), /*#__PURE__*/React.createElement("h5", null, "Gaps and unresolved questions"), s.gaps.map((g, i) => /*#__PURE__*/React.createElement("p", {
    key: i
  }, g))))), /*#__PURE__*/React.createElement("p", null, "Model review is not investor approval. Financial calculations and consensus are not independently verified by this workflow. Investment-map exports omit unresolved claims; the full research retains them with warnings.")));
}