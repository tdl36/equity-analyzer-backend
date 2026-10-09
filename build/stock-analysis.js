function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
import * as React from 'react';
import { sections, label, supported, reportStats, thesisDraft, downloadText } from './stock-analysis-model.mjs';
function Field({
  value,
  path,
  citations,
  sources
}) {
  if (Array.isArray(value)) return value.length ? /*#__PURE__*/React.createElement("div", {
    className: "sa-rows"
  }, value.map((v, i) => /*#__PURE__*/React.createElement("article", {
    key: i
  }, /*#__PURE__*/React.createElement(Field, {
    value: v,
    path: `${path}/${i}`,
    citations: citations,
    sources: sources
  })))) : /*#__PURE__*/React.createElement("p", {
    className: "sa-gap"
  }, "Unavailable in this source pack.");
  if (value && typeof value === 'object') return /*#__PURE__*/React.createElement(React.Fragment, null, Object.entries(value).map(([k, v]) => /*#__PURE__*/React.createElement("div", {
    className: "sa-field",
    key: k
  }, /*#__PURE__*/React.createElement("h5", null, label(k)), /*#__PURE__*/React.createElement(Field, {
    value: v,
    path: `${path}/${k}`,
    citations: citations,
    sources: sources
  }))));
  var c = citations?.[path];
  var unavailable = !value || /^unavailable|^not available|^not recorded/i.test(value);
  return /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", null, value || 'Unavailable'), !unavailable && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("small", {
    className: supported(c) ? 'sa-evidence' : 'sa-warning'
  }, label(c?.basis || 'interpretation'), " \xB7 ", supported(c) ? 'Source matched · model reviewed' : 'Needs review'), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Inspect evidence"), /*#__PURE__*/React.createElement("p", null, c?.reviewReason || 'No source review is available.'), (c?.evidence || []).map((e, i) => /*#__PURE__*/React.createElement("blockquote", {
    key: i
  }, /*#__PURE__*/React.createElement("strong", null, sources?.find(s => s.id === e.sourceId)?.filename || 'Source unavailable', " \xB7 ", e.matched ? 'Passage matched' : 'Unmatched'), /*#__PURE__*/React.createElement("p", null, e.excerpt))), !c?.evidence?.length && /*#__PURE__*/React.createElement("p", null, "No field-level source citation."))));
}
export function StockAnalysis({
  api,
  ticker,
  revision,
  disabled,
  onDraft
}) {
  var [inventory, setInventory] = React.useState(null),
    [run, setRun] = React.useState(null),
    [selected, setSelected] = React.useState(''),
    [reload, setReload] = React.useState(0);
  var [mode, setMode] = React.useState('snapshot'),
    [horizon, setHorizon] = React.useState('12–24 months'),
    [question, setQuestion] = React.useState(''),
    [files, setFiles] = React.useState([]),
    [confirmed, setConfirmed] = React.useState(false),
    [priorId, setPriorId] = React.useState('');
  var [importState, setImportState] = React.useState(null);
  var [error, setError] = React.useState(''),
    [notice, setNotice] = React.useState(''),
    [busy, setBusy] = React.useState(false),
    [view, setView] = React.useState('report'),
    [section, setSection] = React.useState('summary'),
    [ack, setAck] = React.useState(false),
    [verified, setVerified] = React.useState(false),
    [visual, setVisual] = React.useState(null);
  var locked = React.useRef(false),
    pending = React.useRef(null),
    alive = React.useRef(true);
  React.useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  var json = async (path, body) => {
    var r = await fetch(api + '/api/research/' + path, {
      ...(body ? {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
      } : {}),
      signal: AbortSignal.timeout(30000)
    });
    var d;
    try {
      d = await r.json();
    } catch {
      throw Error('Charlie returned an unreadable response. Check saved reports before retrying.');
    }
    if (!r.ok) throw Error(d.error || 'Stock analysis unavailable');
    return d;
  };
  React.useEffect(() => {
    var live = true,
      inflight = false;
    var load = async () => {
      if (inflight) return;
      inflight = true;
      try {
        var d = await json('stock-analysis/' + encodeURIComponent(ticker));
        if (live) {
          setInventory(d);
          setError('');
          setSelected(s => s || d.runs[0]?.id || '');
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
  }, [ticker, api, reload]);
  React.useEffect(() => {
    var live = true;
    var load = async () => {
      try {
        var r = await fetch(api + '/api/collection/control', {
          signal: AbortSignal.timeout(15000)
        });
        if (!r.ok) throw Error();
        var d = await r.json();
        if (live) setImportState({
          rows: (d.snapshot?.originalImports || []).filter(x => x.ticker === ticker),
          updated: d.updatedAt
        });
      } catch {
        if (live) setImportState(null);
      }
    };
    load();
    var timer = setInterval(load, 30000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [ticker, api]);
  React.useEffect(() => {
    var live = true,
      inflight = false;
    setRun(null);
    setVisual(null);
    setVerified(false);
    setAck(false);
    if (!selected) return;
    var load = async () => {
      if (inflight) return;
      inflight = true;
      try {
        var d = await json('stock-analysis-run/' + selected);
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
  }, [selected, api, reload]);
  React.useEffect(() => {
    var live = true;
    if (run?.status === 'complete') json('stock-analysis-run/' + run.id + '/infographic').then(d => {
      if (live && d.html) setVisual(d);
    }).catch(e => {
      if (live) setNotice('Saved infographic could not be loaded: ' + e.message);
    });
    return () => {
      live = false;
    };
  }, [run?.id, run?.status]);
  var mutate = async fn => {
    if (locked.current) return;
    locked.current = true;
    setBusy(true);
    setNotice('');
    try {
      await fn();
    } catch (e) {
      if (alive.current) setNotice(e.message);
    } finally {
      locked.current = false;
      if (alive.current) setBusy(false);
    }
  };
  var key = () => {
    try {
      return localStorage.getItem('equity_analyzer_api_key') || '';
    } catch {
      return '';
    }
  };
  var start = () => mutate(async () => {
    var input = {
      filenames: [...files].sort(),
      revision,
      confirmed,
      mode,
      horizon,
      question,
      priorId
    };
    var signature = JSON.stringify(input);
    if (pending.current?.signature !== signature) pending.current = {
      signature,
      body: {
        ...input,
        requestId: crypto.randomUUID()
      }
    };
    var d = await json('stock-analysis/' + ticker, {
      ...pending.current.body,
      apiKey: key()
    });
    if (alive.current) {
      pending.current = null;
      setSelected(d.id);
      setReload(x => x + 1);
      setNotice('Analysis saved to the queue. You can return to it later.');
    }
  });
  var control = action => mutate(async () => {
    await json(`stock-analysis-run/${run.id}/${action}`, {
      acknowledgeRetry: ack,
      apiKey: key()
    });
    if (alive.current) {
      setReload(x => x + 1);
      setNotice(action === 'stop' ? 'Stop requested. Saved stages are retained.' : 'Resume requested.');
    }
  });
  var unfinished = inventory?.runs.some(r => ['queued', 'running', 'attention'].includes(r.status));
  var stats = reportStats(run);
  var sourceProps = {
    citations: run?.state?.citations || {},
    sources: run?.sources || []
  };
  return /*#__PURE__*/React.createElement("section", {
    className: "stock-analysis",
    "aria-label": "Stock analysis"
  }, /*#__PURE__*/React.createElement("style", null, styles), /*#__PURE__*/React.createElement("header", {
    className: "sa-heading"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("small", null, "CHARLIE / STOCK RESEARCH STUDIO"), /*#__PURE__*/React.createElement("h3", null, "Understand the business.", /*#__PURE__*/React.createElement("br", null), "Test the investment case."), /*#__PURE__*/React.createElement("p", null, "Structured research, a source-linked visual, and a record of what changed.")), /*#__PURE__*/React.createElement("div", {
    className: "sa-company"
  }, /*#__PURE__*/React.createElement("strong", null, ticker), /*#__PURE__*/React.createElement("span", null, "Saved thesis R", revision))), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error, " ", /*#__PURE__*/React.createElement("button", {
    onClick: () => setReload(x => x + 1)
  }, "Reload")), /*#__PURE__*/React.createElement("details", {
    className: "sa-prepare"
  }, /*#__PURE__*/React.createElement("summary", null, "Originals arriving from AlphaSense"), /*#__PURE__*/React.createElement("p", null, "Eligible originals saved by the collector upload automatically while the Mac agent is running and connected. Each upload verifies the original file. Importing does not start research."), importState?.updated ? /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("small", null, "Last Mac update: ", new Date(importState.updated).toLocaleString(), ". This is the last reported state; an offline Mac cannot send new updates."), importState.rows.length ? importState.rows.map(x => /*#__PURE__*/React.createElement("article", {
    key: x.document
  }, /*#__PURE__*/React.createElement("strong", null, x.filename), /*#__PURE__*/React.createElement("p", null, {
    queued: 'Waiting for Mac upload',
    uploading: 'Uploading and verifying',
    imported: 'Original verified in Charlie',
    attention: 'Import needs attention — retry scheduled'
  }[x.status] || x.status), x.issue && /*#__PURE__*/React.createElement("p", null, x.issue))) : /*#__PURE__*/React.createElement("p", null, "No automatic imports reported for this company yet.")) : /*#__PURE__*/React.createElement("p", null, "Import status unavailable. Check the Mac agent and connection; this does not mean there are no queued originals.")), /*#__PURE__*/React.createElement("details", {
    className: "sa-prepare",
    open: !inventory?.runs.length
  }, /*#__PURE__*/React.createElement("summary", null, "Prepare an analysis"), /*#__PURE__*/React.createElement("fieldset", {
    disabled: disabled || busy || !inventory || !!error || unfinished
  }, /*#__PURE__*/React.createElement("div", {
    className: "sa-form-grid"
  }, /*#__PURE__*/React.createElement("label", null, "Research mode", /*#__PURE__*/React.createElement("select", {
    value: mode,
    onChange: e => setMode(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: "snapshot"
  }, "Snapshot"), /*#__PURE__*/React.createElement("option", {
    value: "deep"
  }, "Deep research"), /*#__PURE__*/React.createElement("option", {
    value: "update"
  }, "Update thesis"))), /*#__PURE__*/React.createElement("label", null, "Investment horizon", /*#__PURE__*/React.createElement("input", {
    value: horizon,
    onChange: e => setHorizon(e.target.value),
    maxLength: 100
  }))), /*#__PURE__*/React.createElement("label", null, "Existing thesis or research question ", /*#__PURE__*/React.createElement("span", null, "Optional"), /*#__PURE__*/React.createElement("textarea", {
    value: question,
    onChange: e => setQuestion(e.target.value),
    rows: 3,
    maxLength: 12000,
    placeholder: "What must be true for this business to outperform expectations?"
  })), /*#__PURE__*/React.createElement("label", null, "Compare with an earlier report", /*#__PURE__*/React.createElement("select", {
    value: priorId,
    onChange: e => setPriorId(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Latest completed report, or saved thesis"), inventory?.runs.filter(r => r.status === 'complete').map(r => /*#__PURE__*/React.createElement("option", {
    key: r.id,
    value: r.id
  }, new Date(r.created_at).toLocaleString(), " \xB7 ", r.id.slice(0, 8))))), /*#__PURE__*/React.createElement("h4", null, "Originals for ", ticker), /*#__PURE__*/React.createElement("p", null, "Choose up to eight permitted originals already stored in Charlie. The report uses these documents; missing consensus, price history and financial periods stay unavailable."), /*#__PURE__*/React.createElement("div", {
    className: "sa-source-list"
  }, inventory?.documents.map(d => /*#__PURE__*/React.createElement("label", {
    className: "sa-check",
    key: d.filename
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: files.includes(d.filename),
    disabled: !d.eligible || files.length >= 8 && !files.includes(d.filename),
    onChange: e => {
      setFiles(old => e.target.checked ? [...old, d.filename] : old.filter(n => n !== d.filename));
      setConfirmed(false);
    }
  }), d.filename, !d.eligible ? ' · restricted for AI use' : ''))), !inventory?.documents.length && /*#__PURE__*/React.createElement("p", null, "No originals found. Add permitted company documents through Charlie\u2019s existing collection or import workflow."), /*#__PURE__*/React.createElement("label", {
    className: "sa-check"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: confirmed,
    onChange: e => setConfirmed(e.target.checked)
  }), "I checked that these originals concern ", ticker, " and permit AI research."), /*#__PURE__*/React.createElement("button", {
    className: "sa-primary",
    disabled: !files.length || !confirmed || !horizon.trim() || files.some(n => !inventory?.documents.some(d => d.filename === n && d.eligible)),
    onClick: start
  }, "Generate analysis"), /*#__PURE__*/React.createElement("small", null, files.length, "/8 originals \xB7 up to six research and six source-review calls. Uses Charlie\u2019s configured research model and budget. Snapshot produces shorter answers; all modes retain source checks."))), unfinished && /*#__PURE__*/React.createElement("p", null, "One analysis is unfinished. Open it below to inspect progress, stop or resume."), /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, notice), /*#__PURE__*/React.createElement("label", null, "Saved analyses", /*#__PURE__*/React.createElement("select", {
    value: selected,
    onChange: e => {
      setSelected(e.target.value);
      setNotice('');
    }
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Choose a report"), inventory?.runs.map(r => /*#__PURE__*/React.createElement("option", {
    key: r.id,
    value: r.id
  }, new Date(r.created_at).toLocaleString(), " \xB7 ", r.status, " \xB7 ", r.id.slice(0, 8))))), run && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "sa-report-heading"
  }, /*#__PURE__*/React.createElement("h4", null, label(run.input.mode || 'deep'), " \xB7 ", run.status), /*#__PURE__*/React.createElement("p", null, stats.stages, "/12 stages \xB7 ", stats.supported, " source-supported fields \xB7 ", stats.unresolved, " fields need review"), /*#__PURE__*/React.createElement("small", null, "Report ", run.id, " \xB7 Company ID ", run.input.companyId, " \xB7 ", run.model)), run.error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, run.error), run.baselineStale && /*#__PURE__*/React.createElement("p", {
    className: "sa-warning"
  }, "The saved thesis has changed since this report started. Its frozen baseline remains R", run.baseline.revision, ". Start a new analysis before preparing a thesis draft."), run.status !== 'complete' && /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("label", {
    className: "sa-check"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: ack,
    onChange: e => setAck(e.target.checked)
  }), "If a previous call\u2019s outcome is unknown, I checked usage and acknowledge a retry may incur another charge."), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => control('resume')
  }, "Resume saved stages"), /*#__PURE__*/React.createElement("button", {
    disabled: busy || run.status === 'cancelled',
    onClick: () => control('stop')
  }, "Stop analysis"), /*#__PURE__*/React.createElement("p", null, "After a restart, resume explicitly. A currently active worker will reject a second start.")), /*#__PURE__*/React.createElement("nav", {
    "aria-label": "Stock analysis views"
  }, [['report', 'Research report'], ['changes', 'What changed?'], ['sources', 'Sources & baseline'], ['visual', 'Infographic']].map(([id, title]) => /*#__PURE__*/React.createElement("button", {
    key: id,
    "aria-pressed": view === id,
    onClick: () => setView(id)
  }, title))), /*#__PURE__*/React.createElement("div", {
    className: "sa-actions"
  }, /*#__PURE__*/React.createElement("button", {
    disabled: busy || !run.state?.report,
    onClick: () => mutate(async () => {
      var d = await json(`stock-analysis-run/${run.id}/export`);
      downloadText(d.html, d.filename);
    })
  }, "Export report"), /*#__PURE__*/React.createElement("button", {
    onClick: () => downloadText(JSON.stringify(run, null, 2), `${ticker}-stock-analysis-${run.id}.json`, 'application/json')
  }, "Export data & provenance"), /*#__PURE__*/React.createElement("button", {
    disabled: busy || disabled || run.status !== 'complete' || run.baselineStale,
    onClick: () => {
      try {
        onDraft(thesisDraft(run));
      } catch (e) {
        setNotice(e.message);
      }
    }
  }, "Prepare thesis draft for review")), view === 'report' && /*#__PURE__*/React.createElement("div", {
    className: "sa-report"
  }, /*#__PURE__*/React.createElement("nav", {
    "aria-label": "Report sections"
  }, sections.map(([id, title]) => /*#__PURE__*/React.createElement("button", {
    key: id,
    "aria-pressed": section === id,
    onClick: () => setSection(id)
  }, title))), /*#__PURE__*/React.createElement("div", {
    className: "sa-report-body"
  }, /*#__PURE__*/React.createElement("h4", null, sections.find(s => s[0] === section)?.[1]), run.state?.report?.[section] !== undefined ? /*#__PURE__*/React.createElement(Field, _extends({
    value: run.state.report[section],
    path: '/' + section
  }, sourceProps)) : /*#__PURE__*/React.createElement("p", null, "This section has not been generated yet. Completed stages are saved as they arrive."))), view === 'changes' && /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h4", null, "Evidence changes and changes in interpretation"), /*#__PURE__*/React.createElement("p", null, run.state?.comparison?.note || 'Comparison becomes available when the report completes.'), run.state?.comparison?.sourceChanges?.map((c, i) => /*#__PURE__*/React.createElement("p", {
    key: i
  }, c.change, ": ", c.filename)), run.state?.comparison?.sections?.map(c => /*#__PURE__*/React.createElement("article", {
    className: "sa-comparison",
    key: c.section
  }, /*#__PURE__*/React.createElement("h4", null, label(c.section), " \xB7 ", label(c.kind)), /*#__PURE__*/React.createElement("p", null, c.note), /*#__PURE__*/React.createElement("div", {
    className: "sa-form-grid"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h5", null, "Earlier report"), /*#__PURE__*/React.createElement("pre", null, JSON.stringify(c.before, null, 2))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h5", null, "This report"), /*#__PURE__*/React.createElement("pre", null, JSON.stringify(c.after, null, 2)))))), /*#__PURE__*/React.createElement("h4", null, "Frozen investor thesis \xB7 R", run.baseline.revision), /*#__PURE__*/React.createElement("p", null, run.baseline.body.thesis || 'No saved investor thesis at submission.'), /*#__PURE__*/React.createElement("p", null, "Changes in source selection require review. No thesis breaker is automatically marked as triggered, and no thesis is automatically approved.")), view === 'sources' && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("h4", null, "Frozen originals"), run.sources.map(s => /*#__PURE__*/React.createElement("article", {
    key: s.id
  }, /*#__PURE__*/React.createElement("h5", null, s.filename), /*#__PURE__*/React.createElement("p", null, s.sourceUrl || 'Observed URL unavailable'), /*#__PURE__*/React.createElement("small", null, "Original SHA-256: ", s.originalHash, /*#__PURE__*/React.createElement("br", null), "Extraction SHA-256: ", s.extractionHash))), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Frozen investor case \xB7 R", run.baseline.revision), /*#__PURE__*/React.createElement("pre", null, JSON.stringify(run.baseline.body, null, 2))), /*#__PURE__*/React.createElement("p", null, "Document text is evidence, not instructions. A matching passage and model review do not independently establish factual accuracy.")), view === 'visual' && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("h4", null, "From company overview to investment thesis"), /*#__PURE__*/React.createElement("p", null, "The visual preserves report wording, figures and periods. Unresolved fields are omitted. Comparable source-linked financial series become charts; unsupported series remain unavailable."), !visual && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("label", {
    className: "sa-check"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: verified,
    onChange: e => setVerified(e.target.checked)
  }), "I reviewed the included figures, accounting definitions and periods against the original passages."), /*#__PURE__*/React.createElement("button", {
    disabled: busy || !verified || run.status !== 'complete',
    onClick: () => mutate(async () => {
      var d = await json(`stock-analysis-run/${run.id}/infographic`, {
        verifiedFigures: true
      });
      if (alive.current) {
        setVisual(d);
        setNotice('Infographic saved in Charlie Studio and linked to this report.');
      }
    })
  }, "Generate infographic")), visual && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("button", {
    onClick: () => downloadText(visual.html, `${ticker}-investment-infographic-${run.id}.html`)
  }, "Export infographic \xB7 printable HTML"), /*#__PURE__*/React.createElement("p", null, "Saved in Studio \xB7 #", visual.id, ". Open the exported file to print or save as PDF."), /*#__PURE__*/React.createElement("iframe", {
    title: `${ticker} investment infographic`,
    sandbox: "",
    srcDoc: visual.html,
    style: {
      width: '100%',
      height: 850,
      border: '1px solid #ccd7de',
      background: 'white'
    }
  }))), /*#__PURE__*/React.createElement("p", {
    className: "sa-footnote"
  }, "Draft decision support. Facts, guidance, estimates and interpretations remain distinct. Source-pack coverage is not exhaustive current market coverage. Review before saving a thesis change.")));
}
var styles = `.stock-analysis{font:16px/1.5 Calibri,sans-serif;color:#000;background:#fff;padding:28px;border:1px solid #d8dedd;border-radius:12px}.stock-analysis *{color:#000!important;box-sizing:border-box;overflow-wrap:anywhere}.stock-analysis h3{font-size:34px;font-weight:700;line-height:1.12;margin:12px 0}.stock-analysis h4{font-size:22px;font-weight:700;margin:16px 0 8px}.stock-analysis h5{font-size:17px;font-weight:700;margin:12px 0 6px}.stock-analysis p{margin:8px 0 14px;white-space:pre-wrap}.stock-analysis small{display:block;font-size:13px}.sa-heading{display:flex;justify-content:space-between;gap:24px;border-bottom:1px solid #ccd7de;padding-bottom:24px;margin-bottom:24px}.sa-heading small{letter-spacing:.12em}.sa-company{padding:18px;background:#edf4f6;border-radius:8px;align-self:flex-start;min-width:130px}.sa-company strong{display:block;font-size:30px}.sa-company span{font-size:13px}.stock-analysis label{display:block;margin:14px 0}.stock-analysis input:not([type=checkbox]),.stock-analysis select,.stock-analysis textarea{width:100%;font:inherit;background:#fff!important;border:1px solid #aebcbe;border-radius:5px;padding:10px;margin-top:6px}.stock-analysis button{font:inherit;background:#eef2f2!important;border:1px solid #aebcbe;padding:9px 14px;border-radius:5px;cursor:pointer;max-width:100%}.stock-analysis button[aria-pressed=true],.stock-analysis .sa-primary{background:#d5e8ed!important;border-color:#718e99;font-weight:700}.stock-analysis button:disabled{opacity:.5;cursor:default}.stock-analysis button:focus-visible,.stock-analysis input:focus-visible{outline:2px solid #000;outline-offset:2px}.stock-analysis nav,.sa-actions{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}.sa-form-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.sa-source-list{max-height:270px;overflow:auto;border-top:1px solid #ddd;border-bottom:1px solid #ddd}.stock-analysis .sa-check{display:flex;align-items:flex-start;gap:10px}.sa-check input{margin-top:5px;flex-shrink:0}.stock-analysis summary{cursor:pointer;font-weight:700;padding:12px 0}.sa-prepare{background:#f7f8f6;border:1px solid #dce2de;padding:16px;border-radius:8px}.sa-report{display:grid;grid-template-columns:200px minmax(0,1fr);gap:24px;border-top:1px solid #ccd7de;padding-top:18px}.sa-report>nav{display:flex;flex-direction:column;align-self:start}.sa-report>nav button{text-align:left}.sa-rows{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,250px),1fr));gap:14px}.stock-analysis article{border:1px solid #dce2de;background:#f8faf9;padding:18px;border-radius:6px;margin:12px 0}.sa-field{margin-bottom:20px}.sa-evidence,.sa-warning{background:#ecf4ed;padding:6px 10px;border-radius:4px}.sa-warning{background:#fff1d9}.stock-analysis blockquote{border-left:3px solid #bccbd0;padding:12px;margin:12px 0;background:#fff}.stock-analysis pre{white-space:pre-wrap;font:14px Calibri,sans-serif;max-width:100%;overflow-wrap:anywhere}.sa-footnote{border-top:1px solid #ddd;padding-top:20px;font-size:14px}.stock-analysis fieldset{min-width:0}.sa-report-body{min-width:0}@media(max-width:750px){.stock-analysis{padding:14px}.sa-heading{display:block}.sa-heading h3{font-size:28px}.sa-company{display:inline-block;margin-top:12px}.sa-form-grid,.sa-report{grid-template-columns:1fr}.sa-report>nav{flex-direction:row;flex-wrap:nowrap;overflow-x:auto;max-width:100%}.sa-report>nav button{font-size:14px;flex-shrink:0}.sa-rows{grid-template-columns:1fr}}`;