import { financialWorkbook, saveFinancialWorkbook } from './financial-workbook-download.mjs';
import * as React from 'react';
import { parseTimestamp, routeHash } from './workspace-model.mjs';
import { downloadText } from './stock-analysis-model.mjs';
var outputs = [['summary_note', 'Summary note'], ['stock_summary', 'Stock summary'], ['thesis', 'Thesis draft / update'], ['visual', 'Visual one-pager draft'], ['meeting_prep', 'Meeting questions from this research']];
var date = d => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
export function ResearchAssignment({
  api,
  ticker,
  onReport,
  disabled,
  focusId = '',
  onClearFocus,
  origin = null
}) {
  var [since, setSince] = React.useState(() => {
      var d = new Date();
      d.setDate(d.getDate() - 69);
      return date(d);
    }),
    [until, setUntil] = React.useState(() => date(new Date())),
    [chosen, setChosen] = React.useState(origin ? ['visual'] : outputs.filter(x => x[0] !== 'meeting_prep').map(x => x[0])),
    [horizon, setHorizon] = React.useState('12–24 months'),
    [instruction, setInstruction] = React.useState(''),
    [visualFundamentals, setVisualFundamentals] = React.useState(true),
    [meetingDate, setMeetingDate] = React.useState(''),
    [meetingType, setMeetingType] = React.useState('management');
  var [jobs, setJobs] = React.useState([]),
    [loaded, setLoaded] = React.useState(false),
    [busy, setBusy] = React.useState(false),
    [message, setMessage] = React.useState(''),
    [error, setError] = React.useState(''),
    [ack, setAck] = React.useState(false),
    [preview, setPreview] = React.useState(null);
  var listPath = 'assignments?ticker=' + encodeURIComponent(ticker) + (focusId ? '&requestId=' + encodeURIComponent(focusId) : '');
  var locked = React.useRef(false),
    pending = React.useRef(null),
    generation = React.useRef(0);
  var request = async (path, body, method = 'POST') => {
    var r = await fetch(api + '/api/research/' + path, {
      ...(body ? {
        method,
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
      throw Error('Response unavailable. Your request identity is retained; check saved assignments before retrying.');
    }
    if (!r.ok) throw Error(d.error || 'Research assignment unavailable');
    return d;
  };
  React.useEffect(() => {
    if (origin) return;
    var live = true;
    request('assignment-defaults').then(d => {
      if (live) {
        setChosen(d.outputs);
        setHorizon(d.horizon);
      }
    }).catch(() => {});
    return () => {
      live = false;
    };
  }, [api]);
  React.useEffect(() => {
    var live = true,
      inflight = false;
    generation.current++;
    setJobs([]);
    setLoaded(false);
    setAck(false);
    setPreview(null);
    var load = async () => {
      if (inflight) return;
      inflight = true;
      try {
        var d = await request(listPath);
        if (live) {
          setJobs(d.assignments);
          setLoaded(true);
          setError('');
        }
      } catch (e) {
        if (live) setError(e.message);
      } finally {
        inflight = false;
      }
    };
    load();
    var timer = setInterval(load, 6000);
    return () => {
      live = false;
      generation.current++;
      clearInterval(timer);
    };
  }, [api, ticker, focusId]);
  var mutate = async fn => {
    if (locked.current) return;
    var g = generation.current;
    locked.current = true;
    setBusy(true);
    setMessage('');
    try {
      var msg = await fn();
      if (g === generation.current) {
        var d = await request(listPath);
        if (g === generation.current) {
          setJobs(d.assignments);
          setMessage(msg);
        }
      }
    } catch (e) {
      if (g === generation.current) setMessage(e.message);
    } finally {
      locked.current = false;
      setBusy(false);
    }
  };
  var start = () => mutate(async () => {
    var p = {
      ticker,
      since,
      until,
      ...(origin ? {
        origin
      } : {}),
      outputs: [...chosen].sort(),
      horizon,
      instruction,
      visualFundamentals,
      ...(chosen.includes('meeting_prep') ? {
        meeting: {
          date: meetingDate,
          type: meetingType
        }
      } : {})
    };
    var signature = JSON.stringify(p);
    if (pending.current?.signature !== signature) pending.current = {
      signature,
      body: {
        ...p,
        requestId: crypto.randomUUID()
      }
    };
    await request('assignments', pending.current.body);
    pending.current = null;
    return 'Assignment saved. Charlie will collect originals and prepare your requested drafts. You can close this page.';
  });
  var visibleJobs = jobs.filter(j => (!focusId || j.id === focusId) && j.ticker === ticker);
  var unfinished = jobs.some(j => ['queued', 'running', 'attention'].includes(j.status));
  return /*#__PURE__*/React.createElement("section", {
    className: "sa-prepare",
    "aria-label": "End-to-end research"
  }, /*#__PURE__*/React.createElement("style", null, `@media(max-width:650px){.sa-visual-preview{position:fixed;inset:58px 0 64px;background:white;z-index:80;padding:10px!important;display:flex;flex-direction:column}.sa-visual-preview iframe{flex:1;min-height:0;height:100%!important}.sa-visual-preview button{align-self:flex-start;margin:0!important}}`), /*#__PURE__*/React.createElement("h4", null, origin ? 'Refresh this one-pager from recent sources' : 'One research assignment'), origin && /*#__PURE__*/React.createElement("p", null, "Starting from ", origin.parentId ? 'accepted edition ' + origin.parentId.slice(0, 8) : 'original report ' + origin.runId.slice(0, 8), ". Creates a separate draft and retains this version. Prior editorial corrections are not automatically carried into the new report; compare and review the resulting draft."), /*#__PURE__*/React.createElement("p", null, "For ", /*#__PURE__*/React.createElement("strong", null, ticker), ", collect new originals and prepare the outputs you choose."), focusId && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Opened saved assignment ", focusId.slice(0, 8), ". ", /*#__PURE__*/React.createElement("button", {
    onClick: onClearFocus
  }, "Show all assignments for ", ticker)), /*#__PURE__*/React.createElement("details", {
    open: !focusId
  }, /*#__PURE__*/React.createElement("summary", null, "Prepare a new research assignment"), /*#__PURE__*/React.createElement("fieldset", {
    disabled: disabled || busy || unfinished || !!error || !!focusId
  }, /*#__PURE__*/React.createElement("div", {
    className: "sa-form-grid"
  }, /*#__PURE__*/React.createElement("label", null, "From", /*#__PURE__*/React.createElement("input", {
    "aria-label": "Research from",
    type: "date",
    value: since,
    max: until,
    onChange: e => setSince(e.target.value)
  })), /*#__PURE__*/React.createElement("label", null, "Through", /*#__PURE__*/React.createElement("input", {
    "aria-label": "Research through",
    type: "date",
    value: until,
    min: since,
    max: date(new Date()),
    onChange: e => setUntil(e.target.value)
  }))), /*#__PURE__*/React.createElement("div", {
    className: "sa-form-grid"
  }, outputs.map(([id, title]) => /*#__PURE__*/React.createElement("label", {
    className: "sa-check",
    key: id
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: chosen.includes(id),
    disabled: !!origin && id === 'visual',
    onChange: e => setChosen(p => e.target.checked ? [...p, id] : p.filter(x => x !== id))
  }), title))), chosen.includes('meeting_prep') && /*#__PURE__*/React.createElement("div", {
    className: "sa-form-grid"
  }, /*#__PURE__*/React.createElement("label", null, "Meeting date", /*#__PURE__*/React.createElement("input", {
    type: "date",
    value: meetingDate,
    onChange: e => setMeetingDate(e.target.value)
  })), /*#__PURE__*/React.createElement("label", null, "Meeting type", /*#__PURE__*/React.createElement("select", {
    value: meetingType,
    onChange: e => setMeetingType(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: "management"
  }, "Management"), /*#__PURE__*/React.createElement("option", {
    value: "expert"
  }, "Expert"), /*#__PURE__*/React.createElement("option", {
    value: "conference"
  }, "Conference"), /*#__PURE__*/React.createElement("option", {
    value: "other"
  }, "Other"))), /*#__PURE__*/React.createElement("p", null, "Uses source-matched questions from the same report and originals. No separate generation call. Review their premises before the meeting.")), chosen.includes('visual') && /*#__PURE__*/React.createElement("label", {
    className: "sa-check"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: visualFundamentals,
    onChange: e => setVisualFundamentals(e.target.checked)
  }), "Add historical public filings and dated market data for the visual. Recent events still use your selected date range."), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Research focus and defaults"), /*#__PURE__*/React.createElement("label", null, "Investment horizon", /*#__PURE__*/React.createElement("input", {
    value: horizon,
    maxLength: 100,
    onChange: e => setHorizon(e.target.value)
  })), /*#__PURE__*/React.createElement("label", null, "Focus or questions (optional)", /*#__PURE__*/React.createElement("textarea", {
    value: instruction,
    maxLength: 1800,
    onChange: e => setInstruction(e.target.value),
    rows: 3
  })), /*#__PURE__*/React.createElement("button", {
    disabled: !chosen.length,
    onClick: () => mutate(async () => {
      await request('assignment-defaults', {
        outputs: chosen,
        horizon
      }, 'PUT');
      return 'Output choices and horizon saved for your next assignment.';
    })
  }, "Save my defaults")), /*#__PURE__*/React.createElement("button", {
    className: "sa-primary",
    disabled: !chosen.length || !since || !until || !horizon.trim() || chosen.includes('meeting_prep') && !meetingDate,
    onClick: start
  }, origin ? 'Start source refresh for' : 'Start research for', " ", ticker))), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "One shared research run: up to eight selected originals and twelve model calls within your configured budget. Starting authorizes these calls; explicitly resumed failed stages may add charges. Draft outputs need investor review; a thesis update is never applied automatically.")), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "How automatic collection works"), /*#__PURE__*/React.createElement("p", null, "Charlie queues collection \u2192 Codex searches AlphaSense \u2192 originals are saved in iCloud STOCKS \u2192 your Mac imports and verifies them \u2192 Charlie researches and prepares the selected drafts. Keep your Mac awake, online, the agent running, and Codex with signed-in Chrome available. Collection starts on an available scheduled worker, usually the next 15-minute wake. Source restrictions, sign-in, source choices and uncertain paid-call outcomes can require your attention. The research uses the selected source pack; missing prices, consensus or financial periods remain gaps.")), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error), /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, message), focusId && !loaded && !error && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading the saved assignment\u2026"), focusId && loaded && !error && !visibleJobs.length && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "The requested assignment is not in the loaded results. Check the company and refresh; no other assignment is selected in its place."), visibleJobs.map(j => /*#__PURE__*/React.createElement("article", {
    key: j.id,
    style: {
      borderTop: '1px solid #d8dedd',
      paddingTop: 12,
      marginTop: 12
    }
  }, /*#__PURE__*/React.createElement("strong", null, j.input.since, "\u2013", j.input.until, " \xB7 ", j.status), /*#__PURE__*/React.createElement("p", null, j.result.step), j.input.origin && /*#__PURE__*/React.createElement("p", null, "Refresh from ", /*#__PURE__*/React.createElement("a", {
    href: routeHash('stockanalysis', j.ticker, 'stock-analysis', 'assignment:' + j.input.origin.assignmentId)
  }, "saved one-pager"), " \xB7 ", j.input.origin.parentId ? 'accepted edition ' + j.input.origin.parentId.slice(0, 8) : 'original report ' + j.input.origin.runId.slice(0, 8), ". Outputs are separate drafts for review."), j.result.delayWarning && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, j.result.delayWarning), /*#__PURE__*/React.createElement("small", null, "Saved update: ", parseTimestamp(j.updated_at)?.toLocaleString() || 'Unavailable', j.result.macReportedAt ? ' · Mac report: ' + parseTimestamp(j.result.macReportedAt)?.toLocaleString() || 'Unavailable' : ''), j.result.sources && /*#__PURE__*/React.createElement("p", null, j.result.sources.length, " originals verified \xB7 ", j.result.researchStages || 0, "/12 research stages"), j.error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, j.error), j.result.visualFundamentals?.gaps?.length > 0 && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Public-data gaps: ", j.result.visualFundamentals.gaps.join('; '), ". Retrying retrieves public data only; it does not rerun research. Your previous snapshot stays available if retrieval fails."), /*#__PURE__*/React.createElement("div", {
    className: "sa-actions"
  }, j.result.visualFundamentals && /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => mutate(async () => {
      saveFinancialWorkbook(await financialWorkbook(api, j.id), ticker, j.id);
      return 'Financial workbook exported from the saved snapshot. No data was refreshed.';
    })
  }, "Export financial history \xB7 Excel"), j.status === 'complete' && j.result.artifacts?.some(a => a.kind === 'visual') && (!j.result.visualFundamentals || j.result.visualFundamentals.version < 2 || j.result.visualFundamentals.gaps?.length > 0) && /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => mutate(async () => {
      await request(`assignments/${j.id}/visual-supplement`, {
        expectedRetrievedAt: j.result.visualFundamentals?.retrievedAt
      });
      setPreview(null);
      return 'Public financial history and market snapshot saved. Preview or export the illustrated version. No model calls were made.';
    })
  }, j.result.visualFundamentals?.gaps?.length ? 'Retry missing public data' : 'Add financial charts & market data'), j.result.artifacts?.some(a => a.kind === 'visual') && /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => mutate(async () => {
      var g = generation.current;
      var a = j.result.artifacts.find(a => a.kind === 'visual');
      var r = await fetch(api + a.url, {
        signal: AbortSignal.timeout(30000)
      });
      if (!r.ok) throw Error('Preview unavailable. Saved original remains available.');
      var html = await r.text();
      if (g === generation.current) setPreview({
        id: j.id,
        html
      });
      return 'Visual draft opened below.';
    })
  }, "Preview visual one-pager"), j.result.reportId && /*#__PURE__*/React.createElement("button", {
    onClick: () => onReport(j.result.reportId)
  }, "Open research report"), j.result.artifacts?.map(a => /*#__PURE__*/React.createElement("button", {
    key: a.kind,
    disabled: busy,
    onClick: () => mutate(async () => {
      var r = await fetch(api + a.url, {
        signal: AbortSignal.timeout(30000)
      });
      if (!r.ok) throw Error('Output could not be loaded. Saved output remains in the assignment.');
      downloadText(await r.text(), `${ticker}-${a.kind}-${j.id.slice(0, 8)}.html`);
      return 'Draft exported as printable HTML.';
    })
  }, "Export ", outputs.find(x => x[0] === a.kind)?.[1])), j.result.meetingId && /*#__PURE__*/React.createElement("a", {
    href: routeHash('meetingprep', ticker, '', 'meeting:' + j.result.meetingId)
  }, "Open meeting questions from this pack"), j.result.thesisDraftId && /*#__PURE__*/React.createElement("a", {
    href: `?thesisDraft=${j.result.thesisDraftId}#view=thesisimports&ticker=${encodeURIComponent(ticker)}`
  }, "Review thesis proposal")), preview?.id === j.id && /*#__PURE__*/React.createElement("div", {
    className: "sa-visual-preview"
  }, /*#__PURE__*/React.createElement("button", {
    onClick: () => setPreview(null)
  }, "Close preview"), /*#__PURE__*/React.createElement("iframe", {
    title: `${ticker} visual one-pager preview`,
    sandbox: "",
    srcDoc: preview.html,
    style: {
      width: '100%',
      height: '1100px',
      border: '1px solid #d8dedd',
      marginTop: 12
    }
  })), j.result.separateThesisProposal && /*#__PURE__*/React.createElement("aside", {
    "aria-label": "Separate thesis proposal",
    style: {
      background: "#f4f6f4",
      borderLeft: "3px solid #365b4c",
      padding: "16px 20px",
      margin: "16px 0",
      overflowWrap: "anywhere"
    }
  }, /*#__PURE__*/React.createElement("strong", null, "Rebuilt proposal available for review"), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("a", {
    href: `?thesisDraft=${encodeURIComponent(j.result.separateThesisProposal.id)}#view=thesisimports&ticker=${encodeURIComponent(ticker)}`
  }, "Review rebuilt thesis proposal")), /*#__PURE__*/React.createElement("p", null, "This separately prepared proposal references ", j.result.separateThesisProposal.sharedOriginals, " of this assignment\u2019s ", j.result.separateThesisProposal.assignmentOriginals, " originals and may use supplemental sources. Review its source register and current baseline before approving. The original automated research remains blocked; linking does not approve the thesis or certify its quality.")), j.status === 'attention' && j.result.manualThesisReviewRequired && !j.result.separateThesisProposal && /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("a", {
    href: routeHash('thesisimports', ticker)
  }, "Import & review thesis"), " \xB7 Review the saved report above before preparing a separate proposal. Your accepted thesis is unchanged."), j.status === 'attention' && !j.result.manualThesisReviewRequired && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("label", {
    className: "sa-check"
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: ack,
    onChange: e => setAck(e.target.checked)
  }), "If a paid call is uncertain, I checked usage and accept that retry may charge again."), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => mutate(async () => {
      await request(`assignments/${j.id}/resume`, {
        acknowledgeRetry: ack
      });
      return 'Resume requested; saved stages are retained.';
    })
  }, "Resume assignment"), /*#__PURE__*/React.createElement("a", {
    href: routeHash('desk', ticker, 'collection', j.result.refreshRequestId ? 'collection:' + j.result.refreshRequestId : '')
  }, "Collection and source choices")), ['queued', 'running', 'attention'].includes(j.status) && /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => mutate(async () => {
      await request(`assignments/${j.id}/stop`, {});
      return 'Stopped. Originals and saved drafts are retained; a call already running may finish.';
    })
  }, "Stop assignment"))));
}