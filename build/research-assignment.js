import * as React from 'react';
import { downloadText } from './stock-analysis-model.mjs';
var outputs = [['summary_note', 'Summary note'], ['stock_summary', 'Stock summary'], ['thesis', 'Thesis draft / update'], ['visual', 'Visual one-pager draft']];
var date = d => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
export function ResearchAssignment({
  api,
  ticker,
  onReport,
  disabled
}) {
  var [since, setSince] = React.useState(() => {
      var d = new Date();
      d.setDate(d.getDate() - 69);
      return date(d);
    }),
    [until, setUntil] = React.useState(() => date(new Date())),
    [chosen, setChosen] = React.useState(outputs.map(x => x[0])),
    [horizon, setHorizon] = React.useState('12–24 months'),
    [instruction, setInstruction] = React.useState('');
  var [jobs, setJobs] = React.useState([]),
    [busy, setBusy] = React.useState(false),
    [message, setMessage] = React.useState(''),
    [error, setError] = React.useState(''),
    [ack, setAck] = React.useState(false);
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
    setAck(false);
    var load = async () => {
      if (inflight) return;
      inflight = true;
      try {
        var d = await request('assignments?ticker=' + encodeURIComponent(ticker));
        if (live) {
          setJobs(d.assignments);
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
  }, [api, ticker]);
  var mutate = async fn => {
    if (locked.current) return;
    var g = generation.current;
    locked.current = true;
    setBusy(true);
    setMessage('');
    try {
      var msg = await fn();
      if (g === generation.current) {
        var d = await request('assignments?ticker=' + encodeURIComponent(ticker));
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
      outputs: [...chosen].sort(),
      horizon,
      instruction
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
  var unfinished = jobs.some(j => ['queued', 'running', 'attention'].includes(j.status));
  return /*#__PURE__*/React.createElement("section", {
    className: "sa-prepare",
    "aria-label": "End-to-end research"
  }, /*#__PURE__*/React.createElement("h4", null, "One research assignment"), /*#__PURE__*/React.createElement("p", null, "For ", /*#__PURE__*/React.createElement("strong", null, ticker), ", collect new originals and prepare the outputs you choose."), /*#__PURE__*/React.createElement("fieldset", {
    disabled: disabled || busy || unfinished || !!error
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
    onChange: e => setChosen(p => e.target.checked ? [...p, id] : p.filter(x => x !== id))
  }), title))), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Research focus and defaults"), /*#__PURE__*/React.createElement("label", null, "Investment horizon", /*#__PURE__*/React.createElement("input", {
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
    disabled: !chosen.length || !since || !until || !horizon.trim(),
    onClick: start
  }, "Start research for ", ticker)), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "One shared research run: up to eight selected originals and twelve model calls within your configured budget. Starting authorizes these calls. Draft outputs need investor review; a thesis update is never applied automatically.")), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "How automatic collection works"), /*#__PURE__*/React.createElement("p", null, "Charlie queues collection \u2192 Codex searches AlphaSense \u2192 originals are saved in iCloud STOCKS \u2192 your Mac imports and verifies them \u2192 Charlie researches and prepares the selected drafts. Keep your Mac awake, online, the agent running, and Codex with signed-in Chrome available. Collection starts on an available scheduled worker, usually the next 15-minute wake. Source restrictions, sign-in, source choices and uncertain paid-call outcomes can require your attention. The research uses the selected source pack; missing prices, consensus or financial periods remain gaps.")), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error), /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, message), jobs.map(j => /*#__PURE__*/React.createElement("article", {
    key: j.id,
    style: {
      borderTop: '1px solid #d8dedd',
      paddingTop: 12,
      marginTop: 12
    }
  }, /*#__PURE__*/React.createElement("strong", null, j.input.since, "\u2013", j.input.until, " \xB7 ", j.status), /*#__PURE__*/React.createElement("p", null, j.result.step), j.result.delayWarning && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, j.result.delayWarning), /*#__PURE__*/React.createElement("small", null, "Saved update: ", new Date(j.updated_at).toLocaleString(), j.result.macReportedAt ? ' · Mac report: ' + new Date(j.result.macReportedAt).toLocaleString() : ''), j.result.sources && /*#__PURE__*/React.createElement("p", null, j.result.sources.length, " originals verified \xB7 ", j.result.researchStages || 0, "/12 research stages"), j.error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, j.error), /*#__PURE__*/React.createElement("div", {
    className: "sa-actions"
  }, j.result.reportId && /*#__PURE__*/React.createElement("button", {
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
  }, "Export ", outputs.find(x => x[0] === a.kind)?.[1])), j.result.thesisDraftId && /*#__PURE__*/React.createElement("a", {
    href: `?thesisDraft=${j.result.thesisDraftId}#view=thesisimports&ticker=${encodeURIComponent(ticker)}`
  }, "Review thesis proposal")), j.status === 'attention' && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("label", {
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
    href: "#view=desk"
  }, "Collection and source choices")), ['queued', 'running', 'attention'].includes(j.status) && /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => mutate(async () => {
      await request(`assignments/${j.id}/stop`, {});
      return 'Stopped. Originals and saved drafts are retained; a call already running may finish.';
    })
  }, "Stop assignment"))));
}