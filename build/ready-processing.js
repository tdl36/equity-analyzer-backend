import * as React from 'react';
import { parseTimestamp } from './workspace-model.mjs';
import { CollectionSynthesisStatus } from './collection-synthesis';
export function ReadyProcessing({
  api
}) {
  var [rows, setRows] = React.useState([]),
    [selected, setSelected] = React.useState(new Set()),
    [query, setQuery] = React.useState('');
  var [error, setError] = React.useState(''),
    [message, setMessage] = React.useState(''),
    [loaded, setLoaded] = React.useState(false),
    [busy, setBusy] = React.useState(false);
  var alive = React.useRef(false),
    lock = React.useRef(false),
    serial = React.useRef(0);
  var refresh = async () => {
    var generation = ++serial.current;
    try {
      var r = await fetch(`${api}/api/catalysts/proposals`, {
        signal: AbortSignal.timeout(20000)
      });
      if (!r.ok) throw Error('Could not load ready jobs.');
      var d = await r.json();
      if (!Array.isArray(d.proposals)) throw Error('Ready-job response is incomplete.');
      if (alive.current && generation === serial.current) {
        setRows(d.proposals);
        setSelected(old => new Set([...old].filter(id => d.proposals.some(p => p.jobId === id))));
        setError('');
        setLoaded(true);
      }
    } catch (e) {
      if (alive.current && generation === serial.current) setError(`${e.message} Selection is disabled until a fresh list is available.`);
    }
  };
  React.useEffect(() => {
    alive.current = true;
    refresh();
    var timer = setInterval(() => {
      if (!document.hidden && !lock.current) refresh();
    }, 30000);
    return () => {
      alive.current = false;
      serial.current++;
      clearInterval(timer);
    };
  }, [api]);
  var visible = rows.filter(p => `${p.ticker} ${p.topic}`.toLowerCase().includes(query.trim().toLowerCase()));
  var start = async () => {
    if (lock.current || error || !selected.size) return;
    lock.current = true;
    setBusy(true);
    setMessage('');
    serial.current++;
    var ids = rows.filter(p => selected.has(p.jobId)).map(p => p.jobId);
    try {
      var r = await fetch(`${api}/api/catalysts/proposals/approve`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          jobIds: ids
        }),
        signal: AbortSignal.timeout(30000)
      });
      var d = await r.json();
      if (!r.ok || !Number.isInteger(d.approved)) throw Error('Submission was not confirmed.');
      if (alive.current) setMessage(`${d.approved} job${d.approved === 1 ? '' : 's'} queued for synthesis. ${ids.length - d.approved > 0 ? `${ids.length - d.approved} selections were already handled or no longer ready. ` : ''}Results remain drafts for review.`);
    } catch (e) {
      if (alive.current) setMessage('Submission could not be confirmed. Check Catalyst notes for queued or running jobs before submitting again. Charlie only queues jobs still awaiting processing approval.');
    } finally {
      if (alive.current) {
        setSelected(new Set());
        await refresh();
        setBusy(false);
      }
      lock.current = false;
    }
  };
  return /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel",
    "aria-label": "Ready to process"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "YOU CHOOSE WHAT RUNS"), /*#__PURE__*/React.createElement("h2", null, "Ready to process"), /*#__PURE__*/React.createElement("p", {
    className: "workspace-lead"
  }, "Select the source folders you want Charlie to synthesize."), /*#__PURE__*/React.createElement("p", null, "These are detected event folders awaiting processing approval. Starting selected jobs uses model credits and the Mac agent. It does not enable ongoing automatic synthesis or approve the resulting research."), /*#__PURE__*/React.createElement("div", {
    className: "desk-filter"
  }, /*#__PURE__*/React.createElement("label", null, "Find ready jobs", /*#__PURE__*/React.createElement("input", {
    value: query,
    onChange: e => setQuery(e.target.value),
    placeholder: "Ticker or event"
  })), /*#__PURE__*/React.createElement("button", {
    onClick: refresh,
    disabled: busy
  }, "Refresh ready jobs")), error && /*#__PURE__*/React.createElement("p", {
    role: "alert",
    className: "workspace-error"
  }, error), message && /*#__PURE__*/React.createElement("p", {
    role: "status",
    className: "workspace-notice"
  }, message, " ", /*#__PURE__*/React.createElement("a", {
    href: "#view=catalysts"
  }, "Track synthesis & results \u2192")), !loaded && !error && /*#__PURE__*/React.createElement("p", null, "Loading ready jobs\u2026"), /*#__PURE__*/React.createElement("fieldset", {
    disabled: busy || !!error || !loaded,
    style: {
      border: 0,
      padding: 0,
      minWidth: 0
    }
  }, /*#__PURE__*/React.createElement("legend", null, visible.length, " shown \xB7 ", rows.length, " ready \xB7 ", selected.size, " selected"), /*#__PURE__*/React.createElement("div", {
    className: "desk-filter"
  }, /*#__PURE__*/React.createElement("button", {
    onClick: () => setSelected(new Set([...selected, ...visible.map(p => p.jobId)])),
    disabled: !visible.length
  }, "Select shown"), /*#__PURE__*/React.createElement("button", {
    onClick: () => setSelected(new Set()),
    disabled: !selected.size
  }, "Clear selection")), visible.map(p => /*#__PURE__*/React.createElement("label", {
    className: "workspace-panel",
    key: p.jobId,
    style: {
      display: 'flex',
      gap: 14,
      alignItems: 'flex-start',
      cursor: 'pointer',
      marginBottom: 10
    }
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    "aria-label": `Process ${p.ticker} ${p.topic}`,
    checked: selected.has(p.jobId),
    onChange: e => setSelected(old => {
      var next = new Set(old);
      if (e.target.checked) next.add(p.jobId);else next.delete(p.jobId);
      return next;
    })
  }), /*#__PURE__*/React.createElement("span", {
    style: {
      minWidth: 0,
      overflowWrap: 'anywhere'
    }
  }, /*#__PURE__*/React.createElement("strong", null, p.ticker, " \xB7 ", p.topic || 'Event folder'), /*#__PURE__*/React.createElement("span", {
    style: {
      display: 'block'
    }
  }, Number.isInteger(p.fileCount) ? `${p.fileCount} detected source files` : 'Source count unavailable', " \xB7 ", p.length || 'standard', " synthesis"), /*#__PURE__*/React.createElement("small", null, "Detected ", parseTimestamp(p.proposedAt)?.toLocaleString() || 'date unavailable', " \xB7 CATALYSTS folder")))), loaded && !rows.length && !error && /*#__PURE__*/React.createElement("p", null, "No detected event folders are awaiting processing approval. Collection can continue while automatic synthesis is off."), rows.length > 0 && !visible.length && /*#__PURE__*/React.createElement("p", null, "No ready jobs match this search."), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    onClick: start,
    disabled: !selected.size
  }, busy ? 'Submitting…' : `Approve & process ${selected.size} selected job${selected.size === 1 ? '' : 's'}`)), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Selection applies to every checked job, including those hidden by your search. The list shows the latest 100 proposals; source counts reflect detection, and the agent validates readable files when processing."), /*#__PURE__*/React.createElement("div", {
    className: "desk-filter"
  }, /*#__PURE__*/React.createElement("a", {
    href: "#view=catalysts"
  }, "Queued jobs & synthesis results \u2192"), /*#__PURE__*/React.createElement("a", {
    href: "#view=analysts"
  }, "Analyst tasks & draft review \u2192"), /*#__PURE__*/React.createElement("a", {
    href: "#view=pipelinelab"
  }, "Choose STOCKS sources & outputs in Lab \u2192")), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "An imported document alone is not a ready synthesis job. Lab source libraries, analyst tasks and completed drafts keep their own output and review controls."), /*#__PURE__*/React.createElement(CollectionSynthesisStatus, {
    api: api
  }));
}