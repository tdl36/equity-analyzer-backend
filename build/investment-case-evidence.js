import * as React from 'react';
import { ResearchChat } from './research-chat';
export function CaseEvidence({
  api,
  ticker,
  body,
  revision,
  disabled,
  onCommit
}) {
  var [rows, setRows] = React.useState([]),
    [error, setError] = React.useState(''),
    [loading, setLoading] = React.useState(true),
    [chosen, setChosen] = React.useState(''),
    [assumptionId, setAssumption] = React.useState(''),
    [field, setField] = React.useState('support');
  var [documents, setDocuments] = React.useState([]),
    [files, setFiles] = React.useState([]),
    [instructions, setInstructions] = React.useState(''),
    [working, setWorking] = React.useState(false),
    [notice, setNotice] = React.useState(''),
    [refresh, setRefresh] = React.useState(0);
  var [debate, setDebate] = React.useState(null);
  var requestRef = React.useRef(null),
    mutationLock = React.useRef(false),
    alive = React.useRef(true);
  React.useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  React.useEffect(() => {
    var live = true,
      inflight = false;
    var load = async () => {
      if (inflight) return;
      inflight = true;
      try {
        var r = await fetch(api + '/api/research/investment-case/' + encodeURIComponent(ticker) + '/source-proposals', {
          signal: AbortSignal.timeout(20000)
        });
        var d = await r.json();
        if (!r.ok) throw Error(d.error || 'Sources unavailable');
        if (live) {
          setRows(d.proposals || []);
          setDocuments(d.documents || []);
          setError('');
        }
      } catch (e) {
        if (live) setError(e.message);
      } finally {
        inflight = false;
        if (live) setLoading(false);
      }
    };
    load();
    var timer = setInterval(load, 10000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [api, ticker, refresh]);
  var mutate = async (path, payload) => {
    if (mutationLock.current) return;
    mutationLock.current = true;
    setWorking(true);
    setNotice('Submitting…');
    try {
      var key = '';
      try {
        key = localStorage.getItem('equity_analyzer_api_key') || '';
      } catch {}
      var r = await fetch(api + path, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          ...payload,
          apiKey: key
        }),
        signal: AbortSignal.timeout(20000)
      });
      var d = await r.json();
      if (!r.ok) throw Error(d.error || 'Request failed');
      if (alive.current) {
        setNotice('Request received. Progress and saved proposals appear below.');
        setRefresh(x => x + 1);
        requestRef.current = null;
      }
    } catch (e) {
      if (alive.current) setNotice(e.message + ' Your selections remain here.');
    } finally {
      mutationLock.current = false;
      if (alive.current) setWorking(false);
    }
  };
  var generate = () => {
    var value = {
      revision,
      filenames: files,
      instructions
    };
    var signature = JSON.stringify(value);
    if (requestRef.current?.signature !== signature) requestRef.current = {
      signature,
      payload: {
        ...value,
        requestId: crypto.randomUUID()
      }
    };
    mutate('/api/research/investment-case/' + encodeURIComponent(ticker) + '/proposals', requestRef.current.payload);
  };
  var activeJob = rows.some(j => j.target === 'investment_case' && ['queued', 'running', 'awaiting_approval'].includes(j.status));
  var options = rows.flatMap(job => (job.result?.changes || []).filter(c => ['awaiting_approval', 'applied'].includes(job.status) && c.passageMatched && c.reviewPassed).map(change => ({
    key: job.id + ':' + change.id,
    job,
    change
  })));
  var selected = options.find(o => o.key === chosen);
  var targeted = selected?.change.assumptionId;
  var effectiveAssumption = targeted || assumptionId,
    effectiveField = selected?.change.field || field;
  var assumption = body.assumptions.find(a => a.id === effectiveAssumption);
  return /*#__PURE__*/React.createElement("details", {
    open: true,
    className: "amendment-card"
  }, /*#__PURE__*/React.createElement("summary", null, "Evidence, debate and proposed changes"), /*#__PURE__*/React.createElement("p", null, "Choose an existing research change and review its excerpt before adding it to your investment case. A passage match establishes provenance at generation; it does not prove the interpretation or that the source is still current."), /*#__PURE__*/React.createElement("details", {
    open: true
  }, /*#__PURE__*/React.createElement("summary", null, "Generate proposals for this investment case"), /*#__PURE__*/React.createElement("p", null, "Select up to 10 saved originals. Charlie checks new evidence against your assumptions and proposes only material changes. Model usage is incurred when you generate. Oversized source sets fail visibly rather than being silently clipped."), /*#__PURE__*/React.createElement("fieldset", {
    disabled: disabled || working || activeJob
  }, /*#__PURE__*/React.createElement("label", null, "Focus or instructions", /*#__PURE__*/React.createElement("textarea", {
    value: instructions,
    maxLength: 6000,
    onChange: e => setInstructions(e.target.value),
    placeholder: "Challenge the margin recovery assumption; preserve management qualifiers."
  })), /*#__PURE__*/React.createElement("div", {
    style: {
      maxHeight: 240,
      overflowY: 'auto'
    }
  }, documents.map(name => /*#__PURE__*/React.createElement("label", {
    key: name
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: files.includes(name),
    disabled: !files.includes(name) && files.length >= 10,
    onChange: e => setFiles(e.target.checked ? [...files, name] : files.filter(x => x !== name))
  }), name))), !loading && !documents.length && /*#__PURE__*/React.createElement("p", null, "No saved originals found for this company. Import documents into Charlie first."), /*#__PURE__*/React.createElement("button", {
    disabled: !revision || !body.assumptions.length || !files.length,
    onClick: generate
  }, "Generate assumption proposals \xB7 ", files.length, " sources")), /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, notice), rows.filter(j => j.target === 'investment_case').map(j => /*#__PURE__*/React.createElement("section", {
    key: j.id
  }, /*#__PURE__*/React.createElement("strong", null, j.status.replaceAll('_', ' '), " \xB7 ", new Date(j.created_at).toLocaleString()), j.error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, j.error), j.status === 'awaiting_approval' && /*#__PURE__*/React.createElement("p", null, j.result?.changes?.length || 0, " proposed changes \xB7 ", (j.result?.changes || []).filter(c => c.reviewPassed && c.passageMatched).length, " passed passage and model checks. Review below, then close this proposal to start another."), (j.result?.changes || []).filter(c => !c.reviewPassed || !c.passageMatched).map(c => /*#__PURE__*/React.createElement("p", {
    key: c.id
  }, "Needs review: ", c.after, " \u2014 ", c.reviewIssue || 'Supporting passage did not match.')), j.status === 'failed' && j.result?.checkpoint && /*#__PURE__*/React.createElement("button", {
    disabled: disabled || working,
    onClick: () => mutate('/api/research/amendment/' + j.id + '/resume', {})
  }, "Resume from saved checkpoint"), ['failed', 'awaiting_approval'].includes(j.status) && /*#__PURE__*/React.createElement("button", {
    disabled: disabled || working,
    onClick: () => mutate('/api/research/amendment/' + j.id + '/decide', {
      action: 'dismiss'
    })
  }, "Close proposal \xB7 keep accepted case revisions")))), loading && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading source proposals\u2026"), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error), !loading && !error && !options.length && /*#__PURE__*/React.createElement("p", null, "No eligible source proposals yet. Generate a comparison above, or use an existing thesis comparison. Only changes with a matching passage and a passed model review appear here."), !!options.length && /*#__PURE__*/React.createElement("fieldset", {
    disabled: disabled || working
  }, /*#__PURE__*/React.createElement("label", null, "Reviewed research change", /*#__PURE__*/React.createElement("select", {
    value: chosen,
    onChange: e => setChosen(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Choose a source-supported change"), options.map(o => /*#__PURE__*/React.createElement("option", {
    key: o.key,
    value: o.key
  }, new Date(o.job.created_at).toLocaleDateString(), " \xB7 ", o.change.after.slice(0, 110))))), selected && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("h3", null, "Why Charlie proposes a change"), /*#__PURE__*/React.createElement("p", null, selected.change.reason), /*#__PURE__*/React.createElement("div", {
    className: "lifecycle-tabs"
  }, [['Challenge this interpretation', 'Challenge this proposed interpretation. Separate what the source actually says from inference. Identify missing evidence and explain whether the current thesis should change.'], ['Make the opposing case', 'Make the strongest evidence-based case against this proposed change. Preserve qualifiers and explain what would resolve the disagreement.'], ['What would change my mind?', 'Identify the observable evidence that would support or invalidate this change and the next questions I should investigate.']].map(([label, prompt]) => /*#__PURE__*/React.createElement("button", {
    key: label,
    onClick: () => setDebate({
      prompt,
      content: JSON.stringify({
        caseRevision: revision,
        investmentCase: body,
        proposalId: selected.job.id,
        change: selected.change,
        sources: selected.job.result.sources
      })
    })
  }, label))), selected.change.evidence.filter(e => e.status === 'passage_matched').map((e, i) => /*#__PURE__*/React.createElement("section", {
    key: i
  }, /*#__PURE__*/React.createElement("strong", null, selected.job.result.sources.find(s => s.id === e.sourceId)?.filename || 'Source snapshot'), /*#__PURE__*/React.createElement("blockquote", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, e.excerpt))), /*#__PURE__*/React.createElement("label", null, "Investment assumption", /*#__PURE__*/React.createElement("select", {
    disabled: !!targeted,
    value: effectiveAssumption,
    onChange: e => setAssumption(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Choose an assumption"), body.assumptions.map((a, i) => /*#__PURE__*/React.createElement("option", {
    key: a.id,
    value: a.id
  }, i + 1, ". ", a.claim.slice(0, 120))))), /*#__PURE__*/React.createElement("label", null, "Update which field?", /*#__PURE__*/React.createElement("select", {
    disabled: !!targeted,
    value: effectiveField,
    onChange: e => setField(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: "support"
  }, "Supporting evidence"), /*#__PURE__*/React.createElement("option", {
    value: "contrary"
  }, "Contrary evidence / unresolved"), /*#__PURE__*/React.createElement("option", {
    value: "nextTest"
  }, "Next test and timing"))), assumption && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("h4", null, "Before"), /*#__PURE__*/React.createElement("p", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, assumption[effectiveField] || 'Not recorded'), /*#__PURE__*/React.createElement("h4", null, "Proposed replacement"), /*#__PURE__*/React.createElement("p", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, selected.change.after), /*#__PURE__*/React.createElement("p", null, "This replaces the selected field. Your assumption and its basis remain unchanged; the original quotation is retained separately."), /*#__PURE__*/React.createElement("button", {
    disabled: assumption[effectiveField] === selected.change.after,
    onClick: () => onCommit({
      mode: 'source_change',
      sourceChange: {
        jobId: selected.job.id,
        changeId: selected.change.id,
        assumptionId: effectiveAssumption,
        field: effectiveField
      }
    })
  }, "Accept into a new investment case revision")))), !!body.evidenceLinks?.length && /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, body.evidenceLinks.length, " accepted evidence links"), body.evidenceLinks.map((link, i) => {
    var a = body.assumptions.find(x => x.id === link.assumptionId);
    return /*#__PURE__*/React.createElement("section", {
      key: i
    }, /*#__PURE__*/React.createElement("h4", null, a?.claim || 'Assumption removed in a later revision'), /*#__PURE__*/React.createElement("p", null, a?.[link.field] === link.after ? 'Matches current wording' : 'Historical wording · subsequently edited', " \xB7 ", link.field), /*#__PURE__*/React.createElement("p", {
      style: {
        whiteSpace: 'pre-wrap'
      }
    }, link.after), link.evidence.map((e, j) => /*#__PURE__*/React.createElement("blockquote", {
      key: j
    }, /*#__PURE__*/React.createElement("strong", null, e.source.filename), /*#__PURE__*/React.createElement("br", null), e.excerpt)));
  })), debate && /*#__PURE__*/React.createElement(ResearchChat, {
    api: api,
    context: {
      ticker,
      type: 'review',
      content: debate.content
    },
    initialMessage: debate.prompt,
    allowEdits: false,
    onClose: () => setDebate(null)
  }));
}