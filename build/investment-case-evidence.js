import * as React from 'react';
import { ThesisMonitor } from './thesis-monitor';
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
  var [search, setSearch] = React.useState(''),
    [selectedOnly, setSelectedOnly] = React.useState(false);
  var [debate, setDebate] = React.useState(null),
    [reviews, setReviews] = React.useState({});
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
  var visibleDocuments = [...documents].sort((a, b) => b.localeCompare(a)).filter(name => (!selectedOnly || files.includes(name)) && name.toLowerCase().includes(search.toLowerCase().trim()));
  return /*#__PURE__*/React.createElement("section", {
    className: "case-evidence",
    "aria-label": `${ticker} evidence and proposals`
  }, /*#__PURE__*/React.createElement("header", {
    className: "case-evidence-heading"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, ticker, " / EVIDENCE & PROPOSALS"), /*#__PURE__*/React.createElement("h2", null, "What does the new evidence change?"), /*#__PURE__*/React.createElement("p", null, "Choose the documents you want Charlie to assess against your saved investment case.")), disabled && /*#__PURE__*/React.createElement("p", {
    className: "case-evidence-banner"
  }, "Save your current case edits before generating or accepting changes."), !revision || !body.assumptions.length ? /*#__PURE__*/React.createElement("p", {
    className: "case-evidence-banner"
  }, "Start in Current thesis: save a case with at least one assumption, then return here.") : null, activeJob && /*#__PURE__*/React.createElement("p", {
    className: "case-evidence-banner"
  }, "A comparison is already in progress or awaiting your review. ", /*#__PURE__*/React.createElement("button", {
    onClick: () => document.getElementById(`case-review-${ticker}`)?.scrollIntoView({
      behavior: 'smooth',
      block: 'start'
    })
  }, "Go to results below \u2193")), /*#__PURE__*/React.createElement("section", {
    className: "case-evidence-step"
  }, /*#__PURE__*/React.createElement("div", {
    className: "case-step-title"
  }, /*#__PURE__*/React.createElement("span", null, "1"), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h3", null, "Choose your documents"), /*#__PURE__*/React.createElement("p", null, "Select up to 10 originals. Search by broker, topic, or filename."))), /*#__PURE__*/React.createElement("fieldset", {
    disabled: disabled || working || activeJob || loading || !!error
  }, /*#__PURE__*/React.createElement("div", {
    className: "case-source-tools"
  }, /*#__PURE__*/React.createElement("label", null, "Find documents", /*#__PURE__*/React.createElement("input", {
    type: "search",
    value: search,
    onChange: e => setSearch(e.target.value),
    placeholder: "Search broker, earnings, transcript\u2026"
  })), /*#__PURE__*/React.createElement("button", {
    type: "button",
    "aria-pressed": selectedOnly,
    onClick: () => setSelectedOnly(!selectedOnly)
  }, selectedOnly ? 'Show all documents' : 'Show selected only')), /*#__PURE__*/React.createElement("div", {
    className: "case-source-count"
  }, /*#__PURE__*/React.createElement("strong", null, files.length, " of 10 selected"), /*#__PURE__*/React.createElement("span", null, visibleDocuments.length, " matching documents"), /*#__PURE__*/React.createElement("button", {
    type: "button",
    disabled: !files.length,
    onClick: () => setFiles([])
  }, "Clear selection")), /*#__PURE__*/React.createElement("div", {
    className: "case-source-list",
    role: "group",
    "aria-label": "Available source documents"
  }, visibleDocuments.map(name => /*#__PURE__*/React.createElement("label", {
    className: "case-source-row",
    key: name,
    "data-selected": files.includes(name)
  }, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: files.includes(name),
    disabled: !files.includes(name) && files.length >= 10,
    onChange: e => setFiles(e.target.checked ? [...files, name] : files.filter(x => x !== name))
  }), /*#__PURE__*/React.createElement("span", null, name), /*#__PURE__*/React.createElement("small", null, name.split('.').pop().toUpperCase()))), loading && /*#__PURE__*/React.createElement("p", null, "Loading saved documents\u2026"), !loading && !visibleDocuments.length && /*#__PURE__*/React.createElement("p", null, documents.length ? 'No documents match this filter. Try another search or show all documents.' : 'No saved originals for this company. Import documents through your usual collection workflow first.')), !!files.length && /*#__PURE__*/React.createElement("details", {
    className: "case-selected-list"
  }, /*#__PURE__*/React.createElement("summary", null, "Review your ", files.length, " selected documents"), files.map(name => /*#__PURE__*/React.createElement("div", {
    key: name
  }, /*#__PURE__*/React.createElement("span", null, name, !documents.includes(name) ? ' · No longer in the available list' : ''), /*#__PURE__*/React.createElement("button", {
    type: "button",
    "aria-label": `Remove ${name}`,
    onClick: () => setFiles(files.filter(x => x !== name))
  }, "Remove")))))), /*#__PURE__*/React.createElement("section", {
    className: "case-evidence-step"
  }, /*#__PURE__*/React.createElement("div", {
    className: "case-step-title"
  }, /*#__PURE__*/React.createElement("span", null, "2"), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h3", null, "Set the focus"), /*#__PURE__*/React.createElement("p", null, "Optional. Leave blank for a review across your saved assumptions."))), /*#__PURE__*/React.createElement("fieldset", {
    disabled: disabled || working || activeJob
  }, /*#__PURE__*/React.createElement("label", null, "What should Charlie investigate?", /*#__PURE__*/React.createElement("textarea", {
    rows: 4,
    value: instructions,
    maxLength: 6000,
    onChange: e => setInstructions(e.target.value),
    placeholder: "For example: Does this change the margin recovery case? Preserve management\u2019s qualifiers and highlight contradictory evidence."
  })), /*#__PURE__*/React.createElement("div", {
    className: "case-generate"
  }, /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    disabled: loading || !!error || !revision || !body.assumptions.length || !files.length || files.some(n => !documents.includes(n)),
    onClick: generate
  }, working ? 'Submitting…' : `Assess ${files.length || 'selected'} document${files.length === 1 ? '' : 's'}`), /*#__PURE__*/React.createElement("p", null, "Creates proposed changes for your review. Uses model credits; your saved case is not changed automatically.")))), notice && /*#__PURE__*/React.createElement("p", {
    role: "status",
    className: "case-evidence-banner"
  }, notice), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error, " ", /*#__PURE__*/React.createElement("button", {
    onClick: () => setRefresh(x => x + 1)
  }, "Retry loading documents")), /*#__PURE__*/React.createElement("section", {
    className: "case-evidence-step",
    id: `case-review-${ticker}`
  }, /*#__PURE__*/React.createElement("div", {
    className: "case-step-title"
  }, /*#__PURE__*/React.createElement("span", null, "3"), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h3", null, "Review and decide"), /*#__PURE__*/React.createElement("p", null, "Read the supporting passage, debate the interpretation, then accept changes or record why your view stays the same."))), rows.filter(j => j.target === 'investment_case').map(j => /*#__PURE__*/React.createElement("details", {
    className: "case-result",
    key: j.id,
    open: j.status !== 'dismissed'
  }, /*#__PURE__*/React.createElement("summary", null, j.status.replaceAll('_', ' '), " \xB7 ", new Date(j.created_at).toLocaleString()), j.error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, j.error), j.result?.conditionAssessments?.map(c => /*#__PURE__*/React.createElement("article", {
    className: "workspace-panel",
    key: c.id
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "UNDERWEIGHT CONDITION \xB7 AI SUGGESTION"), /*#__PURE__*/React.createElement("h4", null, c.workTitle), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("strong", null, "Condition:"), " ", c.trigger), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("strong", null, "Suggested assessment:"), " ", c.assessment.replaceAll('_', ' ')), /*#__PURE__*/React.createElement("p", null, c.reason), /*#__PURE__*/React.createElement("p", null, c.passageMatched && c.reviewPassed ? 'Passed excerpt and model checks; your judgment is still required.' : 'Needs further verification · ' + (c.reviewIssue || 'Supporting passage did not match.')), c.evidence?.filter(e => e.status === 'passage_matched').map((e, i) => /*#__PURE__*/React.createElement("blockquote", {
    key: i
  }, /*#__PURE__*/React.createElement("strong", null, j.result.sources?.find(s => s.id === e.sourceId)?.filename), /*#__PURE__*/React.createElement("p", null, e.excerpt))), /*#__PURE__*/React.createElement("p", null, "Based on work revision ", c.workRevision, " and case R", c.caseRevision, ". Review the current record in Decisions & underweights before recording your own assessment."))), j.status === 'awaiting_approval' && /*#__PURE__*/React.createElement("p", null, j.result?.changes?.length || 0, " proposed changes \xB7 ", (j.result?.changes || []).filter(c => c.reviewPassed && c.passageMatched).length, " passed passage and model checks. Review below, then close this proposal to start another."), (j.result?.changes || []).filter(c => !c.reviewPassed || !c.passageMatched).map(c => /*#__PURE__*/React.createElement("p", {
    key: c.id
  }, "Needs review: ", c.after, " \u2014 ", c.reviewIssue || 'Supporting passage did not match.')), j.status === 'failed' && j.result?.checkpoint && /*#__PURE__*/React.createElement("button", {
    disabled: disabled || working,
    onClick: () => mutate('/api/research/amendment/' + j.id + '/resume', {})
  }, "Resume from saved checkpoint"), j.result?.reviewDecision && /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("strong", null, "My review: ", j.result.reviewDecision.outcome.replaceAll('_', ' ')), " \xB7 ", j.result.reviewDecision.rationale), ['failed', 'awaiting_approval'].includes(j.status) && /*#__PURE__*/React.createElement("details", {
    className: "case-close-review"
  }, /*#__PURE__*/React.createElement("summary", null, "Finish this review \xB7 record your conclusion"), /*#__PURE__*/React.createElement("fieldset", {
    disabled: disabled || working
  }, /*#__PURE__*/React.createElement("label", null, "My conclusion", /*#__PURE__*/React.createElement("select", {
    value: reviews[j.id]?.outcome || 'no_change',
    onChange: e => setReviews({
      ...reviews,
      [j.id]: {
        ...reviews[j.id],
        outcome: e.target.value
      }
    })
  }, /*#__PURE__*/React.createElement("option", {
    value: "no_change"
  }, "No thesis change warranted"), /*#__PURE__*/React.createElement("option", {
    value: "rejected"
  }, "Reject proposed interpretation"), /*#__PURE__*/React.createElement("option", {
    value: "changes_reviewed"
  }, "Selected changes reviewed / accepted separately"))), /*#__PURE__*/React.createElement("label", null, "Why I reached this conclusion", /*#__PURE__*/React.createElement("textarea", {
    value: reviews[j.id]?.rationale || '',
    onChange: e => setReviews({
      ...reviews,
      [j.id]: {
        ...reviews[j.id],
        rationale: e.target.value
      }
    })
  })), /*#__PURE__*/React.createElement("button", {
    disabled: !reviews[j.id]?.rationale?.trim(),
    onClick: () => mutate('/api/research/amendment/' + j.id + '/decide', {
      action: 'dismiss',
      reviewDecision: {
        outcome: reviews[j.id]?.outcome || 'no_change',
        rationale: reviews[j.id].rationale
      }
    })
  }, "Record decision and close proposal"))))), loading && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading source proposals\u2026"), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error), !loading && !error && !options.length && /*#__PURE__*/React.createElement("p", null, "Your results will appear here after assessment. Start by selecting documents in step 1."), !!options.length && /*#__PURE__*/React.createElement("fieldset", {
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
  }, "Accept into a new investment case revision"))))), /*#__PURE__*/React.createElement("details", {
    className: "case-automation"
  }, /*#__PURE__*/React.createElement("summary", null, "Automatic monitoring \xB7 optional settings"), /*#__PURE__*/React.createElement(ThesisMonitor, {
    api: api,
    ticker: ticker,
    disabled: disabled || !revision || !body.assumptions.length
  })), !!body.evidenceLinks?.length && /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, body.evidenceLinks.length, " accepted evidence links"), body.evidenceLinks.map((link, i) => {
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