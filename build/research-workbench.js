function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
import * as React from 'react';
import { MeetingAnswerProposal } from './meeting-answer-proposal';
import { ResearchChat } from './research-chat';
import { WorkComparison } from './work-comparison';
import { emptyWorkDraft, workDraft, workRecoveryKey, readWorkRecovery, writeWorkRecovery, clearWorkRecovery } from './research-workbench-recovery.mjs';
var blank = emptyWorkDraft;
async function json(api, path, options = {}) {
  var r = await fetch(api + path, {
    ...options,
    signal: AbortSignal.timeout(20000)
  });
  var d = await r.json();
  if (!r.ok) {
    var e = Error(d.error || 'Work unavailable');
    e.status = r.status;
    throw e;
  }
  return d;
}
export function ResearchWorkbench(props) {
  return /*#__PURE__*/React.createElement(Workbench, _extends({
    key: props.api + '|' + props.ticker
  }, props));
}
function Workbench({
  api,
  ticker,
  caseRevision,
  assumptions = [],
  disabled = false,
  initialKind = 'model'
}) {
  var key = workRecoveryKey(api, ticker);
  var [recovered] = React.useState(() => {
    try {
      return {
        value: readWorkRecovery(sessionStorage, key, ticker),
        error: ''
      };
    } catch (e) {
      return {
        value: null,
        error: e.message
      };
    }
  });
  var [recoveryError, setRecoveryError] = React.useState(recovered.error),
    [dirty, setDirty] = React.useState(!!recovered.value),
    [baseline, setBaseline] = React.useState(recovered.value?.caseRevision ?? caseRevision);
  var initialRecoveryBlocked = React.useRef(!!recovered.error);
  var [submission, setSubmission] = React.useState(recovered.value?.pending || null);
  var [data, setData] = React.useState({
      records: [],
      sources: [],
      answers: []
    }),
    [draft, setDraft] = React.useState(() => recovered.value?.draft || blank(initialKind)),
    [record, setRecord] = React.useState(recovered.value?.record || null),
    [busy, setBusy] = React.useState(false),
    [message, setMessage] = React.useState(''),
    [history, setHistory] = React.useState(null);
  var [debate, setDebate] = React.useState(null);
  var pending = React.useRef(recovered.value?.pending || null),
    lock = React.useRef(false),
    alive = React.useRef(true),
    sequence = React.useRef(0);
  var load = async () => {
    var seq = ++sequence.current;
    try {
      var d = await json(api, '/api/research/workbench/' + ticker);
      if (alive.current && seq === sequence.current) setData(d);
    } catch (e) {
      if (alive.current) setMessage(e.message);
    }
  };
  React.useEffect(() => {
    alive.current = true;
    load();
    return () => {
      alive.current = false;
    };
  }, []);
  var envelope = (draftValue = draft, recordValue = record, pendingValue = pending.current) => ({
    version: 1,
    ticker,
    caseRevision: baseline,
    record: recordValue ? {
      id: recordValue.id,
      revision: recordValue.revision
    } : null,
    draft: draftValue,
    pending: pendingValue
  });
  React.useLayoutEffect(() => {
    if (!dirty || recoveryError) return;
    try {
      writeWorkRecovery(sessionStorage, key, envelope());
    } catch (e) {
      setRecoveryError(e.message + ' Keep this tab open; changes are not safely retained.');
    }
  }, [draft, record, baseline, dirty, submission, recoveryError]);
  React.useEffect(() => {
    if (!dirty && !submission) return;
    var warn = e => {
      e.preventDefault();
      e.returnValue = '';
    };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [dirty, submission]);
  React.useEffect(() => {
    if (!dirty && !submission) setBaseline(caseRevision);
  }, [caseRevision, dirty, submission]);
  var edit = (key, v) => {
    if (pending.current) return;
    setDirty(true);
    setDraft(d => ({
      ...d,
      [key]: v
    }));
  };
  var replaceDraft = (nextRecord, nextDraft, changed = false) => {
    if (pending.current || recoveryError) return;
    if (dirty && !window.confirm('Discard the current unfinished work draft and open this record?')) return;
    try {
      clearWorkRecovery(sessionStorage, key);
    } catch (e) {
      setRecoveryError(e.message);
      return;
    }
    setRecord(nextRecord);
    setDraft(workDraft(nextDraft));
    setBaseline(caseRevision);
    setDirty(changed);
    setMessage('');
  };
  var save = async () => {
    if (lock.current || recoveryError || !pending.current && baseline !== caseRevision) return;
    lock.current = true;
    setBusy(true);
    setMessage('Saving reviewed work…');
    var p = pending.current || {
      requestId: crypto.randomUUID(),
      id: record?.id || crypto.randomUUID(),
      revision: record?.revision || 0,
      body: {
        ...draft,
        caseRevision: baseline
      }
    };
    try {
      try {
        writeWorkRecovery(sessionStorage, key, envelope(draft, record, p));
      } catch (e) {
        setRecoveryError(e.message + ' No new save was sent.');
        throw e;
      }
      pending.current = p;
      setSubmission(p);
      setDirty(true);
      var r = await json(api, '/api/research/workbench/' + ticker, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(p)
      });
      if (r.id !== p.id || r.revision !== p.revision + 1) throw Error('Saved work receipt does not match this request. Keep the original request for recovery.');
      clearWorkRecovery(sessionStorage, key);
      if (alive.current) {
        setRecord({
          id: r.id,
          revision: r.revision
        });
        pending.current = null;
        setSubmission(null);
        setDirty(false);
        setMessage('Saved a work revision. Investment case and portfolio positions were not changed.');
        await load();
      }
    } catch (e) {
      if (e.status === 400 || e.status === 409) {
        try {
          writeWorkRecovery(sessionStorage, key, envelope(draft, record, null));
          pending.current = null;
          setSubmission(null);
        } catch (storageError) {
          setRecoveryError(storageError.message);
        }
      }
      if (alive.current) setMessage(e.message);
    } finally {
      lock.current = false;
      if (alive.current) setBusy(false);
    }
  };
  var input = (key, label, type = 'text') => /*#__PURE__*/React.createElement("label", null, label, /*#__PURE__*/React.createElement("input", {
    type: type,
    value: draft[key] ?? '',
    onChange: e => edit(key, e.target.value)
  }));
  return /*#__PURE__*/React.createElement("details", {
    open: true,
    className: "research-decision-log"
  }, /*#__PURE__*/React.createElement("summary", null, "Research workbench \xB7 ", data.records.length, " records"), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Connect an assumption to a calculation, meeting follow-up or underweight review. Reviewed records preserve your reasoning; they do not execute trades or update a spreadsheet."), dirty && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Unfinished work is kept in this browser tab when storage is available. It is not saved in Charlie until you save a work revision."), recoveryError && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, recoveryError, " Saved work below remains available. ", /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => {
      try {
        var saved = readWorkRecovery(sessionStorage, key, ticker);
        if (initialRecoveryBlocked.current) {
          if (saved) {
            setDraft(saved.draft);
            setRecord(saved.record);
            setBaseline(saved.caseRevision);
            pending.current = saved.pending;
            setSubmission(saved.pending);
            setDirty(true);
          }
        } else {
          if (saved?.pending && JSON.stringify(saved.pending) !== JSON.stringify(pending.current)) throw Error('Another unresolved save is retained. Reload before continuing.');
          if (dirty) writeWorkRecovery(sessionStorage, key, envelope());
        }
        initialRecoveryBlocked.current = false;
        setRecoveryError('');
      } catch (e) {
        setRecoveryError(e.message);
      }
    }
  }, "Retry browser recovery")), submission && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, "A work save is unresolved. Its original request and revision are frozen; retrying saves that same work only. No research generation is involved. ", /*#__PURE__*/React.createElement("button", {
    disabled: busy || !!recoveryError,
    onClick: save
  }, "Retry original work save")), dirty && baseline !== caseRevision && !submission && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, "This draft was based on case revision ", baseline, "; the current case is revision ", caseRevision, ". Review the current assumptions, sources and answers before rebasing. ", /*#__PURE__*/React.createElement("button", {
    disabled: busy || !!recoveryError,
    onClick: () => setBaseline(caseRevision)
  }, "I reviewed the current case \xB7 rebase draft")), record && /*#__PURE__*/React.createElement(WorkComparison, {
    key: record.id + ':' + record.revision,
    api: api,
    ticker: ticker,
    record: record,
    draft: draft,
    baseline: baseline,
    onOpen: replaceDraft,
    replaceDisabled: busy || disabled || !!submission || !!recoveryError
  }), (!caseRevision || !assumptions.length) && /*#__PURE__*/React.createElement("p", null, "Save an investment case with at least one assumption to start connected work."), /*#__PURE__*/React.createElement("fieldset", {
    disabled: busy || disabled || !!submission || !!recoveryError || !caseRevision || !assumptions.length
  }, /*#__PURE__*/React.createElement("label", null, "Work type", /*#__PURE__*/React.createElement("select", {
    "aria-label": "Work type",
    disabled: !!record,
    value: draft.kind,
    onChange: e => edit('kind', e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: "model"
  }, "Source \u2192 model sensitivity"), /*#__PURE__*/React.createElement("option", {
    value: "followup"
  }, "Meeting answer \u2192 follow-up"), /*#__PURE__*/React.createElement("option", {
    value: "underweight"
  }, "Underweight / nonownership review"))), input('title', 'Research question / title'), input('owner', 'Owner'), input('dueDate', 'Next action due', 'date'), /*#__PURE__*/React.createElement("label", null, "Investment assumption", /*#__PURE__*/React.createElement("select", {
    "aria-label": "Investment assumption",
    value: draft.assumptionId,
    onChange: e => edit('assumptionId', e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Choose a saved assumption"), assumptions.map(a => /*#__PURE__*/React.createElement("option", {
    value: a.id,
    key: a.id
  }, a.claim)))), draft.kind === 'model' && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, "Single operating-margin change. Enter revenue and diluted shares in millions, consistent currency and period. Baseline EPS is your model input; all other drivers stay constant."), /*#__PURE__*/React.createElement("label", null, "Supporting original", /*#__PURE__*/React.createElement("select", {
    "aria-label": "Supporting original",
    value: draft.sourceId,
    onChange: e => {
      var s = data.sources.find(x => x.id === Number(e.target.value));
      edit('sourceId', s?.id || '');
      edit('sourceHash', s?.hash || '');
    }
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Choose saved original"), data.sources.map(s => /*#__PURE__*/React.createElement("option", {
    key: s.id,
    value: s.id
  }, s.filename)))), /*#__PURE__*/React.createElement("label", null, "Exact supporting passage", /*#__PURE__*/React.createElement("textarea", {
    value: draft.passage,
    onChange: e => edit('passage', e.target.value)
  })), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Passage matching establishes provenance. Your numerical interpretation still requires review."), /*#__PURE__*/React.createElement("div", {
    className: "workbench-grid"
  }, Object.entries({
    revenueMillions: 'Revenue · millions',
    sharesMillions: 'Diluted shares · millions',
    beforeMarginPct: 'Baseline operating margin %',
    afterMarginPct: 'Proposed operating margin %',
    taxPct: 'Marginal tax rate %',
    baselineEPS: 'Baseline EPS',
    multiple: 'Valuation P/E',
    referencePrice: 'Reference share price',
    currency: 'Currency',
    period: 'Fiscal forecast period',
    basis: 'Accounting / EPS basis',
    asOf: 'Reference date'
  }).map(([key, label]) => /*#__PURE__*/React.createElement("label", {
    key: key
  }, label, /*#__PURE__*/React.createElement("input", {
    type: key === 'asOf' ? 'date' : 'text',
    value: draft.inputs[key] || '',
    onChange: e => edit('inputs', {
      ...draft.inputs,
      [key]: e.target.value
    })
  }))))), draft.kind === 'followup' && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("label", null, "Recorded meeting answer", /*#__PURE__*/React.createElement("select", {
    "aria-label": "Recorded meeting answer",
    value: draft.answerId,
    onChange: e => {
      var a = data.answers.find(x => x.id === Number(e.target.value));
      edit('answerId', a?.id || '');
      edit('answerHash', a?.hash || '');
    }
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Choose an actual answer"), data.answers.map(a => /*#__PURE__*/React.createElement("option", {
    key: a.id,
    value: a.id
  }, a.question)))), /*#__PURE__*/React.createElement("blockquote", null, data.answers.find(a => a.id === draft.answerId)?.response_notes), /*#__PURE__*/React.createElement("label", null, "Does it resolve the issue?", /*#__PURE__*/React.createElement("select", {
    "aria-label": "Does it resolve the issue?",
    value: draft.resolution,
    onChange: e => edit('resolution', e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: "unresolved"
  }, "Unresolved"), /*#__PURE__*/React.createElement("option", {
    value: "partial"
  }, "Partly resolved"), /*#__PURE__*/React.createElement("option", {
    value: "resolved"
  }, "Resolved in my assessment"))), /*#__PURE__*/React.createElement("p", null, "Keep management's recorded answer separate from your interpretation below.")), draft.kind === 'underweight' && /*#__PURE__*/React.createElement(React.Fragment, null, input('mandate', 'Portfolio / mandate'), input('benchmark', 'Benchmark'), input('asOf', 'Holdings and benchmark as of', 'date'), input('holdingPct', 'Portfolio weight %'), input('benchmarkPct', 'Benchmark weight %'), /*#__PURE__*/React.createElement("label", null, "Reason for nonownership / underweight", /*#__PURE__*/React.createElement("select", {
    "aria-label": "Reason for nonownership / underweight",
    value: draft.reason,
    onChange: e => edit('reason', e.target.value)
  }, ['valuation', 'quality', 'recovery', 'constraint', 'research_gap'].map(k => /*#__PURE__*/React.createElement("option", {
    key: k,
    value: k
  }, k.replace('_', ' '))))), /*#__PURE__*/React.createElement("label", null, "Current valuation assessment", /*#__PURE__*/React.createElement("textarea", {
    value: draft.valuationAssessment,
    onChange: e => edit('valuationAssessment', e.target.value)
  })), /*#__PURE__*/React.createElement("p", null, "User-reported point-in-time inputs. Business improvement alone does not establish investment attractiveness."), /*#__PURE__*/React.createElement("h3", null, "What would make us reconsider?"), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Separate business performance, valuation and portfolio constraints. Assessments below are yours; conditions are not automatically monitored yet. Source references are manually recorded."), draft.reviewConditions.map((c, i) => /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel",
    key: c.id
  }, /*#__PURE__*/React.createElement("label", null, "Condition type", /*#__PURE__*/React.createElement("select", {
    value: c.category,
    onChange: e => edit('reviewConditions', draft.reviewConditions.map((v, j) => j === i ? {
      ...v,
      category: e.target.value
    } : v))
  }, ['fundamental', 'valuation', 'constraint', 'research_gap'].map(k => /*#__PURE__*/React.createElement("option", {
    key: k,
    value: k
  }, k.replaceAll('_', ' '))))), /*#__PURE__*/React.createElement("label", null, "Observable condition", /*#__PURE__*/React.createElement("textarea", {
    value: c.trigger,
    onChange: e => edit('reviewConditions', draft.reviewConditions.map((v, j) => j === i ? {
      ...v,
      trigger: e.target.value
    } : v))
  })), /*#__PURE__*/React.createElement("label", null, "My assessment", /*#__PURE__*/React.createElement("select", {
    value: c.state,
    onChange: e => edit('reviewConditions', draft.reviewConditions.map((v, j) => j === i ? {
      ...v,
      state: e.target.value
    } : v))
  }, ['unassessed', 'not_met', 'partly_met', 'met'].map(k => /*#__PURE__*/React.createElement("option", {
    key: k,
    value: k
  }, k.replaceAll('_', ' '))))), [['evidence', 'Evidence and interpretation behind my assessment'], ['sourceReference', 'Source, publication date and passage / page · manual reference']].map(([key, label]) => /*#__PURE__*/React.createElement("label", {
    key: key
  }, label, /*#__PURE__*/React.createElement("textarea", {
    value: c[key],
    onChange: e => edit('reviewConditions', draft.reviewConditions.map((v, j) => j === i ? {
      ...v,
      [key]: e.target.value
    } : v))
  }))), /*#__PURE__*/React.createElement("button", {
    onClick: () => edit('reviewConditions', draft.reviewConditions.filter(v => v.id !== c.id))
  }, "Remove condition from draft"))), /*#__PURE__*/React.createElement("button", {
    disabled: draft.reviewConditions.length >= 12,
    onClick: () => edit('reviewConditions', [...draft.reviewConditions, {
      id: crypto.randomUUID(),
      category: 'fundamental',
      trigger: '',
      state: 'unassessed',
      evidence: '',
      sourceReference: ''
    }])
  }, "Add reconsideration condition"), /*#__PURE__*/React.createElement("label", null, "My underweight decision", /*#__PURE__*/React.createElement("select", {
    value: draft.reviewDecision,
    onChange: e => edit('reviewDecision', e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: "pending"
  }, "Pending review"), /*#__PURE__*/React.createElement("option", {
    value: "maintain"
  }, "Maintain underweight"), /*#__PURE__*/React.createElement("option", {
    value: "investigate"
  }, "Investigate further"), /*#__PURE__*/React.createElement("option", {
    value: "propose_change"
  }, "Propose portfolio change for consideration")))), /*#__PURE__*/React.createElement("label", null, "My interpretation and rationale", /*#__PURE__*/React.createElement("textarea", {
    value: draft.rationale,
    onChange: e => edit('rationale', e.target.value)
  })), /*#__PURE__*/React.createElement("label", null, "Next action / evidence that would change my view", /*#__PURE__*/React.createElement("textarea", {
    value: draft.nextAction,
    onChange: e => edit('nextAction', e.target.value)
  })), /*#__PURE__*/React.createElement("label", null, "State", /*#__PURE__*/React.createElement("select", {
    "aria-label": "State",
    value: draft.status,
    onChange: e => edit('status', e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: "open"
  }, "Open work"), /*#__PURE__*/React.createElement("option", {
    value: "reviewed"
  }, "Reviewed \xB7 conclusion recorded"), /*#__PURE__*/React.createElement("option", {
    value: "closed"
  }, "Closed \xB7 outcome recorded"))), draft.status !== 'open' && /*#__PURE__*/React.createElement("label", null, "Review outcome", /*#__PURE__*/React.createElement("textarea", {
    value: draft.outcome,
    onChange: e => edit('outcome', e.target.value)
  })), record && /*#__PURE__*/React.createElement("p", null, "Editing work revision ", record.revision, ". Draft case revision ", baseline, "; current case revision ", caseRevision, ". Recheck changed assumptions."), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    disabled: baseline !== caseRevision,
    onClick: save
  }, busy ? 'Saving…' : draft.kind === 'model' ? 'Save work and calculate' : 'Save review revision'), /*#__PURE__*/React.createElement("button", {
    onClick: () => replaceDraft(null, blank(initialKind))
  }, "New work record"), /*#__PURE__*/React.createElement("button", {
    onClick: load
  }, "Reload source and answer choices")), /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, message), data.records.map(r => /*#__PURE__*/React.createElement("article", {
    className: "workspace-panel",
    key: r.id
  }, /*#__PURE__*/React.createElement("h4", null, r.body.title, " \xB7 ", r.body.status), /*#__PURE__*/React.createElement("p", null, r.body.owner, " \xB7 due ", r.body.dueDate, " \xB7 case r", r.body.caseRevision, r.body.caseRevision !== caseRevision ? ' · Case has changed; review baseline' : ''), r.evidenceState && /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Evidence snapshot: ", r.evidenceState.replaceAll('_', ' '), ". Saved text matching does not establish currentness of the original disclosure."), /*#__PURE__*/React.createElement("p", null, r.body.rationale), /*#__PURE__*/React.createElement("p", null, "Next: ", r.body.nextAction), r.body.outcome && /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("strong", null, "Review outcome: "), r.body.outcome), r.body.resolution && /*#__PURE__*/React.createElement("p", null, "Answer resolution \xB7 analyst assessment: ", r.body.resolution), r.body.calculation && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("table", null, /*#__PURE__*/React.createElement("thead", null, /*#__PURE__*/React.createElement("tr", null, /*#__PURE__*/React.createElement("th", null, "Metric"), /*#__PURE__*/React.createElement("th", null, "Before"), /*#__PURE__*/React.createElement("th", null, "After"))), /*#__PURE__*/React.createElement("tbody", null, /*#__PURE__*/React.createElement("tr", null, /*#__PURE__*/React.createElement("th", null, "EPS"), /*#__PURE__*/React.createElement("td", null, r.body.calculation.beforeEPS), /*#__PURE__*/React.createElement("td", null, r.body.calculation.afterEPS)), /*#__PURE__*/React.createElement("tr", null, /*#__PURE__*/React.createElement("th", null, "Implied price \xB7 ", r.body.inputs.currency), /*#__PURE__*/React.createElement("td", null, r.body.calculation.beforeValue ?? 'Not applicable'), /*#__PURE__*/React.createElement("td", null, r.body.calculation.afterValue ?? 'Not applicable')))), /*#__PURE__*/React.createElement("p", null, "Operating profit change: ", r.body.calculation.operatingProfitDeltaMillions, "m \xB7 EPS change: ", r.body.calculation.epsDelta), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, r.body.calculation.scope), /*#__PURE__*/React.createElement("blockquote", null, r.body.passage, /*#__PURE__*/React.createElement("cite", null, " \u2014 ", r.body.sourceFilename))), r.body.answerSnapshot && /*#__PURE__*/React.createElement("blockquote", null, r.body.answerSnapshot.response_notes), r.body.kind === 'underweight' && /*#__PURE__*/React.createElement("p", null, "Active weight: ", r.body.activeWeightPct, "% \xB7 ", r.body.mandate, " vs ", r.body.benchmark, " \xB7 as of ", r.body.asOf, (Date.now() - Date.parse(r.body.asOf)) / 86400000 > 30 ? ' · Inputs older than 30 days' : ''), r.body.kind === 'followup' && /*#__PURE__*/React.createElement(MeetingAnswerProposal, {
    key: r.id + ':' + r.revision,
    api: api,
    ticker: ticker,
    record: r,
    disabled: disabled || busy
  }), r.body.kind === 'underweight' && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("button", {
    onClick: () => setDebate(r)
  }, "Challenge my underweight rationale"), /*#__PURE__*/React.createElement("p", null, "Review decision: ", (r.body.reviewDecision || 'pending').replaceAll('_', ' ')), r.body.reviewConditions?.map(c => /*#__PURE__*/React.createElement("p", {
    key: c.id
  }, /*#__PURE__*/React.createElement("strong", null, c.category, " \xB7 ", c.state.replaceAll('_', ' ')), ": ", c.trigger, /*#__PURE__*/React.createElement("br", null), c.evidence, /*#__PURE__*/React.createElement("br", null), /*#__PURE__*/React.createElement("small", null, c.sourceReference)))), /*#__PURE__*/React.createElement("button", {
    disabled: busy || !!submission || !!recoveryError,
    onClick: () => replaceDraft(r, r.body)
  }, "Review / update"), /*#__PURE__*/React.createElement("button", {
    onClick: async () => {
      try {
        setHistory({
          ...(await json(api, `/api/research/workbench/${ticker}/${r.id}/history`)),
          id: r.id
        });
      } catch (e) {
        setMessage(e.message);
      }
    }
  }, "Version history"), /*#__PURE__*/React.createElement("button", {
    onClick: () => {
      var text = [r.body.title, `Owner: ${r.body.owner} | Due: ${r.body.dueDate}`, `Interpretation: ${r.body.rationale}`, `Next action: ${r.body.nextAction}`, `Assumption: ${r.body.assumptionSnapshot?.claim || 'See saved case'} (case revision ${r.body.caseRevision})`, r.body.outcome ? `Reviewed outcome: ${r.body.outcome}` : '', r.body.kind === 'underweight' ? `Portfolio context: ${r.body.mandate} versus ${r.body.benchmark}; holding ${r.body.holdingPct}%, benchmark ${r.body.benchmarkPct}%, active weight ${r.body.activeWeightPct}%, as of ${r.body.asOf}.\nReason: ${r.body.reason}. Valuation assessment: ${r.body.valuationAssessment}` : '', r.body.resolution ? `Answer resolution (analyst assessment): ${r.body.resolution}` : '', r.body.calculation ? `Sensitivity: EPS ${r.body.calculation.beforeEPS} → ${r.body.calculation.afterEPS}; implied price ${r.body.calculation.beforeValue ?? 'N/A'} → ${r.body.calculation.afterValue ?? 'N/A'} ${r.body.inputs.currency}.\n${r.body.calculation.scope}` : '', r.body.answerSnapshot ? `Recorded answer: ${r.body.answerSnapshot.response_notes}` : '', r.body.passage ? `Source: ${r.body.sourceFilename}\n${r.body.passage}` : ''].filter(Boolean).join('\n\n');
      navigator.clipboard.writeText(text).then(() => setMessage('Research brief copied.'), () => setMessage('Clipboard unavailable.'));
    }
  }, "Copy research brief"))), history && /*#__PURE__*/React.createElement("details", {
    open: true
  }, /*#__PURE__*/React.createElement("summary", null, "Immutable work history"), history.versions.map(v => /*#__PURE__*/React.createElement("details", {
    key: v.revision
  }, /*#__PURE__*/React.createElement("summary", null, "Revision ", v.revision, " \xB7 ", v.body.status), /*#__PURE__*/React.createElement("button", {
    disabled: busy || !!submission || !!recoveryError || !data.records.some(r => r.id === history.id),
    onClick: () => replaceDraft(data.records.find(r => r.id === history.id), {
      ...v.body,
      status: 'open'
    }, true)
  }, "Use this version as a new draft"), /*#__PURE__*/React.createElement("pre", {
    style: {
      whiteSpace: 'pre-wrap',
      overflowWrap: 'anywhere'
    }
  }, JSON.stringify(v.body, null, 2)))), /*#__PURE__*/React.createElement("button", {
    onClick: () => setHistory(null)
  }, "Close history")), debate && /*#__PURE__*/React.createElement(ResearchChat, {
    api: api,
    context: {
      ticker,
      type: 'review',
      content: JSON.stringify({
        workId: debate.id,
        workRevision: debate.revision,
        review: debate.body
      })
    },
    initialMessage: "Challenge my underweight rationale using the recorded assumption and reconsideration conditions. Distinguish fundamental improvement, valuation, constraints and research gaps. Present the strongest opposing case and what additional evidence is needed. Do not claim a condition has been automatically verified or a portfolio change executed.",
    allowEdits: false,
    onClose: () => setDebate(null)
  }));
}
export function ResearchDueQueue({
  api,
  onOpen
}) {
  var [data, setData] = React.useState(null),
    [error, setError] = React.useState('');
  var sequence = React.useRef(0),
    alive = React.useRef(true);
  var load = async () => {
    var seq = ++sequence.current;
    try {
      var result = await json(api, '/api/research/workbench-queue');
      if (alive.current && seq === sequence.current) {
        setData(result);
        setError('');
      }
    } catch (e) {
      if (alive.current && seq === sequence.current) setError(e.message);
    }
  };
  React.useEffect(() => {
    alive.current = true;
    load();
    var timer = setInterval(load, 60000);
    return () => {
      alive.current = false;
      clearInterval(timer);
    };
  }, []);
  return /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("h2", null, "Research follow-through"), /*#__PURE__*/React.createElement("p", null, "Due issue reviews, meeting follow-ups, model work and underweight reviews."), /*#__PURE__*/React.createElement("button", {
    onClick: load
  }, "Refresh due work"), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error), data && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, data.items.filter(r => r.due).length, " due \xB7 as of ", data.asOf), !data.items.length && /*#__PURE__*/React.createElement("p", null, "No dated work is queued. Add a work record inside an investment case."), /*#__PURE__*/React.createElement("details", {
    open: data.items.some(r => r.due)
  }, /*#__PURE__*/React.createElement("summary", null, data.items.length, " open or scheduled reviews"), data.items.map(r => /*#__PURE__*/React.createElement("p", {
    key: r.kind + r.id
  }, /*#__PURE__*/React.createElement("button", {
    onClick: () => onOpen(r.ticker)
  }, r.ticker, " \xB7 ", r.title), " \xB7 ", r.due ? 'Due' : 'Upcoming', " ", r.dueDate, " \xB7 ", r.body.owner || 'Issue review'))), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, data.scope)));
}