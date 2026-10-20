import * as React from 'react';
import { rewriteScope, rewriteStillMatches } from './package-rewrite-model.mjs';
export function PackageRewrite({
  runId,
  storageKey,
  fields,
  onKeep,
  api,
  disabled
}) {
  var [show, setShow] = React.useState(false),
    [scope, setScope] = React.useState('*'),
    [instructions, setInstructions] = React.useState(''),
    [plan, setPlan] = React.useState(null),
    [job, setJob] = React.useState(null),
    [pending, setPending] = React.useState(null),
    [busy, setBusy] = React.useState(false),
    [error, setError] = React.useState('');
  var lock = React.useRef(false),
    key = storageKey + ':rewrite';
  React.useEffect(() => {
    try {
      var saved = JSON.parse(sessionStorage.getItem(key) || 'null');
      if (saved) {
        setPending(saved);
        setShow(true);
      }
    } catch {
      setError('Rewrite recovery could not be loaded. Keep this tab open.');
    }
  }, [key]);
  async function call(path, body) {
    var r = await fetch(api + path, {
      ...(body ? {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
      } : {}),
      signal: AbortSignal.timeout(60000)
    });
    var value = await r.json();
    if (!r.ok) throw Error(value.error || 'Unable to read the saved rewrite.');
    return value;
  }
  async function act(fn) {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    setError('');
    try {
      await fn();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
      lock.current = false;
    }
  }
  function store(value) {
    sessionStorage.setItem(key, JSON.stringify(value));
    if (sessionStorage.getItem(key) !== JSON.stringify(value)) throw Error('Browser recovery storage is unavailable; generation was not started.');
    setPending(value);
  }
  async function prepare() {
    var body = {
      requestId: crypto.randomUUID(),
      writingPolicy: 'synthesis-v2',
      instructions,
      ...rewriteScope(fields, scope)
    };
    var quote = await call(`/api/research/stock-analysis-run/${runId}/rewrite/prepare`, body);
    setPlan({
      body,
      quote
    });
  }
  async function start() {
    var body = {
      ...plan.body,
      fingerprint: plan.quote.fingerprint,
      confirmed: true
    };
    store({
      body,
      runId
    });
    setPlan(null);
    await call(`/api/research/stock-analysis-run/${runId}/rewrite/start`, body);
    setJob(await call('/api/research/stock-analysis-run/' + body.requestId));
  }
  async function check() {
    setJob(await call('/api/research/stock-analysis-run/' + pending.body.requestId));
  }
  React.useEffect(() => {
    if (!pending || ['complete', 'attention', 'cancelled'].includes(job?.status)) return;
    var active = true,
      count = 0;
    var timer = setInterval(async () => {
      if (++count > 24) {
        clearInterval(timer);
        return;
      }
      try {
        var value = await call('/api/research/stock-analysis-run/' + pending.body.requestId);
        if (active) setJob(value);
      } catch {/* Explicit Check saved result remains available. */}
    }, 5000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [pending?.body.requestId, job?.status]);
  function clear() {
    sessionStorage.removeItem(key);
    setPending(null);
    setJob(null);
    setPlan(null);
  }
  function keep() {
    if (!rewriteStillMatches(pending.body, fields)) throw Error('Your package wording or context changed after generation. Copy the alternative you want to retain, or restore the original wording before keeping this version.');
    onKeep(job.state.rewrite);
    clear();
    setShow(false);
  }
  return /*#__PURE__*/React.createElement("aside", {
    className: "pe-note"
  }, /*#__PURE__*/React.createElement("button", {
    disabled: disabled || busy,
    onClick: () => setShow(!show)
  }, "Revise with AI"), show && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, "Create a new alternative from the same saved originals. Charlie re-reads their returned text before showing the cost. No new source collection, automatic approval, or changes to your saved package. The whole-package option revises the thesis and stock summary together with the one-pager, using one consistent investment case. Individual section changes still need a consistency review."), !pending && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("label", null, "What should change?", /*#__PURE__*/React.createElement("select", {
    style: {
      maxWidth: "100%"
    },
    disabled: busy,
    value: scope,
    onChange: e => {
      setScope(e.target.value);
      setPlan(null);
    }
  }, /*#__PURE__*/React.createElement("option", {
    value: "*"
  }, "Whole package \xB7 thesis, summary & one-pager"), fields.map(f => /*#__PURE__*/React.createElement("option", {
    key: f.id,
    value: f.id
  }, f.label)))), /*#__PURE__*/React.createElement("label", null, "Instructions", /*#__PURE__*/React.createElement("textarea", {
    rows: 3,
    maxLength: 4000,
    disabled: busy,
    value: instructions,
    onChange: e => {
      setInstructions(e.target.value);
      setPlan(null);
    },
    placeholder: "For example: explain why the business is attractive, make the investment case specific, and keep uncertainties visible."
  })), /*#__PURE__*/React.createElement("button", {
    disabled: disabled || busy || !instructions.trim(),
    onClick: () => act(prepare)
  }, "Review scope & cost"), plan && /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", null, plan.quote.fieldCount, " fields \xB7 ", plan.quote.sourceCount, " saved sources \xB7 ", plan.quote.calls, " model call \xB7 estimated $", plan.quote.estimatedUsd.toFixed(3), " USD"), /*#__PURE__*/React.createElement("p", null, plan.quote.notice), plan.quote.contextFieldCount > 0 && /*#__PURE__*/React.createElement("p", null, plan.quote.contextFieldCount, " other sections are included as context only and will not be rewritten."), plan.quote.sourceLimitations?.length > 0 && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Saved source limitations: ", Array.from(new Set(plan.quote.sourceLimitations)).join(" "), " Review these limitations before authorizing."), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Source coverage for this revision"), /*#__PURE__*/React.createElement("p", null, "Extracted text is not proof of complete visual, table or chart review. The coverage below describes the evidence prepared for this specific revision."), plan.quote.sourceCoverage?.map((s, i) => /*#__PURE__*/React.createElement("div", {
    key: s.id || i
  }, /*#__PURE__*/React.createElement("strong", null, s.filename), /*#__PURE__*/React.createElement("p", null, s.status === 'omitted_text' ? 'Text omitted' : s.status === 'recorded_extraction' ? 'Returned text extraction recorded' : 'Coverage unknown'), s.warnings.map((w, j) => /*#__PURE__*/React.createElement("p", {
    key: j
  }, w))))), /*#__PURE__*/React.createElement("p", null, "Model: ", plan.quote.model, ". Your saved writing preferences are included."), /*#__PURE__*/React.createElement("button", {
    disabled: disabled || busy || !rewriteStillMatches(plan.body, fields),
    onClick: () => act(start)
  }, "Authorize cost & generate alternative"))), pending && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, "Saved rewrite request: ", pending.body.requestId, ". ", job?.status || 'Submission awaiting confirmation'), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => act(check)
  }, "Check saved result"), !job && /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => act(async () => {
      await call(`/api/research/stock-analysis-run/${pending.runId}/rewrite/start`, pending.body);
      await check();
    })
  }, "Recover this exact request"), job?.error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, job.error, " No automatic paid retry will occur."), job?.status === 'complete' && job.state?.rewrite && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, "Review every change against the evidence. Keeping it only places it in your editable draft; save a package version separately."), pending.body.fields.map(f => /*#__PURE__*/React.createElement("article", {
    key: f.id
  }, /*#__PURE__*/React.createElement("h4", null, f.label), /*#__PURE__*/React.createElement("strong", null, "Before"), /*#__PURE__*/React.createElement("blockquote", null, f.text), /*#__PURE__*/React.createElement("strong", null, "Alternative"), /*#__PURE__*/React.createElement("blockquote", null, job.state.rewrite[f.id]))), /*#__PURE__*/React.createElement("button", {
    disabled: busy || disabled,
    onClick: () => act(keep)
  }, "Keep alternative in my draft"), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: clear
  }, "Discard alternative")), job?.status === 'attention' && /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => act(async () => {
      await call('/api/research/stock-analysis-run/' + pending.body.requestId + '/stop', {});
      await check();
    })
  }, "Stop this rewrite \xB7 preserve response"), job?.status === 'cancelled' && /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: clear
  }, "Dismiss stopped rewrite"), ['attention', 'cancelled'].includes(job?.status) && /*#__PURE__*/React.createElement("p", null, "Inspect ", /*#__PURE__*/React.createElement("a", {
    href: api + "/api/research/stock-analysis-run/" + pending.body.requestId + "/usage-evidence",
    target: "_blank",
    rel: "noreferrer"
  }, "recorded usage"), " and ", /*#__PURE__*/React.createElement("a", {
    href: api + "/api/research/stock-analysis-run/" + pending.body.requestId + "/response/package-rewrite",
    target: "_blank",
    rel: "noreferrer"
  }, "the saved response, if received"), ", before authorizing any new request. Your original wording is unchanged.")), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error)));
}