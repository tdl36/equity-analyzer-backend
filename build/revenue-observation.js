import * as React from 'react';
export function RevenueObservation({
  api,
  ticker,
  model,
  onChange,
  onResearch,
  metric = "revenue"
}) {
  var isEbitda = metric === 'ebitda',
    label = isEbitda ? 'EBITDA' : 'revenue',
    field = isEbitda ? 'baseEbitda' : 'baseRevenue';
  var [runs, setRuns] = React.useState(null),
    [runId, setRunId] = React.useState(''),
    [options, setOptions] = React.useState([]),
    [choice, setChoice] = React.useState(''),
    [busy, setBusy] = React.useState(false),
    [error, setError] = React.useState('');
  var [form, setForm] = React.useState({
    token: '',
    unit: '',
    fiscalYear: '',
    currency: '',
    basis: '',
    locator: '',
    confirmed: false
  });
  var sequence = React.useRef(0),
    latest = React.useRef(model);
  latest.current = model;
  React.useEffect(() => () => {
    sequence.current++;
  }, []);
  var json = async (path, init = {}) => {
    var r = await fetch(api + path, {
      ...init,
      signal: AbortSignal.timeout(20000)
    });
    var d = await r.json();
    if (!r.ok) throw Error(d.error || 'Evidence unavailable');
    return d;
  };
  var load = async () => {
    var seq = ++sequence.current;
    setBusy(true);
    setError('');
    setOptions([]);
    setChoice('');
    setRunId('');
    try {
      var d = await json('/api/research/company/' + encodeURIComponent(ticker));
      if (seq === sequence.current) setRuns(d.runs.filter(r => r.status === 'complete'));
    } catch (e) {
      if (seq === sequence.current) setError(e.message);
    } finally {
      if (seq === sequence.current) setBusy(false);
    }
  };
  var chooseRun = async id => {
    var seq = ++sequence.current;
    setRunId(id);
    setOptions([]);
    setChoice('');
    setError('');
    setForm(f => ({
      ...f,
      token: '',
      confirmed: false
    }));
    if (!id) return;
    setBusy(true);
    try {
      var d = await json('/api/research/operating-model/' + encodeURIComponent(ticker) + '/' + metric + '-observation?runId=' + encodeURIComponent(id));
      if (seq === sequence.current) setOptions(d.candidates);
    } catch (e) {
      if (seq === sequence.current) setError(e.message);
    } finally {
      if (seq === sequence.current) setBusy(false);
    }
  };
  var selected = options[Number(choice)];
  var link = async () => {
    var seq = ++sequence.current,
      captured = JSON.stringify(model);
    setBusy(true);
    setError('');
    try {
      var d = await json('/api/research/operating-model/' + encodeURIComponent(ticker) + '/' + metric + '-observation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          ...selected,
          ...form
        })
      });
      if (seq !== sequence.current) return;
      if (JSON.stringify(latest.current) !== captured) throw Error('Model inputs changed during the check. Review them, then link again.');
      var o = d.observation;
      if (isEbitda && (Number(model.baseYear) !== o.fiscalYear || model.currency !== o.currency)) throw Error('EBITDA must match the model base fiscal year and currency. Update those inputs first.');
      onChange({
        ...model,
        baseEbitdaComparable: false,
        [field]: o.valueMillions,
        ...(isEbitda ? {
          ebitdaBasis: o.basis,
          baseEbitdaComparable: false
        } : {
          baseYear: o.fiscalYear,
          currency: o.currency
        }),
        [isEbitda ? 'baseEbitdaReference' : 'revenueReference']: `${o.filename} · ${o.locator} · FY${o.fiscalYear} · ${o.basis}`,
        [field + 'Observation']: o
      });
    } catch (e) {
      if (seq === sequence.current) setError(e.message);
    } finally {
      if (seq === sequence.current) setBusy(false);
    }
  };
  var observation = model[field + 'Observation'];
  return /*#__PURE__*/React.createElement("section", {
    "aria-label": "Base " + label + " evidence",
    className: "revenue-observation"
  }, /*#__PURE__*/React.createElement("style", null, `.revenue-observation{border:1px solid #b8c5ce;border-left:4px solid #667f90;padding:18px;margin:20px 0;background:#f6f8fa}.revenue-observation select{display:block;width:100%;max-width:100%;padding:10px;background:white;color:black;border:1px solid #888}.revenue-observation blockquote{white-space:pre-wrap;overflow-wrap:anywhere;margin:12px 0;border-left:2px solid #aaa;padding:12px}.revenue-observation code{overflow-wrap:anywhere;font-size:12px}`), /*#__PURE__*/React.createElement("h4", null, "Base ", label, " \xB7 source observation"), /*#__PURE__*/React.createElement("p", null, "Choose a printed number from a reviewed research passage. Charlie checks the passage, number and conversion; you confirm what that number means. Future growth, margins and the equity bridge remain your assumptions."), observation ? /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("strong", null, "Passage and printed number matched \xB7 analyst meaning confirmed")), /*#__PURE__*/React.createElement("p", null, observation.token, " ", observation.unit, " ", observation.currency, " \u2192 ", observation.valueMillions, " million \xB7 FY", observation.fiscalYear), /*#__PURE__*/React.createElement("p", null, observation.filename, " \xB7 ", observation.locator, " \xB7 ", observation.basis), /*#__PURE__*/React.createElement("blockquote", null, observation.excerpt), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Frozen evidence receipt"), /*#__PURE__*/React.createElement("p", null, "Research: ", /*#__PURE__*/React.createElement("code", null, observation.researchRunId)), /*#__PURE__*/React.createElement("p", null, "Original SHA-256: ", /*#__PURE__*/React.createElement("code", null, observation.originalHash)), /*#__PURE__*/React.createElement("p", null, "Extracted text SHA-256: ", /*#__PURE__*/React.createElement("code", null, observation.extractionHash)), /*#__PURE__*/React.createElement("p", null, observation.interpretation)), /*#__PURE__*/React.createElement("p", null, "Changing the base ", label, ", fiscal year, currency or linked EBITDA definition requires relinking or unlinking. Every calculation and new save rechecks the original. Historical revisions retain their original receipt."), /*#__PURE__*/React.createElement("button", {
    onClick: () => {
      var next = {
        ...model
      };
      delete next[field + 'Observation'];
      onChange(next);
    }
  }, "Unlink ", label, " evidence \xB7 keep the number")) : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: load
  }, busy ? 'Checking sources…' : 'Find ' + label + ' in completed research'), onResearch && /*#__PURE__*/React.createElement("button", {
    onClick: onResearch
  }, "Open Deep Research"), runs && runs.length === 0 && /*#__PURE__*/React.createElement("p", null, "No completed research runs for ", ticker, ". Save your draft, then select permitted originals in Deep Research. You can keep using a manual ", label, " reference meanwhile."), runs?.length > 0 && /*#__PURE__*/React.createElement("fieldset", {
    disabled: busy
  }, /*#__PURE__*/React.createElement("label", null, "Completed research", /*#__PURE__*/React.createElement("select", {
    value: runId,
    onChange: e => chooseRun(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Choose a run"), runs.map(r => /*#__PURE__*/React.createElement("option", {
    key: r.id,
    value: r.id
  }, r.created_at ? new Date(r.created_at).toLocaleString() : 'Date unavailable', " \xB7 ", r.id.slice(0, 8), " \xB7 case baseline R", r.baseline.revision)))), runId && options.length === 0 && !busy && /*#__PURE__*/React.createElement("p", null, "No eligible numeric passages. Claims must be reported facts with matched passages and supported review. Missing ", label, " stays manual; guidance, estimates and unresolved claims cannot be linked here."), options.length > 0 && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("label", null, "Reported passage \xB7 verify it is consolidated annual ", label, /*#__PURE__*/React.createElement("select", {
    value: choice,
    onChange: e => {
      setChoice(e.target.value);
      setForm(f => ({
        ...f,
        token: '',
        confirmed: false
      }));
    }
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Choose a passage"), options.map((o, i) => /*#__PURE__*/React.createElement("option", {
    key: i,
    value: i
  }, o.filename, " \xB7 ", o.statement.slice(0, 160))))), choice !== '' && selected && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("blockquote", null, selected.excerpt), /*#__PURE__*/React.createElement("label", null, "Exact printed ", label, " number", /*#__PURE__*/React.createElement("select", {
    value: form.token,
    onChange: e => setForm({
      ...form,
      token: e.target.value,
      confirmed: false
    })
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Choose the ", label, " figure, not a year or other metric"), selected.numbers.map(n => /*#__PURE__*/React.createElement("option", {
    key: n,
    value: n
  }, n)))), /*#__PURE__*/React.createElement("label", null, "Units printed in source", /*#__PURE__*/React.createElement("select", {
    value: form.unit,
    onChange: e => setForm({
      ...form,
      unit: e.target.value,
      confirmed: false
    })
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Choose source units"), ['units', 'thousands', 'millions', 'billions'].map(u => /*#__PURE__*/React.createElement("option", {
    key: u
  }, u)))), [['fiscalYear', label + ' fiscal year'], ['currency', label + ' currency · three-letter code'], ['basis', label + ' accounting basis and definition'], ['locator', 'Original page or section']].map(([k, label]) => /*#__PURE__*/React.createElement("label", {
    key: k
  }, label, /*#__PURE__*/React.createElement("input", {
    value: form[k],
    onChange: e => setForm({
      ...form,
      [k]: e.target.value,
      confirmed: false
    })
  }))), /*#__PURE__*/React.createElement("label", null, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: form.confirmed,
    onChange: e => setForm({
      ...form,
      confirmed: e.target.checked
    })
  }), "I inspected the original and confirm this is ", ticker, " consolidated annual ", isEbitda ? 'positive EBITDA' : label, ", with the stated year, currency, units and accounting basis. It is not a forecast, segment figure or a different issuer."), /*#__PURE__*/React.createElement("button", {
    disabled: !form.confirmed || !form.token || !form.unit,
    onClick: link
  }, "Check and use ", label, " in draft"), /*#__PURE__*/React.createElement("p", null, isEbitda ? 'This sets positive base EBITDA and its definition in the draft. Confirm comparability with forecast margins before calculating.' : 'This changes base revenue, base fiscal year and currency in your draft.', " It does not save a case or launch research."))))), /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error));
}