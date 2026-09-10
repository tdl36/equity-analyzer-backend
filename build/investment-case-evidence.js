import * as React from 'react';
export function CaseEvidence({
  api,
  ticker,
  body,
  disabled,
  onCommit
}) {
  var [rows, setRows] = React.useState([]),
    [error, setError] = React.useState(''),
    [loading, setLoading] = React.useState(true),
    [chosen, setChosen] = React.useState(''),
    [assumptionId, setAssumption] = React.useState(''),
    [field, setField] = React.useState('support');
  React.useEffect(() => {
    var live = true;
    setRows([]);
    setChosen('');
    setAssumption('');
    setError('');
    setLoading(true);
    fetch(api + '/api/research/investment-case/' + encodeURIComponent(ticker) + '/source-proposals', {
      signal: AbortSignal.timeout(20000)
    }).then(async r => {
      var d = await r.json();
      if (!r.ok) throw Error(d.error || 'Sources unavailable');
      if (live) setRows(d.proposals || []);
    }).catch(e => {
      if (live) setError(e.message);
    }).finally(() => {
      if (live) setLoading(false);
    });
    return () => {
      live = false;
    };
  }, [api, ticker]);
  var options = rows.flatMap(job => (job.result?.changes || []).filter(c => c.passageMatched && c.reviewPassed).map(change => ({
    key: job.id + ':' + change.id,
    job,
    change
  })));
  var selected = options.find(o => o.key === chosen),
    assumption = body.assumptions.find(a => a.id === assumptionId);
  return /*#__PURE__*/React.createElement("details", {
    className: "amendment-card"
  }, /*#__PURE__*/React.createElement("summary", null, "Connect reviewed research to an assumption"), /*#__PURE__*/React.createElement("p", null, "Choose an existing research change and review its excerpt before adding it to your investment case. A passage match establishes provenance at generation; it does not prove the interpretation or that the source is still current."), loading && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading source proposals\u2026"), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error), !loading && !error && !options.length && /*#__PURE__*/React.createElement("p", null, "No eligible source proposals yet. Generate a source comparison in Evidence & changes first. Only changes with a matching passage and a passed model review appear here."), !!options.length && /*#__PURE__*/React.createElement("fieldset", {
    disabled: disabled
  }, /*#__PURE__*/React.createElement("label", null, "Reviewed research change", /*#__PURE__*/React.createElement("select", {
    value: chosen,
    onChange: e => setChosen(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Choose a source-supported change"), options.map(o => /*#__PURE__*/React.createElement("option", {
    key: o.key,
    value: o.key
  }, new Date(o.job.created_at).toLocaleDateString(), " \xB7 ", o.change.after.slice(0, 110))))), selected && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, selected.change.reason), selected.change.evidence.filter(e => e.status === 'passage_matched').map((e, i) => /*#__PURE__*/React.createElement("section", {
    key: i
  }, /*#__PURE__*/React.createElement("strong", null, selected.job.result.sources.find(s => s.id === e.sourceId)?.filename || 'Source snapshot'), /*#__PURE__*/React.createElement("blockquote", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, e.excerpt))), /*#__PURE__*/React.createElement("label", null, "Investment assumption", /*#__PURE__*/React.createElement("select", {
    value: assumptionId,
    onChange: e => setAssumption(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Choose an assumption"), body.assumptions.map((a, i) => /*#__PURE__*/React.createElement("option", {
    key: a.id,
    value: a.id
  }, i + 1, ". ", a.claim.slice(0, 120))))), /*#__PURE__*/React.createElement("label", null, "Update which field?", /*#__PURE__*/React.createElement("select", {
    value: field,
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
  }, assumption[field] || 'Not recorded'), /*#__PURE__*/React.createElement("h4", null, "Proposed replacement"), /*#__PURE__*/React.createElement("p", {
    style: {
      whiteSpace: 'pre-wrap'
    }
  }, selected.change.after), /*#__PURE__*/React.createElement("p", null, "This replaces the selected field. Your assumption and its basis remain unchanged; the original quotation is retained separately."), /*#__PURE__*/React.createElement("button", {
    disabled: assumption[field] === selected.change.after,
    onClick: () => onCommit({
      mode: 'source_change',
      sourceChange: {
        jobId: selected.job.id,
        changeId: selected.change.id,
        assumptionId,
        field
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
  })));
}