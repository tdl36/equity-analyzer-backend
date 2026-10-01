import * as React from 'react';
import { RevenueObservation } from './revenue-observation';
var names = ['bear', 'base', 'bull'];
var numeric = [['growthPct', 'Revenue CAGR (%)'], ['marginPct', 'Target EBITDA margin (%)'], ['multiple', 'Target EV / EBITDA (×)'], ['netDebt', 'Target net debt (millions; negative = net cash)'], ['otherClaims', 'Other senior claims (millions)'], ['nonOperatingAssets', 'Nonoperating assets (millions)'], ['shares', 'Target diluted shares (millions)']];
export var newOperatingModel = () => ({
  version: 'ev-ebitda-v1',
  method: 'ev_ebitda',
  units: 'millions',
  suitable: false,
  currency: '',
  baseYear: '',
  targetYear: '',
  asOf: '',
  baseRevenue: '',
  referencePrice: '',
  revenueReference: '',
  priceReference: '',
  ebitdaBasis: '',
  scenarios: Object.fromEntries(names.map(n => [n, {
    ...Object.fromEntries(numeric.map(([k]) => [k, ''])),
    rationale: ''
  }]))
});
export function OperatingModel({
  api,
  ticker,
  body,
  revision,
  busy,
  dirty,
  onChange,
  onSave,
  onResearch
}) {
  var model = body.operatingModel,
    signature = JSON.stringify(model);
  var [preview, setPreview] = React.useState(null),
    [error, setError] = React.useState(''),
    [calculating, setCalculating] = React.useState(false);
  var latest = React.useRef(signature),
    sequence = React.useRef(0);
  latest.current = signature;
  React.useEffect(() => () => {
    sequence.current++;
  }, []);
  var edit = m => {
    setError('');
    onChange({
      ...body,
      operatingModel: m
    });
  };
  var set = (key, value) => edit({
    ...model,
    [key]: value
  });
  var scenario = (name, key, value) => edit({
    ...model,
    scenarios: {
      ...model.scenarios,
      [name]: {
        ...model.scenarios[name],
        [key]: value
      }
    }
  });
  var calculate = async () => {
    var mine = ++sequence.current,
      captured = signature;
    setCalculating(true);
    setError('');
    try {
      var r = await fetch(api + '/api/research/operating-model/preview', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          ...model,
          ticker
        }),
        signal: AbortSignal.timeout(20000)
      });
      var d = await r.json();
      if (!r.ok) throw Error(d.error || 'Calculation unavailable');
      if (mine === sequence.current && latest.current === captured) setPreview({
        signature: captured,
        model: d.model
      });
    } catch (e) {
      if (mine === sequence.current) setError(e.message);
    } finally {
      if (mine === sequence.current) setCalculating(false);
    }
  };
  var output = preview && preview.signature === signature ? preview.model : !dirty ? model : null;
  return /*#__PURE__*/React.createElement("section", {
    className: "operating-model"
  }, /*#__PURE__*/React.createElement("style", null, `.operating-model{font-family:Calibri,sans-serif;color:#000;background:#fff;border-radius:12px;padding:24px}.operating-model *{color:#000!important}.operating-model p{margin:12px 0}.operating-model h3{font-size:24px}.operating-model h4{font-size:20px;font-weight:bold;margin:16px 0}.operating-model label{display:block;margin:12px 0}.operating-model input:not([type=checkbox]),.operating-model textarea{display:block;background:#fff!important;border:1px solid #888;border-radius:4px;padding:10px;width:100%;min-width:0}.operating-model textarea{min-height:90px}.operating-model button{background:#eee!important;border:1px solid #999;border-radius:5px;padding:10px;margin:8px 8px 8px 0}.operating-model button:disabled{opacity:.5}.operating-model article{border:1px solid #ccc;padding:16px;border-radius:8px;margin:12px 0}.operating-model-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.operating-model td,.operating-model th{padding:10px;border-bottom:1px solid #ddd;text-align:left}.operating-model small{display:block}.operating-model table{width:100%}@media(max-width:850px){.operating-model-grid{grid-template-columns:1fr}.operating-model{padding:16px}}`), /*#__PURE__*/React.createElement("h3", null, ticker, " \xB7 Operating scenarios"), /*#__PURE__*/React.createElement("p", null, "Translate explicit assumptions into revenue, EBITDA and equity value. All calculations run in code and use no model credits. Inputs are your assumptions; source references remain unverified."), !model ? /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, "The first template uses EV/EBITDA for businesses where positive EBITDA is an appropriate valuation basis. It is not a bank, insurer, REIT, loss-making or recovery model. Existing EPS/P-E sensitivities remain in Current thesis."), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => edit(newOperatingModel())
  }, "Add EV/EBITDA model")) : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, dirty ? 'Unsaved model / case edits' : `Saved with case R${revision}`, " \xB7 ", model.version, ". Money and shares must both be in millions, in the same currency and forecast period. Share prices are per share."), /*#__PURE__*/React.createElement("fieldset", {
    disabled: busy
  }, /*#__PURE__*/React.createElement("label", null, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: model.suitable,
    onChange: e => set('suitable', e.target.checked)
  }), "I consider positive-EBITDA valuation appropriate for this company and scenario horizon."), /*#__PURE__*/React.createElement("div", {
    className: "operating-model-grid"
  }, [['currency', 'Currency (three-letter code)'], ['baseYear', 'Base fiscal year'], ['targetYear', 'Target fiscal year · 1–5 years later'], ['baseRevenue', 'Base-year revenue (millions)'], ['referencePrice', 'Reference share price'], ['asOf', 'Reference-price date']].map(([key, label]) => /*#__PURE__*/React.createElement("label", {
    key: key
  }, label, /*#__PURE__*/React.createElement("input", {
    type: key === 'asOf' ? 'date' : 'text',
    value: model[key],
    inputMode: ['baseYear', 'targetYear', 'baseRevenue', 'referencePrice'].includes(key) ? 'decimal' : undefined,
    onChange: e => set(key, e.target.value)
  })))), [['revenueReference', 'Base revenue · source, period and definition'], ['priceReference', 'Reference price · source and date'], ['ebitdaBasis', 'EBITDA basis · reported/adjusted definition and adjustments']].map(([key, label]) => /*#__PURE__*/React.createElement("label", {
    key: key
  }, label, /*#__PURE__*/React.createElement("textarea", {
    value: model[key],
    maxLength: 1800,
    onChange: e => set(key, e.target.value)
  }))), /*#__PURE__*/React.createElement(RevenueObservation, {
    api: api,
    ticker: ticker,
    model: model,
    onChange: edit,
    onResearch: onResearch
  }), /*#__PURE__*/React.createElement("div", {
    className: "operating-model-grid"
  }, names.map(name => /*#__PURE__*/React.createElement("article", {
    key: name
  }, /*#__PURE__*/React.createElement("h4", null, name[0].toUpperCase() + name.slice(1)), numeric.map(([key, label]) => /*#__PURE__*/React.createElement("label", {
    key: key
  }, label, /*#__PURE__*/React.createElement("input", {
    inputMode: "decimal",
    value: model.scenarios[name][key],
    onChange: e => scenario(name, key, e.target.value)
  }))), /*#__PURE__*/React.createElement("label", null, "Assumptions, evidence and source references", /*#__PURE__*/React.createElement("textarea", {
    value: model.scenarios[name].rationale,
    maxLength: 1800,
    onChange: e => scenario(name, 'rationale', e.target.value)
  }))))), /*#__PURE__*/React.createElement("p", null, "Revenue = base revenue \xD7 (1 + CAGR) ^ fiscal-year interval. EBITDA = target revenue \xD7 target margin. Enterprise value = EBITDA \xD7 multiple. Equity value = max(0, enterprise value \u2212 net debt \u2212 other senior claims + nonoperating assets). Per-share value = equity value \xF7 diluted shares."), /*#__PURE__*/React.createElement("button", {
    disabled: calculating,
    onClick: calculate
  }, calculating ? 'Calculating…' : 'Calculate draft'), /*#__PURE__*/React.createElement("button", {
    disabled: calculating || !dirty || preview?.signature !== signature,
    onClick: () => onSave({
      body
    })
  }, "Save case with these model inputs"), /*#__PURE__*/React.createElement("button", {
    onClick: () => {
      var next = {
        ...body
      };
      delete next.operatingModel;
      onChange(next);
      setPreview(null);
    }
  }, "Remove model from draft")), /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error), output?.results ? /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("h4", null, dirty ? 'Draft calculations · not saved' : `Calculated from saved case R${revision}`), /*#__PURE__*/React.createElement("div", {
    style: {
      overflowX: 'auto'
    }
  }, /*#__PURE__*/React.createElement("table", null, /*#__PURE__*/React.createElement("thead", null, /*#__PURE__*/React.createElement("tr", null, /*#__PURE__*/React.createElement("th", null, "Scenario"), /*#__PURE__*/React.createElement("th", null, "Revenue (m)"), /*#__PURE__*/React.createElement("th", null, "EBITDA (m)"), /*#__PURE__*/React.createElement("th", null, "EV (m)"), /*#__PURE__*/React.createElement("th", null, "Equity (m)"), /*#__PURE__*/React.createElement("th", null, "Price (", output.currency, ")"), /*#__PURE__*/React.createElement("th", null, "Price return"))), /*#__PURE__*/React.createElement("tbody", null, names.map(name => {
    var r = output.results[name];
    return /*#__PURE__*/React.createElement("tr", {
      key: name
    }, /*#__PURE__*/React.createElement("th", null, name), ['revenue', 'ebitda', 'enterpriseValue', 'equityValue', 'impliedPrice'].map(k => /*#__PURE__*/React.createElement("td", {
      key: k
    }, r[k])), /*#__PURE__*/React.createElement("td", null, r.priceReturnPct, "%"));
  })))), /*#__PURE__*/React.createElement("h4", null, "Conditional reverse valuation"), /*#__PURE__*/React.createElement("p", null, "At the reference price, base-case EBITDA required is ", output.results.base.reverseValid ? `${output.results.base.impliedEbitdaAtReferencePrice} million ${output.currency}` : 'not interpretable (nonpositive implied EBITDA)', ". This holds the base multiple, net debt, other claims, nonoperating assets and shares fixed. It is not a consensus estimate or a unique market-implied forecast."), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Base-case EBITDA / multiple sensitivity"), /*#__PURE__*/React.createElement("div", {
    style: {
      overflowX: 'auto'
    }
  }, /*#__PURE__*/React.createElement("table", null, /*#__PURE__*/React.createElement("thead", null, /*#__PURE__*/React.createElement("tr", null, /*#__PURE__*/React.createElement("th", null, "EBITDA change"), /*#__PURE__*/React.createElement("th", null, "EV/EBITDA"), /*#__PURE__*/React.createElement("th", null, "Implied price (", output.currency, ")"))), /*#__PURE__*/React.createElement("tbody", null, output.sensitivity.map((s, i) => /*#__PURE__*/React.createElement("tr", {
    key: i
  }, /*#__PURE__*/React.createElement("td", null, s.ebitdaChangePct, "%"), /*#__PURE__*/React.createElement("td", null, s.multiple, "\xD7"), /*#__PURE__*/React.createElement("td", null, s.impliedPrice))))))), output.warnings.map(w => /*#__PURE__*/React.createElement("p", {
    key: w
  }, w))) : /*#__PURE__*/React.createElement("p", null, "Calculate after changing inputs. Earlier results are hidden until they match the current inputs.")));
}