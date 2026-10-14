import * as React from 'react';
import { historicalCandidate, historicalPlan } from './historical-review-model.mjs';
export function HistoricalCandidate({
  url,
  baseline,
  originals
}) {
  var [date, setDate] = React.useState(''),
    [price, setPrice] = React.useState(''),
    [earnings, setEarnings] = React.useState(''),
    [state, setState] = React.useState({}),
    active = React.useRef(null),
    generation = React.useRef(0);
  var identity = JSON.stringify([url, baseline, originals]);
  var clear = () => {
    generation.current++;
    active.current?.abort();
    setState({});
  };
  React.useEffect(() => {
    clear();
    setDate('');
    setPrice('');
    setEarnings('');
    return () => {
      generation.current++;
      active.current?.abort();
    };
  }, [identity]);
  var prices = originals.filter(r => r.kind === 'daily_prices'),
    facts = originals.filter(r => r.kind === 'sec_companyfacts');
  if (!prices.length || !facts.length) return null;
  async function preview(e, automatic = false) {
    e.preventDefault();
    clear();
    var token = ++generation.current,
      controller = new AbortController();
    active.current = controller;
    var chosenPrice = prices.find(r => r.id === price),
      chosenEarnings = facts.find(r => r.id === earnings);
    if (!automatic && (!chosenPrice || !chosenEarnings)) {
      setState({
        error: 'Reload saved evidence and choose the retained originals again.'
      });
      return;
    }
    var receipt = r => ({
      id: r.id,
      sha256: r.sha256
    });
    var selection = automatic ? {
      baseline,
      date
    } : {
      baseline,
      observation: {
        date,
        priceSource: receipt(chosenPrice),
        earningsSource: receipt(chosenEarnings)
      }
    };
    setState({
      busy: true
    });
    try {
      var response = await fetch(url + (automatic ? '/historical-plan?' : '/historical-candidate?') + new URLSearchParams({
        selection: JSON.stringify(selection)
      }), {
        signal: AbortSignal.any([controller.signal, AbortSignal.timeout(30000)])
      });
      if (!response.ok) throw Error('These historical inputs could not be verified. Reload the saved evidence and check the selected date.');
      var data = await response.json();
      var result = automatic ? historicalPlan(data, baseline, date) : {
        value: historicalCandidate(data, selection)
      };
      if (token === generation.current) setState(result);
    } catch (error) {
      if (token === generation.current) setState({
        error: error.message
      });
    }
  }
  return /*#__PURE__*/React.createElement("section", {
    "aria-label": "Preview historical inputs"
  }, /*#__PURE__*/React.createElement("h5", null, "Preview retained inputs"), /*#__PURE__*/React.createElement("p", null, "Choose an exact completed trading date and retained originals. This reads existing evidence only."), /*#__PURE__*/React.createElement("form", {
    onSubmit: preview,
    style: {
      display: 'grid',
      gap: 12,
      maxWidth: 600
    }
  }, /*#__PURE__*/React.createElement("label", null, "Trading date", /*#__PURE__*/React.createElement("input", {
    style: {
      display: 'block',
      width: '100%',
      minWidth: 0
    },
    type: "date",
    required: true,
    value: date,
    onChange: e => {
      clear();
      setDate(e.target.value);
    }
  })), /*#__PURE__*/React.createElement("button", {
    type: "button",
    disabled: state.busy || !date,
    onClick: e => preview(e, true)
  }, "Find matching retained inputs"), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Choose originals manually"), /*#__PURE__*/React.createElement("label", null, "Price original", /*#__PURE__*/React.createElement("select", {
    style: {
      display: 'block',
      width: '100%',
      minWidth: 0
    },
    "aria-label": "Price original",
    required: true,
    value: price,
    onChange: e => {
      clear();
      setPrice(e.target.value);
    }
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Select retained prices"), prices.map((r, i) => /*#__PURE__*/React.createElement("option", {
    key: r.id,
    value: r.id
  }, "Daily prices \xB7 ", r.retrievedDate, " \xB7 ", i + 1)))), /*#__PURE__*/React.createElement("label", null, "Earnings original", /*#__PURE__*/React.createElement("select", {
    style: {
      display: 'block',
      width: '100%',
      minWidth: 0
    },
    "aria-label": "Earnings original",
    required: true,
    value: earnings,
    onChange: e => {
      clear();
      setEarnings(e.target.value);
    }
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Select retained earnings"), facts.map((r, i) => /*#__PURE__*/React.createElement("option", {
    key: r.id,
    value: r.id
  }, "SEC company facts \xB7 ", r.retrievedDate, " \xB7 ", i + 1)))), /*#__PURE__*/React.createElement("button", {
    disabled: state.busy || !date || !price || !earnings
  }, "Preview retained inputs"))), state.coverage && /*#__PURE__*/React.createElement("div", {
    role: "status"
  }, state.coverage.map((gap, i) => /*#__PURE__*/React.createElement("p", {
    key: i
  }, gap))), state.busy && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Checking retained inputs\u2026"), state.error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, state.error), state.value && /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h5", null, "Unapproved input preview \xB7 ", state.value.date), /*#__PURE__*/React.createElement("p", null, "Raw closing price: ", state.value.price), /*#__PURE__*/React.createElement("p", null, "Annual diluted EPS: ", state.value.earnings), /*#__PURE__*/React.createElement("p", null, "Annual period ending ", state.value.periodEnd, " \xB7 filed ", state.value.publishedDate), state.value.gaps.map((g, i) => /*#__PURE__*/React.createElement("p", {
    key: i
  }, g)), /*#__PURE__*/React.createElement("p", null, "These amounts do not establish a valuation. Currency and share-class mapping, corporate actions, filing completeness and source provenance still require review. Nothing has been saved or approved.")));
}