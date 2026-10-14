import * as React from 'react';
import { HistoricalCandidate } from './historical-candidate';
import { historicalReview, historicalInventory } from './historical-review-model.mjs';

// Read-only inspection; never retrieves providers or writes research.
export function HistoricalReview({
  api,
  assignmentId,
  editionId
}) {
  var [state, setState] = React.useState({
      busy: false
    }),
    generation = React.useRef(0),
    activeRequest = React.useRef(null);
  var url = `${api}/api/research/assignments/${encodeURIComponent(assignmentId)}/onepager-editions/${encodeURIComponent(editionId)}`;
  React.useEffect(() => {
    generation.current++;
    activeRequest.current?.abort();
    setState({
      busy: false
    });
    return () => {
      generation.current++;
      activeRequest.current?.abort();
    };
  }, [url]);
  async function read(path, signal) {
    var response = await fetch(path, {
      signal: AbortSignal.any([signal, AbortSignal.timeout(30000)])
    });
    var data = await response.json();
    if (!response.ok) throw Error(data.error || 'Historical evidence is unavailable.');
    return data;
  }
  async function load(id) {
    var token = ++generation.current;
    activeRequest.current?.abort();
    var controller = new AbortController();
    activeRequest.current = controller;
    setState({
      busy: true
    });
    try {
      if (id) {
        var review = historicalReview(await read(`${url}/historical-reviews/${encodeURIComponent(id)}`, controller.signal), assignmentId, editionId, id);
        if (token === generation.current) setState({
          busy: false,
          review
        });
      } else {
        var [sources, retrievals, reviews] = await Promise.all(['historical-originals', 'historical-retrievals', 'historical-reviews'].map(path => read(`${url}/${path}`, controller.signal)));
        var inventory = historicalInventory(sources, retrievals, reviews, assignmentId, editionId);
        if (token === generation.current) setState({
          busy: false,
          ...inventory
        });
      }
    } catch (error) {
      controller.abort();
      if (token === generation.current) setState({
        busy: false,
        error: error.message
      });
    }
  }
  return /*#__PURE__*/React.createElement("details", {
    className: "sa-prepare"
  }, /*#__PURE__*/React.createElement("summary", null, "Historical valuation evidence"), /*#__PURE__*/React.createElement("p", null, "Inspect retained sources and pending checks for this edition. Matching source amounts alone does not establish a valid historical valuation. Share classes, currencies, corporate actions and filing coverage still require review."), /*#__PURE__*/React.createElement("button", {
    disabled: state.busy,
    onClick: () => load()
  }, "Load saved evidence"), state.busy && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Checking saved evidence\u2026"), state.error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, state.error), state.sources && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(HistoricalCandidate, {
    url: url,
    baseline: state.sources.baseline,
    originals: state.sources.originals
  }), /*#__PURE__*/React.createElement("h5", null, "Retained originals"), !state.sources.originals.length && /*#__PURE__*/React.createElement("p", null, "No historical originals retained for this assignment."), state.sources.originals.map(source => /*#__PURE__*/React.createElement("p", {
    key: source.id
  }, source.kind === 'daily_prices' ? 'Daily prices' : 'SEC company facts', " \xB7 retrieved ", source.retrievedDate)), /*#__PURE__*/React.createElement("h5", null, "Retrieval progress"), !state.retrievals.retrievals.length && /*#__PURE__*/React.createElement("p", null, "No retrieval attempts recorded."), state.retrievals.retrievals.map(row => /*#__PURE__*/React.createElement("article", {
    key: row.id
  }, /*#__PURE__*/React.createElement("strong", null, row.status === 'evidence_retained' ? 'Evidence retained' : 'Retrieval needs investigation'), row.baselineChanged && /*#__PURE__*/React.createElement("p", null, "This attempt belongs to an earlier edition or source baseline."), /*#__PURE__*/React.createElement("p", null, row.nextAction))), /*#__PURE__*/React.createElement("h5", null, "Saved pending reviews"), !state.reviews.reviews.length && /*#__PURE__*/React.createElement("p", null, "No historical reviews saved for this edition."), state.reviews.reviews.map(row => /*#__PURE__*/React.createElement("article", {
    key: row.id
  }, /*#__PURE__*/React.createElement("p", null, new Date(row.createdAt).toLocaleString(), " \xB7 Pending investor review"), row.baselineChanged ? /*#__PURE__*/React.createElement("p", null, "The edition baseline changed. Prepare a new review version.") : /*#__PURE__*/React.createElement("button", {
    disabled: state.busy,
    onClick: () => load(row.id)
  }, "Recheck and open review"))), (state.sources.truncated || state.retrievals.truncated || state.reviews.truncated) && /*#__PURE__*/React.createElement("p", null, "Showing a bounded recent history; older records remain saved.")), state.review && /*#__PURE__*/React.createElement("section", {
    "aria-label": "Pending historical evidence checks"
  }, /*#__PURE__*/React.createElement("h5", null, "Pending evidence checks \xB7 not approved"), state.review.observations.map(row => /*#__PURE__*/React.createElement("article", {
    key: row.index
  }, /*#__PURE__*/React.createElement("h5", null, "Observation ", row.index + 1, " \xB7 ", row.priceDate), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("b", null, "Proposed price:"), " ", row.price, " \xB7 ", row.priceBasis), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("b", null, "Proposed earnings per share:"), " ", row.earnings, " \xB7 ", row.earningsBasis), /*#__PURE__*/React.createElement("p", null, "Period ", row.periodStart, " to ", row.periodEnd, " \xB7 published ", row.publishedDate), /*#__PURE__*/React.createElement("p", null, row.message))), /*#__PURE__*/React.createElement("p", null, "These checks do not update the one-pager or approve a valuation.")));
}