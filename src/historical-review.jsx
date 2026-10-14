import {acquisitionDetails} from './historical-acquisition.mjs';
import * as React from 'react';
import {HistoricalCandidate} from './historical-candidate';
import {historicalReview,historicalInventory,historicalEditionReadiness} from './historical-review-model.mjs';

// Read-only inspection; never retrieves providers or writes research.
export function HistoricalReview({api, assignmentId, editionId}) {
  const [state,setState] = React.useState({busy:false}), generation = React.useRef(0), activeRequest = React.useRef(null);
  const url = `${api}/api/research/assignments/${encodeURIComponent(assignmentId)}/onepager-editions/${encodeURIComponent(editionId)}`;
  React.useEffect(() => {generation.current++;activeRequest.current?.abort();setState({busy:false});return () => {generation.current++;activeRequest.current?.abort();};},[url]);
  async function read(path, signal) {
    const response = await fetch(path,{signal:AbortSignal.any([signal,AbortSignal.timeout(30000)])});
    const data = await response.json();
    if (!response.ok) throw Error(data.error || 'Historical evidence is unavailable.');
    return data;
  }
  async function load(id) {
    const token = ++generation.current;
    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;
    setState({busy:true});
    try {
      if (id) {
        const review = historicalReview(await read(`${url}/historical-reviews/${encodeURIComponent(id)}`,controller.signal),assignmentId,editionId,id);
        if (token === generation.current) setState({busy:false,review});
      } else {
        const [sources,retrievals,reviews] = await Promise.all(['historical-originals','historical-retrievals','historical-reviews'].map(path=>read(`${url}/${path}`,controller.signal)));
        const inventory = historicalInventory(sources,retrievals,reviews,assignmentId,editionId);
        if (token === generation.current) setState({busy:true,...inventory});
        try {
          const selection=encodeURIComponent(JSON.stringify({baseline:sources.baseline}));
          const coverage=historicalEditionReadiness(await read(`${url}/historical-edition-readiness?selection=${selection}`,controller.signal),sources.baseline);
          if (token === generation.current) setState({busy:false,...inventory,coverage});
        } catch (error) {
          if (token === generation.current) setState({busy:false,...inventory,coverageError:error.message});
        }
      }
    } catch (error) {
      controller.abort();
      if (token === generation.current) setState({busy:false,error:error.message});
    }
  }
  return <details className="sa-prepare">
    <summary>Historical valuation evidence</summary>
    <p>Inspect retained sources and pending checks for this edition. Matching source amounts alone does not establish a valid historical valuation. Share classes, currencies, corporate actions and filing coverage still require review.</p>
    <button disabled={state.busy} onClick={()=>load()}>Load saved evidence</button>
    {state.busy && <p role="status">Checking saved evidence…</p>}
    {state.error && <p role="alert">{state.error}</p>}
    {state.sources && <>
      {state.coverageError && <p role="alert">Saved date coverage unavailable: {state.coverageError}</p>}
      {state.coverage && <section aria-label="Saved financial date coverage">
        <h5>Coverage at saved financial dates</h5>
        <p>Fiscal year ends are calendar anchors, not verified trading days. Missing prices stay missing; Charlie does not substitute a nearby trading session.</p>
        {state.coverage.needsDateReview ? <p>The saved annual periods need review before dates can be checked.</p> : state.coverage.rows.map(row=><p key={row.date}><strong>{row.date}</strong> · {row.message}</p>)}
        <p>These checks do not establish a valuation or update saved research.</p>
      </section>}
      <HistoricalCandidate url={url} baseline={state.sources.baseline} originals={state.sources.originals}/>
      <h5>Retained originals</h5>
      {!state.sources.originals.length && <p>No historical originals retained for this assignment.</p>}
      {state.sources.originals.map(source=>{const acquisition=acquisitionDetails(source.acquisition);return <article key={source.id} style={{overflowWrap:'anywhere'}}><p>{source.kind === 'daily_prices' ? 'Daily prices' : 'SEC company facts'} · retrieved {source.retrievedDate}</p><p>{acquisition.message}</p>{acquisition.provider&&<details><summary>Acquisition details</summary><p>Provider: {acquisition.provider}</p><p>Observed endpoint: {acquisition.url}</p><p>Acquired: {acquisition.time}</p><p>Collector version: {acquisition.version}</p></details>}</article>;})}
      <h5>Retrieval progress</h5>
      {!state.retrievals.retrievals.length && <p>No retrieval attempts recorded.</p>}
      {state.retrievals.retrievals.map(row=><article key={row.id}>
        <strong>{row.status === 'evidence_retained' ? 'Evidence retained' : 'Retrieval needs investigation'}</strong>
        {row.baselineChanged && <p>This attempt belongs to an earlier edition or source baseline.</p>}
        <p>{row.nextAction}</p>
      </article>)}
      <h5>Saved pending reviews</h5>
      {!state.reviews.reviews.length && <p>No historical reviews saved for this edition.</p>}
      {state.reviews.reviews.map(row=><article key={row.id}>
        <p>{new Date(row.createdAt).toLocaleString()} · Pending investor review</p>
        {row.baselineChanged ? <p>The edition baseline changed. Prepare a new review version.</p> : <button disabled={state.busy} onClick={()=>load(row.id)}>Recheck and open review</button>}
      </article>)}
      {(state.sources.truncated || state.retrievals.truncated || state.reviews.truncated) && <p>Showing a bounded recent history; older records remain saved.</p>}
    </>}
    {state.review && <section aria-label="Pending historical evidence checks">
      <h5>Pending evidence checks · not approved</h5>
      {state.review.observations.map(row=><article key={row.index}>
        <h5>Observation {row.index+1} · {row.priceDate}</h5>
        <p><b>Proposed price:</b> {row.price} · {row.priceBasis}</p>
        <p><b>Proposed earnings per share:</b> {row.earnings} · {row.earningsBasis}</p>
        <p>Period {row.periodStart} to {row.periodEnd} · published {row.publishedDate}</p>
        <p>{row.message}</p>
      </article>)}
      <p>These checks do not update the one-pager or approve a valuation.</p>
    </section>}
  </details>;
}
