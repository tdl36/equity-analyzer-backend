import * as React from 'react';
import {historicalCandidate} from './historical-review-model.mjs';

export function HistoricalCandidate({url,baseline,originals}) {
 const [date,setDate]=React.useState(''),[price,setPrice]=React.useState(''),[earnings,setEarnings]=React.useState(''),[state,setState]=React.useState({}),active=React.useRef(null),generation=React.useRef(0);
 const identity=JSON.stringify([url,baseline,originals]);
 const clear=()=>{generation.current++;active.current?.abort();setState({});};
 React.useEffect(()=>{clear();setDate('');setPrice('');setEarnings('');return()=>{generation.current++;active.current?.abort();};},[identity]);
 const prices=originals.filter(r=>r.kind==='daily_prices'),facts=originals.filter(r=>r.kind==='sec_companyfacts');
 if(!prices.length||!facts.length)return null;
 async function preview(e){
  e.preventDefault();clear();const token=++generation.current,controller=new AbortController();active.current=controller;
  const chosenPrice=prices.find(r=>r.id===price),chosenEarnings=facts.find(r=>r.id===earnings);
  if(!chosenPrice||!chosenEarnings){setState({error:'Reload saved evidence and choose the retained originals again.'});return;}
  const receipt=r=>({id:r.id,sha256:r.sha256});
  const selection={baseline,observation:{date,priceSource:receipt(chosenPrice),earningsSource:receipt(chosenEarnings)}};
  setState({busy:true});
  try{const response=await fetch(url+'/historical-candidate?'+new URLSearchParams({selection:JSON.stringify(selection)}),{signal:AbortSignal.any([controller.signal,AbortSignal.timeout(30000)])});
   if(!response.ok)throw Error('These historical inputs could not be verified. Reload the saved evidence and check the selected date.');
   const value=historicalCandidate(await response.json(),selection);if(token===generation.current)setState({value});
  }catch(error){if(token===generation.current)setState({error:error.message});}
 }
 return <section aria-label="Preview historical inputs"><h5>Preview retained inputs</h5>
 <p>Choose an exact completed trading date and retained originals. This reads existing evidence only.</p>
 <form onSubmit={preview} style={{display:'grid',gap:12,maxWidth:600}}>
 <label>Trading date<input style={{display:'block',width:'100%',minWidth:0}} type="date" required value={date} onChange={e=>{clear();setDate(e.target.value);}}/></label>
 <label>Price original<select style={{display:'block',width:'100%',minWidth:0}} aria-label="Price original" required value={price} onChange={e=>{clear();setPrice(e.target.value);}}><option value="">Select retained prices</option>{prices.map((r,i)=><option key={r.id} value={r.id}>Daily prices · {r.retrievedDate} · {i+1}</option>)}</select></label>
 <label>Earnings original<select style={{display:'block',width:'100%',minWidth:0}} aria-label="Earnings original" required value={earnings} onChange={e=>{clear();setEarnings(e.target.value);}}><option value="">Select retained earnings</option>{facts.map((r,i)=><option key={r.id} value={r.id}>SEC company facts · {r.retrievedDate} · {i+1}</option>)}</select></label>
 <button disabled={state.busy||!date||!price||!earnings}>Preview retained inputs</button></form>
 {state.busy&&<p role="status">Checking retained inputs…</p>}{state.error&&<p role="alert">{state.error}</p>}
 {state.value&&<div><h5>Unapproved input preview · {state.value.date}</h5><p>Raw closing price: {state.value.price}</p><p>Annual diluted EPS: {state.value.earnings}</p><p>Annual period ending {state.value.periodEnd} · filed {state.value.publishedDate}</p>{state.value.gaps.map((g,i)=><p key={i}>{g}</p>)}<p>These amounts do not establish a valuation. Currency and share-class mapping, corporate actions, filing completeness and source provenance still require review. Nothing has been saved or approved.</p></div>}
 </section>;
}
