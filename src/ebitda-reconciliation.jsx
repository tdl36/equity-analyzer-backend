import * as React from 'react';
export function EbitdaReconciliation({model,onChange}) {
 const bridge=model.ebitdaReconciliation;
 const edit=b=>onChange({...model,baseEbitdaComparable:false,ebitdaReconciliation:{...b,confirmed:false}});
 const rowEdit=(i,key,value)=>edit({...bridge,adjustments:bridge.adjustments.map((r,j)=>j===i?{...r,[key]:value}:r)});
 return <section aria-label="EBITDA reconciliation"><h4>Reported-to-adjusted EBITDA · optional</h4><p>Explain the bridge into your base-year EBITDA. Use the same consolidated fiscal year, currency and millions as the model. Positive adjustments add to EBITDA; negative adjustments subtract. References and accounting judgments remain your responsibility.</p>
 {!bridge?<button onClick={()=>edit({startingEbitda:'',startingBasis:'',sourceReference:'',adjustments:[{label:'',amount:'',reference:'',recurrence:'uncertain'}]})}>Add EBITDA reconciliation</button>:<>
 <label>Starting EBITDA (millions; signed)<input inputMode="decimal" value={bridge.startingEbitda} onChange={e=>edit({...bridge,startingEbitda:e.target.value})}/></label>
 <label>Starting EBITDA definition<textarea maxLength={1800} value={bridge.startingBasis} onChange={e=>edit({...bridge,startingBasis:e.target.value})}/></label>
 <label>Starting EBITDA source and fiscal period<textarea maxLength={1800} value={bridge.sourceReference} onChange={e=>edit({...bridge,sourceReference:e.target.value})}/></label>
 {bridge.adjustments.map((row,i)=><article key={i} aria-label={`Adjustment ${i+1}`}><h4>Adjustment {i+1}</h4>
 <label>Adjustment name<input maxLength={200} value={row.label} onChange={e=>rowEdit(i,'label',e.target.value)}/></label>
 <label>Signed amount (millions)<input inputMode="decimal" value={row.amount} onChange={e=>rowEdit(i,'amount',e.target.value)}/></label>
 <label>Recurrence<select value={row.recurrence} onChange={e=>rowEdit(i,'recurrence',e.target.value)}><option value="uncertain">Uncertain</option><option value="recurring">Recurring</option><option value="nonrecurring">Nonrecurring</option></select></label>
 <label>Adjustment source and rationale<textarea maxLength={1800} value={row.reference} onChange={e=>rowEdit(i,'reference',e.target.value)}/></label>
 <button onClick={()=>edit({...bridge,adjustments:bridge.adjustments.filter((_,j)=>j!==i)})}>Remove adjustment {i+1}</button></article>)}
 <button disabled={bridge.adjustments.length>=20} onClick={()=>edit({...bridge,adjustments:[...bridge.adjustments,{label:'',amount:'',reference:'',recurrence:'uncertain'}]})}>Add adjustment</button>
 <p>Required total: base-year EBITDA {model.baseEbitda||'not entered'} million {model.currency}. Calculate draft to verify the exact sum. Arithmetic agreement does not establish source accuracy or acceptable accounting treatment.</p>
 <label><input type="checkbox" checked={bridge.confirmed===true} onChange={e=>onChange({...model,ebitdaReconciliation:{...bridge,confirmed:e.target.checked}})}/>I checked the reconciliation against its sources, confirmed the model period and currency, and checked for double-counting.</label>
 <button onClick={()=>{const next={...model,baseEbitdaComparable:false};delete next.ebitdaReconciliation;onChange(next);}}>Remove EBITDA reconciliation</button>
 </>}
 </section>;
}
