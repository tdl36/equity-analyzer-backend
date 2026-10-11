import * as React from 'react';
import {researchCoverage,coverageDocument} from './research-coverage-model.mjs';
import {downloadText} from './stock-analysis-model.mjs';
export function ResearchCoverage({run,onSection}) {
 const c=React.useMemo(()=>researchCoverage(run),[run]);
 const [filter,setFilter]=React.useState('review');
 return <section aria-label="Research coverage and review"><h4>What this research can support</h4><p>{c.matched} matched fields · {c.review} need source review · {c.unavailable} unavailable items</p><p>Counts include labels and periods. A matched excerpt and a model review do not establish factual accuracy, complete market coverage or investor approval.</p>
 {c.baselineStale&&<p className="sa-warning">Your investment case changed after this report started. Compare the current case with the frozen baseline before using this report.</p>}
 {c.noEvidence.length>0&&<p className="sa-warning"><strong>No matched fields:</strong> {c.noEvidence.join(' · ')}. Open these sections to inspect the gaps before relying on the conclusions.</p>}
 <p>{c.sources.length} frozen originals · {c.limitedSources} with recorded extraction/selection limits · {c.unknownCoverage} without coverage metadata</p>
 <div className="sa-actions"><button aria-pressed={filter==='review'} onClick={()=>setFilter('review')}>Needs source review</button><button aria-pressed={filter==='all'} onClick={()=>setFilter('all')}>All sections</button><button onClick={()=>downloadText(coverageDocument(run),`${run.ticker}-review-checklist-${run.id}.html`)}>Export review checklist</button></div>
 <div style={{overflowX:'auto'}}><table style={{width:'100%',borderCollapse:'collapse'}}><thead><tr><th>Section</th><th>Matched</th><th>Review</th><th>Unavailable</th></tr></thead><tbody>{c.rows.filter(r=>filter==='all'||r.review||!r.matched||r.unavailable).map(r=><tr key={r.id}><th><button onClick={()=>onSection(r.id)}>Open {r.title}</button></th><td>{r.matched}</td><td>{r.review}</td><td>{r.unavailable}</td></tr>)}</tbody></table></div>
 {filter==='review'&&!c.rows.some(r=>r.review||!r.matched||r.unavailable)&&<p>No field-level review gaps recorded. Investor review is still required.</p>}
 {c.rows.filter(r=>r.review).map(r=><details key={r.id}><summary>{r.title} · {r.review} fields to check</summary>{r.fields.filter(f=>f.status==='review').map(f=><article key={f.path}><small>{f.path} · {f.basis.replaceAll('_',' ')}</small><p>{f.value}</p><p className="sa-warning">{f.reason}</p></article>)}<button onClick={()=>onSection(r.id)}>Open section and source passages</button></details>)}
 <details><summary>Source coverage limits · {c.sources.length} originals</summary>{c.sources.map(s=><article key={s.id}><strong>{s.name}</strong><p>{s.used?'Referenced by at least one matched field.':'No matched report field references this source.'}</p><p>{s.limitation||(!s.coverageKnown?'Extraction coverage metadata unavailable; full-document review is not established.':'No recorded selection omission; this is not proof of complete document review.')}</p>{s.omitted!=null&&<p>{s.omitted.toLocaleString()} extracted characters omitted from model input.</p>}{s.extraction.map((v,i)=><p key={i}>{v}</p>)}</article>)}</details>
 </section>;
}
