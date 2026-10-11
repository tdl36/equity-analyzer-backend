import * as React from 'react';
import {recoveryEvidence} from './research-recovery-model.mjs';

export function ResearchRecovery({run}) {
 const evidence=recoveryEvidence(run);
 if(!['attention','cancelled'].includes(run.status))return null;
 return <aside className="research-recovery" aria-label="Research recovery evidence">
 <style>{`.research-recovery{margin:20px 0;padding:18px 20px;background:#f6f7f3;border:1px solid #cbd2ca;border-left:4px solid #85968b;border-radius:6px;font-family:Calibri,sans-serif;color:#000}.research-recovery .rr-eyebrow{font-size:12px;letter-spacing:.1em;text-transform:uppercase;margin:0 0 8px}.research-recovery .rr-title{font-size:19px;font-weight:700;margin:0 0 8px}.research-recovery .rr-counts{display:flex;flex-wrap:wrap;gap:8px 24px;padding:10px 0;border-top:1px solid #d9dfd7}.research-recovery details{margin-top:8px}.research-recovery summary{cursor:pointer}.research-recovery li{margin:12px 0;overflow-wrap:anywhere}.research-recovery code{font:13px Calibri,sans-serif;overflow-wrap:anywhere}@media(max-width:600px){.research-recovery{padding:14px}.research-recovery .rr-counts{display:flex;flex-direction:column;gap:4px}}`}</style>
 <p className="rr-eyebrow">Recovery evidence</p><p className="rr-title">{evidence.title}</p>
 <p>{evidence.next}</p><div className="rr-counts"><span>{evidence.savedCount} saved responses</span><span>{evidence.previousAttempts} archived responses</span>{evidence.stage&&<span>Stopped stage: {evidence.stage}</span>}</div>
 <details><summary>Inspect saved response records</summary><p>These records confirm preservation only. They do not establish source validation, investor approval or provider billing. Response text stays private.</p>
 {!evidence.receipts.length?<p>No detailed response records are available for this run. Older jobs may predate response preservation.</p>:<ul>{evidence.receipts.map(r=><li key={r.stage}><strong>{r.stage}</strong> · {Number.isInteger(r.characters)?r.characters.toLocaleString()+' characters':'Text size unavailable'} · Stop reason: {r.stopReason||'unavailable'}{r.sha256&&<div>Response fingerprint: <code>{r.sha256}</code></div>}</li>)}</ul>}
 </details></aside>;
}
