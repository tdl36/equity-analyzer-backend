import * as React from 'react';
export function CommitteeFindings({run}) {
 const claims=new Map((run.state?.sections||[]).flatMap(s=>s.claims.map(c=>[c.id,{...c,role:s.title}])));
 return <section aria-label="Committee challenges"><h4>Challenge round · draft responses</h4><p>Initial assessments are preserved below. These are five roles of the same configured model, not five independent people or a vote. An addressed challenge means the lead proposed a response; it is not evidence verification or investor approval.</p>
 {(run.state?.challenges||[]).map(c=>{const response=run.state?.responses?.find(r=>r.challengeId===c.id);return <article key={c.id}><h4>{c.question}</h4>{c.claimIds.map(id=><p key={id}><strong>{claims.get(id)?.role||id}:</strong> {claims.get(id)?.statement||'Assessment unavailable'}</p>)}<p><strong>{response?.status||'Response pending'}</strong></p>{response&&<><p>{response.reason}</p><p>Next test: {response.nextTest}</p><p>Unaccepted proposal: {response.proposedChange}</p></>}</article>;})}
 {run.state?.completed?.includes('challenges')&&!run.state.challenges.length&&<p>No challenges returned in this round. This is not proof of consensus or complete coverage.</p>}
 <p>Review proposed changes in Current thesis or Evidence & proposals. This committee cannot approve or edit your case.</p></section>;
}
