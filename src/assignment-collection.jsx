import * as React from 'react';
import {assignmentCollection} from './assignment-collection-model.mjs';
export function AssignmentCollection({job}){
 const c=assignmentCollection(job);if(!c)return null;
 return <section aria-label="Assignment collection status" style={{padding:'12px 0',overflowWrap:'anywhere'}}>
  <strong>{c.label}</strong>
  <p><small>Last saved Mac report: {c.reportedAt}. {c.fresh?'This is the latest saved report, not a live browser observation.':'Report is stale or its time is unavailable. Refresh before interpreting current browser progress.'}</small></p>
  {c.issue&&<p role="status" style={{whiteSpace:'pre-wrap'}}><strong>Reported collection issue: </strong>{c.issue}</p>}
  <p>{c.verified} verified originals recorded by this assignment. Collection progress does not establish complete coverage or evidence incorporated into the accepted thesis.</p>
  <details><summary>Reported searches and handoffs</summary>
   {c.counts.map(r=><p key={r.label}>{r.label}: {r.count}</p>)}
   {c.tasks.length?<ul>{c.tasks.map((t,i)=><li key={i}>{t.kind.replaceAll('-',' ')} · {t.status.replaceAll('_',' ')}{t.expected!==null?` · ${t.expected} selected`:''}</li>)}</ul>:<p>No search-progress receipt available.</p>}
   <p>Staging, iCloud handoff and existing-copy counts are not verified cloud-import receipts or completed research.</p>
  </details>
  {c.href&&<p><a href={c.href}>Open this collection in Operations</a></p>}
  <p><small>Collection uses the existing Operations worker. It needs the Mac awake and online, the agent and Codex/browser available, and AlphaSense signed in. Resolve the reported issue there; this view starts no collection or research.</small></p>
 </section>;
}
