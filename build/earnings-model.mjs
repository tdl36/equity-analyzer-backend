// These checks describe stored output; they never certify investment conclusions.
export function eventResearch(activities = []) {
  const seen = new Set();
  return activities.filter(a => {
    if (!a?.id || seen.has(a.id) || !['earnings_recap','takeaway'].includes(a.activityType) || !a.input?.topic) return false;
    seen.add(a.id); return true;
  }).map(a => {
    const output = a.output || {};
    const sources = [...new Set((Array.isArray(output.sourceFiles) ? output.sourceFiles : []).filter(x => typeof x === 'string' && x.trim()))];
    const draft = typeof output.synthesisMarkdown === 'string' && !!output.synthesisMarkdown.trim();
    const coverage = [
      ['transcript','Call transcript',/transcript|earnings.call|ET_E/i],
      ['presentation','Presentation',/presentation|slides|deck/i],
      ['release','Earnings release',/earnings.release|press.release|results.release/i],
    ].map(([id,label,pattern]) => ({id,label,files:sources.filter(s=>pattern.test(s))}));
    const state = a.status === 'failed' ? 'failed' : a.status === 'running' ? 'running' : draft ? 'draft' : 'queued';
    const expected = Number(output.fileCount || a.input.fileCount) || null;
    return {...a, sources, coverage, state, draft, expected,
      sourceMismatch: expected !== null && draft && expected !== sources.length,
      provenance: output.sourceProvenance || null};
  });
}
