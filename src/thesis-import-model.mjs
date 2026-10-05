export const MAX_DRAFT_BYTES = 2_000_000;
export function parseDraft(text) {
  if (new TextEncoder().encode(text).length > MAX_DRAFT_BYTES) throw new Error('Choose a draft smaller than 2 MB. Originals are not needed for import.');
  let parsed;
  try { parsed = JSON.parse(text.trim().replace(/^```(?:json)?\s*\n?/, '').replace(/\n?```$/, '')); }
  catch { throw new Error('This file is not valid JSON. Ask ChatGPT to return the Charlie draft template as a JSON file.'); }
  if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object' || !parsed.analysis) throw new Error('Choose the structured thesis draft, not the review PDF or source register.');
  return parsed;
}
export function canApprove(draft, checked, typedTicker, busy) {
  return !!draft && draft.status === 'pending' && !draft.stale && checked && typedTicker === draft.ticker && !busy;
}
export function downloadFile(name, body) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(body, null, 2)], {type: 'application/json'}));
  const a = document.createElement('a'); a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
