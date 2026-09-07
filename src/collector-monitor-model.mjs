export function documentStatus(doc) {
  if (doc.usage === 'reference_only') return 'held';
  if (doc.status === 'handed_off') return 'saved';
  if (doc.status === 'duplicate') return 'duplicate';
  return 'pending';
}

export function filterDocuments(documents, {ticker = '', status = '', query = ''} = {}) {
  const needle = query.trim().toLocaleLowerCase();
  return documents.filter(doc => (!ticker || doc.ticker === ticker) &&
    (!status || documentStatus(doc) === status) &&
    (!needle || [doc.ticker, doc.filename, doc.publisher, doc.destinationFolder]
      .some(value => String(value || '').toLocaleLowerCase().includes(needle))));
}

export function collectionCounts(documents) {
  const counts = {saved: 0, held: 0, pending: 0, duplicate: 0};
  for (const doc of documents) counts[documentStatus(doc)]++;
  return counts;
}
