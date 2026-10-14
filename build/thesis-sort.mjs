// Shared ordering for the thesis sidebar and compact mobile ticker list.
export function sortTheses(rows, order) {
    const ticker = row => typeof row.ticker === 'string' ? row.ticker : '';
    const updated = row => {
        const value = row.updated ? Date.parse(row.updated) : NaN;
        return Number.isFinite(value) ? value : -Infinity;
    };
    return [...rows].sort((a, b) => {
        if (order === 'updated' && updated(a) !== updated(b)) {
            return updated(a) > updated(b) ? -1 : 1;
        }
        return ticker(a).localeCompare(ticker(b));
    });
}
