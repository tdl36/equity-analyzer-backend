// Explain dispatches immediately and polls; it does not need a direct backend connection.
// Never retry a POST: a lost response may already have started a paid job.
export async function explainRequest(base, path = '', options = {}, fetcher = globalThis.fetch) {
    let response;
    try {
        response = await fetcher(`${base}/api/decipher${path}`, options);
    } catch {
        throw new Error('Charlie could not be reached. Your attachment is still here. Check your connection and try again when Charlie is available. If you already clicked Decipher, the request may have started; it has not been automatically resubmitted.');
    }
    if (!response.ok) {
        let detail;
        try { detail = (await response.json()).error; } catch {}
        const message = response.status === 413
            ? 'This upload is too large. Try fewer attachments or a smaller screenshot.'
            : response.status === 404
                ? 'This explanation session is no longer available. Your attachments are still here; start a new explanation when ready.'
                : response.status >= 500
                    ? 'Charlie is temporarily unavailable. Your attachments are still here. Please try again shortly.'
                    : typeof detail === 'string' ? detail : `Charlie could not accept the request (HTTP ${response.status}).`;
        const error = new Error(message);
        error.status = response.status;
        throw error;
    }
    return response;
}
