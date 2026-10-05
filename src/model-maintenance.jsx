import React from 'react';

export function ModelMaintenance({apiUrl}) {
    const [data, setData] = React.useState(null);
    const [error, setError] = React.useState('');
    const load = React.useCallback(async () => {
        setError('');
        try {
            const response = await fetch(`${apiUrl}/api/models/maintenance`);
            if (!response.ok) throw new Error('Model status could not be loaded.');
            setData(await response.json());
        } catch (e) { setError(e.message); }
    }, [apiUrl]);
    React.useEffect(() => { load(); }, [load]);
    return <ModelPolicyView data={data} error={error} load={load} />;
}

export function ModelPolicyView({data, error = '', load = () => {}}) {
    return <section className="workspace-panel max-w-3xl space-y-4" style={{overflowWrap:'anywhere',color:'var(--ink, #202018)'}}>
        <div className="flex justify-between gap-3"><h2>AI model maintenance</h2><button onClick={load} className="text-sm underline">Refresh status</button></div>
        <p className="text-sm text-slate-300">Charlie checks for model releases and retirements through scheduled maintenance. Eligible changes are tested and deployed together; saved model selections are preserved. Retired selections stop with an actionable error.</p>
        {error && <p role="alert">{error}</p>}
        {!data && !error && <p>Loading model policy…</p>}
        {data && <>
            <p className="text-sm">Last maintenance attempt: <strong>{data.lastCheck?.checkedAt || data.verifiedAt}</strong></p>
            {data.lastCheck && <p className="text-sm" role="status">{data.lastCheck.status}: {data.lastCheck.summary}</p>}
            <p className="text-sm text-slate-400">{data.policy}</p>
            <div className="space-y-2">
                {Object.entries(data.roles).map(([role, model]) => <div key={role} className="border-b border-white/10 pb-2 flex flex-wrap justify-between gap-2 text-sm"><span>{role.replaceAll('_',' ')}</span><span>{model}</span></div>)}
            </div>
            {!!data.notices.length && <div><h3 className="font-semibold mb-2">Retirement notices</h3>{data.notices.map(n => <p className="text-sm mb-2" key={n.model}>{n.model} · {n.date} · {n.message}</p>)}</div>}
            <details><summary className="cursor-pointer">Verified changes and sources</summary><div className="space-y-2 mt-3 text-sm">{[...data.history].reverse().map((h,i) => <p key={i}>{h.date}: {h.summary}</p>)}{data.sources.map(url => <p key={url}><a className="underline" href={url} target="_blank" rel="noreferrer">{url}</a></p>)}</div></details>
            <p className="text-xs text-slate-400">{data.maintenanceDependency}</p>
            <p className="text-xs text-slate-400">Registry {data.revision}. Prices are estimates, not invoices. Unknown prices are excluded from totals and flagged in API usage.</p>
        </>}
    </section>;
}
