import { ResearchEdits } from './research-edits';
import * as React from 'react';
export function ResearchChat({
  api,
  context,
  onClose,
  onApplied
}) {
  var [conversation, setConversation] = React.useState(() => crypto.randomUUID());
  var [messages, setMessages] = React.useState([]),
    [history, setHistory] = React.useState([]),
    [analysts, setAnalysts] = React.useState([]);
  var [analyst, setAnalyst] = React.useState(''),
    [text, setText] = React.useState(''),
    [job, setJob] = React.useState(null);
  var [error, setError] = React.useState(''),
    [busy, setBusy] = React.useState(false),
    [uncertain, setUncertain] = React.useState(false);
  var panelRef = React.useRef(null);
  var alive = React.useRef(true),
    lock = React.useRef(false),
    pending = React.useRef(null),
    current = React.useRef(conversation);
  current.current = conversation;
  var fetchJson = async (path, options = {}) => {
    var controller = new AbortController(),
      timer = setTimeout(() => controller.abort(), 20000);
    try {
      var r = await fetch(`${api}${path}`, {
        ...options,
        signal: controller.signal
      });
      var d = await r.json();
      if (!r.ok) {
        var e = new Error(d.error || `Request failed (${r.status})`);
        e.status = r.status;
        throw e;
      }
      return d;
    } finally {
      clearTimeout(timer);
    }
  };
  var list = async () => {
    var d = await fetchJson(`/api/research/conversations?ticker=${encodeURIComponent(context.ticker)}&type=${encodeURIComponent(context.type)}`);
    if (alive.current) setHistory(d.conversations || []);
  };
  var refresh = async (id = conversation) => {
    try {
      var d = await fetchJson(`/api/research/conversations/${id}`);
      if (!alive.current || current.current !== id) return;
      setMessages(d.messages || []);
      setJob(d.job);
      if (pending.current && (d.messages || []).some(m => m.requestId === pending.current.requestId)) {
        pending.current = null;
        setUncertain(false);
        setText('');
      }
    } catch (e) {
      if (alive.current && current.current === id && e.status !== 404) setError(`Could not refresh conversation: ${e.message}. Saved work may still be processing.`);
    }
  };
  React.useEffect(() => {
    alive.current = true;
    list().catch(e => {
      if (alive.current) setError(e.message);
    });
    fetchJson('/api/analysts').then(d => {
      if (alive.current) setAnalysts(d.analysts || []);
    }).catch(e => {
      if (alive.current) setError(e.message);
    });
    return () => {
      alive.current = false;
    };
  }, [api, context.ticker, context.type]);
  React.useEffect(() => {
    refresh();
    var timer = setInterval(() => {
      if (!document.hidden) refresh();
    }, 4000);
    return () => clearInterval(timer);
  }, [conversation]);
  var active = job && ['queued', 'running'].includes(job.status);
  var send = async () => {
    if (lock.current || active || !text.trim() && !pending.current) return;
    lock.current = true;
    setBusy(true);
    setError('');
    var payload = pending.current || {
      requestId: crypto.randomUUID(),
      conversationId: conversation,
      ticker: context.ticker,
      contentType: context.type,
      content: context.content,
      message: text.trim(),
      analystId: analyst
    };
    pending.current = payload;
    try {
      await fetchJson('/api/research/conversations/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });
      if (alive.current && current.current === payload.conversationId) {
        pending.current = null;
        setText('');
        setUncertain(false);
        await refresh(payload.conversationId);
        await list();
      }
    } catch (e) {
      if (alive.current) {
        setError(e.message);
        if (e.status) {
          pending.current = null;
          setUncertain(false);
        } else setUncertain(true);
        await refresh(payload.conversationId);
      }
    } finally {
      lock.current = false;
      if (alive.current) setBusy(false);
    }
  };
  var select = id => {
    if (busy || uncertain) return;
    setConversation(id);
    setMessages([]);
    setJob(null);
    setText('');
    setError('');
    pending.current = null;
  };
  var stop = async () => {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    try {
      await fetchJson(`/api/research/conversations/${conversation}/stop`, {
        method: 'POST'
      });
      await refresh();
    } catch (e) {
      if (alive.current) setError(e.message);
    } finally {
      lock.current = false;
      if (alive.current) setBusy(false);
    }
  };
  React.useEffect(() => {
    if (!onClose) return;
    var prior = document.activeElement;
    var trap = e => {
      if (e.key !== 'Tab') return;
      var items = [...panelRef.current.querySelectorAll('button:not(:disabled),select:not(:disabled),textarea:not(:disabled)')];
      var first = items[0],
        last = items[items.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last?.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first?.focus();
      }
    };
    panelRef.current?.addEventListener('keydown', trap);
    var panel = panelRef.current;
    return () => {
      panel?.removeEventListener('keydown', trap);
      prior?.focus();
    };
  }, []);
  var panel = /*#__PURE__*/React.createElement("section", {
    ref: panelRef,
    className: "research-chat",
    "aria-label": `${context.ticker} analyst conversation`
  }, /*#__PURE__*/React.createElement("header", null, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "RESEARCH CONVERSATION / ", context.ticker), /*#__PURE__*/React.createElement("h3", null, "Work through the investment case.")), onClose && /*#__PURE__*/React.createElement("button", {
    autoFocus: true,
    "aria-label": "Close analyst conversation",
    onClick: onClose
  }, "Close \xD7")), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Ask questions or request replacement wording for this ", context.type, ". Replies use the research shown here and the last 20 conversation messages; original source files are not automatically consulted. Saved research is unchanged until you review and apply edits in its workflow."), /*#__PURE__*/React.createElement("div", {
    className: "research-chat-controls"
  }, /*#__PURE__*/React.createElement("label", null, "Replying analyst", /*#__PURE__*/React.createElement("select", {
    value: analyst,
    disabled: busy || active || uncertain,
    onChange: e => setAnalyst(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Research analyst"), analysts.map(a => /*#__PURE__*/React.createElement("option", {
    key: a.id,
    value: a.id
  }, a.name)))), /*#__PURE__*/React.createElement("label", null, "Conversation", /*#__PURE__*/React.createElement("select", {
    value: history.some(h => h.id === conversation) ? conversation : '',
    disabled: busy || uncertain,
    onChange: e => {
      if (e.target.value) select(e.target.value);
    }
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "New conversation"), history.map(h => /*#__PURE__*/React.createElement("option", {
    key: h.id,
    value: h.id
  }, h.updatedAt)))), /*#__PURE__*/React.createElement("button", {
    disabled: busy || uncertain,
    onClick: () => select(crypto.randomUUID())
  }, "New conversation")), error && /*#__PURE__*/React.createElement("p", {
    role: "alert",
    className: "workspace-error"
  }, error), /*#__PURE__*/React.createElement("div", {
    className: "research-chat-messages",
    role: "log",
    "aria-label": "Analyst conversation messages"
  }, messages.length ? messages.map((m, i) => /*#__PURE__*/React.createElement("article", {
    key: i,
    className: m.role === 'user' ? 'chat-user' : 'chat-analyst'
  }, /*#__PURE__*/React.createElement("strong", null, m.role === 'user' ? 'You' : m.analystName || 'Research analyst'), /*#__PURE__*/React.createElement("p", null, m.content))) : /*#__PURE__*/React.createElement("p", {
    className: "workspace-empty"
  }, "Start with a specific instruction: \u201CChallenge the second thesis pillar,\u201D or \u201CRewrite the conclusion to distinguish facts from estimates.\u201D")), /*#__PURE__*/React.createElement("div", {
    role: "status"
  }, active ? `Analyst ${job.status === 'queued' ? 'queued' : 'working'} · you can close this panel and reopen the saved conversation.` : job?.error || ''), active && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "If processing was interrupted, stop this reply before sending another message. A provider request already running may still incur charges."), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: stop
  }, "Stop reply delivery")), /*#__PURE__*/React.createElement("label", {
    className: "earnings-search"
  }, "Your instruction", /*#__PURE__*/React.createElement("textarea", {
    rows: 3,
    maxLength: 6000,
    value: text,
    disabled: busy || active || uncertain,
    onChange: e => setText(e.target.value),
    placeholder: "Ask the analyst to explain, challenge or revise\u2026"
  })), /*#__PURE__*/React.createElement(ResearchEdits, {
    key: context.ticker,
    api: api,
    ticker: context.ticker,
    instruction: text,
    analystId: analyst,
    onApplied: onApplied
  }), /*#__PURE__*/React.createElement("footer", null, /*#__PURE__*/React.createElement("span", null, "Uses configured model API credits. Conversation replies are proposed research, not verified source claims."), /*#__PURE__*/React.createElement("button", {
    onClick: () => refresh(),
    disabled: busy
  }, "Check status"), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    disabled: busy || active || !text.trim() && !uncertain,
    onClick: send
  }, busy ? 'Submitting…' : uncertain ? 'Retry same request' : 'Send to analyst')));
  return onClose ? /*#__PURE__*/React.createElement("div", {
    role: "dialog",
    "aria-modal": "true",
    "aria-label": "Research analyst conversation",
    className: "research-chat-backdrop",
    onKeyDown: e => {
      if (e.key === 'Escape') onClose();
    }
  }, panel) : panel;
}