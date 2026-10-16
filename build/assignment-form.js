import * as React from 'react';
import { readAssignmentForm, writeAssignmentForm, clearAssignmentForm } from './assignment-form-draft.mjs';
export function useAssignmentForm(key, initial, ready) {
  var [recovery] = React.useState(() => {
    try {
      return {
        value: readAssignmentForm(sessionStorage, key),
        error: ''
      };
    } catch (e) {
      return {
        value: null,
        error: e.message
      };
    }
  });
  var [form, setForm] = React.useState(() => recovery.value || initial);
  var [restored, setRestored] = React.useState(!!recovery.value),
    [issue, setIssue] = React.useState(recovery.error);
  var baseline = React.useRef(null),
    signature = JSON.stringify(form);
  React.useLayoutEffect(() => {
    if (!ready) return;
    if (baseline.current === null) baseline.current = signature;
    if (issue || !restored && baseline.current === signature) return;
    try {
      writeAssignmentForm(sessionStorage, key, form);
    } catch (e) {
      setIssue(e.message || 'Browser form recovery is unavailable. Keep this screen open.');
    }
  }, [ready, key, signature, issue, restored]);
  React.useEffect(() => {
    if (!issue) return;
    var warn = e => {
      e.preventDefault();
      e.returnValue = '';
    };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [issue]);
  var clear = () => {
    try {
      clearAssignmentForm(sessionStorage, key);
      baseline.current = signature;
      setRestored(false);
      setIssue('');
    } catch (e) {
      setIssue(e.message || 'Browser form recovery could not be cleared.');
    }
  };
  var edit = name => value => setForm(old => ({
    ...old,
    [name]: typeof value === 'function' ? value(old[name]) : value
  }));
  return {
    form,
    edit,
    restored,
    issue,
    clear
  };
}