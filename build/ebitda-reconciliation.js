import * as React from 'react';
export function EbitdaReconciliation({
  model,
  onChange
}) {
  var bridge = model.ebitdaReconciliation;
  var edit = b => onChange({
    ...model,
    baseEbitdaComparable: false,
    ebitdaReconciliation: {
      ...b,
      confirmed: false
    }
  });
  var rowEdit = (i, key, value) => edit({
    ...bridge,
    adjustments: bridge.adjustments.map((r, j) => j === i ? {
      ...r,
      [key]: value
    } : r)
  });
  return /*#__PURE__*/React.createElement("section", {
    "aria-label": "EBITDA reconciliation"
  }, /*#__PURE__*/React.createElement("h4", null, "Reported-to-adjusted EBITDA \xB7 optional"), /*#__PURE__*/React.createElement("p", null, "Explain the bridge into your base-year EBITDA. Use the same consolidated fiscal year, currency and millions as the model. Positive adjustments add to EBITDA; negative adjustments subtract. References and accounting judgments remain your responsibility."), !bridge ? /*#__PURE__*/React.createElement("button", {
    onClick: () => edit({
      startingEbitda: '',
      startingBasis: '',
      sourceReference: '',
      adjustments: [{
        label: '',
        amount: '',
        reference: '',
        recurrence: 'uncertain'
      }]
    })
  }, "Add EBITDA reconciliation") : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("label", null, "Starting EBITDA (millions; signed)", /*#__PURE__*/React.createElement("input", {
    inputMode: "decimal",
    value: bridge.startingEbitda,
    onChange: e => edit({
      ...bridge,
      startingEbitda: e.target.value
    })
  })), /*#__PURE__*/React.createElement("label", null, "Starting EBITDA definition", /*#__PURE__*/React.createElement("textarea", {
    maxLength: 1800,
    value: bridge.startingBasis,
    onChange: e => edit({
      ...bridge,
      startingBasis: e.target.value
    })
  })), /*#__PURE__*/React.createElement("label", null, "Starting EBITDA source and fiscal period", /*#__PURE__*/React.createElement("textarea", {
    maxLength: 1800,
    value: bridge.sourceReference,
    onChange: e => edit({
      ...bridge,
      sourceReference: e.target.value
    })
  })), bridge.adjustments.map((row, i) => /*#__PURE__*/React.createElement("article", {
    key: i,
    "aria-label": `Adjustment ${i + 1}`
  }, /*#__PURE__*/React.createElement("h4", null, "Adjustment ", i + 1), /*#__PURE__*/React.createElement("label", null, "Adjustment name", /*#__PURE__*/React.createElement("input", {
    maxLength: 200,
    value: row.label,
    onChange: e => rowEdit(i, 'label', e.target.value)
  })), /*#__PURE__*/React.createElement("label", null, "Signed amount (millions)", /*#__PURE__*/React.createElement("input", {
    inputMode: "decimal",
    value: row.amount,
    onChange: e => rowEdit(i, 'amount', e.target.value)
  })), /*#__PURE__*/React.createElement("label", null, "Recurrence", /*#__PURE__*/React.createElement("select", {
    value: row.recurrence,
    onChange: e => rowEdit(i, 'recurrence', e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: "uncertain"
  }, "Uncertain"), /*#__PURE__*/React.createElement("option", {
    value: "recurring"
  }, "Recurring"), /*#__PURE__*/React.createElement("option", {
    value: "nonrecurring"
  }, "Nonrecurring"))), /*#__PURE__*/React.createElement("label", null, "Adjustment source and rationale", /*#__PURE__*/React.createElement("textarea", {
    maxLength: 1800,
    value: row.reference,
    onChange: e => rowEdit(i, 'reference', e.target.value)
  })), /*#__PURE__*/React.createElement("button", {
    onClick: () => edit({
      ...bridge,
      adjustments: bridge.adjustments.filter((_, j) => j !== i)
    })
  }, "Remove adjustment ", i + 1))), /*#__PURE__*/React.createElement("button", {
    disabled: bridge.adjustments.length >= 20,
    onClick: () => edit({
      ...bridge,
      adjustments: [...bridge.adjustments, {
        label: '',
        amount: '',
        reference: '',
        recurrence: 'uncertain'
      }]
    })
  }, "Add adjustment"), /*#__PURE__*/React.createElement("p", null, "Required total: base-year EBITDA ", model.baseEbitda || 'not entered', " million ", model.currency, ". Calculate draft to verify the exact sum. Arithmetic agreement does not establish source accuracy or acceptable accounting treatment."), /*#__PURE__*/React.createElement("label", null, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: bridge.confirmed === true,
    onChange: e => onChange({
      ...model,
      ebitdaReconciliation: {
        ...bridge,
        confirmed: e.target.checked
      }
    })
  }), "I checked the reconciliation against its sources, confirmed the model period and currency, and checked for double-counting."), /*#__PURE__*/React.createElement("button", {
    onClick: () => {
      var next = {
        ...model,
        baseEbitdaComparable: false
      };
      delete next.ebitdaReconciliation;
      onChange(next);
    }
  }, "Remove EBITDA reconciliation")));
}