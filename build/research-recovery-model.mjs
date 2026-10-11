// Saved receipts are evidence of a returned response, not validated research or billing.
export function recoveryEvidence(run={}) {
 const state=run.state||{}, recovery=run.responseRecovery||{};
 const stage=state.knownFailure?.stage||state.inFlight||'';
 const saved=Array.isArray(recovery.savedStages)?recovery.savedStages:[];
 const hasReturn=!!stage&&saved.includes(stage);
 const known=!!state.knownFailure;
 let title='Saved work is preserved';
 let next='Resume continues from saved stages. Remaining stages may use model credits.';
 if(stage&&known){title='Response returned; this stage failed';next='Resume requests a new paid attempt for this stage and archives its saved response, if available. Review usage before retrying.';}
 else if(stage&&hasReturn){title='Response saved before interruption';next='Resume can reuse this saved response for the interrupted stage. It still requires the existing retry acknowledgment; later stages may use model credits. A saved response may still fail validation.';}
 else if(stage){title='Previous call outcome is unresolved';next='No saved response is available for this stage. Inspect provider usage and backend execution before acknowledging a retry. Missing evidence does not establish that no charge occurred.';}
 return {title,next,stage,hasReturn,savedCount:saved.length,previousAttempts:Number.isInteger(recovery.previousAttempts)?recovery.previousAttempts:0,
  receipts:Array.isArray(recovery.receipts)?recovery.receipts:[]};
}
