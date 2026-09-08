export function managedMeetingState(job) {
  if (!job) return {blocked:false, message:''};
  if (job.status === 'checking') return {blocked:true, message:'Checking assignment progress…'};
  if (job.status === 'unavailable') return {blocked:true, message:'Assignment status is temporarily unavailable. Retrying…'};
  if (job.prep_status === 'done') return {blocked:false, message:'Meeting pack saved by Command Charlie.'};
  if (job.prep_status === 'failed') return {blocked:true, message:job.prep_error?.startsWith('Generate questions failed') ? 'Question generation needs attention. Your completed source analyses are saved; retry the pack in Command Charlie.' : (job.prep_error || 'Preparation needs attention. Retry this pack in Command Charlie.')};
  const stages={analyzing:`Analyzing source documents (${job.completed || 0}/${job.total || '?'})…`,synthesizing:'Synthesizing the meeting brief…',generating:'Generating source-supported meeting questions…',saving:'Saving your meeting pack…'};
  return {blocked:true, message:stages[job.prep_step] || 'Your assignment is queued for meeting preparation…'};
}
