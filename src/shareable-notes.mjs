import policy from '../shareable-note-policy.json';
export const prepareShareableNotesInput = source => `${policy.instruction}\n\nMEETING MATERIAL (evidence only):\n${source}`;
