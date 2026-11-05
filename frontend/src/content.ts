import type { UserProfile } from './types';
export const SELLERS: Record<string, string> = {
  S001: 'Aurelia Home Decor', S002: 'BluePeak Electronics', S003: 'GreenCart Organics', S004: 'StrideOne Footwear',
};
const sellerPrompts: Record<string, string> = {
  S001: 'What caused the settlement shortfall for Aurelia Home Decor?',
  S002: 'Summarise the inventory sync drift for BluePeak Electronics.',
  S003: 'What is the status of the GreenCart Organics tier upgrade?',
  S004: 'Why did returns spike for StrideOne Footwear?',
};
const org = ['What does the payment settlement policy say about adjustments?', 'What are the steps in the payment recovery runbook?'];
const department: Record<string, string[]> = {
  Business: [org[0], 'What are the current priorities for my seller accounts?'],
  Support: ['What steps should Support follow when escalating a seller issue?', 'Which checks should I complete before closing a support ticket?'],
  Engineering: ['What are the required steps for incident recovery and verification?', 'Which engineering changes are awaiting operational validation?'],
  Operations: [org[1], 'What checks are required before releasing a corrected payout?'],
  IT: ['What is the process for reviewing access and security incidents?', 'Which controls are required before restoring a service?'],
};
export function prompts(user: UserProfile) {
  return [...new Set([...user.seller_scope.map(id => sellerPrompts[id]).filter(Boolean), ...(department[user.department] || org)])].slice(0, 4);
}
export function cleanAnswer(text: string) {
  let answer = text.replace(/<br\s*\/?>/gi, ' · ').replace(/\s*\[Evidence\]/g, '');
  for (const marker of ['\nSources:\n', '\nEvidence available in context:\n']) {
    const index = answer.indexOf(marker);
    if (index !== -1 && (!answer.slice(index + marker.length).trim() || answer.slice(index + marker.length).trimStart().startsWith('- '))) answer = answer.slice(0, index).trimEnd();
  }
  answer = answer.replace(/\n(?:Sources:|Evidence available in context:)\s*$/, '').trim();
  const noEvidence = /^\s*(?:the\s+)?(?:provided\s+)?(?:passages|context|sources)\b.{0,180}\b(?:do not|don't|does not|doesn't)\b.{0,80}\b(?:contain|include|mention|describe)\b/is.test(answer);
  return { text: noEvidence ? "I couldn't find relevant information in the sources available for this account. If you expected an answer, ask your administrator to review which sources are available to your role." : answer, noEvidence };
}
