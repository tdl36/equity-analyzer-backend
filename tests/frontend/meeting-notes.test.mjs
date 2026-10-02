import test from 'node:test';
import assert from 'node:assert/strict';
import {documentHtml,sourceMarkers,emailDocument} from '../../src/summary-lab-format.mjs';
test('shorthand survives Lab rendering and email without losing qualifiers or citations',()=>{
 const notes='## Growth / timing\n- mgmt: ~5% possible → not committed [P1]\n- Open: annual cadence?';
 const html=documentHtml(notes,x=>x);
 assert.match(html,/<li>/);assert.match(html,/not committed/);assert.match(html,/annual cadence\?/);
 assert.match(sourceMarkers(html,'inline'),/Source part 1/);
 const mail=emailDocument('Fixture',[['Meeting Notes',notes]],x=>x,'inline');
 assert.match(mail,/Meeting Notes/);assert.match(mail,/not committed/);assert.match(mail,/Calibri/);
});
