import test from 'node:test';
import assert from 'node:assert/strict';
import { GROUPS, VIEWS, readRoute, routeHash, parseTimestamp, companyIndex, selectedProjectSlide } from '../../src/workspace-model.mjs';

test('all existing workflows remain reachable through the shared navigation', () => {
  const required = ['dashboard','portfolio','overview','chat','summary','research','meetingprep','slides','studio','formats','deepdive','explain','onepager','pipeline','review','agents','feed','analysts','alerts','settings'];
  for (const view of required) assert.ok(VIEWS.includes(view), view);
  assert.equal(new Set(VIEWS).size, VIEWS.length);
  assert.ok(GROUPS.some(g => g.items.some(([id]) => id === 'review')));
});
test('routes preserve company context and reject unknown destinations', () => {
  assert.deepEqual(readRoute(routeHash('review','BRK.B')), { view:'review',ticker:'BRK.B' });
  assert.deepEqual(readRoute('#view=unknown&ticker=%20de%20'), { view:'today',ticker:'DE' });
  assert.deepEqual(readRoute(''), {view:'today',ticker:''});
  assert.deepEqual(readRoute(routeHash('companies','A&B')), {view:'companies',ticker:'A&B'});
});
test('Flask RFC dates, SQL timestamps and zoned ISO dates represent the same instant', () => {
  const dates = ['Sun, 06 Sep 2026 17:00:00 GMT','2026-09-06 17:00:00','2026-09-06T17:00:00Z','2026-09-06T13:00:00-04:00'];
  for (const value of dates) assert.equal(parseTimestamp(value).toISOString(),'2026-09-06T17:00:00.000Z');
  for (const value of [null,undefined,'','not a date',new Date(NaN)]) assert.equal(parseTimestamp(value),null);
  assert.equal(parseTimestamp('2026-09-06').toLocaleDateString('en-CA',{timeZone:'America/New_York'}),'2026-09-06');
});
test('coverage combines thesis-only and overview-only companies without losing either', () => {
  const theses = [{ticker:'de',company:'Deere'},{ticker:'ABT',companyName:'Abbott'}];
  const overviews = [{ticker:'DE',company:'Deere & Co'},{ticker:'AMZN',company:'Amazon'},{}];
  const result=companyIndex(theses,overviews);
  assert.deepEqual(result.map(x=>x.ticker),['ABT','AMZN','DE']);
  assert.equal(result[0].hasThesis,true); assert.equal(result[0].hasOverview,false);
  assert.equal(result[1].hasThesis,false); assert.equal(result[1].hasOverview,true);
  assert.equal(result[2].hasThesis,true); assert.equal(result[2].hasOverview,true);
  assert.equal(theses[0].ticker,'de');
});

test('project refresh preserves selected slide while new projects select their own first slide', () => {
  const project={id:2,slides:[{slide_number:1},{slide_number:3}]};
  assert.equal(selectedProjectSlide(project,2,3).slide_number,3);
  assert.equal(selectedProjectSlide(project,1,3).slide_number,1);
  assert.equal(selectedProjectSlide(project,2,7).slide_number,1);
  assert.equal(selectedProjectSlide({id:2,slides:[]},2,3),null);
});
