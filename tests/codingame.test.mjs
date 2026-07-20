import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(new URL('../engine/codingame/othello.js', import.meta.url), 'utf8');

assert.doesNotMatch(source, /^\s*(?:import|export)\s/m, 'CodinGame submission must be standalone.');
assert.match(source, /TURN_RESPONSE_LIMIT_MS\s*=\s*150/);
assert.match(source, /SEARCH_TIME_LIMIT_MS\s*=\s*100/);
assert.match(source, /refereeActions\.includes\(output\)/);
assert.match(source, /console\.log\(output\)/);

console.log('CodinGame standalone submission checks passed.');
