#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';

function parseOutput(argv) {
  let output = 'vendor/onnxruntime-web';
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === '--output') {
      if (!argv[index + 1]) throw new Error('Missing value for --output.');
      output = argv[index + 1];
      index += 1;
    } else {
      throw new Error(`Unknown option: ${argv[index]}`);
    }
  }
  return output;
}

const source = path.resolve('node_modules/onnxruntime-web/dist');
const output = path.resolve(parseOutput(process.argv.slice(2)));
if (!fs.existsSync(source)) {
  throw new Error('onnxruntime-web is not installed. Run npm install first.');
}

fs.rmSync(output, { recursive: true, force: true });
fs.mkdirSync(output, { recursive: true });

const selected = fs.readdirSync(source)
  .filter((name) => (
    /^ort\.wasm.*\.mjs$/.test(name)
    || /^ort-wasm.*\.wasm$/.test(name)
  ))
  .sort();

if (!selected.includes('ort.wasm.min.mjs')) {
  throw new Error('onnxruntime-web did not contain ort.wasm.min.mjs.');
}
if (!selected.some((name) => name.endsWith('.wasm'))) {
  throw new Error('onnxruntime-web did not contain a WASM binary.');
}

for (const name of selected) {
  fs.copyFileSync(path.join(source, name), path.join(output, name));
}

fs.writeFileSync(
  path.join(output, 'manifest.json'),
  `${JSON.stringify({ version: 1, files: selected }, null, 2)}\n`,
  'utf8',
);
console.log(`Prepared ${selected.length} ONNX Runtime Web assets in ${output}`);
