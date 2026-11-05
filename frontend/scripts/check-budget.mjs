import { readFileSync } from 'node:fs';
import { gzipSync } from 'node:zlib';

const html = readFileSync(new URL('../dist/index.html', import.meta.url), 'utf8');
const entry = html.match(/src="(\/assets\/[^"\s]+\.js)"/);
if (!entry) throw new Error('Missing production entry script');
const size = gzipSync(readFileSync(new URL(`../dist${entry[1]}`, import.meta.url))).length;
console.log(`Initial JavaScript: ${(size / 1024).toFixed(1)} KiB gzip (budget: 200 KiB)`);
if (size > 200 * 1024) throw new Error('Initial JavaScript exceeded the loading budget');
