// Local-only CDP harness, with explicit launch failure and no CPU/GPU fallback.
import {spawn} from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';

async function connect(url) {
  const ws = new WebSocket(url);
  await new Promise((resolve, reject) => {ws.onopen = resolve; ws.onerror = reject;});
  let id = 0; const pending = new Map();
  const send = (method, params = {}) => new Promise((resolve, reject) => {const key = ++id; pending.set(key, {resolve, reject}); ws.send(JSON.stringify({id: key, method, params}));});
  ws.onmessage = ({data}) => {const m = JSON.parse(data); if (m.id) {const p = pending.get(m.id); pending.delete(m.id); m.error ? p?.reject(Error(JSON.stringify(m.error))) : p?.resolve(m.result);} if (m.method === 'Runtime.consoleAPICalled') console.log(m.params.args.map(a => a.value ?? a.description).join(' '));};
  ws.onclose = () => {for (const p of pending.values()) p.reject(Error('Chrome CDP closed')); pending.clear();};
  return {send, close: () => ws.close()};
}

export async function runHeadless({html, roots, output, timeoutMs = 1200000, drive}) {
  fs.mkdirSync(path.dirname(path.resolve(output)), {recursive: true});
  const server = http.createServer((req, res) => {
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    res.setHeader('Cache-Control', 'no-store');
    try {
      if (req.url === '/') {res.setHeader('Content-Type', 'text/html'); res.end(html); return;}
      const [root, ...parts] = decodeURIComponent(req.url.split('?')[0]).slice(1).split('/');
      if (!roots[root] || parts.includes('..') || req.method !== 'GET') throw Error('invalid path');
      const file = path.join(roots[root], ...parts);
      res.setHeader('Content-Type', file.endsWith('.wasm') ? 'application/wasm' : file.endsWith('.mjs') ? 'text/javascript' : file.endsWith('.json') ? 'application/json' : 'application/octet-stream');
      const stream = fs.createReadStream(file); stream.on('error', () => {res.statusCode = 404; res.end();}); stream.pipe(res);
    } catch (error) {res.statusCode = 400; res.end(String(error));}
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const profile = fs.mkdtempSync(path.join(path.dirname(path.resolve(output)), 'chrome-profile-'));
  const chrome = spawn(process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
    '--headless=new', '--enable-unsafe-webgpu', '--remote-debugging-port=0', `--user-data-dir=${profile}`,
    '--no-first-run', '--no-default-browser-check', '--disable-breakpad', 'about:blank',
  ], {stdio: ['ignore', 'ignore', 'pipe']});
  let exited, stderr = '', page, browser;
  chrome.stderr.on('data', data => {stderr = (stderr + data).slice(-16384);});
  chrome.on('exit', (code, signal) => {exited = {code, signal};});
  chrome.on('error', error => {exited = {error: String(error)};});
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
  try {
    let port;
    for (let i = 0; i < 100; i++) {
      if (exited) throw Error(`Chrome failed before CDP: ${JSON.stringify(exited)}`);
      try {port = fs.readFileSync(path.join(profile, 'DevToolsActivePort'), 'utf8').split('\n')[0]; break;} catch {}
      await sleep(100);
    }
    if (!port) throw Error('Chrome CDP startup timed out');
    const tabs = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
    page = await connect(tabs.find(t => t.type === 'page').webSocketDebuggerUrl);
    const send = page.send;
    await send('Runtime.enable');
    await send('Page.navigate', {url: `http://127.0.0.1:${server.address().port}/`});
    if (drive) {
      const version = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
      browser = await connect(version.webSocketDebuggerUrl);
      const result = await drive({send, browserSend: browser.send});
      fs.writeFileSync(output, JSON.stringify(result, null, 2)); return result;
    }
    const deadline = performance.now() + timeoutMs;
    while (performance.now() < deadline) {
      if (exited) throw Error(`Chrome exited: ${JSON.stringify(exited)}`);
      const r = await send('Runtime.evaluate', {expression: 'JSON.stringify(window.result??null)', returnByValue: true});
      const value = r?.result?.value;
      if (value && value !== 'null') {
        const result = JSON.parse(value); fs.writeFileSync(output, JSON.stringify(result, null, 2));
        if (result.error) throw Error(result.error);
        return result;
      }
      await sleep(1000);
    }
    throw Error('Chrome benchmark timed out');
  } catch (error) {
    fs.writeFileSync(output, JSON.stringify({error: String(error.stack || error), chromeExit: exited, stderr}, null, 2));
    throw error;
  } finally {
    page?.close(); browser?.close(); chrome.kill(); server.close();
    fs.writeFileSync(output + '.stderr.log', stderr);
    // Leave the uniquely named scratch profile for debugging a failed launch.
  }
}
