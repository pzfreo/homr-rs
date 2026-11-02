// Shared Node WASM / Chrome WebGPU model comparison. All fixtures are scratch.
// IO uses relative paths, allowing the exact same loop in both runtimes.
export async function compareModels({ort, pages, models, read, write, log = console.log,
  threads = 1, ep = 'wasm', optimization = 'all', instructions = () => null}) {
  ort.env.wasm.numThreads = threads;
  const report = {ort: ort.env.versions, ep, threads, optimization, pages: [],
    counter: ep === 'wasm' ? 'proc_pid_rusage RUSAGE_INFO_V4 (when supplied)' : 'GPU speed not measured'};
  const floats = bytes => new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  const json = async name => JSON.parse(new TextDecoder().decode(await read(name)));
  const save = async (name, value) => write(name, new TextEncoder().encode(JSON.stringify(value, null, 2)));
  for (const [kind, modelPath] of Object.entries(models)) {
    const session = await ort.InferenceSession.create(await read(modelPath), {
      executionProviders: ep === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'], graphOptimizationLevel: optimization,
    });
    for (const page of pages) {
      const meta = await json(`native/${page}/models.json`);
      let row = report.pages.find(p => p.page === page);
      if (!row) { row = {page, segnet: {different: 0, pixels: 0, instructions: 0}, encoder: []}; report.pages.push(row); }
      if (kind === 'segnet') {
        for (const [index, batch] of meta.batches.entries()) {
          const input = Float32Array.from(await read(`native/${page}/seg-${index}.input.u8`));
          const tensor = new ort.Tensor('float32', input, batch.shape);
          const before = instructions();
          const output = (await session.run({[session.inputNames[0]]: tensor}))[session.outputNames[0]];
          const after = instructions();
          if (before !== null) row.segnet.instructions += after - before;
          const [n, c, h, w] = output.dims, plane = h * w;
          if (output.dims.join() !== batch.output_shape.join()) throw Error('segnet output shape mismatch');
          const got = new Uint8Array(n * plane), data = output.data;
          const ref = await read(`native/${page}/seg-${index}.native.u8`);
          if (ref.length !== got.length) throw Error('class map size mismatch');
          for (let k = 0; k < n; k++) for (let p = 0; p < plane; p++) {
            let best = 0, v = data[k * c * plane + p];
            for (let ch = 1; ch < c; ch++) {
              const next = data[(k * c + ch) * plane + p];
              if (!Number.isFinite(next)) throw Error('nonfinite segnet output');
              if (next > v) { best = ch; v = next; }
            }
            got[k * plane + p] = best;
            row.segnet.different += Number(best !== ref[k * plane + p]);
          }
          row.segnet.pixels += got.length;
          await write(`output/${page}/seg-${index}.classes.u8`, got);
          output.dispose(); tensor.dispose();
        }
        row.segnet.fraction = row.segnet.different / row.segnet.pixels;
      } else {
        for (let staff = 0; staff < meta.staffs; staff++) {
          const d = `native/${page}/${staff}`;
          const shape = (await json(`${d}/input.json`)).shape;
          const input = new ort.Tensor('float32', floats(await read(`${d}/input.f32`)), shape);
          const before = instructions();
          const output = (await session.run({[session.inputNames[0]]: input}))[session.outputNames[0]];
          const after = instructions();
          if (output.dims.join() !== '1,1280,512') throw Error('encoder output shape mismatch');
          const ref = floats(await read(`${d}/context.bin`)), got = output.data;
          if (ref.length !== got.length) throw Error('context size mismatch');
          let max = 0, sum = 0, different = 0;
          for (let i = 0; i < ref.length; i++) {
            const e = Math.abs(got[i] - ref[i]);
            if (!Number.isFinite(e)) throw Error('nonfinite encoder output');
            max = Math.max(max, e); sum += e; different += Number(e !== 0);
          }
          row.encoder.push({staff, max_abs: max, mean_abs: sum / got.length, different,
            values: got.length, instructions: before === null ? null : after - before});
          await write(`output/${page}/${staff}/context.bin`, new Uint8Array(got.buffer, got.byteOffset, got.byteLength));
          output.dispose(); input.dispose();
        }
      }
      await save(`output/${page}/metrics.json`, row);
      log(kind, page, kind === 'segnet' ? row.segnet.fraction : row.encoder.at(-1)?.max_abs);
    }
    await session.release();
  }
  const seg = report.pages.map(p => ({page: p.page, ...p.segnet}));
  const enc = report.pages.flatMap(p => p.encoder.map(s => ({page: p.page, ...s})));
  report.summary = {
    pages: pages.length, staffs: enc.length,
    segnet_different: seg.reduce((n, p) => n + p.different, 0),
    segnet_pixels: seg.reduce((n, p) => n + p.pixels, 0),
    segnet_pages_different: seg.filter(p => p.different).length,
    segnet_worst: seg.toSorted((a, b) => b.fraction - a.fraction)[0],
    encoder_max_abs: Math.max(...enc.map(s => s.max_abs)),
    encoder_mean_abs: enc.reduce((n, s) => n + s.mean_abs, 0) / enc.length,
    encoder_worst: enc.toSorted((a, b) => b.max_abs - a.max_abs)[0],
  };
  await save('output/comparison.json', report);
  return report.summary;
}
