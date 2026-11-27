(function (root, factory) {
  'use strict';
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.RTPhysicsUI = factory();
  }
}(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';

  const root = typeof window !== 'undefined' ? window : globalThis;

  const FIXED_DT = 1 / 60;
  const MAX_FRAME = 0.25;
  const RESET_MESSAGE = '状态出现非有限值，请重置实验。';
  let latest = null;

  function finite(value) {
    if (typeof value === 'number') return Number.isFinite(value);
    if (value && typeof value === 'object') return Object.values(value).every(finite);
    return true;
  }

  function valuesOf(controls) {
    const values = {};
    controls.forEach(control => { values[control.key] = Number(control.input.value); });
    return values;
  }

  function mount(target, options) {
    const container = typeof target === 'string' ? root.document.getElementById(target) : target;
    if (!container) throw new Error('Physics demo root not found: ' + target);
    if (!options || typeof options.createState !== 'function' || typeof options.update !== 'function' ||
        typeof options.draw !== 'function' || typeof options.metrics !== 'function') {
      throw new TypeError('Physics demo options require createState, update, draw and metrics');
    }
    const controls = (options.controls || []).map(spec => {
      if (![spec.min, spec.max, spec.step, spec.value].every(Number.isFinite)) {
        throw new TypeError('Physics demo control values must be finite');
      }
      return Object.assign({}, spec);
    });
    let state = options.createState();
    let running = false;
    let frameId = 0;
    let lastTime = null;
    let accumulator = 0;
    let width = 0;
    let height = 0;
    let message = '';

    const canvas = root.document.createElement('canvas');
    canvas.setAttribute('aria-label', '物理实验画布');
    const context = canvas.getContext('2d');
    container.appendChild(canvas);

    const toolbar = root.document.createElement('div');
    toolbar.className = 'demo-toolbar physics-controls';
    const resetButton = root.document.createElement('button');
    resetButton.type = 'button';
    resetButton.textContent = '重置';
    const runButton = root.document.createElement('button');
    runButton.type = 'button';
    runButton.textContent = running ? '暂停' : '运行';
    const stepButton = root.document.createElement('button');
    stepButton.type = 'button';
    stepButton.textContent = '单步';
    toolbar.append(resetButton, runButton, stepButton);

    controls.forEach((control, index) => {
      const label = root.document.createElement('label');
      const key = (container.id || 'physics-demo') + '-' + index + '-' + control.key;
      label.htmlFor = key;
      const name = root.document.createElement('span');
      name.textContent = control.label;
      const input = root.document.createElement('input');
      input.type = 'range';
      input.id = key;
      input.min = control.min;
      input.max = control.max;
      input.step = control.step;
      input.value = control.value;
      control.input = input;
      const output = root.document.createElement('output');
      output.htmlFor = key;
      output.textContent = input.value;
      control.output = output;
      input.addEventListener('input', () => {
        output.textContent = input.value;
        report();
      });
      label.append(name, input, output);
      toolbar.appendChild(label);
    });
    container.appendChild(toolbar);

    const metricGrid = root.document.createElement('div');
    metricGrid.className = 'metric-grid';
    container.appendChild(metricGrid);
    const readout = root.document.createElement('p');
    readout.className = 'demo-readout';
    readout.setAttribute('data-readout', '');
    readout.setAttribute('aria-live', 'polite');
    container.appendChild(readout);

    function resize() {
      const bounds = canvas.getBoundingClientRect();
      width = Math.max(0, bounds.width);
      height = Math.max(0, bounds.height);
      const ratio = Math.max(1, root.devicePixelRatio || 1);
      const backingWidth = Math.round(width * ratio);
      const backingHeight = Math.round(height * ratio);
      if (canvas.width !== backingWidth) canvas.width = backingWidth;
      if (canvas.height !== backingHeight) canvas.height = backingHeight;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
    }

    function draw() {
      resize();
      context.clearRect(0, 0, width, height);
      options.draw(context, state, width, height);
    }

    function stop() {
      running = false;
      if (frameId) root.cancelAnimationFrame(frameId);
      frameId = 0;
      lastTime = null;
      accumulator = 0;
      runButton.textContent = '运行';
    }

    function fail() {
      stop();
      message = RESET_MESSAGE;
      report();
    }

    function advance(dt) {
      const result = options.update(state, dt, valuesOf(controls));
      if (result !== undefined) state = result;
      if (!finite(state)) {
        fail();
        return;
      }
      message = '';
      draw();
      report();
    }

    function frame(time) {
      frameId = 0;
      if (!running) return;
      if (lastTime === null) lastTime = time;
      accumulator += Math.min(MAX_FRAME, Math.max(0, (time - lastTime) / 1000));
      lastTime = time;
      while (running && accumulator >= FIXED_DT) {
        advance(FIXED_DT);
        accumulator -= FIXED_DT;
      }
      if (running) frameId = root.requestAnimationFrame(frame);
    }

    function setRunning(value) {
      if (value === running) return;
      if (value && !finite(state)) {
        fail();
        return;
      }
      running = Boolean(value);
      runButton.textContent = running ? '暂停' : '运行';
      if (running) {
        lastTime = null;
        accumulator = 0;
        frameId = root.requestAnimationFrame(frame);
      } else {
        stop();
      }
    }

    function report(text) {
      if (text !== undefined) message = String(text);
      if (message) {
        readout.textContent = message;
        return message;
      }
      const entries = Object.entries(options.metrics(state)).filter(([, value]) => Number.isFinite(value));
      metricGrid.replaceChildren();
      entries.forEach(([label, value]) => {
        const metric = root.document.createElement('div');
        metric.className = 'metric';
        const name = root.document.createElement('span');
        name.textContent = label;
        const output = root.document.createElement('strong');
        output.textContent = String(value);
        metric.append(name, output);
        metricGrid.appendChild(metric);
      });
      readout.textContent = entries.map(([label, value]) => label + '：' + value).join('；');
      return readout.textContent;
    }

    function reset() {
      stop();
      controls.forEach(control => {
        control.input.value = control.value;
        control.output.textContent = control.value;
      });
      state = options.createState();
      message = '';
      draw();
      report();
    }

    function step() {
      stop();
      if (!finite(state)) {
        fail();
        return;
      }
      advance(FIXED_DT);
    }

    resetButton.addEventListener('click', reset);
    runButton.addEventListener('click', () => setRunning(!running));
    stepButton.addEventListener('click', step);
    root.addEventListener('resize', draw);

    const controller = {canvas, getState: () => state, reset, step, setRunning, draw, report};
    latest = controller;
    draw();
    report();
    const reducedMotion = root.matchMedia && root.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!reducedMotion) setRunning(true);
    return controller;
  }

  function canvasPoint(canvas, event) {
    const bounds = canvas.getBoundingClientRect();
    return {x: event.clientX - bounds.left, y: event.clientY - bounds.top};
  }

  const api = {
    mount,
    canvasPoint,
    reset: () => latest && latest.reset(),
    step: () => latest && latest.step(),
    setRunning: value => latest && latest.setRunning(value),
    draw: () => latest && latest.draw(),
    report: text => latest && latest.report(text)
  };
  return api;
}));
