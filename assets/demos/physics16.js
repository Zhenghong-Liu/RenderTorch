(() => {
  'use strict';
  const container = document.getElementById('substep-invariance');
  if (!container) return;
  const {projectDistance, projectDistanceXPBD} = window.RTPhysics;
  const ui = window.RTPhysicsUI;
  const defaults = Object.freeze({substeps: 2, iterations: 8, compliance: 0.01});
  const restLength = 2;
  const outerDt = 1 / 60;
  const totalDuration = 1;
  const duration = 60;
  const referenceSubsteps = 8;
  const epsilon = 1e-12;

  function initialPair() {
    return [
      {x: -1.5, y: 0, vx: -0.6, vy: 0, w: 1},
      {x: 1.5, y: 0, vx: 0.6, vy: 0, w: 1},
    ];
  }

  function createRun() {
    const pair = initialPair();
    return {
      a: Object.assign({}, pair[0]), b: Object.assign({}, pair[1]),
      lambda: 0, length: 3, error: 0.5, maxRelativeError: 0.5,
    };
  }

  function measure(run) {
    run.length = Math.hypot(run.b.x - run.a.x, run.b.y - run.a.y);
    run.error = Math.abs(run.length - restLength) / restLength;
    run.maxRelativeError = Math.max(run.maxRelativeError, run.error);
  }

  function validPair(pair) {
    return pair.every(particle => ['x', 'y', 'vx', 'vy', 'w'].every(key =>
      Number.isFinite(particle[key])) && particle.w === 1) &&
      Math.hypot(pair[1].x - pair[0].x, pair[1].y - pair[0].y) > epsilon;
  }

  function createState(values = defaults) {
    return {
      substeps: values.substeps, selectedSubsteps: values.substeps,
      iterations: values.iterations,
      compliance: values.compliance, referenceSubsteps,
      selectedH: outerDt / values.substeps,
      referenceH: outerDt / referenceSubsteps,
      restLength, outerDt, totalDuration, errorBarScale: 0.5,
      pbd: {selected: createRun(), reference: createRun()},
      xpbd: {selected: createRun(), reference: createRun()},
      lengthDifference: {pbd: 0, xpbd: 0},
      time: 0, frame: 0, finite: true, invalid: false, finished: false,
    };
  }

  function pauseWithGuidance(state, message) {
    state.invalid = true;
    state.finished = true;
    state.finite = false;
    controller.setRunning(false);
    warning.hidden = false;
    warning.textContent = message;
  }

  function solveRun(run, method, substeps, h, iterations, compliance) {
    for (let substep = 0; substep < substeps; substep += 1) {
      const oldA = Object.assign({}, run.a);
      const oldB = Object.assign({}, run.b);
      let pair = [Object.assign({}, run.a, {
        x: run.a.x + run.a.vx * h, y: run.a.y + run.a.vy * h,
      }), Object.assign({}, run.b, {
        x: run.b.x + run.b.vx * h, y: run.b.y + run.b.vy * h,
      })];
      if (method === 'xpbd') run.lambda = 0;
      for (let iteration = 0; iteration < iterations; iteration += 1) {
        let projected;
        if (method === 'pbd') {
          projected = projectDistance(pair[0], pair[1], restLength, 1);
          pair = [projected.a, projected.b];
        } else {
          projected = projectDistanceXPBD(
            pair[0], pair[1], restLength, compliance, h, run.lambda);
          pair = [projected.a, projected.b];
          run.lambda = projected.lambda;
        }
      }
      run.a = Object.assign({}, pair[0], {
        vx: (pair[0].x - oldA.x) / h, vy: (pair[0].y - oldA.y) / h,
      });
      run.b = Object.assign({}, pair[1], {
        vx: (pair[1].x - oldB.x) / h, vy: (pair[1].y - oldB.y) / h,
      });
      measure(run);
    }
  }

  function update(state, dt, values) {
    if (state.finished || state.invalid) {
      controller.setRunning(false);
      return state;
    }
    if (![dt, state.substeps, state.iterations, state.compliance].every(Number.isFinite) ||
      dt !== outerDt || ![1, 2, 4, 8].includes(state.substeps) ||
      !Number.isInteger(state.iterations) || state.iterations < 1 || state.iterations > 40 ||
      state.compliance < 0 || state.compliance > 1 ||
      !['pbd', 'xpbd'].every(method => ['selected', 'reference'].every(group =>
        validPair([state[method][group].a, state[method][group].b])))) {
      pauseWithGuidance(state, '控件、时间步或距离几何无效，已暂停。请重置实验。');
      return state;
    }
    state.selectedH = outerDt / state.substeps;
    state.referenceH = outerDt / referenceSubsteps;
    for (const method of ['pbd', 'xpbd']) {
      solveRun(state[method].selected, method, state.substeps, state.selectedH,
        state.iterations, state.compliance);
      solveRun(state[method].reference, method, referenceSubsteps, state.referenceH,
        state.iterations, state.compliance);
    }
    for (const method of ['pbd', 'xpbd']) {
      state.lengthDifference[method] = Math.abs(
        state[method].selected.length - state[method].reference.length);
    }
    state.time += dt;
    state.frame += 1;
    state.finite = ['pbd', 'xpbd'].every(method => ['selected', 'reference'].every(group => {
      const run = state[method][group];
      return validPair([run.a, run.b]) && [run.length, run.error, run.maxRelativeError,
        run.lambda, state.lengthDifference[method]].every(Number.isFinite);
    })) && [state.selectedH, state.referenceH, state.time].every(Number.isFinite);
    if (!state.finite) {
      pauseWithGuidance(state, '状态出现非有限值，已暂停。请重置，不要用裁剪隐藏数值错误。');
      return state;
    }
    if (state.frame >= duration) {
      state.finished = true;
      controller.setRunning(false);
      warning.hidden = false;
      warning.textContent = '四条运行均已完成 1 s 模拟时间并暂停。点击重置可复现。';
    }
    return state;
  }

  function drawErrorBar(ctx, label, color, value, y) {
    const left = 112;
    const width = 154;
    const baseline = 248;
    ctx.font = '11px system-ui';
    ctx.fillStyle = '#9db6b8';
    ctx.fillText(label, 8, y + 4);
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#547079';
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(left + width, y);
    ctx.stroke();
    const fraction = Math.max(0, Math.min(1, value / 0.5));
    ctx.strokeStyle = color;
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(left + width * fraction, y);
    ctx.stroke();
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(baseline, y - 5);
    ctx.lineTo(baseline, y + 5);
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.fillText(value.toFixed(3), baseline + 7, y + 4);
  }

  function draw(ctx, state, width, height) {
    ctx.fillStyle = '#10242c';
    ctx.fillRect(0, 0, width, height);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.font = '11px system-ui';
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('相同总时间 T=1 s；参考固定 8 子步', 8, 14);
    ctx.fillText('所选 q=' + state.substeps + ' h=' + state.selectedH.toFixed(6) +
      ' 迭代=' + state.iterations, 8, 30);
    ctx.fillText('柔度=' + state.compliance.toFixed(2) +
      ' 参考h=' + state.referenceH.toFixed(6), 8, 46);
    ctx.fillStyle = '#dce8df';
    ctx.fillText('相对约束误差条：共同 0..0.5 标尺', 8, 64);
    drawErrorBar(ctx, 'PBD 所选', '#8fb7ff', state.pbd.selected.error, 82);
    drawErrorBar(ctx, 'PBD 参考', '#587aad', state.pbd.reference.error, 106);
    drawErrorBar(ctx, 'XPBD 所选', '#8fd1b0', state.xpbd.selected.error, 130);
    drawErrorBar(ctx, 'XPBD 参考', '#527f6a', state.xpbd.reference.error, 154);
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('差 PBD=' + state.lengthDifference.pbd.toFixed(4) +
      ' XPBD=' + state.lengthDifference.xpbd.toFixed(4), 8, 174);
  }

  const controller = ui.mount(container, {
    createState,
    controls: [
      {key: 'substeps', label: '所选每次更新子步数', min: 1, max: 8, step: 1, value: defaults.substeps},
      {key: 'iterations', label: '每实际子步迭代数', min: 1, max: 40, step: 1, value: defaults.iterations},
      {key: 'compliance', label: 'XPBD 柔度 alpha（0..1）', min: 0, max: 1, step: 0.01, value: defaults.compliance},
    ],
    update,
    draw,
    metrics(state) {
      const rounded = value => Number(value.toFixed(8));
      const runMetrics = (method, group, label) => ({
        [label + '最终长度（m）']: rounded(state[method][group].length),
        [label + '相对约束误差']: rounded(state[method][group].error),
        [label + '最大相对约束误差']: rounded(state[method][group].maxRelativeError),
      });
      return Object.assign({
        '所选子步数': state.substeps,
        '参考子步数': referenceSubsteps,
        '每实际子步迭代数': state.iterations,
        'XPBD 柔度 alpha': state.compliance,
        '所选实际子步长 h（s）': rounded(state.selectedH),
        '参考实际子步长 h（s）': rounded(state.referenceH),
        '共同目标长度（m）': restLength,
      }, runMetrics('pbd', 'selected', 'PBD 所选'),
      runMetrics('pbd', 'reference', 'PBD 参考'),
      runMetrics('xpbd', 'selected', 'XPBD 所选'),
      runMetrics('xpbd', 'reference', 'XPBD 参考'), {
        'PBD 所选与参考绝对长度差（m）': rounded(state.lengthDifference.pbd),
        'XPBD 所选与参考绝对长度差（m）': rounded(state.lengthDifference.xpbd),
        'XPBD 所选当前 lambda': rounded(state.xpbd.selected.lambda),
        'XPBD 参考当前 lambda': rounded(state.xpbd.reference.lambda),
        '总模拟时间（s）': rounded(state.time),
        '有限值状态（1是/0否）': state.finite ? 1 : 0,
      });
    },
  });
  controller.setRunning(false);
  controller.canvas.setAttribute('aria-label', '子步不变性实验：PBD 与 XPBD 所选及八子步参考运行，显示最终长度差和共同标尺误差条');

  const toolbar = Array.from(container.children).find(element => element.className === 'demo-toolbar physics-controls');
  const warning = document.createElement('p');
  warning.className = 'demo-readout';
  warning.setAttribute('role', 'status');
  warning.setAttribute('aria-live', 'polite');
  warning.hidden = true;
  container.appendChild(warning);
  function clearWarning() {
    warning.hidden = true;
    warning.textContent = '';
  }
  const resetController = controller.reset;
  controller.reset = () => {
    resetController();
    clearWarning();
  };
  toolbar.children[0].addEventListener('click', clearWarning);
  const ranges = Array.from(toolbar.children).filter(element => element.tagName.toLowerCase() === 'label')
    .map(element => Array.from(element.children).find(child => child.tagName.toLowerCase() === 'input'));
  toolbar.addEventListener('input', () => {
    controller.setRunning(false);
    const values = {
      substeps: Number(ranges[0].value),
      iterations: Number(ranges[1].value),
      compliance: Number(ranges[2].value),
    };
    const valid = [1, 2, 4, 8].includes(values.substeps) &&
      Number.isInteger(values.iterations) && values.iterations >= 1 && values.iterations <= 40 &&
      Number.isFinite(values.compliance) && values.compliance >= 0 && values.compliance <= 1;
    if (!valid) {
      pauseWithGuidance(controller.getState(), '控件含非有限值、越界值或非允许子步数，已暂停。请重置实验。');
      return;
    }
    Object.assign(controller.getState(), createState(values));
    clearWarning();
    controller.draw();
    controller.report();
  });
})();
