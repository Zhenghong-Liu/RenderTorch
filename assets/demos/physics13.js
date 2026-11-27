(() => {
  'use strict';
  const container = document.getElementById('xpbd-compliance');
  if (!container) return;
  const {projectDistance, projectDistanceXPBD} = window.RTPhysics;
  const ui = window.RTPhysicsUI;
  const defaults = Object.freeze({compliance: 0.01, substeps: 2, iterations: 4});
  const restLength = 2;
  const duration = 60;
  const epsilon = 1e-12;

  function initialPair() {
    return [
      {x: 0, y: 0, vx: 0, vy: 0, w: 1},
      {x: 3, y: 0, vx: 0, vy: 0, w: 1},
    ];
  }

  function clonePair(pair) {
    return pair.map(particle => Object.assign({}, particle));
  }

  function measure(a, b) {
    const length = Math.hypot(b.x - a.x, b.y - a.y);
    return {length, error: Math.abs(length - restLength) / restLength};
  }

  function validParticle(particle) {
    return particle && ['x', 'y', 'vx', 'vy', 'w'].every(key => Number.isFinite(particle[key])) &&
      particle.w >= 0;
  }

  function validGeometry(state) {
    return [state.pbd.a, state.pbd.b, state.xpbd.a, state.xpbd.b].every(validParticle) &&
      Math.hypot(state.pbd.b.x - state.pbd.a.x, state.pbd.b.y - state.pbd.a.y) > epsilon &&
      Math.hypot(state.xpbd.b.x - state.xpbd.a.x, state.xpbd.b.y - state.xpbd.a.y) > epsilon;
  }

  function createState(values = defaults) {
    const pbdPair = clonePair(initialPair());
    const xpbdPair = clonePair(initialPair());
    const h = 1 / (60 * values.substeps);
    const pbdMeasurement = measure(pbdPair[0], pbdPair[1]);
    const xpbdMeasurement = measure(xpbdPair[0], xpbdPair[1]);
    return {
      compliance: values.compliance,
      substeps: values.substeps,
      iterations: values.iterations,
      restLength,
      h,
      alphaTilde: values.compliance / (h * h),
      pbd: {a: pbdPair[0], b: pbdPair[1], length: pbdMeasurement.length, error: pbdMeasurement.error},
      xpbd: {a: xpbdPair[0], b: xpbdPair[1], lambda: 0,
        length: xpbdMeasurement.length, error: xpbdMeasurement.error},
      maxRelativeError: Math.max(pbdMeasurement.error, xpbdMeasurement.error),
      time: 0,
      frame: 0,
      finite: true,
      invalid: false,
      finished: false,
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

  function update(state, dt, values) {
    if (state.finished || state.invalid) {
      controller.setRunning(false);
      return state;
    }
    if (![dt, state.compliance, state.substeps, state.iterations].every(Number.isFinite) ||
        dt <= 0 || state.compliance < 0 || !Number.isInteger(state.substeps) ||
        state.substeps < 1 || state.substeps > 8 || !Number.isInteger(state.iterations) ||
        state.iterations < 1 || state.iterations > 40 || !validGeometry(state)) {
      pauseWithGuidance(state, '时间步、控件或距离几何无效（包括非有限值或重合），已暂停。请重置实验。');
      return state;
    }

    state.h = 1 / (60 * state.substeps);
    state.alphaTilde = state.compliance / (state.h * state.h);
    for (let substep = 0; substep < state.substeps; substep += 1) {
      let lambda = 0;
      for (let iteration = 0; iteration < state.iterations; iteration += 1) {
        const pbd = projectDistance(state.pbd.a, state.pbd.b, restLength, 1);
        state.pbd.a = pbd.a;
        state.pbd.b = pbd.b;
        const xpbd = projectDistanceXPBD(
          state.xpbd.a, state.xpbd.b, restLength, state.compliance, state.h, lambda);
        state.xpbd.a = xpbd.a;
        state.xpbd.b = xpbd.b;
        lambda = xpbd.lambda;
      }
      state.xpbd.lambda = lambda;
    }

    const pbdMeasurement = measure(state.pbd.a, state.pbd.b);
    const xpbdMeasurement = measure(state.xpbd.a, state.xpbd.b);
    state.pbd.length = pbdMeasurement.length;
    state.pbd.error = pbdMeasurement.error;
    state.xpbd.length = xpbdMeasurement.length;
    state.xpbd.error = xpbdMeasurement.error;
    state.maxRelativeError = Math.max(state.pbd.error, state.xpbd.error);
    state.time += dt;
    state.frame += 1;
    state.finite = validGeometry(state) && Number.isFinite(state.xpbd.lambda) &&
      [state.h, state.alphaTilde, state.pbd.length, state.pbd.error,
        state.xpbd.length, state.xpbd.error, state.maxRelativeError, state.time].every(Number.isFinite);
    if (!state.finite) {
      pauseWithGuidance(state, '状态出现非有限值，已暂停。请重置，不要用裁剪隐藏数值错误。');
      return state;
    }
    if (state.frame >= duration) {
      state.finished = true;
      controller.setRunning(false);
      warning.hidden = false;
      warning.textContent = '已完成 60 次共享界面更新（1 s 模拟时间）并暂停。点击重置可复现。';
    }
    return state;
  }

  function panelPoint(particle, panelStart, panelWidth, width, height) {
    const scale = Math.min((panelWidth - 28) / 3.4, (height - 136) / 1.6);
    return {
      x: panelStart + panelWidth / 2 + (particle.x - 1.5) * scale,
      y: Math.min(height - 48, Math.max(82, height / 2 - particle.y * scale)),
    };
  }

  function drawPair(ctx, run, panelStart, panelWidth, width, height, color) {
    const a = panelPoint(run.a, panelStart, panelWidth, width, height);
    const b = panelPoint(run.b, panelStart, panelWidth, width, height);
    ctx.strokeStyle = '#547079';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(panelStart + panelWidth / 2 - restLength *
      Math.min((panelWidth - 28) / 3.4, (height - 136) / 1.6) / 2, height / 2);
    ctx.lineTo(panelStart + panelWidth / 2 + restLength *
      Math.min((panelWidth - 28) / 3.4, (height - 136) / 1.6) / 2, height / 2);
    ctx.stroke();
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.lineWidth = 1;
    for (const point of [a, b]) {
      ctx.beginPath();
      ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = '#10242c';
      ctx.stroke();
    }
  }

  function draw(ctx, state, width, height) {
    ctx.fillStyle = '#10242c';
    ctx.fillRect(0, 0, width, height);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.font = '11px system-ui';
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('PBD / XPBD 距离对照', 8, 14);
    ctx.fillText('alpha=' + state.compliance.toFixed(4) + ' h=' + state.h.toFixed(6), 8, 30);
    ctx.fillText('子步=' + state.substeps + ' 迭代=' + state.iterations, 8, 46);
    const panelWidth = (width - 28) / 2;
    const leftStart = 8;
    const rightStart = 20 + panelWidth;
    ctx.fillStyle = '#8fb7ff';
    ctx.fillText('PBD 全刚度', leftStart + 4, 68);
    ctx.fillStyle = '#ef9a67';
    ctx.fillText('XPBD 柔度', rightStart + 4, 68);
    ctx.strokeStyle = '#547079';
    ctx.beginPath();
    ctx.moveTo(width / 2, 58);
    ctx.lineTo(width / 2, height - 40);
    ctx.stroke();
    drawPair(ctx, state.pbd, leftStart, panelWidth, width, height, '#8fb7ff');
    drawPair(ctx, state.xpbd, rightStart, panelWidth, width, height, '#ef9a67');
    ctx.font = '11px system-ui';
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('L=' + state.pbd.length.toFixed(4), leftStart + 4, height - 28);
    ctx.fillText('E=' + state.pbd.error.toFixed(4), leftStart + 4, height - 12);
    ctx.fillText('L=' + state.xpbd.length.toFixed(4), rightStart + 4, height - 28);
    ctx.fillText('E=' + state.xpbd.error.toFixed(4), rightStart + 4, height - 12);
  }

  const controller = ui.mount(container, {
    createState,
    controls: [
      {key: 'compliance', label: '柔度 alpha（0..1）', min: 0, max: 1, step: 0.01, value: defaults.compliance},
      {key: 'substeps', label: '每次更新子步数', min: 1, max: 8, step: 1, value: defaults.substeps},
      {key: 'iterations', label: '每子步迭代数', min: 1, max: 40, step: 1, value: defaults.iterations},
    ],
    update,
    draw,
    metrics(state) {
      const rounded = value => Number(value.toFixed(8));
      return {
        '柔度 alpha': rounded(state.compliance),
        '每次更新子步数': state.substeps,
        '每子步迭代数': state.iterations,
        '实际子步长 h（s）': rounded(state.h),
        '有效柔度 alpha/h^2': rounded(state.alphaTilde),
        'PBD 最终长度（m）': rounded(state.pbd.length),
        'PBD 最大相对距离误差': rounded(state.pbd.error),
        'XPBD 当前 lambda': rounded(state.xpbd.lambda),
        'XPBD 最终长度（m）': rounded(state.xpbd.length),
        'XPBD 最大相对距离误差': rounded(state.xpbd.error),
        '最大相对距离误差': rounded(state.maxRelativeError),
        '模拟时间（s）': rounded(state.time),
        '有限值状态（1是/0否）': state.finite ? 1 : 0,
      };
    },
  });
  controller.setRunning(false);
  controller.canvas.setAttribute('aria-label', 'PBD 与 XPBD 距离柔度对照：独立相同初值，显示最终长度、误差和当前乘子');

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
    Object.assign(controller.getState(), createState({
      compliance: Number(ranges[0].value),
      substeps: Number(ranges[1].value),
      iterations: Number(ranges[2].value),
    }));
    clearWarning();
    controller.draw();
    controller.report();
  });
})();
