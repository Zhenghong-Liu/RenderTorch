(() => {
  'use strict';
  const container = document.getElementById('solver-race');
  if (!container) return;
  const {springForce, stableSpringStep, projectDistance, projectDistanceXPBD} = window.RTPhysics;
  const ui = window.RTPhysicsUI;
  const defaults = Object.freeze({mass: 1, restLength: 2, iterations: 8});
  const stiffness = 120;
  const compliance = 0.01;
  const outerDt = 1 / 60;
  const duration = 60;
  const epsilon = 1e-12;

  function initialPair(mass, restLength) {
    return [
      {x: -0.75 * restLength, y: 0, vx: -0.6, vy: 0, w: 1 / mass},
      {x: 0.75 * restLength, y: 0, vx: 0.6, vy: 0, w: 1 / mass},
    ];
  }

  function clonePair(pair) {
    return pair.map(particle => Object.assign({}, particle));
  }

  function measure(run, restLength) {
    run.length = Math.hypot(run.b.x - run.a.x, run.b.y - run.a.y);
    run.error = Math.abs(run.length - restLength) / restLength;
    run.maxRelativeError = Math.max(run.maxRelativeError, run.error);
    run.kinetic = 0.5 / run.a.w * (run.a.vx ** 2 + run.a.vy ** 2 +
      run.b.vx ** 2 + run.b.vy ** 2);
    run.kineticDelta = run.kinetic - run.initialKinetic;
  }

  function validPair(pair, mass) {
    return pair.every(particle => ['x', 'y', 'vx', 'vy', 'w'].every(key =>
      Number.isFinite(particle[key])) && particle.w === 1 / mass) &&
      Math.hypot(pair[1].x - pair[0].x, pair[1].y - pair[0].y) > epsilon;
  }

  function createState(values = defaults) {
    const mass = values.mass;
    const restLength = values.restLength;
    const pair = initialPair(mass, restLength);
    const initialKinetic = 0.5 * mass * (0.6 ** 2 + 0.6 ** 2);
    const initialPotential = 0.5 * stiffness * (0.5 * restLength) ** 2;
    const run = () => ({
      a: clonePair(pair)[0], b: clonePair(pair)[1],
      length: 1.5 * restLength, error: 0.5, maxRelativeError: 0.5,
      kinetic: initialKinetic, kineticDelta: 0, initialKinetic,
      elasticPotential: initialPotential,
      totalMechanical: initialKinetic + initialPotential,
    });
    return {
      mass, restLength, iterations: values.iterations,
      stiffness, compliance, outerDt, reducedMass: mass / 2,
      stableStep: stableSpringStep(stiffness, mass / 2),
      massSpring: run(), pbd: run(), xpbd: Object.assign(run(), {lambda: 0}),
      initialKinetic, time: 0, frame: 0, finite: true, invalid: false, finished: false,
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

  function predict(pair) {
    return pair.map(particle => Object.assign({}, particle, {
      x: particle.x + particle.vx * outerDt,
      y: particle.y + particle.vy * outerDt,
    }));
  }

  function rebuildVelocity(run, oldPair) {
    run.a = Object.assign({}, run.a, {
      vx: (run.a.x - oldPair[0].x) / outerDt,
      vy: (run.a.y - oldPair[0].y) / outerDt,
    });
    run.b = Object.assign({}, run.b, {
      vx: (run.b.x - oldPair[1].x) / outerDt,
      vy: (run.b.y - oldPair[1].y) / outerDt,
    });
  }

  function update(state, dt, values) {
    if (state.finished || state.invalid) {
      controller.setRunning(false);
      return state;
    }
    if (![dt, state.mass, state.restLength, state.iterations, state.stiffness,
      state.compliance].every(Number.isFinite) || dt !== outerDt || state.mass < 0.25 ||
      state.mass > 2 || state.restLength < 1 || state.restLength > 3 ||
      !Number.isInteger(state.iterations) || state.iterations < 1 || state.iterations > 40 ||
      ![state.massSpring, state.pbd, state.xpbd].every(run =>
        validPair([run.a, run.b], state.mass))) {
      pauseWithGuidance(state, '控件、时间步或距离几何无效，已暂停。请重置实验。');
      return state;
    }
    state.reducedMass = state.mass / 2;
    state.stableStep = stableSpringStep(state.stiffness, state.reducedMass);
    if (!(outerDt < state.stableStep)) {
      pauseWithGuidance(state, '当前显式质量弹簧步长未满足稳定条件，已暂停。请减小质量或重置。');
      return state;
    }

    const spring = state.massSpring;
    const dx = spring.b.x - spring.a.x;
    const dy = spring.b.y - spring.a.y;
    const length = Math.hypot(dx, dy);
    const relativeVelocity = ((spring.b.vx - spring.a.vx) * dx +
      (spring.b.vy - spring.a.vy) * dy) / length;
    const force = springForce(length - state.restLength, relativeVelocity, state.stiffness, 0);
    const accelerationMagnitude = force / state.mass * outerDt;
    spring.a.vx -= accelerationMagnitude * dx / length;
    spring.a.vy -= accelerationMagnitude * dy / length;
    spring.b.vx += accelerationMagnitude * dx / length;
    spring.b.vy += accelerationMagnitude * dy / length;
    spring.a.x += spring.a.vx * outerDt;
    spring.a.y += spring.a.vy * outerDt;
    spring.b.x += spring.b.vx * outerDt;
    spring.b.y += spring.b.vy * outerDt;

    const pbd = state.pbd;
    const oldPbd = [Object.assign({}, pbd.a), Object.assign({}, pbd.b)];
    let pair = predict([pbd.a, pbd.b]);
    for (let iteration = 0; iteration < state.iterations; iteration += 1) {
      const projected = projectDistance(pair[0], pair[1], state.restLength, 1);
      pair = [projected.a, projected.b];
    }
    if (!validPair(pair, state.mass)) {
      pauseWithGuidance(state, 'PBD 投影产生非有限状态，已暂停。请重置实验。');
      return state;
    }
    pbd.a = pair[0];
    pbd.b = pair[1];
    rebuildVelocity(pbd, oldPbd);

    const xpbd = state.xpbd;
    const oldXpbd = [Object.assign({}, xpbd.a), Object.assign({}, xpbd.b)];
    pair = predict([xpbd.a, xpbd.b]);
    xpbd.lambda = 0;
    for (let iteration = 0; iteration < state.iterations; iteration += 1) {
      const projected = projectDistanceXPBD(
        pair[0], pair[1], state.restLength, state.compliance, outerDt, xpbd.lambda);
      pair = [projected.a, projected.b];
      xpbd.lambda = projected.lambda;
    }
    if (!validPair(pair, state.mass) || !Number.isFinite(xpbd.lambda)) {
      pauseWithGuidance(state, 'XPBD 投影产生非有限状态，已暂停。请重置实验。');
      return state;
    }
    xpbd.a = pair[0];
    xpbd.b = pair[1];
    rebuildVelocity(xpbd, oldXpbd);

    measure(spring, state.restLength);
    measure(pbd, state.restLength);
    measure(xpbd, state.restLength);
    spring.elasticPotential = 0.5 * state.stiffness *
      (spring.length - state.restLength) ** 2;
    spring.totalMechanical = spring.kinetic + spring.elasticPotential;
    state.time += dt;
    state.frame += 1;
    state.finite = state.stableStep > outerDt && Number.isFinite(xpbd.lambda) &&
      [spring, pbd, xpbd].every(run => validPair([run.a, run.b], state.mass) &&
        [run.length, run.error, run.maxRelativeError, run.kinetic, run.kineticDelta,
          run.initialKinetic].every(Number.isFinite)) &&
      [spring.elasticPotential, spring.totalMechanical, state.time].every(Number.isFinite);
    if (!state.finite) {
      pauseWithGuidance(state, '状态出现非有限值，已暂停。请重置，不要用裁剪隐藏数值错误。');
      return state;
    }
    if (state.frame >= duration) {
      state.finished = true;
      controller.setRunning(false);
      warning.hidden = false;
      warning.textContent = '已完成 60 次共同更新（1 s 模拟时间）并暂停。点击重置可复现。';
    }
    return state;
  }

  function panelPoint(particle, start, panelWidth, height) {
    const scale = Math.min((panelWidth - 18) / 3.6, 38 / 1.6);
    return {
      x: start + panelWidth / 2 + particle.x * scale,
      y: Math.min(height - 52, Math.max(82, height / 2 - particle.y * scale)),
    };
  }

  function drawRun(ctx, run, name, color, start, panelWidth, state, height) {
    const scale = Math.min((panelWidth - 18) / 3.6, 38 / 1.6);
    const centerY = height / 2;
    ctx.fillStyle = color;
    ctx.font = '11px system-ui';
    ctx.fillText(name, start + 3, 64);
    ctx.strokeStyle = '#547079';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(start + panelWidth / 2 - state.restLength * scale / 2, centerY);
    ctx.lineTo(start + panelWidth / 2 + state.restLength * scale / 2, centerY);
    ctx.stroke();
    const a = panelPoint(run.a, start, panelWidth, height);
    const b = panelPoint(run.b, start, panelWidth, height);
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.lineWidth = 1;
    for (const point of [a, b]) {
      ctx.beginPath();
      ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = '#10242c';
      ctx.stroke();
    }
    ctx.font = '10px system-ui';
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('L=' + run.length.toFixed(3), start + 3, height - 30);
    ctx.fillText('E=' + run.maxRelativeError.toFixed(2), start + 3, height - 16);
  }

  function draw(ctx, state, width, height) {
    ctx.fillStyle = '#10242c';
    ctx.fillRect(0, 0, width, height);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.font = '11px system-ui';
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('同题三路：固定 H=1/60 s', 8, 14);
    ctx.fillText('m=' + state.mass.toFixed(2) + ' L0=' + state.restLength.toFixed(2) +
      ' k=120', 8, 30);
    ctx.fillText('柔度=' + state.compliance.toFixed(2) +
      ' 迭代=' + state.iterations + '；弹簧力积分', 8, 46);
    const panelWidth = (width - 28) / 3;
    drawRun(ctx, state.massSpring, '质量弹簧', '#ef9a67', 8, panelWidth, state, height);
    drawRun(ctx, state.pbd, 'PBD', '#8fb7ff', 20 + panelWidth, panelWidth, state, height);
    drawRun(ctx, state.xpbd, 'XPBD', '#8fd1b0', 32 + 2 * panelWidth, panelWidth, state, height);
  }

  const controller = ui.mount(container, {
    createState,
    controls: [
      {key: 'mass', label: '公共质量（kg）', min: 0.25, max: 2, step: 0.25, value: defaults.mass},
      {key: 'restLength', label: '共同目标长度（m）', min: 1, max: 3, step: 0.5, value: defaults.restLength},
      {key: 'iterations', label: 'PBD/XPBD 每子步迭代', min: 1, max: 40, step: 1, value: defaults.iterations},
    ],
    update,
    draw,
    metrics(state) {
      const rounded = value => Number(value.toFixed(8));
      const runMetrics = (prefix, label) => ({
        [label + ' 最终长度（m）']: rounded(state[prefix].length),
        [label + ' 最大相对距离误差']: rounded(state[prefix].maxRelativeError),
        [label + ' 动能（J）']: rounded(state[prefix].kinetic),
        [label + ' 动能变化（J）']: rounded(state[prefix].kineticDelta),
      });
      return Object.assign({
        '公共质量（kg）': rounded(state.mass),
        '共同目标长度（m）': state.restLength,
        '质量弹簧劲度系数 k（N/m）': state.stiffness,
        'XPBD 柔度 alpha': state.compliance,
        '固定外层步长 H（s）': rounded(state.outerDt),
        '两质点约化质量（kg）': rounded(state.reducedMass),
        '显式积分稳定步长（s）': rounded(state.stableStep),
        'PBD/XPBD 每子步迭代数': state.iterations,
        '质量弹簧力积分次数（每更新）': 1,
      }, runMetrics('massSpring', '质量弹簧'), runMetrics('pbd', 'PBD'),
      runMetrics('xpbd', 'XPBD'), {
        '质量弹簧弹性势能（J）': rounded(state.massSpring.elasticPotential),
        '质量弹簧总机械能（J）': rounded(state.massSpring.totalMechanical),
        'XPBD 当前 lambda': rounded(state.xpbd.lambda),
        '模拟时间（s）': rounded(state.time),
        '有限值状态（1是/0否）': state.finite ? 1 : 0,
      });
    },
  });
  controller.setRunning(false);
  controller.canvas.setAttribute('aria-label', '质量弹簧、PBD 与 XPBD 三路同题竞赛：显示共同初值下的最终长度、最大相对距离误差和实际动能');

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
      mass: Number(ranges[0].value),
      restLength: Number(ranges[1].value),
      iterations: Number(ranges[2].value),
    };
    const valid = Number.isFinite(values.mass) && values.mass >= 0.25 && values.mass <= 2 &&
      Number.isFinite(values.restLength) && values.restLength >= 1 && values.restLength <= 3 &&
      Number.isInteger(values.iterations) && values.iterations >= 1 && values.iterations <= 40;
    if (!valid) {
      pauseWithGuidance(controller.getState(), '控件含非有限值或越界值，已暂停。请重置实验。');
      return;
    }
    Object.assign(controller.getState(), createState(values));
    clearWarning();
    controller.draw();
    controller.report();
  });
})();
