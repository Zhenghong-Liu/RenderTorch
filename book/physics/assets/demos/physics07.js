(() => {
  'use strict';
  const container = document.getElementById('spring-oscillator');
  if (!container) return;
  const {springForce, energy1d, stableSpringStep} = window.RTPhysics;
  const ui = window.RTPhysicsUI;
  const defaults = Object.freeze({mass: 1, k: 20, damping: 0, initialStretch: 0.5});
  const gravity = 9.81;
  const duration = 12;
  const frameLimit = duration * 60;
  const substeps = 4;
  const substepDt = 1 / 60 / substeps;

  function createState(values = defaults) {
    const energy = energy1d(values.mass, 0, values.k, values.initialStretch);
    return {
      mass: values.mass,
      k: values.k,
      damping: values.damping,
      initialStretch: values.initialStretch,
      x: values.initialStretch,
      v: 0,
      force: springForce(values.initialStretch, 0, values.k, values.damping),
      kinetic: energy.kinetic,
      potential: energy.potential,
      totalEnergy: energy.kinetic + energy.potential,
      periodApprox: 2 * Math.PI * Math.sqrt(values.mass / values.k),
      staticExtension: values.mass * gravity / values.k,
      stepHint: stableSpringStep(values.k, values.mass),
      substeps,
      substepDt,
      time: 0,
      frame: 0,
      finished: false,
    };
  }

  function drawArrow(ctx, x, y, force) {
    if (Math.abs(force) <= 1e-12) {
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fill();
      return;
    }
    const direction = force > 0 ? 1 : -1;
    const length = Math.min(30, 6 + Math.abs(force) * 1.5);
    const endY = y + direction * length;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x, endY);
    ctx.lineTo(x - 4, endY - direction * 6);
    ctx.moveTo(x, endY);
    ctx.lineTo(x + 4, endY - direction * 6);
    ctx.stroke();
  }

  function draw(ctx, state, width, height) {
    ctx.fillStyle = '#10242c';
    ctx.fillRect(0, 0, width, height);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.font = '11px system-ui';
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('竖直弹簧—质点：x 从静平衡量起', 10, 16);
    ctx.fillText('4 子步 × 1/240 s；重力不重复', 10, 32);

    const centerX = width / 2;
    const anchorY = 42;
    const equilibriumY = 88;
    const ballY = equilibriumY + state.x * 24;
    ctx.strokeStyle = '#8fb7ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(centerX, anchorY);
    for (let index = 1; index < 8; index += 1) {
      const ratio = index / 8;
      ctx.lineTo(centerX + (index % 2 ? -6 : 6),
        anchorY + (ballY - anchorY) * ratio);
    }
    ctx.lineTo(centerX, ballY);
    ctx.stroke();

    ctx.strokeStyle = '#d9e3df';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(centerX - 16, anchorY);
    ctx.lineTo(centerX + 16, anchorY);
    ctx.stroke();
    for (let index = 0; index < 5; index += 1) {
      const hatchX = centerX - 14 + index * 7;
      ctx.beginPath();
      ctx.moveTo(hatchX, anchorY - 5);
      ctx.lineTo(hatchX + 4, anchorY - 9);
      ctx.stroke();
    }

    ctx.strokeStyle = 'rgba(140,164,170,.55)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(centerX - 42, equilibriumY);
    ctx.lineTo(centerX + 42, equilibriumY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.beginPath();
    ctx.arc(centerX, ballY, 7, 0, Math.PI * 2);
    ctx.fillStyle = '#f4f1e8';
    ctx.fill();
    ctx.strokeStyle = '#10242c';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.lineWidth = 1;

    ctx.strokeStyle = '#ffd48a';
    ctx.fillStyle = '#ffd48a';
    ctx.lineWidth = 2;
    drawArrow(ctx, centerX + 20, ballY, state.force);
    ctx.lineWidth = 1;

    ctx.fillStyle = '#9db6b8';
    ctx.fillText('固定锚点', 10, 50);
    ctx.fillText('静平衡 x=0', centerX + 12, equilibriumY - 3);
    ctx.fillText('质点 x=' + state.x.toFixed(3) + ' m', 10, 112);
    ctx.fillText('静止伸长 mg/k=' + state.staticExtension.toFixed(3) + ' m', 10, 128);
    const direction = state.force > 1e-12 ? '↓' : state.force < -1e-12 ? '↑' : '·';
    ctx.fillStyle = '#ffd48a';
    ctx.fillText('F=' + state.force.toFixed(1) + ' N ' + direction + '；箭头只示方向', 10, 148);
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('T≈' + state.periodApprox.toFixed(3) + ' s；小振幅无阻尼', 10, 168);
  }

  const controller = ui.mount(container, {
    createState,
    controls: [
      {key: 'mass', label: '质量 m（kg）', min: 0.5, max: 2, step: 0.1, value: defaults.mass},
      {key: 'k', label: '刚度 k（N/m）', min: 1, max: 80, step: 1, value: defaults.k},
      {key: 'damping', label: '阻尼 c（N·s/m）', min: 0, max: 8, step: 0.1, value: defaults.damping},
      {key: 'initialStretch', label: '初始伸长 x0（m，相对静平衡）', min: 0, max: 1.5, step: 0.1, value: defaults.initialStretch},
    ],
    update(state, dt, values) {
      if (state.finished) {
        controller.setRunning(false);
        return state;
      }
      for (let index = 0; index < state.substeps; index += 1) {
        state.force = springForce(state.x, state.v, state.k, state.damping);
        state.v += state.force / state.mass * state.substepDt;
        state.x += state.v * state.substepDt;
      }
      state.force = springForce(state.x, state.v, state.k, state.damping);
      const energy = energy1d(state.mass, state.v, state.k, state.x);
      state.kinetic = energy.kinetic;
      state.potential = energy.potential;
      state.totalEnergy = state.kinetic + state.potential;
      state.time += dt;
      state.frame += 1;
      if (![state.x, state.v, state.force, state.totalEnergy].every(Number.isFinite)) {
        state.finished = true;
        controller.setRunning(false);
        warning.hidden = false;
        warning.textContent = '状态出现非有限值，已暂停。请重置，不要用裁剪隐藏数值错误。';
        return state;
      }
      if (state.frame >= frameLimit) {
        state.time = duration;
        state.finished = true;
        controller.setRunning(false);
        warning.hidden = false;
        warning.textContent = '已完成 12 s 观察并暂停。周期值是小振幅无阻尼近似；点击重置可复现。';
      }
      return state;
    },
    draw,
    metrics(state) {
      const rounded = value => Number(value.toFixed(6));
      return {
        '质量（kg）': state.mass,
        '刚度（N/m）': state.k,
        '阻尼（N·s/m）': state.damping,
        '初始伸长 x0（m，相对静平衡）': state.initialStretch,
        '当前伸长 x（m，向下为正）': rounded(state.x),
        '速度 v（m/s，向下为正）': rounded(state.v),
        '当前合力（N，向下为正）': rounded(state.force),
        '总能量 K+1/2kx²（J）': rounded(state.totalEnergy),
        '静止伸长 mg/k（m）': rounded(state.staticExtension),
        '小振幅无阻尼周期近似（s）': rounded(state.periodApprox),
        '稳定步长上限（s，非保证）': rounded(state.stepHint),
        '内部子步数': state.substeps,
        '内部子步长（s）': state.substepDt,
        '时间（s）': rounded(state.time),
      };
    },
  });
  controller.setRunning(false);
  controller.canvas.setAttribute('aria-label', '竖直弹簧质点振子：x 从重力静平衡量起，绘制固定锚点、当前伸长与合力方向');

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
      mass: Number(ranges[0].value),
      k: Number(ranges[1].value),
      damping: Number(ranges[2].value),
      initialStretch: Number(ranges[3].value),
    }));
    clearWarning();
    controller.draw();
    controller.report();
  });
})();
