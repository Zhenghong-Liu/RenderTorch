(() => {
  'use strict';
  const container = document.getElementById('force-balance');
  if (!container) return;
  const ui = window.RTPhysicsUI;
  const defaults = Object.freeze({mass: 1, drag: 2, buoyancy: 0});
  const gravity = 9.81;
  const buoyantForce = 4;
  const duration = 12;
  const frameLimit = duration * 60;

  function createState(values = defaults) {
    const weight = values.mass * gravity;
    const upward = values.buoyancy ? buoyantForce : 0;
    return {
      mass: values.mass,
      drag: values.drag,
      buoyancy: values.buoyancy,
      gravityForce: weight,
      buoyantForce: upward,
      dragForce: 0,
      netForce: weight - upward,
      acceleration: (weight - upward) / values.mass,
      terminalSpeed: (weight - upward) / values.drag,
      velocity: 0,
      distance: 0,
      ball: {x: 0, y: 6, vx: 0, vy: 0, w: 1 / values.mass},
      time: 0,
      frame: 0,
      finished: false,
    };
  }

  function drawArrow(ctx, x, y, magnitude, direction, maximum, color, extent) {
    if (magnitude <= 1e-12) {
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      return;
    }
    const length = Math.max(3, magnitude / maximum * extent);
    const endY = y + direction * length;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x, endY);
    ctx.lineTo(x - 4, endY - direction * 6);
    ctx.moveTo(x, endY);
    ctx.lineTo(x + 4, endY - direction * 6);
    ctx.stroke();
    ctx.lineWidth = 1;
  }

  function draw(ctx, state, width, height) {
    ctx.fillStyle = '#10242c';
    ctx.fillRect(0, 0, width, height);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.font = '11px system-ui';
    const legend = [
      {text: '■ 重力 mg', color: '#ef9a67'},
      {text: '■ 空气阻力 cv', color: '#58c7bd'},
      {text: '■ 浮力 F_b', color: '#8fb7ff'},
      {text: '■ 合力 F_net', color: '#ffd48a'},
    ];
    legend.forEach((row, index) => {
      ctx.fillStyle = row.color;
      ctx.fillText(row.text, 10, 16 + index * 16);
    });

    const plotTop = 88;
    const plotBottom = Math.max(plotTop + 1, height - 64);
    const plotHeight = plotBottom - plotTop;
    const centerX = width / 2;
    const progress = Math.min(1, state.distance / 30);
    const ballY = plotTop + progress * plotHeight * 0.72;
    const maximum = Math.max(1, state.gravityForce, state.buoyantForce);
    const extent = Math.max(3, plotHeight * 0.22);

    ctx.fillStyle = '#9db6b8';
    ctx.fillText('向下为正；箭头长度表示力', 10, plotTop - 8);
    ctx.strokeStyle = 'rgba(140,164,170,.22)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(centerX, plotTop);
    ctx.lineTo(centerX, plotBottom);
    ctx.stroke();

    drawArrow(ctx, centerX - 42, ballY, state.gravityForce, 1, maximum, '#ef9a67', extent);
    drawArrow(ctx, centerX - 14, ballY, state.dragForce, -1, maximum, '#58c7bd', extent);
    drawArrow(ctx, centerX + 14, ballY, state.buoyantForce, -1, maximum, '#8fb7ff', extent);
    drawArrow(ctx, centerX + 42, ballY, Math.abs(state.netForce),
      state.netForce < 0 ? -1 : 1, maximum, '#ffd48a', extent);

    ctx.beginPath();
    ctx.arc(centerX, ballY, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#f4f1e8';
    ctx.fill();
    ctx.strokeStyle = '#10242c';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.lineWidth = 1;

    ctx.fillStyle = '#9db6b8';
    ctx.fillText('30 m 观察窗；12 s 后暂停', 10, height - 52);
    ctx.fillStyle = '#ffd48a';
    ctx.fillText(state.buoyancy ? '浮力开启：v_t=(mg-F_b)/c' : '浮力关闭：v_t=mg/c',
      10, height - 34);
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('F_b=4 N；向下为正', 10, height - 16);
  }

  const controller = ui.mount(container, {
    createState,
    controls: [
      {key: 'mass', label: '质量 m（kg）', min: 0.5, max: 2, step: 0.1, value: defaults.mass},
      {key: 'drag', label: '阻力 c（N·s/m）', min: 0.5, max: 5, step: 0.1, value: defaults.drag},
      {key: 'buoyancy', label: '浮力开关（0关/1开）', min: 0, max: 1, step: 1, value: defaults.buoyancy},
    ],
    update(state, dt, values) {
      if (state.finished) return state;
      const netBefore = state.gravityForce - state.buoyantForce - state.drag * state.velocity;
      state.velocity += netBefore / state.mass * dt;
      state.distance += state.velocity * dt;
      state.dragForce = state.drag * state.velocity;
      state.netForce = state.gravityForce - state.buoyantForce - state.dragForce;
      state.acceleration = state.netForce / state.mass;
      state.ball.y = 6 - state.distance;
      state.ball.vy = -state.velocity;
      state.time += dt;
      state.frame += 1;
      if (state.frame >= frameLimit) {
        state.time = duration;
        state.finished = true;
        controller.setRunning(false);
        warning.hidden = false;
        warning.textContent = '已完成 12 s 下落观察并暂停。合力仍由当前重力、浮力和阻力计算；点击重置可复现。';
      }
      return state;
    },
    draw,
    metrics(state) {
      const rounded = value => Number(value.toFixed(6));
      return {
        '质量（kg）': state.mass,
        '阻力系数（N·s/m）': state.drag,
        '浮力开关（0关/1开）': state.buoyancy,
        '重力（N）': rounded(state.gravityForce),
        '浮力（N）': rounded(state.buoyantForce),
        '空气阻力（N）': rounded(state.dragForce),
        '合力（N，向下为正）': rounded(state.netForce),
        '加速度（m/s²，向下为正）': rounded(state.acceleration),
        '当前速度（m/s，向下）': rounded(state.velocity),
        '终端速度（m/s，向下）': rounded(state.terminalSpeed),
        '下落距离（m）': rounded(state.distance),
        '时间（s）': rounded(state.time),
      };
    },
  });
  controller.setRunning(false);
  controller.canvas.setAttribute('aria-label', '受力平衡实验：绘制小球的重力、空气阻力、浮力与合力向量');

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
      drag: Number(ranges[1].value),
      buoyancy: Number(ranges[2].value),
    }));
    clearWarning();
    controller.draw();
    controller.report();
  });
})();
