(() => {
  'use strict';
  const container = document.getElementById('fixed-step');
  if (!container) return;
  const {integrateProjectile} = window.RTPhysics;
  const ui = window.RTPhysicsUI;
  const defaults = Object.freeze({fps: 60, fixedRate: 120, stallMs: 120});
  const gravity = Object.freeze({x: 0, y: -9.81});
  const frameLimit = 12;
  const stallEvery = 5;

  function particle() {
    return {x: 0, y: 0, vx: 5, vy: 20};
  }

  function integrate(state, h) {
    return integrateProjectile(state, gravity, h, 'semi-implicit');
  }

  function createState(values = defaults) {
    return {
      fps: values.fps,
      fixedRate: values.fixedRate,
      fixedDt: 1 / values.fixedRate,
      stallMs: values.stallMs,
      frame: 0,
      elapsed: 0,
      directTime: 0,
      fixedTime: 0,
      substeps: 0,
      currentFrameSteps: 0,
      accumulator: 0,
      maxCatchupError: 0,
      direct: particle(),
      fixed: particle(),
      directTrail: [{x: 0, y: 0}],
      fixedTrail: [{x: 0, y: 0}],
      finished: false,
    };
  }

  function drawPanel(ctx, state, panel, title, color, trail, current, time, subtitle) {
    ctx.fillStyle = '#132b34';
    ctx.fillRect(panel.left, panel.top, panel.width, panel.height);
    ctx.strokeStyle = 'rgba(140,164,170,.28)';
    ctx.lineWidth = 1;
    ctx.strokeRect(panel.left + 0.5, panel.top + 0.5, panel.width - 1, panel.height - 1);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.font = '12px system-ui';
    ctx.fillStyle = color;
    ctx.fillText(title, panel.left, 17);
    ctx.fillStyle = '#b7c9cb';
    ctx.fillText('t=' + time.toFixed(3) + ' s', panel.left, 32);
    ctx.fillStyle = '#9db6b8';
    ctx.font = '11px system-ui';
    ctx.fillText(subtitle, panel.left, panel.top + panel.height + 17);

    const innerLeft = panel.left + 12;
    const innerTop = panel.top + 10;
    const innerWidth = Math.max(1, panel.width - 24);
    const innerHeight = Math.max(1, panel.height - 20);
    ctx.strokeStyle = 'rgba(140,164,170,.16)';
    for (let index = 0; index <= 3; index += 1) {
      const x = innerLeft + innerWidth * index / 3;
      const y = innerTop + innerHeight * index / 3;
      ctx.beginPath();
      ctx.moveTo(x, innerTop);
      ctx.lineTo(x, innerTop + innerHeight);
      ctx.moveTo(innerLeft, y);
      ctx.lineTo(innerLeft + innerWidth, y);
      ctx.stroke();
    }
    const allPoints = [...state.directTrail, ...state.fixedTrail];
    const maxX = Math.max(1, ...allPoints.map(point => point.x)) * 1.08;
    const minY = Math.min(0, ...allPoints.map(point => point.y));
    const maxY = 0.5;
    const rangeY = Math.max(1, maxY - minY) * 1.08;
    const toScreen = point => ({
      x: innerLeft + point.x / maxX * innerWidth,
      y: innerTop + (maxY - point.y) / rangeY * innerHeight,
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    trail.forEach((point, index) => {
      const screen = toScreen(point);
      if (index === 0) ctx.moveTo(screen.x, screen.y);
      else ctx.lineTo(screen.x, screen.y);
    });
    ctx.stroke();
    const marker = toScreen(current);
    ctx.beginPath();
    ctx.arc(marker.x, marker.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = '#10242c';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.lineWidth = 1;
    ctx.font = '11px system-ui';
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('x →', innerLeft + innerWidth - 22, innerTop + innerHeight - 5);
    ctx.fillText('y ↑', innerLeft + 3, innerTop + 12);
  }

  function draw(ctx, state, width, height) {
    ctx.fillStyle = '#10242c';
    ctx.fillRect(0, 0, width, height);
    const gap = 12;
    const panelWidth = Math.max(1, (width - 24 - gap) / 2);
    const panelHeight = Math.max(1, height - 64);
    const panel = {left: 12, top: 38, width: panelWidth, height: panelHeight};
    drawPanel(ctx, state, panel, '每帧直接 dt', '#ef9a67', state.directTrail,
      state.direct, state.directTime, '一帧积分一次');
    drawPanel(ctx, state, {left: panel.left + panelWidth + gap, top: panel.top,
      width: panelWidth, height: panelHeight}, '累加器固定步长', '#58c7bd',
      state.fixedTrail, state.fixed, state.fixedTime, state.substeps + ' 个固定子步');
    ctx.fillStyle = '#9db6b8';
    ctx.font = '11px system-ui';
    ctx.fillText('两图共用坐标尺度；圆点为当前状态', 12, height - 5);
  }

  const controller = ui.mount(container, {
    createState,
    controls: [
      {key: 'fps', label: '显示帧率（帧/s）', min: 10, max: 120, step: 1, value: defaults.fps},
      {key: 'fixedRate', label: '固定步长（1/n s）', min: 30, max: 240, step: 1, value: defaults.fixedRate},
      {key: 'stallMs', label: '卡顿帧（ms）', min: 0, max: 250, step: 10, value: defaults.stallMs},
    ],
    update(state, _tick, values) {
      if (state.finished) {
        controller.setRunning(false);
        return state;
      }
      state.fps = values.fps;
      state.fixedRate = values.fixedRate;
      state.fixedDt = 1 / values.fixedRate;
      state.stallMs = values.stallMs;
      state.frame += 1;
      const frameDt = 1 / state.fps + (state.frame % stallEvery === 0 ? state.stallMs / 1000 : 0);
      state.elapsed += frameDt;
      state.directTime = state.elapsed;
      state.direct = integrate(state.direct, frameDt);
      state.directTrail.push({x: state.direct.x, y: state.direct.y});

      state.accumulator += Math.max(0, Math.min(frameDt, 0.25));
      state.currentFrameSteps = 0;
      while (state.accumulator + 1e-9 >= state.fixedDt) {
        state.fixed = integrate(state.fixed, state.fixedDt);
        state.substeps += 1;
        state.currentFrameSteps += 1;
        state.accumulator -= state.fixedDt;
      }
      state.fixedTime = state.substeps * state.fixedDt;
      state.fixedTrail.push({x: state.fixed.x, y: state.fixed.y});
      state.maxCatchupError = Math.max(
        state.maxCatchupError,
        Math.abs(state.elapsed - state.fixedTime),
      );
      if (state.frame >= frameLimit) {
        state.finished = true;
        controller.setRunning(false);
        warning.hidden = false;
        warning.textContent = '已完成 12 个显示帧并暂停。直接 dt 与固定步长用同一组帧时间；点击重置后可改变参数复现。';
      }
      return state;
    },
    draw,
    metrics(state) {
      const rounded = value => Number(value.toFixed(6));
      return {
        '显示帧序号': state.frame,
        '显示帧上限': frameLimit,
        '显示帧率（帧/s）': state.fps,
        '固定步长（s）': rounded(state.fixedDt),
        '目标经过时间（s）': rounded(state.elapsed),
        '直接 dt 模拟时间（s）': rounded(state.directTime),
        '固定步模拟时间（s）': rounded(state.fixedTime),
        '固定子步数': state.substeps,
        '本帧固定子步数': state.currentFrameSteps,
        '累加器余量（s）': rounded(state.accumulator),
        '最大追帧误差（s）': rounded(state.maxCatchupError),
      };
    },
  });
  controller.setRunning(false);
  controller.canvas.setAttribute('aria-label', '固定步长实验：并排比较每帧直接积分与累加器固定步长，两图共用坐标尺度');

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
      fps: Number(ranges[0].value),
      fixedRate: Number(ranges[1].value),
      stallMs: Number(ranges[2].value),
    }));
    warning.hidden = true;
    warning.textContent = '';
    controller.draw();
    controller.report();
  });
})();
