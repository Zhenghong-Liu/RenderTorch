(() => {
  'use strict';
  const container = document.getElementById('integration-race');
  if (!container) return;
  const {integrateProjectile} = window.RTPhysics;
  const ui = window.RTPhysicsUI;
  const defaults = Object.freeze({dt: 0.1, v0: 20, gravity: -9.81});
  const methods = Object.freeze(['forward', 'semi-implicit', 'exact']);
  const colors = Object.freeze({forward: '#ef9a67', 'semi-implicit': '#58c7bd', exact: '#ffd48a'});
  const horizontalVelocity = 5;

  function initialParticle(v0) {
    return {x: 0, y: 0, vx: horizontalVelocity, vy: v0};
  }

  function createTrajectory(v0, gravity, dt) {
    const duration = 2 * v0 / Math.abs(gravity);
    const paths = {forward: [], 'semi-implicit': [], exact: []};
    const current = {
      forward: initialParticle(v0),
      'semi-implicit': initialParticle(v0),
      exact: initialParticle(v0),
    };
    methods.forEach(method => paths[method].push({x: 0, y: 0}));
    let time = 0;
    let frame = 0;
    while (time < duration - 1e-12) {
      const h = Math.min(dt, duration - time);
      methods.forEach(method => {
        current[method] = integrateProjectile(current[method], {x: 0, y: gravity}, h, method);
        paths[method].push({x: current[method].x, y: current[method].y});
      });
      time += h;
      frame += 1;
    }
    const endpoints = {};
    const errors = {};
    methods.forEach(method => {
      const point = current[method];
      endpoints[method] = {x: point.x, y: point.y};
    });
    errors.forward = Math.hypot(
      endpoints.forward.x - endpoints.exact.x,
      endpoints.forward.y - endpoints.exact.y,
    );
    errors['semi-implicit'] = Math.hypot(
      endpoints['semi-implicit'].x - endpoints.exact.x,
      endpoints['semi-implicit'].y - endpoints.exact.y,
    );
    return {duration, plannedSteps: frame, paths, endpoints, errors};
  }

  function createState(values = defaults) {
    return Object.assign({
      dt: values.dt,
      v0: values.v0,
      gravity: values.gravity,
      frame: 0,
      time: 0,
      finished: false,
      current: {
        forward: initialParticle(values.v0),
        'semi-implicit': initialParticle(values.v0),
        exact: initialParticle(values.v0),
      },
      trails: {forward: [{x: 0, y: 0}], 'semi-implicit': [{x: 0, y: 0}], exact: [{x: 0, y: 0}]},
    }, createTrajectory(values.v0, values.gravity, values.dt));
  }

  function drawPath(ctx, points, toScreen, color, dashed, lineWidth) {
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.setLineDash(dashed ? [6, 4] : []);
    ctx.beginPath();
    points.forEach((point, index) => {
      const screen = toScreen(point);
      if (index === 0) ctx.moveTo(screen.x, screen.y);
      else ctx.lineTo(screen.x, screen.y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.lineWidth = 1;
  }

  function marker(ctx, point, color, hollow) {
    ctx.beginPath();
    ctx.arc(point.x, point.y, hollow ? 5 : 4, 0, Math.PI * 2);
    ctx.fillStyle = hollow ? '#10242c' : color;
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.lineWidth = 1;
  }

  function draw(ctx, state, width, height) {
    ctx.fillStyle = '#10242c';
    ctx.fillRect(0, 0, width, height);
    ctx.font = '12px system-ui';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    const legend = [
      {text: '■ 前向欧拉', color: colors.forward},
      {text: '■ 半隐式欧拉', color: colors['semi-implicit']},
      {text: '○ 精确解（参考）', color: colors.exact},
    ];
    legend.forEach((row, index) => {
      ctx.fillStyle = row.color;
      ctx.fillText(row.text, 10, 16 + index * 16);
    });

    const left = 38;
    const top = 64;
    const plotWidth = Math.max(1, width - left - 16);
    const plotHeight = Math.max(1, height - top - 30);
    const allPoints = methods.flatMap(method => state.paths[method]);
    const xs = allPoints.map(point => point.x);
    const ys = allPoints.map(point => point.y);
    let minX = Math.min(...xs);
    let maxX = Math.max(...xs);
    let minY = Math.min(...ys);
    let maxY = Math.max(...ys);
    const xPadding = Math.max(0.5, (maxX - minX) * 0.04);
    const yPadding = Math.max(0.5, (maxY - minY) * 0.08);
    minX -= xPadding;
    maxX += xPadding;
    minY -= yPadding;
    maxY += yPadding;
    const toScreen = point => ({
      x: left + (point.x - minX) / Math.max(1e-12, maxX - minX) * plotWidth,
      y: top + (maxY - point.y) / Math.max(1e-12, maxY - minY) * plotHeight,
    });

    ctx.strokeStyle = 'rgba(140,164,170,.2)';
    ctx.lineWidth = 1;
    for (let index = 0; index <= 4; index += 1) {
      const x = left + plotWidth * index / 4;
      const y = top + plotHeight * index / 4;
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, top + plotHeight);
      ctx.moveTo(left, y);
      ctx.lineTo(left + plotWidth, y);
      ctx.stroke();
    }
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('x / m →', left + plotWidth - 43, top + plotHeight + 18);
    ctx.fillText('y / m ↑', 2, top - 8);

    drawPath(ctx, state.paths.exact, toScreen, colors.exact, true, 3);
    drawPath(ctx, state.paths.forward, toScreen, colors.forward, false, 2);
    drawPath(ctx, state.paths['semi-implicit'], toScreen, colors['semi-implicit'], false, 2);
    const start = toScreen(state.paths.exact[0]);
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('共同起点', start.x + 7, start.y - 7);
    methods.forEach(method => {
      const point = state.current[method];
      marker(ctx, toScreen(point), colors[method], method === 'exact');
    });
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('实线：完整轨迹；圆点：当前状态', left, height - 5);
  }

  const controller = ui.mount(container, {
    createState,
    controls: [
      {key: 'dt', label: '积分步长 dt（s）', min: 0.01, max: 0.5, step: 0.01, value: defaults.dt},
      {key: 'v0', label: '初速度 v0（m/s）', min: 5, max: 30, step: 0.5, value: defaults.v0},
      {key: 'gravity', label: '重力 g（m/s²）', min: -20, max: -1.62, step: 0.01, value: defaults.gravity},
    ],
    update(state, _tick, values) {
      if (state.finished) {
        controller.setRunning(false);
        return state;
      }
      const h = Math.min(state.dt, state.duration - state.time);
      methods.forEach(method => {
        state.current[method] = integrateProjectile(state.current[method], {x: 0, y: state.gravity}, h, method);
        state.trails[method].push({x: state.current[method].x, y: state.current[method].y});
      });
      state.time += h;
      state.frame += 1;
      if (state.time >= state.duration - 1e-10) {
        state.time = state.duration;
        state.finished = true;
        controller.setRunning(false);
        warning.hidden = false;
        warning.textContent = '已到共同终点，实验暂停。比较终点位置与两条数值误差，然后点击重置复现。';
      }
      return state;
    },
    draw,
    metrics(state) {
      const rounded = value => Number(value.toFixed(6));
      return {
        '积分步长 dt（s）': rounded(state.dt),
        '当前时间（s）': rounded(state.time),
        '共同终点时间（s）': rounded(state.duration),
        '当前积分步序号': state.frame,
        '共同终点 x（m）': rounded(state.endpoints.exact.x),
        '精确终点 y（m）': rounded(state.endpoints.exact.y),
        '前向终点 y（m）': rounded(state.endpoints.forward.y),
        '半隐式终点 y（m）': rounded(state.endpoints['semi-implicit'].y),
        '前向欧拉误差（m）': rounded(state.errors.forward),
        '半隐式欧拉误差（m）': rounded(state.errors['semi-implicit']),
      };
    },
  });
  controller.setRunning(false);
  controller.canvas.setAttribute('aria-label', '积分竞赛：前向欧拉、半隐式欧拉和精确自由落体的三条完整轨迹');

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
    const state = controller.getState();
    Object.assign(state, createState({
      dt: Number(ranges[0].value),
      v0: Number(ranges[1].value),
      gravity: Number(ranges[2].value),
    }));
    warning.hidden = true;
    warning.textContent = '';
    controller.draw();
    controller.report();
  });
})();
