(() => {
  'use strict';
  const container = document.getElementById('particle-lab');
  if (!container) return;
  const {vec2} = window.RTPhysics;
  const ui = window.RTPhysicsUI;
  const bounds = {minX: -1, maxX: 6, minY: -1, maxY: 5};
  const boundaryMessage = '下一帧会越出实验边界，已暂停。请重置实验，或拖拽小球回到范围内。';
  let dragging = false;

  function inBounds(point) {
    return point.x >= bounds.minX && point.x <= bounds.maxX &&
      point.y >= bounds.minY && point.y <= bounds.maxY;
  }

  function createState() {
    return {p: {x: 1, y: 2}, v: {x: 2, y: -1}, a: {x: 0, y: 0}, dt: 0.5, frame: 0};
  }

  function plotFor(width, height) {
    return {left: 45, top: 24, width: Math.max(1, width - 65), height: Math.max(1, height - 59)};
  }

  function toScreen(point, width, height) {
    const plot = plotFor(width, height);
    return {
      x: plot.left + ((point.x - bounds.minX) / (bounds.maxX - bounds.minX)) * plot.width,
      y: plot.top + ((bounds.maxY - point.y) / (bounds.maxY - bounds.minY)) * plot.height,
    };
  }

  function toWorld(point, width, height) {
    const plot = plotFor(width, height);
    const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
    return {
      x: clamp(bounds.minX + ((point.x - plot.left) / plot.width) * (bounds.maxX - bounds.minX), bounds.minX, bounds.maxX),
      y: clamp(bounds.maxY - ((point.y - plot.top) / plot.height) * (bounds.maxY - bounds.minY), bounds.minY, bounds.maxY),
    };
  }

  function arrow(ctx, start, end, color) {
    const angle = Math.atan2(end.y - start.y, end.x - start.x);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(end.x, end.y);
    ctx.lineTo(end.x - 8 * Math.cos(angle - 0.45), end.y - 8 * Math.sin(angle - 0.45));
    ctx.lineTo(end.x - 8 * Math.cos(angle + 0.45), end.y - 8 * Math.sin(angle + 0.45));
    ctx.closePath();
    ctx.fill();
    ctx.lineWidth = 1;
  }

  function marker(ctx, point, color, label, hollow = false) {
    ctx.beginPath();
    ctx.arc(point.x, point.y, hollow ? 8 : 10, 0, Math.PI * 2);
    ctx.fillStyle = hollow ? '#10242c' : color;
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.font = '13px system-ui';
    ctx.fillText(label, point.x + 12, point.y - 10);
  }

  function draw(ctx, state, width, height) {
    const plot = plotFor(width, height);
    ctx.fillStyle = '#10242c';
    ctx.fillRect(0, 0, width, height);
    ctx.font = '12px system-ui';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let x = Math.ceil(bounds.minX); x <= bounds.maxX; x += 1) {
      const screen = toScreen({x, y: 0}, width, height);
      ctx.strokeStyle = x === 0 ? '#8ca4aa' : 'rgba(140,164,170,.2)';
      ctx.beginPath();
      ctx.moveTo(screen.x, plot.top);
      ctx.lineTo(screen.x, plot.top + plot.height);
      ctx.stroke();
      ctx.fillStyle = '#9db6b8';
      ctx.fillText(String(x), screen.x, plot.top + plot.height + 15);
    }
    for (let y = Math.ceil(bounds.minY); y <= bounds.maxY; y += 1) {
      const screen = toScreen({x: 0, y}, width, height);
      ctx.strokeStyle = y === 0 ? '#8ca4aa' : 'rgba(140,164,170,.2)';
      ctx.beginPath();
      ctx.moveTo(plot.left, screen.y);
      ctx.lineTo(plot.left + plot.width, screen.y);
      ctx.stroke();
      ctx.fillStyle = '#9db6b8';
      ctx.fillText(String(y), plot.left - 16, screen.y);
    }
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('x / m', plot.left + plot.width - 34, plot.top + plot.height + 28);
    ctx.fillText('y / m', plot.left - 30, plot.top - 8);

    const current = toScreen(state.p, width, height);
    const nextPosition = vec2.add(state.p, vec2.scale(state.v, state.dt));
    const next = toScreen(nextPosition, width, height);
    const velocityTip = toScreen(vec2.add(state.p, vec2.scale(state.v, 0.65)), width, height);
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = '#789096';
    ctx.beginPath();
    ctx.moveTo(current.x, current.y);
    ctx.lineTo(next.x, next.y);
    ctx.stroke();
    ctx.setLineDash([]);
    if (vec2.length(state.v) > 1e-12) arrow(ctx, current, velocityTip, '#ffd48a');
    marker(ctx, current, '#58c7bd', '质点 p（可拖拽）');
    marker(ctx, next, '#ef9a67', '下一帧', true);
    ctx.fillStyle = '#ffd48a';
    if (vec2.length(state.v) > 1e-12) ctx.fillText('速度 v', velocityTip.x + 7, velocityTip.y - 7);
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('虚线圆：p + v·dt；箭头长度按 0.65 s 比例绘制', plot.left, 14);
  }

  const controller = ui.mount(container, {
    createState,
    controls: [
      {key: 'vx', label: '速度 x（m/s）', min: -5, max: 5, step: 0.1, value: 2},
      {key: 'vy', label: '速度 y（m/s）', min: -5, max: 5, step: 0.1, value: -1},
      {key: 'dt', label: '时间步（s）', min: 0.05, max: 1, step: 0.05, value: 0.5},
    ],
    update(state, _tick, values) {
      state.v = {x: values.vx, y: values.vy};
      state.dt = values.dt;
      const next = vec2.add(state.p, vec2.scale(state.v, state.dt));
      if (!inBounds(next)) {
        controller.setRunning(false);
        warning.hidden = false;
        warning.textContent = boundaryMessage;
        return;
      }
      clearWarning();
      state.p = next;
      state.frame += 1;
      return state;
    },
    draw,
    metrics(state) {
      const next = vec2.add(state.p, vec2.scale(state.v, state.dt));
      const rounded = value => Number(value.toFixed(6));
      return {
        '帧号': state.frame,
        '位置 x（m）': rounded(state.p.x),
        '位置 y（m）': rounded(state.p.y),
        '速度 x（m/s）': rounded(state.v.x),
        '速度 y（m/s）': rounded(state.v.y),
        '下一帧位置 x（m）': rounded(next.x),
        '下一帧位置 y（m）': rounded(next.y),
        '速度大小（m/s）': rounded(vec2.length(state.v)),
      };
    },
  });
  controller.setRunning(false);
  controller.canvas.setAttribute('aria-label', '质点实验：坐标网格、可拖拽小球、速度向量和下一帧预测');

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
  toolbar.children[0].addEventListener('click', clearWarning);
  const ranges = Array.from(toolbar.children).filter(element => element.tagName.toLowerCase() === 'label')
    .map(element => Array.from(element.children).find(child => child.tagName.toLowerCase() === 'input'));
  function syncControls() {
    clearWarning();
    const state = controller.getState();
    state.v = {x: Number(ranges[0].value), y: Number(ranges[1].value)};
    state.dt = Number(ranges[2].value);
    controller.draw();
    controller.report();
  }
  toolbar.addEventListener('input', syncControls);

  controller.canvas.addEventListener('pointerdown', event => {
    const point = ui.canvasPoint(controller.canvas, event);
    const width = controller.canvas.getBoundingClientRect().width;
    const height = controller.canvas.getBoundingClientRect().height;
    const current = toScreen(controller.getState().p, width, height);
    if (Math.hypot(point.x - current.x, point.y - current.y) > 18) return;
    dragging = true;
    clearWarning();
    controller.setRunning(false);
    if (controller.canvas.setPointerCapture) controller.canvas.setPointerCapture(event.pointerId);
  });
  controller.canvas.addEventListener('pointermove', event => {
    if (!dragging) return;
    const point = ui.canvasPoint(controller.canvas, event);
    const width = controller.canvas.getBoundingClientRect().width;
    const height = controller.canvas.getBoundingClientRect().height;
    controller.getState().p = toWorld(point, width, height);
    controller.draw();
    controller.report();
  });
  function stopDragging() { dragging = false; }
  controller.canvas.addEventListener('pointerup', stopDragging);
  controller.canvas.addEventListener('pointercancel', stopDragging);
})();
