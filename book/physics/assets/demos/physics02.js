(() => {
  'use strict';
  const container = document.getElementById('frame-states');
  if (!container) return;
  const {vec2} = window.RTPhysics;
  const ui = window.RTPhysicsUI;
  const p0 = {x: 1, y: 2};
  const velocity = {x: 2, y: -1};

  const table = document.createElement('table');
  table.setAttribute('aria-label', '围绕当前帧的五帧状态表');
  const caption = document.createElement('caption');
  caption.textContent = '五帧状态（随当前帧滑动；位置单位：m）';
  const head = document.createElement('thead');
  const headRow = document.createElement('tr');
  for (const label of ['帧号 n', '时刻 t（s）', 'x（m）', 'y（m）']) {
    const cell = document.createElement('th');
    cell.scope = 'col';
    cell.textContent = label;
    headRow.appendChild(cell);
  }
  head.appendChild(headRow);
  const body = document.createElement('tbody');
  table.append(caption, head, body);
  container.appendChild(table);

  function positionAt(time) {
    return vec2.add(p0, vec2.scale(velocity, time));
  }

  function frameRows(center, fps) {
    const start = Math.max(0, center - 2);
    return Array.from({length: 5}, (_, index) => {
      const frame = start + index;
      const time = frame / fps;
      return {frame, time};
    });
  }

  function createState() {
    return {
      time: 0, frame: 0, fps: 10, timeScale: 1,
      frames: frameRows(0, 10), actual: {...p0}, interpolated: {...p0}, error: 0,
    };
  }

  function drawGrid(ctx, width, height, points) {
    const left = 48;
    const top = 38;
    const plotWidth = Math.max(1, width - 70);
    const plotHeight = Math.max(1, height - 82);
    const xs = points.map(point => point.x);
    const ys = points.map(point => point.y);
    let minX = Math.min(...xs) - 0.5;
    let maxX = Math.max(...xs) + 0.5;
    let minY = Math.min(...ys) - 0.5;
    let maxY = Math.max(...ys) + 0.5;
    if (maxX - minX < 4) {
      const middle = (maxX + minX) / 2;
      minX = middle - 2;
      maxX = middle + 2;
    }
    if (maxY - minY < 4) {
      const middle = (maxY + minY) / 2;
      minY = middle - 2;
      maxY = middle + 2;
    }
    const toScreen = point => ({
      x: left + ((point.x - minX) / (maxX - minX)) * plotWidth,
      y: top + ((maxY - point.y) / (maxY - minY)) * plotHeight,
    });
    ctx.fillStyle = '#10242c';
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = 'rgba(140,164,170,.2)';
    ctx.lineWidth = 1;
    for (let index = 0; index <= 5; index += 1) {
      const x = left + plotWidth * index / 5;
      const y = top + plotHeight * index / 5;
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, top + plotHeight);
      ctx.moveTo(left, y);
      ctx.lineTo(left + plotWidth, y);
      ctx.stroke();
    }
    ctx.fillStyle = '#9db6b8';
    ctx.font = '12px system-ui';
    ctx.fillText('x / m →', left + plotWidth - 42, top + plotHeight + 24);
    ctx.fillText('y / m ↑', left - 36, top - 12);
    return {toScreen, left, top, plotWidth, plotHeight};
  }

  function draw(ctx, state, width, height) {
    state.frames = frameRows(state.frame, state.fps);
    body.replaceChildren();
    for (const row of state.frames) {
      const point = positionAt(row.time);
      const tr = document.createElement('tr');
      for (const value of [row.frame, row.time.toFixed(3), point.x.toFixed(3), point.y.toFixed(3)]) {
        const cell = document.createElement('td');
        cell.textContent = String(value);
        tr.appendChild(cell);
      }
      body.appendChild(tr);
    }

    const sampled = state.frames.map(row => positionAt(row.time));
    const view = drawGrid(ctx, width, height, [...sampled, state.actual, state.interpolated]);
    ctx.strokeStyle = '#789096';
    ctx.lineWidth = 2;
    ctx.beginPath();
    sampled.forEach((point, index) => {
      const screen = view.toScreen(point);
      if (index === 0) ctx.moveTo(screen.x, screen.y);
      else ctx.lineTo(screen.x, screen.y);
    });
    ctx.stroke();

    sampled.forEach((point, index) => {
      const screen = view.toScreen(point);
      ctx.fillStyle = '#58c7bd';
      ctx.fillRect(screen.x - 4, screen.y - 4, 8, 8);
      ctx.fillStyle = '#9db6b8';
      ctx.font = '11px system-ui';
      ctx.fillText('n=' + state.frames[index].frame, screen.x + 7, screen.y - 7);
    });
    const actual = view.toScreen(state.actual);
    const interpolated = view.toScreen(state.interpolated);
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = '#ef9a67';
    ctx.beginPath();
    ctx.moveTo(actual.x, actual.y);
    ctx.lineTo(interpolated.x, interpolated.y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(actual.x, actual.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#58c7bd';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(interpolated.x, interpolated.y, 7, 0, Math.PI * 2);
    ctx.fillStyle = '#10242c';
    ctx.fill();
    ctx.strokeStyle = '#ffd48a';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = '#edf5f3';
    ctx.font = '13px system-ui';
    ctx.fillText('■ 实际帧位置（方块/实心点）', view.left, 18);
    ctx.fillStyle = '#ffd48a';
    ctx.fillText('○ 线性插值位置；虚线为两者距离', view.left, 36);
    ctx.lineWidth = 1;
  }

  const controller = ui.mount(container, {
    createState,
    controls: [
      {key: 'fps', label: '采样帧率（帧/s）', min: 1, max: 60, step: 1, value: 10},
      {key: 'scale', label: '时间缩放', min: 0.25, max: 2, step: 0.25, value: 1},
    ],
    update(state, tick, values) {
      state.fps = values.fps;
      state.timeScale = values.scale;
      state.time += tick * state.timeScale;
      state.frame = Math.floor(state.time * state.fps + 1e-10);
      state.actual = positionAt(state.frame / state.fps);
      state.interpolated = positionAt(state.time);
      state.error = vec2.length(vec2.sub(state.interpolated, state.actual));
      return state;
    },
    draw,
    metrics(state) {
      const rounded = value => Number(value.toFixed(6));
      return {
        '帧号': state.frame,
        '模拟时间（s）': rounded(state.time),
        '当前状态 x（m）': rounded(state.actual.x),
        '当前状态 y（m）': rounded(state.actual.y),
        '插值位置 x（m）': rounded(state.interpolated.x),
        '插值位置 y（m）': rounded(state.interpolated.y),
        '插值误差（m）': rounded(state.error),
      };
    },
  });
  controller.canvas.setAttribute('aria-label', '帧状态实验：五帧采样、实际帧位置和线性插值位置');
  const toolbar = Array.from(container.children).find(element => element.className === 'demo-toolbar physics-controls');
  const ranges = Array.from(toolbar.children).filter(element => element.tagName.toLowerCase() === 'label')
    .map(element => Array.from(element.children).find(child => child.tagName.toLowerCase() === 'input'));
  function syncControls() {
    const state = controller.getState();
    state.fps = Number(ranges[0].value);
    state.timeScale = Number(ranges[1].value);
    state.frame = Math.floor(state.time * state.fps + 1e-10);
    state.actual = positionAt(state.frame / state.fps);
    state.interpolated = positionAt(state.time);
    state.error = vec2.length(vec2.sub(state.interpolated, state.actual));
    controller.draw();
    controller.report();
  }
  toolbar.addEventListener('input', syncControls);
})();
