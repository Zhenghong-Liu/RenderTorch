(() => {
  'use strict';
  const container = document.getElementById('pbd-rope');
  if (!container) return;
  const {projectDistance} = window.RTPhysics;
  const ui = window.RTPhysicsUI;
  const defaults = Object.freeze({iterations: 8, stiffness: 1, dt: 1 / 120});
  const particleCount = 8;
  const restLength = 0.35;
  const ropeWidth = (particleCount - 1) * restLength;
  const gravity = 9.81;
  const duration = 8;
  const bounds = Object.freeze({minX: 0, maxX: ropeWidth, minY: 0.95, maxY: 2});

  function createState(values = defaults) {
    const particles = Array.from({length: particleCount}, (_, index) => ({
      x: index * restLength,
      y: 1.5,
      vx: 0,
      vy: 0,
      w: index === 0 || index === particleCount - 1 ? 0 : 1,
    }));
    return {
      iterations: values.iterations,
      stiffness: values.stiffness,
      dt: values.dt,
      gravity,
      particles,
      constraints: Array.from({length: particleCount - 1}, (_, index) => ({
        a: index,
        b: index + 1,
        restLength,
      })),
      fixedCount: 2,
      errorBefore: 0,
      errorAfter: 0,
      maxError: 0,
      finite: true,
      invalid: false,
      dragging: null,
      time: 0,
      frame: 0,
      finished: false,
    };
  }

  function relativeError(state) {
    return Math.max(...state.constraints.map(constraint => {
      const a = state.particles[constraint.a];
      const b = state.particles[constraint.b];
      return Math.abs(Math.hypot(b.x - a.x, b.y - a.y) - constraint.restLength) /
        constraint.restLength;
    }));
  }

  function hasValidGeometry(state) {
    return state.constraints.every(constraint => {
      const a = state.particles[constraint.a];
      const b = state.particles[constraint.b];
      const distance = Math.hypot(b.x - a.x, b.y - a.y);
      return Number.isFinite(constraint.restLength) && constraint.restLength > 0 &&
        Number.isFinite(distance) && distance > 1e-12;
    });
  }

  function restoreEndpoints(state) {
    const left = state.particles[0];
    const right = state.particles[particleCount - 1];
    left.x = 0;
    left.y = 1.5;
    right.x = ropeWidth;
    right.y = 1.5;
    left.vx = 0;
    left.vy = 0;
    right.vx = 0;
    right.vy = 0;
  }

  function update(state, dt, values) {
    if (state.finished || state.invalid) {
      controller.setRunning(false);
      return state;
    }
    if (!Number.isFinite(state.dt) || state.dt <= 0) {
      state.invalid = true;
      state.finite = false;
      state.finished = true;
      controller.setRunning(false);
      warning.hidden = false;
      warning.textContent = '物理时间步无效，已暂停。请重置实验。';
      return state;
    }

    // A drag updates the current position and zeroes its velocity, so this
    // snapshot is the dragged reference, not the position before the drag.
    const before = state.particles.map(particle => ({x: particle.x, y: particle.y}));
    for (const particle of state.particles) {
      if (particle.w === 0) continue;
      particle.vy += gravity * state.dt;
      particle.x += particle.vx * state.dt;
      particle.y += particle.vy * state.dt;
    }
    if (!hasValidGeometry(state)) {
      state.invalid = true;
      state.finite = false;
      state.finished = true;
      controller.setRunning(false);
      warning.hidden = false;
      warning.textContent = '约束出现零长度、重合或非有限几何，已暂停。请重置并检查状态。';
      return state;
    }
    state.errorBefore = relativeError(state);

    for (let iteration = 0; iteration < state.iterations; iteration += 1) {
      for (const constraint of state.constraints) {
        const projected = projectDistance(
          state.particles[constraint.a], state.particles[constraint.b],
          constraint.restLength, state.stiffness);
        state.particles[constraint.a] = projected.a;
        state.particles[constraint.b] = projected.b;
      }
    }
    restoreEndpoints(state);
    state.errorAfter = relativeError(state);
    state.maxError = state.errorAfter;

    state.particles.forEach((particle, index) => {
      if (particle.w === 0) return;
      particle.vx = (particle.x - before[index].x) / state.dt;
      particle.vy = (particle.y - before[index].y) / state.dt;
    });
    state.fixedCount = state.particles.filter(particle => particle.w === 0).length;
    state.time += state.dt;
    state.frame += 1;
    state.finite = state.particles.every(particle =>
      [particle.x, particle.y, particle.vx, particle.vy, particle.w].every(Number.isFinite)) &&
      [state.errorBefore, state.errorAfter, state.maxError, state.time].every(Number.isFinite);
    if (!state.finite) {
      state.invalid = true;
      state.finished = true;
      controller.setRunning(false);
      warning.hidden = false;
      warning.textContent = '状态出现非有限值，已暂停。请重置，不要用裁剪隐藏数值错误。';
      return state;
    }
    if (state.particles.some(particle => particle.w !== 0 && particle.y > bounds.maxY)) {
      state.finished = true;
      controller.setRunning(false);
      warning.hidden = false;
      warning.textContent = '自由质点越过观察视图下边界 y=2 m，已因观察视图边界暂停；不是碰撞或地面，位置未裁剪。请重置实验。';
    }
    if (state.time >= duration - 1e-12) {
      state.time = duration;
      state.finished = true;
      controller.setRunning(false);
      warning.hidden = false;
      warning.textContent = '已完成 8 s 物理时间观察并暂停。共享界面运输步长仍为 1/60 s；点击重置可复现。';
    }
    return state;
  }

  function screenPoint(particle, width, height) {
    const scale = Math.min((width - 48) / ropeWidth,
      (height - 64) / (bounds.maxY - bounds.minY));
    return {
      x: (width - ropeWidth * scale) / 2 + particle.x * scale,
      y: 40 + (particle.y - bounds.minY) * scale,
    };
  }

  function draw(ctx, state, width, height) {
    ctx.fillStyle = '#10242c';
    ctx.fillRect(0, 0, width, height);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.font = '11px system-ui';
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('8 点 / 7 约束；两端固定', 8, 14);
    ctx.fillText('拖拽中点；播放观察投影', 8, 30);
    ctx.fillText('物理 dt=' + state.dt.toFixed(6) + ' s', 8, 46);

    const points = state.particles.map(particle => screenPoint(particle, width, height));
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#8fb7ff';
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let index = 1; index < points.length; index += 1) {
      ctx.lineTo(points[index].x, points[index].y);
    }
    ctx.stroke();
    ctx.lineWidth = 1;

    state.particles.forEach((particle, index) => {
      const point = points[index];
      if (particle.w === 0) {
        ctx.fillStyle = '#ffd48a';
        ctx.fillRect(point.x - 4, point.y - 4, 8, 8);
      }
      ctx.beginPath();
      ctx.arc(point.x, point.y, particle.w === 0 ? 2.5 : 4, 0, Math.PI * 2);
      ctx.fillStyle = particle.w === 0 ? '#ffd48a' : '#f4f1e8';
      ctx.fill();
      ctx.strokeStyle = '#10242c';
      ctx.stroke();
    });

    ctx.fillStyle = '#9db6b8';
    ctx.fillText('误差 ' + state.errorBefore.toFixed(4) + ' → ' + state.errorAfter.toFixed(4), 8, 166);
    ctx.fillText('固定端 2；自由点 6', 8, 178);
  }

  const controller = ui.mount(container, {
    createState,
    controls: [
      {key: 'iterations', label: '迭代数', min: 1, max: 40, step: 1, value: defaults.iterations},
      {key: 'stiffness', label: '投影刚度（0..1）', min: 0, max: 1, step: 0.05, value: defaults.stiffness},
      {key: 'dt', label: '物理时间步 dt（s）', min: 1 / 240, max: 1 / 60, step: 1 / 240, value: defaults.dt},
    ],
    update,
    draw,
    metrics(state) {
      const rounded = value => Number(value.toFixed(8));
      return {
        '迭代数': state.iterations,
        '投影刚度': state.stiffness,
        '物理时间步（s）': rounded(state.dt),
        '最大相对约束误差': rounded(state.maxError),
        '迭代前最大相对约束误差': rounded(state.errorBefore),
        '迭代后最大相对约束误差': rounded(state.errorAfter),
        '固定端数量': state.fixedCount,
        '自由质点数': state.particles.length - state.fixedCount,
        '距离约束数': state.constraints.length,
        '有限值状态（1是/0否）': state.finite ? 1 : 0,
        '模拟时间（s）': rounded(state.time),
      };
    },
  });
  controller.setRunning(false);
  controller.canvas.setAttribute('aria-label', '八点绳子距离投影：两端固定，六个中点可拖拽，显示迭代前后约束误差');

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
      iterations: Number(ranges[0].value),
      stiffness: Number(ranges[1].value),
      dt: Number(ranges[2].value),
    }));
    clearWarning();
    controller.draw();
    controller.report();
  });

  function dragParticle(state, canvas, event) {
    const point = ui.canvasPoint(canvas, event);
    let selected = null;
    let nearest = 12;
    state.particles.forEach((particle, index) => {
      if (particle.w === 0) return;
      const screen = screenPoint(particle, canvas.clientWidth, canvas.clientHeight);
      const distance = Math.hypot(screen.x - point.x, screen.y - point.y);
      if (distance <= nearest) {
        nearest = distance;
        selected = index;
      }
    });
    return selected;
  }

  function moveParticle(state, canvas, index, event) {
    const point = ui.canvasPoint(canvas, event);
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const scale = Math.min((width - 48) / ropeWidth,
      (height - 64) / (bounds.maxY - bounds.minY));
    const particle = state.particles[index];
    particle.x = Math.max(bounds.minX, Math.min(bounds.maxX,
      (point.x - (width - ropeWidth * scale) / 2) / scale));
    particle.y = Math.max(bounds.minY, Math.min(bounds.maxY,
      bounds.minY + (point.y - 40) / scale));
    particle.vx = 0;
    particle.vy = 0;
  }

  controller.canvas.addEventListener('pointerdown', event => {
    const state = controller.getState();
    if (state.finished || state.invalid) return;
    const index = dragParticle(state, controller.canvas, event);
    if (index === null) return;
    controller.setRunning(false);
    state.dragging = index;
    controller.canvas.setPointerCapture(event.pointerId);
    moveParticle(state, controller.canvas, index, event);
    clearWarning();
    controller.draw();
    controller.report();
  });
  controller.canvas.addEventListener('pointermove', event => {
    const state = controller.getState();
    if (state.dragging === null) return;
    moveParticle(state, controller.canvas, state.dragging, event);
    controller.draw();
    controller.report();
  });
  function finishDrag(event) {
    const state = controller.getState();
    if (state.dragging === null) return;
    const particle = state.particles[state.dragging];
    particle.vx = 0;
    particle.vy = 0;
    state.dragging = null;
    if (controller.canvas.releasePointerCapture) {
      controller.canvas.releasePointerCapture(event.pointerId);
    }
    controller.draw();
    controller.report();
  }
  controller.canvas.addEventListener('pointerup', finishDrag);
  controller.canvas.addEventListener('pointercancel', finishDrag);
})();
