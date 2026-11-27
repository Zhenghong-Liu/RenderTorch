(() => {
  'use strict';
  const container = document.getElementById('soft-area');
  if (!container) return;
  const {projectAreaXPBD} = window.RTPhysics;
  const ui = window.RTPhysicsUI;
  const defaults = Object.freeze({mode: 0, compliance: 0.01, iterations: 8});
  const restArea = 3, restPerimeter = 8, h = 1 / 60, duration = 60;
  const epsilon = 1e-10;
  const bounds = Object.freeze({minX: -0.6, maxX: 3.6, minY: -0.6, maxY: 1.6});

  function initialParticles() {
    return [
      {x: 0, y: 0, vx: 0, vy: 0, w: 1},
      {x: 3, y: 0, vx: 0, vy: 0, w: 1},
      {x: 3, y: 1, vx: 0, vy: 0, w: 1},
      {x: 0, y: 1, vx: 0, vy: 0, w: 1},
    ];
  }

  function signedArea(particles) {
    let twiceArea = 0;
    for (let index = 0; index < 4; index += 1) {
      const next = (index + 1) % 4;
      twiceArea += particles[index].x * particles[next].y -
        particles[next].x * particles[index].y;
    }
    return twiceArea / 2;
  }

  function perimeter(particles) {
    let value = 0;
    for (let index = 0; index < 4; index += 1) {
      const next = (index + 1) % 4;
      value += Math.hypot(particles[next].x - particles[index].x,
        particles[next].y - particles[index].y);
    }
    return value;
  }

  function cross(a, b, c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
  }

  function onSegment(a, b, point) {
    return Math.min(a.x, b.x) - epsilon <= point.x && point.x <= Math.max(a.x, b.x) + epsilon &&
      Math.min(a.y, b.y) - epsilon <= point.y && point.y <= Math.max(a.y, b.y) + epsilon;
  }

  function segmentsIntersect(a, b, c, d) {
    const abC = cross(a, b, c), abD = cross(a, b, d), cdA = cross(c, d, a), cdB = cross(c, d, b);
    if (((abC > epsilon && abD < -epsilon) || (abC < -epsilon && abD > epsilon)) &&
        ((cdA > epsilon && cdB < -epsilon) || (cdA < -epsilon && cdB > epsilon))) return true;
    return (Math.abs(abC) <= epsilon && onSegment(a, b, c)) ||
      (Math.abs(abD) <= epsilon && onSegment(a, b, d)) ||
      (Math.abs(cdA) <= epsilon && onSegment(c, d, a)) ||
      (Math.abs(cdB) <= epsilon && onSegment(c, d, b));
  }

  function validParticle(particle) {
    return particle && ['x', 'y', 'vx', 'vy', 'w'].every(key => Number.isFinite(particle[key])) &&
      particle.w > 0;
  }

  function geometryProblem(particles) {
    if (!Array.isArray(particles) || particles.length !== 4 || !particles.every(validParticle)) return '非有限粒子';
    const area = signedArea(particles);
    if (!Number.isFinite(area)) return '非有限面积';
    if (area <= epsilon) return '面积退化或翻面';
    for (let index = 0; index < 4; index += 1) {
      const a = particles[index], b = particles[(index + 1) % 4], c = particles[(index + 2) % 4];
      const turn = cross(a, b, c);
      const edge = Math.hypot(b.x - a.x, b.y - a.y);
      if (!Number.isFinite(turn) || edge <= epsilon || turn <= epsilon) return '共线、凹入或翻面';
    }
    if (segmentsIntersect(particles[0], particles[1], particles[2], particles[3]) ||
        segmentsIntersect(particles[1], particles[2], particles[3], particles[0])) return '边交叉';
    return null;
  }

  function validGeometry(particles) {
    return geometryProblem(particles) === null;
  }

  function measure(state) {
    state.area = signedArea(state.particles);
    state.absoluteError = Math.abs(state.area - restArea);
    state.relativeError = state.absoluteError / restArea;
    state.perimeter = perimeter(state.particles);
    state.perimeterRatio = state.perimeter / restPerimeter;
  }

  function createState(values = defaults) {
    const particles = initialParticles();
    return {
      mode: values.mode === 0 ? '硬' : '软', modeIndex: values.mode,
      selectedCompliance: values.compliance,
      compliance: values.mode === 0 ? 0 : values.compliance,
      iterations: values.iterations, restArea, restPerimeter, h, orientation: 1,
      particles, initialParticles: initialParticles(), area: restArea, absoluteError: 0,
      relativeError: 0, perimeter: restPerimeter, perimeterRatio: 1,
      lambda: 0, maxAbsLambda: 0, dragging: null, time: 0, frame: 0,
      finite: true, invalid: false, finished: false,
    };
  }

  function pauseWithGuidance(state, message) {
    state.invalid = true; state.finished = true; state.finite = false; state.dragging = null;
    controller.setRunning(false);
    warning.hidden = false; warning.textContent = message;
  }

  function update(state, dt) {
    if (state.finished || state.invalid) {
      controller.setRunning(false);
      return state;
    }
    if (dt !== h) {
      pauseWithGuidance(state, '实际子步不是 1/60 s，已暂停。请重置实验。');
      return state;
    }
    if (![0, 1].includes(state.modeIndex) || !Number.isInteger(state.iterations) ||
        state.iterations < 1 || state.iterations > 40 || !Number.isFinite(state.selectedCompliance) ||
        state.selectedCompliance < 0.001 || state.selectedCompliance > 1 ||
        (state.modeIndex === 0 ? state.compliance !== 0 : state.compliance !== state.selectedCompliance)) {
      pauseWithGuidance(state, '模式、迭代或柔度控件无效，已暂停。请重置实验。');
      return state;
    }
    const problem = geometryProblem(state.particles);
    if (problem !== null) {
      pauseWithGuidance(state, `面积几何${problem}，已暂停。请重置实验。`);
      return state;
    }
    const previous = state.particles.map(particle => Object.assign({}, particle));
    state.particles.forEach((particle, index) => {
      const horizontal = index === 0 || index === 3 ? -0.8 : 0.8;
      particle.vx += horizontal * h;
      particle.vy -= 0.4 * h;
      particle.x += particle.vx * h;
      particle.y += particle.vy * h;
    });
    if (!validGeometry(state.particles)) {
      pauseWithGuidance(state, '外力预测产生退化或非有限面积几何，已暂停。请重置实验。');
      return state;
    }
    state.lambda = 0; state.maxAbsLambda = 0;
    for (let iteration = 0; iteration < state.iterations; iteration += 1) {
      const result = projectAreaXPBD(state.particles[0], state.particles[1],
        state.particles[2], state.particles[3], restArea, state.compliance, h, state.lambda);
      state.particles = result.particles;
      state.lambda = result.lambda;
      state.maxAbsLambda = Math.max(state.maxAbsLambda, Math.abs(state.lambda));
      if (!validGeometry(state.particles) || !Number.isFinite(state.lambda)) {
        pauseWithGuidance(state, '面积投影产生重合、共线、交叉或非有限状态，已暂停。请重置实验。');
        return state;
      }
    }
    state.particles.forEach((particle, index) => {
      particle.vx = (particle.x - previous[index].x) / h;
      particle.vy = (particle.y - previous[index].y) / h;
    });
    measure(state); state.time += dt; state.frame += 1;
    state.finite = validGeometry(state.particles) && [state.area, state.absoluteError,
      state.relativeError, state.perimeter, state.perimeterRatio, state.lambda,
      state.maxAbsLambda, state.time].every(Number.isFinite);
    if (!state.finite) {
      pauseWithGuidance(state, '面积读数出现非有限值，已暂停。请重置，不要编造梯度。');
      return state;
    }
    if (state.frame >= duration) {
      state.finished = true; controller.setRunning(false);
      warning.hidden = false; warning.textContent = '已完成 1 s 面积约束实验并暂停。点击重置可复现。';
    }
    return state;
  }

  function view(width, height) {
    const top = 52;
    const scale = Math.max(0.001, Math.min((width - 32) / (bounds.maxX - bounds.minX),
      (height - top - 40) / (bounds.maxY - bounds.minY)));
    const left = (width - (bounds.maxX - bounds.minX) * scale) / 2;
    return {top, scale, left};
  }

  function screenPoint(particle, width, height) {
    const transform = view(width, height);
    return {
      x: Math.max(5, Math.min(width - 5, transform.left + (particle.x - bounds.minX) * transform.scale)),
      y: Math.max(48, Math.min(height - 24, transform.top + (particle.y - bounds.minY) * transform.scale)),
    };
  }

  function worldPoint(canvas, event) {
    const point = ui.canvasPoint(canvas, event), transform = view(canvas.clientWidth, canvas.clientHeight);
    return {x: bounds.minX + (point.x - transform.left) / transform.scale,
      y: bounds.minY + (point.y - transform.top) / transform.scale};
  }

  function draw(ctx, state, width, height) {
    ctx.fillStyle = '#10242c'; ctx.fillRect(0, 0, width, height);
    ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.font = '11px system-ui';
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('2D area analogy', 8, 14);
    ctx.fillText(state.mode === '硬' ? 'hard a=0' : 'soft a=' + state.compliance.toFixed(3), 8, 30);
    ctx.fillText('h=1/60 iter=' + state.iterations, 8, 46);
    const points = state.particles.map(particle => screenPoint(particle, width, height));
    ctx.beginPath(); ctx.moveTo(points[0].x, points[0].y);
    for (let index = 1; index < 4; index += 1) ctx.lineTo(points[index].x, points[index].y);
    ctx.closePath(); ctx.fillStyle = 'rgba(143,183,255,0.18)'; ctx.fill();
    ctx.strokeStyle = '#8fb7ff'; ctx.lineWidth = 2; ctx.stroke(); ctx.lineWidth = 1;
    points.forEach((point, index) => {
      ctx.beginPath(); ctx.arc(point.x, point.y, state.dragging === index ? 7 : 5, 0, Math.PI * 2);
      ctx.fillStyle = state.dragging === index ? '#ffd48a' : '#f4f1e8'; ctx.fill();
      ctx.strokeStyle = '#10242c'; ctx.stroke();
    });
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('A=' + state.area.toFixed(3) + ' E=' + state.relativeError.toFixed(4), 8, height - 28);
    ctx.fillText('P/P0=' + state.perimeterRatio.toFixed(3) + ' l=' + state.lambda.toFixed(4), 8, height - 12);
  }

  const controller = ui.mount(container, {
    createState,
    controls: [
      {key: 'mode', label: '模式 0硬/1软', min: 0, max: 1, step: 1, value: defaults.mode},
      {key: 'compliance', label: '软模式柔度 alpha', min: 0.001, max: 1, step: 0.001, value: defaults.compliance},
      {key: 'iterations', label: '每次更新迭代数', min: 1, max: 40, step: 1, value: defaults.iterations},
    ],
    update,
    draw,
    metrics(state) {
      const rounded = value => Number(value.toFixed(8));
      return {
        '二维四粒子面积类比（非三维体积求解器）': 1,
        [state.mode + '面积约束模式']: 1,
        '绝对面积误差': rounded(state.absoluteError),
        '相对面积误差': rounded(state.relativeError),
        '当前面积': rounded(state.area),
        '静止面积': state.restArea,
        '周长': rounded(state.perimeter),
        '静止周长': state.restPerimeter,
        '周长比': rounded(state.perimeterRatio),
        '柔度': rounded(state.compliance),
        '所选有限柔度': rounded(state.selectedCompliance),
        '迭代': state.iterations,
        'lambda': rounded(state.lambda),
        '最大乘子绝对值': rounded(state.maxAbsLambda),
        '模拟时间（s）': rounded(state.time),
        '有限状态（1是/0否）': state.finite ? 1 : 0,
      };
    },
  });
  controller.setRunning(false);
  controller.canvas.setAttribute('aria-label', '二维四粒子面积类比：硬/软 XPBD、合法拖拽、面积误差与周长比，不是三维体积求解器');

  const toolbar = Array.from(container.children).find(element => element.className === 'demo-toolbar physics-controls');
  const warning = document.createElement('p');
  warning.className = 'demo-readout'; warning.setAttribute('role', 'status');
  warning.setAttribute('aria-live', 'polite'); warning.hidden = true; container.appendChild(warning);
  function clearWarning() { warning.hidden = true; warning.textContent = ''; }
  const resetController = controller.reset;
  controller.reset = () => { resetController(); clearWarning(); };
  toolbar.children[0].addEventListener('click', clearWarning);
  const ranges = Array.from(toolbar.children).filter(element => element.tagName.toLowerCase() === 'label')
    .map(element => Array.from(element.children).find(child => child.tagName.toLowerCase() === 'input'));
  toolbar.addEventListener('input', () => {
    controller.setRunning(false);
    const values = {mode: Number(ranges[0].value), compliance: Number(ranges[1].value),
      iterations: Number(ranges[2].value)};
    const valid = [0, 1].includes(values.mode) && Number.isFinite(values.compliance) &&
      values.compliance >= 0.001 && values.compliance <= 1 && Number.isInteger(values.iterations) &&
      values.iterations >= 1 && values.iterations <= 40;
    if (!valid) {
      pauseWithGuidance(controller.getState(), '控件含非有限值、零柔度或越界值，已暂停。请重置实验。');
      return;
    }
    Object.assign(controller.getState(), createState(values));
    clearWarning(); controller.draw(); controller.report();
  });

  function nearestParticle(state, canvas, event) {
    const point = ui.canvasPoint(canvas, event);
    let selected = null, nearest = 14;
    state.particles.forEach((particle, index) => {
      const screen = screenPoint(particle, canvas.clientWidth, canvas.clientHeight);
      const distance = Math.hypot(screen.x - point.x, screen.y - point.y);
      if (distance <= nearest) { nearest = distance; selected = index; }
    });
    return selected;
  }

  function rejectDrag(state, message) {
    state.dragging = null;
    controller.setRunning(false);
    warning.hidden = false; warning.textContent = message + ' 已保留最后有效几何；可继续拖拽或重置。';
  }

  function moveParticle(state, canvas, index, event) {
    const raw = worldPoint(canvas, event);
    if (!Number.isFinite(raw.x) || !Number.isFinite(raw.y)) {
      rejectDrag(state, '拖拽坐标非有限。');
      return;
    }
    const target = {x: Math.max(bounds.minX, Math.min(bounds.maxX, raw.x)),
      y: Math.max(bounds.minY, Math.min(bounds.maxY, raw.y)), vx: 0, vy: 0, w: 1};
    const candidate = state.particles.map((particle, particleIndex) =>
      particleIndex === index ? target : Object.assign({}, particle, {vx: 0, vy: 0}));
    if (!validGeometry(candidate)) {
      rejectDrag(state, '拖拽会使顶点重合、共线、交叉、翻面或面积退化。');
      return;
    }
    state.particles = candidate; state.lambda = 0; state.maxAbsLambda = 0;
    measure(state); clearWarning();
  }

  controller.canvas.addEventListener('pointerdown', event => {
    const state = controller.getState();
    if (state.finished || state.invalid) return;
    const index = nearestParticle(state, controller.canvas, event);
    if (index === null) return;
    controller.setRunning(false); state.dragging = index;
    state.lambda = 0; state.maxAbsLambda = 0;
    controller.canvas.setPointerCapture(event.pointerId);
    moveParticle(state, controller.canvas, index, event); controller.draw(); controller.report();
  });
  controller.canvas.addEventListener('pointermove', event => {
    const state = controller.getState();
    if (state.dragging === null || state.invalid) return;
    moveParticle(state, controller.canvas, state.dragging, event); controller.draw(); controller.report();
  });
  controller.canvas.addEventListener('pointerup', () => {
    controller.getState().dragging = null; controller.draw(); controller.report();
  });
  controller.canvas.addEventListener('pointercancel', () => {
    controller.getState().dragging = null; controller.draw(); controller.report();
  });
})();
