(() => {
  'use strict';
  const container = document.getElementById('area-constraint');
  if (!container) return;
  const {projectAreaXPBD} = window.RTPhysics;
  const ui = window.RTPhysicsUI;
  const defaults = Object.freeze({compliance: 0.01, iterations: 8});
  const restArea = 2;
  const h = 1 / 60;
  const duration = 60;
  const epsilon = 1e-10;
  const orientation = 1;
  const bounds = Object.freeze({minX: -0.5, maxX: 3.5, minY: -0.5, maxY: 1.5});

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
    for (let index = 0; index < particles.length; index += 1) {
      const next = particles[(index + 1) % particles.length];
      twiceArea += particles[index].x * next.y - next.x * particles[index].y;
    }
    return twiceArea / 2;
  }

  function validParticle(particle) {
    return particle && ['x', 'y', 'vx', 'vy', 'w'].every(key => Number.isFinite(particle[key])) &&
      particle.w >= 0;
  }

  function validGeometry(particles) {
    if (!Array.isArray(particles) || particles.length !== 4 || !particles.every(validParticle)) return false;
    const area = signedArea(particles);
    if (!Number.isFinite(area) || Math.abs(area) <= epsilon || area * orientation <= 0) return false;
    for (let index = 0; index < particles.length; index += 1) {
      const first = particles[index];
      const second = particles[(index + 1) % particles.length];
      const third = particles[(index + 2) % particles.length];
      const e1 = {x: second.x - first.x, y: second.y - first.y};
      const e2 = {x: third.x - second.x, y: third.y - second.y};
      const length1 = Math.hypot(e1.x, e1.y);
      const length2 = Math.hypot(e2.x, e2.y);
      if (length1 <= epsilon || length2 <= epsilon ||
          Math.abs(e1.x * e2.y - e1.y * e2.x) <= epsilon * length1 * length2) return false;
    }
    for (let index = 0; index < particles.length; index += 1) {
      const next = particles[(index + 1) % particles.length];
      if (Math.hypot(next.x - particles[index].x, next.y - particles[index].y) <= epsilon) return false;
    }
    return true;
  }

  function measure(state) {
    state.area = signedArea(state.particles);
    state.relativeError = Math.abs(state.area - orientation * restArea) / restArea;
  }

  function createState(values = defaults) {
    const particles = initialParticles();
    const state = {
      compliance: values.compliance,
      iterations: values.iterations,
      restArea,
      h,
      alphaTilde: values.compliance / (h * h),
      orientation,
      particles,
      area: signedArea(particles),
      relativeError: Math.abs(signedArea(particles) - restArea) / restArea,
      lambda: 0,
      maxAbsLambda: 0,
      dragging: null,
      time: 0,
      frame: 0,
      finite: true,
      invalid: false,
      finished: false,
    };
    return state;
  }

  function pauseWithGuidance(state, message) {
    state.invalid = true;
    state.finished = true;
    state.finite = false;
    state.dragging = null;
    controller.setRunning(false);
    warning.hidden = false;
    warning.textContent = message;
  }

  function update(state, dt, values) {
    if (state.finished || state.invalid) {
      controller.setRunning(false);
      return state;
    }
    if (![dt, state.compliance, state.iterations].every(Number.isFinite) || dt <= 0 ||
        state.compliance < 0 || !Number.isInteger(state.iterations) ||
        state.iterations < 1 || state.iterations > 40 || !validGeometry(state.particles)) {
      pauseWithGuidance(state, '面积几何出现重合、共线、翻面、非有限值或无效控件，已暂停。请重置实验。');
      return state;
    }

    state.lambda = 0;
    state.maxAbsLambda = 0;
    for (let iteration = 0; iteration < state.iterations; iteration += 1) {
      const result = projectAreaXPBD(
        state.particles[0], state.particles[1], state.particles[2], state.particles[3],
        restArea, state.compliance, h, state.lambda);
      state.particles = result.particles;
      state.lambda = result.lambda;
      state.maxAbsLambda = Math.max(state.maxAbsLambda, Math.abs(state.lambda));
      if (!validGeometry(state.particles) || !Number.isFinite(state.lambda)) {
        pauseWithGuidance(state, '面积投影产生退化或非有限状态，已暂停。请重置，不要编造梯度。');
        return state;
      }
    }
    measure(state);
    state.time += dt;
    state.frame += 1;
    state.finite = validGeometry(state.particles) &&
      [state.area, state.relativeError, state.lambda, state.maxAbsLambda,
        state.alphaTilde, state.time].every(Number.isFinite);
    if (!state.finite) {
      pauseWithGuidance(state, '面积读数出现非有限值，已暂停。请重置实验。');
      return state;
    }
    if (state.frame >= duration) {
      state.finished = true;
      controller.setRunning(false);
      warning.hidden = false;
      warning.textContent = '已完成 60 次面积投影更新（1 s 模拟时间）并暂停。点击重置可复现。';
    }
    return state;
  }

  function view(width, height) {
    const top = 56;
    const availableWidth = Math.max(1, width - 32);
    const availableHeight = Math.max(1, height - 92);
    const scale = Math.min(availableWidth / (bounds.maxX - bounds.minX),
      availableHeight / (bounds.maxY - bounds.minY));
    const left = (width - (bounds.maxX - bounds.minX) * scale) / 2;
    return {top, scale, left};
  }

  function screenPoint(particle, width, height) {
    const transform = view(width, height);
    return {
      x: transform.left + (particle.x - bounds.minX) * transform.scale,
      y: transform.top + (particle.y - bounds.minY) * transform.scale,
    };
  }

  function worldPoint(canvas, event) {
    const point = ui.canvasPoint(canvas, event);
    const transform = view(canvas.clientWidth, canvas.clientHeight);
    return {
      x: bounds.minX + (point.x - transform.left) / transform.scale,
      y: bounds.minY + (point.y - transform.top) / transform.scale,
    };
  }

  function draw(ctx, state, width, height) {
    ctx.fillStyle = '#10242c';
    ctx.fillRect(0, 0, width, height);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.font = '11px system-ui';
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('二维面积 / 体积梯度', 8, 14);
    ctx.fillText('a=' + state.compliance.toFixed(4) + ' l=' + state.lambda.toFixed(4), 8, 30);
    ctx.fillText('h=0.016667 迭代=' + state.iterations, 8, 46);
    const points = state.particles.map(particle => screenPoint(particle, width, height));
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let index = 1; index < points.length; index += 1) ctx.lineTo(points[index].x, points[index].y);
    ctx.closePath();
    ctx.fillStyle = 'rgba(143,183,255,0.16)';
    ctx.fill();
    ctx.strokeStyle = '#8fb7ff';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.lineWidth = 1;
    points.forEach((point, index) => {
      ctx.beginPath();
      ctx.arc(point.x, point.y, state.dragging === index ? 7 : 5, 0, Math.PI * 2);
      ctx.fillStyle = state.dragging === index ? '#ffd48a' : '#f4f1e8';
      ctx.fill();
      ctx.strokeStyle = '#10242c';
      ctx.stroke();
    });
    ctx.font = '11px system-ui';
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('A=' + state.area.toFixed(4) + ' A0=' + restArea, 8, height - 28);
    ctx.fillText('E=' + state.relativeError.toFixed(4) + ' |l|max=' + state.maxAbsLambda.toFixed(4), 8, height - 12);
  }

  const controller = ui.mount(container, {
    createState,
    controls: [
      {key: 'compliance', label: '柔度 alpha（0..1）', min: 0, max: 1, step: 0.01, value: defaults.compliance},
      {key: 'iterations', label: '每次更新迭代数', min: 1, max: 40, step: 1, value: defaults.iterations},
    ],
    update,
    draw,
    metrics(state) {
      const rounded = value => Number(value.toFixed(8));
      return {
        '柔度 alpha': rounded(state.compliance),
        '每次更新迭代数': state.iterations,
        '实际子步长 h（s）': rounded(state.h),
        '有效柔度 alpha/h^2': rounded(state.alphaTilde),
        '当前面积': rounded(state.area),
        '静止面积': state.restArea,
        '相对面积误差': rounded(state.relativeError),
        '当前 lambda': rounded(state.lambda),
        '最大乘子绝对值': rounded(state.maxAbsLambda),
        '模拟时间（s）': rounded(state.time),
        '有限值状态（1是/0否）': state.finite ? 1 : 0,
      };
    },
  });
  controller.setRunning(false);
  controller.canvas.setAttribute('aria-label', '二维四边形面积约束：四顶点可拖拽，显示当前面积、相对误差与乘子');

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
      iterations: Number(ranges[1].value),
    }));
    clearWarning();
    controller.draw();
    controller.report();
  });

  function nearestParticle(state, canvas, event) {
    const point = ui.canvasPoint(canvas, event);
    let selected = null;
    let nearest = 14;
    state.particles.forEach((particle, index) => {
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
    const raw = worldPoint(canvas, event);
    if (!Number.isFinite(raw.x) || !Number.isFinite(raw.y)) {
      pauseWithGuidance(state, '拖拽坐标出现非有限值，已暂停并保留最后有效几何。请重置实验。');
      return;
    }
    const target = {
      x: Math.max(bounds.minX, Math.min(bounds.maxX, raw.x)),
      y: Math.max(bounds.minY, Math.min(bounds.maxY, raw.y)),
      vx: 0,
      vy: 0,
      w: state.particles[index].w,
    };
    const candidate = state.particles.map((particle, particleIndex) =>
      particleIndex === index ? target : Object.assign({}, particle));
    if (!validGeometry(candidate)) {
      pauseWithGuidance(state, '拖拽会使顶点重合、共线、翻面或面积退化，已暂停并保留最后有效几何。请重置实验。');
      return;
    }
    state.particles = candidate;
    state.lambda = 0;
    state.maxAbsLambda = 0;
    measure(state);
    clearWarning();
  }

  controller.canvas.addEventListener('pointerdown', event => {
    const state = controller.getState();
    if (state.finished || state.invalid) return;
    const index = nearestParticle(state, controller.canvas, event);
    if (index === null) return;
    controller.setRunning(false);
    state.dragging = index;
    state.lambda = 0;
    state.maxAbsLambda = 0;
    controller.canvas.setPointerCapture(event.pointerId);
    moveParticle(state, controller.canvas, index, event);
    controller.draw();
    controller.report();
  });
  controller.canvas.addEventListener('pointermove', event => {
    const state = controller.getState();
    if (state.dragging === null || state.invalid) return;
    moveParticle(state, controller.canvas, state.dragging, event);
    controller.draw();
    controller.report();
  });
  controller.canvas.addEventListener('pointerup', () => {
    controller.getState().dragging = null;
    controller.draw();
    controller.report();
  });
  controller.canvas.addEventListener('pointercancel', () => {
    controller.getState().dragging = null;
    controller.draw();
    controller.report();
  });
})();
