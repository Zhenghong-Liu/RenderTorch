(() => {
  'use strict';
  const container = document.getElementById('cloth-showcase');
  if (!container) return;
  const {springForce, projectDistance, projectDistanceXPBD, collideParticleSphere} = window.RTPhysics;
  const ui = window.RTPhysicsUI;
  const defaults = Object.freeze({algorithm: 2, iterations: 8, strength: 0.5, wind: 1.2});
  const algorithmNames = ['质量弹簧', 'PBD', 'XPBD'];
  const cols = 7, rows = 5, spacingX = 0.45, spacingY = 0.42;
  const outerDt = 1 / 60, substeps = 4, substepDt = outerDt / substeps;
  const duration = 360, totalDuration = 6, mass = 0.2;
  const circleRadius = 0.36, particleRadius = 0.035, epsilon = 1e-12;
  const bounds = Object.freeze({minX: -2, maxX: 2, minY: -0.1, maxY: 2.8});
  const sphereBounds = Object.freeze({minX: -2, maxX: 2, minY: 0, maxY: 2.8});

  function initialParticles() {
    const particles = [];
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        particles.push({x: (col - 3) * spacingX, y: 2.25 - row * spacingY,
          vx: 0, vy: 0, w: row === 0 ? 0 : 1 / mass});
      }
    }
    return particles;
  }

  function initialLinks(particles) {
    const links = [], seen = new Set();
    function add(a, b, kind) {
      const key = Math.min(a, b) + ':' + Math.max(a, b);
      if (seen.has(key)) return;
      seen.add(key);
      links.push({a, b, kind, restLength: Math.hypot(
        particles[b].x - particles[a].x, particles[b].y - particles[a].y)});
    }
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const index = row * cols + col;
        if (col + 1 < cols) add(index, index + 1, 'structural');
        if (row + 1 < rows) add(index, index + cols, 'structural');
        if (row + 1 < rows && col + 1 < cols) {
          add(index, index + cols + 1, 'shear');
          add(index + 1, index + cols, 'shear');
        }
      }
    }
    return links;
  }

  function initialTriangles() {
    const triangles = [];
    for (let row = 0; row < rows - 1; row += 1) {
      for (let col = 0; col < cols - 1; col += 1) {
        const a = row * cols + col, b = a + 1, c = a + cols, d = c + 1;
        triangles.push([a, c, b], [b, c, d]);
      }
    }
    return triangles;
  }

  function sphereReference(time) {
    return {x: 1.45 * Math.sin(2 * Math.PI * time / totalDuration),
      y: 0.1 + 0.08 * Math.sin(4 * Math.PI * time / totalDuration + 0.4),
      vx: 1.45 * (2 * Math.PI / 6) * Math.cos(2 * Math.PI * time / 6),
      vy: 0.08 * (4 * Math.PI / 6) * Math.cos(4 * Math.PI * time / 6 + 0.4)};
  }

  function cloneParticles(particles) {
    return particles.map(particle => Object.assign({}, particle));
  }

  function xpbdCompliance(strength) {
    return (1 - strength) * 0.001 + 0.00001;
  }

  function createState(values = defaults) {
    const particles = initialParticles();
    const links = initialLinks(particles);
    const sphere = sphereReference(0);
    return {
      algorithm: algorithmNames[values.algorithm], algorithmIndex: values.algorithm,
      iterations: values.iterations, strength: values.strength, wind: values.wind,
      structuralStiffness: 300 + 600 * values.strength,
      shearStiffness: 150 + 300 * values.strength,
      compliance: xpbdCompliance(values.strength), damping: 0.8, gravity: -9.81,
      particles, initialParticles: cloneParticles(particles), links,
      triangles: initialTriangles(), lambdas: links.map(() => 0), inContact: particles.map(() => false),
      lastDistance: particles.map(particle => Math.hypot(particle.x - sphere.x, particle.y - sphere.y)),
      sphere, circleRadius, particleRadius,
      restitution: 0.2, friction: 0.15,
      combinedRadius: circleRadius + particleRadius,
      contacts: 0, fixedCount: 7, freeCount: 28,
      maxRelativeError: 0, outerDt, substepDt, substeps, totalDuration, duration,
      time: 0, frame: 0, dragging: false, finite: true, invalid: false, finished: false,
    };
  }

  function pauseWithGuidance(state, message) {
    state.invalid = true;
    state.finished = true;
    state.finite = false;
    state.dragging = false;
    controller.setRunning(false);
    warning.hidden = false;
    warning.textContent = message;
  }

  function validParticle(particle) {
    return particle && ['x', 'y', 'vx', 'vy', 'w'].every(key => Number.isFinite(particle[key])) &&
      particle.w >= 0;
  }

  function validState(state) {
    return state.particles.length === 35 && state.particles.every(validParticle) &&
      state.particles.slice(0, 7).every(particle => particle.w === 0) &&
      state.particles.slice(7).every(particle => particle.w === 1 / mass) &&
      state.links.every(link => link.a >= 0 && link.b < 35 && link.a !== link.b &&
        Number.isFinite(link.restLength) && link.restLength > epsilon) &&
      state.triangles.every(triangle => triangle.length === 3 &&
        triangle.every(index => index >= 0 && index < 35)) &&
      [state.sphere.x, state.sphere.y, state.sphere.vx, state.sphere.vy,
        state.compliance, state.time].every(Number.isFinite) &&
      state.links.every((link, index) => Number.isFinite(state.lambdas[index]));
  }

  function measure(state) {
    let maximum = 0;
    for (const link of state.links) {
      const a = state.particles[link.a], b = state.particles[link.b];
      const length = Math.hypot(b.x - a.x, b.y - a.y);
      const relative = Math.abs(length - link.restLength) / link.restLength;
      if (!Number.isFinite(relative)) return false;
      maximum = Math.max(maximum, relative);
    }
    state.maxRelativeError = maximum;
    return true;
  }

  function applyForces(state, h, time) {
    const ax = state.wind;
    const ay = state.wind * 0.3 * Math.sin(2 * Math.PI * time / totalDuration);
    for (const particle of state.particles) {
      if (particle.w === 0) continue;
      particle.vy += (state.gravity + ay) * h;
      particle.vx += ax * h;
    }
    if (state.algorithm === '质量弹簧') {
      for (const link of state.links) {
        const a = state.particles[link.a], b = state.particles[link.b];
        const dx = b.x - a.x, dy = b.y - a.y, length = Math.hypot(dx, dy);
        if (length <= epsilon || a.w + b.w <= 0) continue;
        const relativeVelocity = ((b.vx - a.vx) * dx + (b.vy - a.vy) * dy) / length;
        const stiffness = link.kind === 'structural' ? state.structuralStiffness : state.shearStiffness;
        const force = springForce(length - link.restLength, relativeVelocity, stiffness, state.damping);
        const nx = dx / length, ny = dy / length;
        a.vx -= a.w * force * nx * h;
        a.vy -= a.w * force * ny * h;
        b.vx += b.w * force * nx * h;
        b.vy += b.w * force * ny * h;
      }
    }
  }

  function predict(state, h) {
    for (const particle of state.particles) {
      if (particle.w === 0) continue;
      particle.x += particle.vx * h;
      particle.y += particle.vy * h;
    }
  }

  function solveDistances(state) {
    for (const particle of state.particles) {
      if (particle.w === 0) { particle.vx = 0; particle.vy = 0; }
    }
    if (state.algorithm === '质量弹簧') return;
    const previous = state.previousPositions;
    if (state.algorithm === 'XPBD') state.lambdas = state.links.map(() => 0);
    for (let iteration = 0; iteration < state.iterations; iteration += 1) {
      for (let index = 0; index < state.links.length; index += 1) {
        const link = state.links[index];
        const a = state.particles[link.a], b = state.particles[link.b];
        if (state.algorithm === 'PBD') {
          const projected = projectDistance(a, b, link.restLength, 1);
          state.particles[link.a] = projected.a;
          state.particles[link.b] = projected.b;
        } else {
          const projected = projectDistanceXPBD(a, b, link.restLength, state.compliance,
            substepDt, state.lambdas[index]);
          state.particles[link.a] = projected.a;
          state.particles[link.b] = projected.b;
          state.lambdas[index] = projected.lambda;
        }
      }
    }
    state.particles.forEach((particle, index) => {
      if (particle.w === 0) return;
      particle.vx = (particle.x - previous[index].x) / substepDt;
      particle.vy = (particle.y - previous[index].y) / substepDt;
    });
  }

  function collide(state) {
    for (let index = 7; index < state.particles.length; index += 1) {
      const particle = state.particles[index];
      const distance = Math.hypot(particle.x - state.sphere.x, particle.y - state.sphere.y);
      if (!Number.isFinite(distance)) return false;
      if (distance <= epsilon && distance <= state.combinedRadius) return false;
      if (distance > state.combinedRadius + epsilon) {
        state.inContact[index] = false;
        state.lastDistance[index] = distance;
        continue;
      }
      if (distance <= epsilon) continue;
      const nx = (particle.x - state.sphere.x) / distance;
      const ny = (particle.y - state.sphere.y) / distance;
      const approaching = (particle.vx - state.sphere.vx) * nx +
        (particle.vy - state.sphere.vy) * ny < 0;
      const wasInContact = state.inContact[index];
      const previousDistance = state.lastDistance[index];
      const relativeParticle = Object.assign({}, particle,
        {vx: particle.vx - state.sphere.vx, vy: particle.vy - state.sphere.vy});
      const stationaryCenter = Object.assign({}, state.sphere, {vx: 0, vy: 0});
      const projected = collideParticleSphere(relativeParticle, stationaryCenter, state.circleRadius,
        state.particleRadius, state.restitution, state.friction);
      projected.vx += state.sphere.vx;
      projected.vy += state.sphere.vy;
      state.particles[index] = projected;
      if (!validParticle(projected)) return false;
      const afterDistance = Math.hypot(projected.x - state.sphere.x, projected.y - state.sphere.y);
      if (previousDistance > state.combinedRadius && distance <= state.combinedRadius + epsilon &&
          approaching && !wasInContact) state.contacts += 1;
      state.inContact[index] = afterDistance <= state.combinedRadius + epsilon;
      state.lastDistance[index] = afterDistance;
    }
    return true;
  }

  function update(state, dt, values) {
    if (state.finished || state.invalid) {
      controller.setRunning(false);
      return state;
    }
    if (dt !== outerDt || ![0, 1, 2].includes(state.algorithmIndex) ||
        !Number.isInteger(state.iterations) || state.iterations < 1 || state.iterations > 40 ||
        !Number.isFinite(state.strength) || state.strength < 0 || state.strength > 1 ||
        !Number.isFinite(state.wind) || state.wind < -3 || state.wind > 3 ||
        !(state.compliance > 0) || !validState(state)) {
      pauseWithGuidance(state, '控件、子步或网格状态无效，已暂停。请重置实验。');
      return state;
    }
    const outerTime = state.time;
    for (let substep = 0; substep < substeps; substep += 1) {
      const sampleTime = outerTime + substep * substepDt;
      state.sphere = sphereReference(sampleTime);
      applyForces(state, substepDt, sampleTime);
      state.previousPositions = cloneParticles(state.particles);
      predict(state, substepDt);
      solveDistances(state);
      if (!collide(state) || !validState(state) || !measure(state)) {
        pauseWithGuidance(state, '出现非有限值、球心与粒子重合或退化几何，已暂停。请重置实验。');
        return state;
      }
      state.time += substepDt;
    }
    state.frame += 1;
    state.finite = validState(state) && Number.isFinite(state.contacts) &&
      Number.isFinite(state.maxRelativeError) && Number.isFinite(state.time);
    if (!state.finite) {
      pauseWithGuidance(state, '状态出现非有限值，已暂停。请重置，不要用裁剪隐藏数值错误。');
      return state;
    }
    if (state.frame >= duration) {
      state.finished = true;
      controller.setRunning(false);
      warning.hidden = false;
      warning.textContent = '已完成 6 s 模拟时间并暂停。点击重置可复现。';
    }
    return state;
  }

  function view(width, height) {
    const left = 8, top = 54;
    const scale = Math.max(0.001, Math.min((width - 16) / (bounds.maxX - bounds.minX),
      (height - top - 24) / (bounds.maxY - bounds.minY)));
    return {left, top, scale};
  }

  function screenPoint(point, width, height) {
    const transform = view(width, height);
    return {
      x: Math.max(5, Math.min(width - 5, transform.left + (point.x - bounds.minX) * transform.scale)),
      y: Math.max(48, Math.min(height - 22, transform.top + (point.y - bounds.minY) * transform.scale)),
    };
  }

  function worldPoint(canvas, event) {
    const point = ui.canvasPoint(canvas, event), transform = view(canvas.clientWidth, canvas.clientHeight);
    return {x: bounds.minX + (point.x - transform.left) / transform.scale,
      y: bounds.minY + (point.y - transform.top) / transform.scale};
  }

  function draw(ctx, state, width, height) {
    ctx.fillStyle = '#10242c';
    ctx.fillRect(0, 0, width, height);
    ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.font = '11px system-ui';
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('7x5 cloth: 7 pinned', 8, 14);
    ctx.fillText('106 links; bend omitted', 8, 30);
    ctx.fillText('6 s; h=1/240 s', 8, 46);
    const points = state.particles.map(particle => screenPoint(particle, width, height));
    for (const triangle of state.triangles) {
      ctx.beginPath();
      ctx.moveTo(points[triangle[0]].x, points[triangle[0]].y);
      ctx.lineTo(points[triangle[1]].x, points[triangle[1]].y);
      ctx.lineTo(points[triangle[2]].x, points[triangle[2]].y);
      ctx.closePath();
      ctx.fillStyle = 'rgba(143,183,255,0.10)'; ctx.fill();
      ctx.strokeStyle = '#547079'; ctx.lineWidth = 1; ctx.stroke();
    }
    for (const link of state.links) {
      if (link.kind !== 'structural') continue;
      ctx.beginPath(); ctx.moveTo(points[link.a].x, points[link.a].y);
      ctx.lineTo(points[link.b].x, points[link.b].y);
      ctx.strokeStyle = '#8fb7ff'; ctx.lineWidth = 1; ctx.stroke();
    }
    const sphere = screenPoint(state.sphere, width, height);
    const radius = Math.min(18, state.circleRadius * view(width, height).scale);
    ctx.beginPath(); ctx.arc(sphere.x, sphere.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = state.dragging ? '#ffd48a' : 'rgba(239,154,103,0.32)'; ctx.fill();
    ctx.strokeStyle = '#ef9a67'; ctx.lineWidth = 2; ctx.stroke(); ctx.lineWidth = 1;
    points.forEach((point, index) => {
      ctx.beginPath(); ctx.arc(point.x, point.y, index < 7 ? 3.5 : 2.5, 0, Math.PI * 2);
      ctx.fillStyle = index < 7 ? '#ef9a67' : '#dce8df'; ctx.fill();
    });
    ctx.fillStyle = '#9db6b8';
    ctx.fillText(state.algorithm + ' E=' + state.maxRelativeError.toFixed(3), 8, height - 28);
    ctx.fillText('contacts=' + state.contacts + ' t=' + state.time.toFixed(2), 8, height - 12);
  }

  const controller = ui.mount(container, {
    createState,
    controls: [
      {key: 'algorithm', label: '算法 0弹簧/1PBD/2XPBD', min: 0, max: 2, step: 1, value: defaults.algorithm},
      {key: 'iterations', label: 'PBD/XPBD 每子步迭代', min: 1, max: 40, step: 1, value: defaults.iterations},
      {key: 'strength', label: '弹簧k/XPBD柔度强度', min: 0, max: 1, step: 0.1, value: defaults.strength},
      {key: 'wind', label: '共享风力加速度', min: -3, max: 3, step: 0.1, value: defaults.wind},
    ],
    update,
    draw,
    metrics(state) {
      const rounded = value => Number(value.toFixed(8));
      return {
        [state.algorithm + ' 最大约束误差']: rounded(state.maxRelativeError),
        [state.algorithm === '质量弹簧' ? '质量弹簧：误差仅作诊断' : 'PBD/XPBD：结构/剪切投影残差']: 1,
        '粒子数': state.particles.length,
        '固定点数': state.fixedCount,
        '自由点数': state.freeCount,
        '结构边数': state.links.filter(link => link.kind === 'structural').length,
        '剪切边数': state.links.filter(link => link.kind === 'shear').length,
        '新接近接触数': state.contacts,
        '每子步迭代数': state.algorithm === '质量弹簧' ? 0 : state.iterations,
        '结构劲度 k（N/m）': state.structuralStiffness,
        '剪切劲度 k（N/m）': state.shearStiffness,
        'XPBD 柔度 alpha': rounded(state.compliance),
        '实际子步长 h（s）': rounded(state.substepDt),
        '模拟时间（s）': rounded(state.time),
        '有限状态（1是/0否）': state.finite ? 1 : 0,
      };
    },
  });
  controller.setRunning(false);
  controller.canvas.setAttribute('aria-label', '7×5 三角布料：质量弹簧、PBD、XPBD 与移动球，明确省略 bend 链接');

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
    const values = {algorithm: Number(ranges[0].value), iterations: Number(ranges[1].value),
      strength: Number(ranges[2].value), wind: Number(ranges[3].value)};
    const valid = [0, 1, 2].includes(values.algorithm) && Number.isInteger(values.iterations) &&
      values.iterations >= 1 && values.iterations <= 40 && Number.isFinite(values.strength) &&
      values.strength >= 0 && values.strength <= 1 && Number.isFinite(values.wind) &&
      values.wind >= -3 && values.wind <= 3;
    if (!valid) {
      pauseWithGuidance(controller.getState(), '控件含非有限值或越界值，已暂停。请重置实验。');
      return;
    }
    Object.assign(controller.getState(), createState(values));
    clearWarning(); controller.draw(); controller.report();
  });

  function moveSphere(state, canvas, event) {
    const raw = worldPoint(canvas, event);
    if (!Number.isFinite(raw.x) || !Number.isFinite(raw.y)) {
      pauseWithGuidance(state, '拖拽坐标非有限，已暂停并保留最后有效位置。请重置实验。');
      return;
    }
    state.sphere = {x: Math.max(sphereBounds.minX, Math.min(sphereBounds.maxX, raw.x)),
      y: Math.max(sphereBounds.minY, Math.min(sphereBounds.maxY, raw.y)), vx: 0, vy: 0};
    state.lastDistance = state.particles.map(particle =>
      Math.hypot(particle.x - state.sphere.x, particle.y - state.sphere.y));
    clearWarning();
  }

  controller.canvas.addEventListener('pointerdown', event => {
    const state = controller.getState();
    if (state.finished || state.invalid) return;
    const point = ui.canvasPoint(controller.canvas, event);
    const sphere = screenPoint(state.sphere, controller.canvas.clientWidth, controller.canvas.clientHeight);
    if (Math.hypot(point.x - sphere.x, point.y - sphere.y) > 18) return;
    controller.setRunning(false); state.dragging = true; state.inContact.fill(false);
    controller.canvas.setPointerCapture(event.pointerId);
    moveSphere(state, controller.canvas, event); controller.draw(); controller.report();
  });
  controller.canvas.addEventListener('pointermove', event => {
    const state = controller.getState();
    if (!state.dragging || state.invalid) return;
    moveSphere(state, controller.canvas, event); controller.draw(); controller.report();
  });
  controller.canvas.addEventListener('pointerup', () => {
    controller.getState().dragging = false; controller.draw(); controller.report();
  });
  controller.canvas.addEventListener('pointercancel', () => {
    controller.getState().dragging = false; controller.draw(); controller.report();
  });
})();
