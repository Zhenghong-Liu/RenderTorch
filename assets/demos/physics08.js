(() => {
  'use strict';
  const container = document.getElementById('spring-cloth');
  if (!container) return;
  const {springForce} = window.RTPhysics;
  const ui = window.RTPhysicsUI;
  const defaults = Object.freeze({wind: 1.2, k: 60});
  const rows = 4;
  const cols = 6;
  const spacingX = 0.35;
  const spacingY = 0.3;
  const particleMass = 0.08;
  const gravity = 9.81;
  const damping = 0.8;
  const duration = 12;
  const frameLimit = duration * 60;
  const substeps = 4;
  const substepDt = 1 / 60 / substeps;

  function createState(values = defaults) {
    const particles = [];
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const fixed = row === 0;
        particles.push({
          row,
          col,
          x: col * spacingX,
          y: row * spacingY,
          vx: 0,
          vy: 0,
          fx: 0,
          fy: 0,
          mass: fixed ? 0 : particleMass,
          inverseMass: fixed ? 0 : 1 / particleMass,
          fixed,
        });
      }
    }
    const springs = [];
    function addSpring(a, b, restLength) {
      springs.push({a, b, restLength, kind: 'structural', extension: 0, force: 0});
    }
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const index = row * cols + col;
        if (col + 1 < cols) addSpring(index, index + 1, spacingX);
        if (row + 1 < rows) addSpring(index, index + cols, spacingY);
      }
    }
    return {
      wind: values.wind,
      k: values.k,
      damping,
      particleMass,
      particles,
      springs,
      maxStretch: 0,
      maxSpeed: 0,
      fixedCount: cols,
      finite: true,
      invalid: false,
      substeps,
      substepDt,
      time: 0,
      frame: 0,
      finished: false,
    };
  }

  function integrateSubstep(state) {
    for (const particle of state.particles) {
      particle.fx = particle.fixed ? 0 : state.wind;
      particle.fy = particle.fixed ? 0 : gravity * particle.mass;
    }
    for (const spring of state.springs) {
      const a = state.particles[spring.a];
      const b = state.particles[spring.b];
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const distance = Math.hypot(dx, dy);
      if (!Number.isFinite(distance) || distance <= 1e-12) return false;
      const nx = dx / distance;
      const ny = dy / distance;
      const relativeVelocity = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny;
      spring.extension = distance - spring.restLength;
      spring.force = springForce(spring.extension, relativeVelocity, state.k, state.damping);
      a.fx -= spring.force * nx;
      a.fy -= spring.force * ny;
      b.fx += spring.force * nx;
      b.fy += spring.force * ny;
    }
    for (const particle of state.particles) {
      if (particle.fixed) continue;
      particle.vx += particle.fx / particle.mass * state.substepDt;
      particle.vy += particle.fy / particle.mass * state.substepDt;
      particle.x += particle.vx * state.substepDt;
      particle.y += particle.vy * state.substepDt;
    }
    for (const particle of state.particles) {
      if (!particle.fixed) continue;
      particle.x = particle.col * spacingX;
      particle.y = 0;
      particle.vx = 0;
      particle.vy = 0;
    }
    return true;
  }

  function drawArrow(ctx, startX, startY, endX, endY, color) {
    const length = Math.hypot(endX - startX, endY - startY);
    if (length <= 1e-9) return;
    const nx = (endX - startX) / length;
    const ny = (endY - startY) / length;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.lineTo(endX - nx * 6 - ny * 4, endY - ny * 6 + nx * 4);
    ctx.moveTo(endX, endY);
    ctx.lineTo(endX - nx * 6 + ny * 4, endY - ny * 6 - nx * 4);
    ctx.stroke();
    ctx.lineWidth = 1;
  }

  function draw(ctx, state, width, height) {
    ctx.fillStyle = '#10242c';
    ctx.fillRect(0, 0, width, height);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.font = '11px system-ui';
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('6×4 结构弹簧；顶行 6 点固定', 8, 14);
    ctx.fillText('剪切/弯曲未启用；无自碰撞', 8, 30);
    ctx.fillText('蓝=伸长 橙=压缩；箭头示弹力', 8, 46);

    const xs = state.particles.map(particle => particle.x);
    const ys = state.particles.map(particle => particle.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const spanX = Math.max(0.1, maxX - minX);
    const spanY = Math.max(0.1, maxY - minY);
    const scale = Math.min((width - 40) / spanX, 74 / spanY);
    const offsetX = (width - spanX * scale) / 2 - minX * scale;
    const offsetY = 58 - minY * scale;
    const points = state.particles.map(particle => ({
      x: particle.x * scale + offsetX,
      y: particle.y * scale + offsetY,
    }));

    for (const spring of state.springs) {
      const a = points[spring.a];
      const b = points[spring.b];
      ctx.strokeStyle = spring.extension > 1e-6 ? '#8fb7ff' :
        spring.extension < -1e-6 ? '#ef9a67' : '#6f8f98';
      ctx.lineWidth = Math.abs(spring.extension) > 1e-6 ? 2 : 1;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
    ctx.lineWidth = 1;

    state.particles.forEach((particle, index) => {
      const point = points[index];
      if (particle.fixed) {
        ctx.fillStyle = '#ffd48a';
        ctx.fillRect(point.x - 4, point.y - 4, 8, 8);
        ctx.strokeStyle = '#10242c';
        ctx.strokeRect(point.x - 4, point.y - 4, 8, 8);
      }
      ctx.beginPath();
      ctx.arc(point.x, point.y, particle.fixed ? 2.5 : 4, 0, Math.PI * 2);
      ctx.fillStyle = particle.fixed ? '#ffd48a' : '#f4f1e8';
      ctx.fill();
      ctx.strokeStyle = '#10242c';
      ctx.stroke();
    });

    let selected = null;
    for (const spring of state.springs) {
      if (selected === null || Math.abs(spring.extension) > Math.abs(selected.extension)) {
        selected = spring;
      }
    }
    if (selected && Math.abs(selected.extension) > 1e-6) {
      const a = points[selected.a];
      const b = points[selected.b];
      const distance = Math.hypot(b.x - a.x, b.y - a.y);
      const nx = (b.x - a.x) / distance;
      const ny = (b.y - a.y) / distance;
      const direction = selected.extension > 0 ? -1 : 1;
      const startX = (a.x + b.x) / 2 - ny * 7;
      const startY = (a.y + b.y) / 2 + nx * 7;
      const length = Math.min(18, distance * 0.35);
      drawArrow(ctx, startX, startY, startX + direction * nx * length,
        startY + direction * ny * length, '#ffd48a');
    }

    ctx.fillStyle = '#9db6b8';
    ctx.fillText('静止长度：横0.35m 竖0.30m', 8, 148);
    ctx.fillText('固定点 ' + state.fixedCount + '/6；有限值：' + (state.finite ? '是' : '否'), 8, 164);
    ctx.fillText('视图自适应；不改写物理值', 8, 176);
  }

  const controller = ui.mount(container, {
    createState,
    controls: [
      {key: 'wind', label: '水平风力（N/自由质点）', min: 0, max: 2, step: 0.1, value: defaults.wind},
      {key: 'k', label: '结构弹簧刚度 k（N/m）', min: 20, max: 120, step: 5, value: defaults.k},
    ],
    update(state, dt, values) {
      if (state.finished) {
        controller.setRunning(false);
        return state;
      }
      for (let index = 0; index < state.substeps; index += 1) {
        if (!integrateSubstep(state)) {
          state.invalid = true;
          state.finite = false;
          state.finished = true;
          controller.setRunning(false);
          warning.hidden = false;
          warning.textContent = '弹簧长度为零或方向非有限，已暂停。请重置并检查状态。';
          return state;
        }
      }
      state.maxStretch = Math.max(...state.springs.map(spring => Math.abs(spring.extension)));
      state.maxSpeed = Math.max(...state.particles.map(particle => Math.hypot(particle.vx, particle.vy)));
      state.fixedCount = state.particles.filter(particle => particle.fixed).length;
      state.finite = state.particles.every(particle =>
        [particle.x, particle.y, particle.vx, particle.vy, particle.fx, particle.fy].every(Number.isFinite)) &&
        state.springs.every(spring => [spring.extension, spring.force].every(Number.isFinite)) &&
        Number.isFinite(state.maxStretch) && Number.isFinite(state.maxSpeed);
      state.time += dt;
      state.frame += 1;
      if (!state.finite) {
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
        warning.textContent = '已完成 12 s 结构弹簧观察并暂停。剪切、弯曲与自碰撞未启用；点击重置可复现。';
      }
      return state;
    },
    draw,
    metrics(state) {
      const rounded = value => Number(value.toFixed(6));
      return {
        '水平风力（N/自由质点）': state.wind,
        '结构弹簧刚度（N/m）': state.k,
        '弹簧阻尼（N·s/m）': state.damping,
        '最大伸长幅度 |L-L0|（m）': rounded(state.maxStretch),
        '最大速度（m/s）': rounded(state.maxSpeed),
        '固定点数': state.fixedCount,
        '自由质点数': state.particles.length - state.fixedCount,
        '结构弹簧数': state.springs.length,
        '有限值状态（1是/0否）': state.finite ? 1 : 0,
        '内部子步数': state.substeps,
        '内部子步长（s）': state.substepDt,
        '时间（s）': rounded(state.time),
      };
    },
  });
  controller.setRunning(false);
  controller.canvas.setAttribute('aria-label', '结构弹簧布片：顶行六点固定，显示当前伸长与弹力方向');

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
      wind: Number(ranges[0].value),
      k: Number(ranges[1].value),
    }));
    clearWarning();
    controller.draw();
    controller.report();
  });
})();
