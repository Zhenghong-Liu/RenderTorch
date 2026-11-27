(() => {
  'use strict';
  const container = document.getElementById('collision-friction');
  if (!container) return;
  const {collideSpherePlane, collideParticleSphere} = window.RTPhysics;
  const ui = window.RTPhysicsUI;
  const defaults = Object.freeze({restitution: 0.5, friction: 0.2, vx: 4, vy: -5});
  const radius = 0.5;
  const circleRadius = 0.3;
  const circleCenter = Object.freeze({x: 0, y: 0});
  const dt = 1 / 60;
  const duration = 2;

  function emptyContact() {
    return {correction: 0, normalBefore: 0, normalAfter: 0, tangentBefore: 0, tangentAfter: 0};
  }

  function createRun(particle) {
    return {particle, contacts: 0, contact: emptyContact(), projected: false};
  }

  function createState(values = defaults) {
    return {
      restitution: values.restitution,
      friction: values.friction,
      vx: values.vx,
      vy: values.vy,
      dt,
      ground: createRun({x: 0, y: 0.2, vx: values.vx, vy: values.vy, w: 1}),
      circle: createRun({x: 0.36, y: 0.48, vx: values.vx, vy: values.vy, w: 1}),
      time: 0,
      frame: 0,
      finite: true,
      invalid: false,
      finished: false,
    };
  }

  function measure(run, normal, correction, before) {
    const after = run.particle;
    const normalBefore = before.vx * normal.x + before.vy * normal.y;
    const normalAfter = after.vx * normal.x + after.vy * normal.y;
    const tangentBefore = Math.hypot(before.vx - normalBefore * normal.x,
      before.vy - normalBefore * normal.y);
    const tangentAfter = Math.hypot(after.vx - normalAfter * normal.x,
      after.vy - normalAfter * normal.y);
    const projected = after.x !== before.x || after.y !== before.y;
    const responded = before.vx !== after.vx || before.vy !== after.vy;
    if (projected || responded) {
      run.contact = {correction, normalBefore, normalAfter, tangentBefore, tangentAfter};
      run.projected = projected;
    }
    return normalBefore < 0 && responded;
  }

  function update(state, transportDt, values) {
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

    const beforeGround = Object.assign({}, state.ground.particle);
    const groundNormal = {x: 0, y: 1};
    const groundOverlap = Math.max(0, radius - beforeGround.y);
    state.ground.particle = collideSpherePlane(beforeGround, radius, 0,
      state.restitution, state.friction);
    if (measure(state.ground, groundNormal, groundOverlap, beforeGround)) {
      state.ground.contacts += 1;
    }

    const beforeCircle = Object.assign({}, state.circle.particle);
    const offset = {x: beforeCircle.x - circleCenter.x, y: beforeCircle.y - circleCenter.y};
    const distance = Math.hypot(offset.x, offset.y);
    const circleOverlap = Math.max(0, circleRadius + radius - distance);
    const circleNormal = distance > 1e-12 ? {x: offset.x / distance, y: offset.y / distance} : {x: 1, y: 0};
    state.circle.particle = collideParticleSphere(beforeCircle, circleCenter,
      circleRadius, radius, state.restitution, state.friction);
    if (measure(state.circle, circleNormal, circleOverlap, beforeCircle)) {
      state.circle.contacts += 1;
    }

    for (const run of [state.ground, state.circle]) {
      run.particle.x += run.particle.vx * state.dt;
      run.particle.y += run.particle.vy * state.dt;
    }
    state.time += state.dt;
    state.frame += 1;
    state.finite = [state.time, state.restitution, state.friction].every(Number.isFinite) &&
      [state.ground, state.circle].every(run =>
        ['x', 'y', 'vx', 'vy', 'w'].every(key => Number.isFinite(run.particle[key])) &&
        Object.values(run.contact).every(Number.isFinite) && Number.isFinite(run.contacts));
    if (!state.finite) {
      state.invalid = true;
      state.finished = true;
      controller.setRunning(false);
      warning.hidden = false;
      warning.textContent = '状态出现非有限值，已暂停。请重置，不要用裁剪隐藏数值错误。';
      return state;
    }
    if (state.time >= duration - 1e-12) {
      state.finished = true;
      controller.setRunning(false);
      warning.hidden = false;
      warning.textContent = '已完成 2 s 无重力观察并暂停。初速度只用于重建初始状态；点击重置可复现。';
    }
    return state;
  }

  function drawArrow(ctx, start, vector, color) {
    const length = Math.hypot(vector.x, vector.y);
    if (!Number.isFinite(length) || length <= 1e-12) return;
    const end = {x: start.x + vector.x, y: start.y + vector.y};
    const ux = vector.x / length;
    const uy = vector.y / length;
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.moveTo(end.x, end.y);
    ctx.lineTo(end.x - 5 * ux + 3 * uy, end.y - 5 * uy - 3 * ux);
    ctx.moveTo(end.x, end.y);
    ctx.lineTo(end.x - 5 * ux - 3 * uy, end.y - 5 * uy + 3 * ux);
    ctx.stroke();
  }

  function draw(ctx, state, width, height) {
    ctx.fillStyle = '#10242c';
    ctx.fillRect(0, 0, width, height);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.font = '11px system-ui';
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('地面 / 静态圆；无重力', 8, 14);
    ctx.fillText('共享乘性切向衰减；非 Coulomb', 8, 30);
    ctx.fillText('真实调用前后测量；dt=1/60 s', 8, 46);

    const minX = -0.75 - Math.max(0, -state.vx) * duration;
    const maxX = 1.5 + Math.max(0, state.vx) * duration;
    const maxY = Math.max(2, 1.5 + Math.max(0, state.vy) * duration);
    const scaleY = 48 / (maxY + 0.5);
    const map = (point, baseline) => ({
      x: 70 + (point.x - minX) / (maxX - minX) * (width - 84),
      y: baseline - point.y * scaleY,
    });

    const rows = [
      {name: '地面', run: state.ground, baseline: 100, normal: {x: 0, y: 1}},
      {name: '静态圆', run: state.circle, baseline: 150, normal: null},
    ];
    rows.forEach(row => {
      const particle = map(row.run.particle, row.baseline);
      const origin = map({x: 0, y: 0}, row.baseline);
      ctx.fillStyle = '#9db6b8';
      ctx.fillText(row.name, 8, row.baseline + 4);
      ctx.strokeStyle = '#718f94';
      ctx.lineWidth = 1;
      if (row.name === '地面') {
        ctx.beginPath();
        ctx.moveTo(64, origin.y);
        ctx.lineTo(width - 10, origin.y);
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.arc(origin.x, origin.y, circleRadius * scaleY, 0, Math.PI * 2);
        ctx.stroke();
      }
      if (row.normal === null) {
        const dx = row.run.particle.x - circleCenter.x;
        const dy = row.run.particle.y - circleCenter.y;
        const distance = Math.hypot(dx, dy);
        row.normal = distance > 1e-12 ? {x: dx / distance, y: dy / distance} : {x: 1, y: 0};
      }
      drawArrow(ctx, particle, {x: row.normal.x * 0.7, y: -row.normal.y * 0.7}, '#ef9a67');
      ctx.beginPath();
      ctx.arc(particle.x, particle.y, Math.max(2, radius * scaleY), 0, Math.PI * 2);
      ctx.fillStyle = '#8fb7ff';
      ctx.fill();
      ctx.strokeStyle = '#f4f1e8';
      ctx.stroke();
    });
    ctx.lineWidth = 1;
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('箭头：法线；接触次数只数 approaching', 8, 170);
    ctx.fillText('摩擦=f 时切向速度乘以 max(0,1-f)', 8, 178);
  }

  const controller = ui.mount(container, {
    createState,
    controls: [
      {key: 'restitution', label: '恢复系数 e', min: 0, max: 1, step: 0.05, value: defaults.restitution},
      {key: 'friction', label: '摩擦 f（乘性衰减）', min: 0, max: 1, step: 0.05, value: defaults.friction},
      {key: 'vx', label: '初始水平速度（m/s）', min: -6, max: 6, step: 0.5, value: defaults.vx},
      {key: 'vy', label: '初始竖直速度（m/s）', min: -8, max: 8, step: 0.5, value: defaults.vy},
    ],
    update,
    draw,
    metrics(state) {
      const rounded = value => Number(value.toFixed(8));
      const metrics = {
        '恢复系数': state.restitution,
        '摩擦（乘性衰减）': state.friction,
        '物理时间步（s）': rounded(state.dt),
      };
      for (const [name, run] of [['地面', state.ground], ['静态圆', state.circle]]) {
        metrics[name + '穿透修正量（m）'] = rounded(run.contact.correction);
        metrics[name + '碰撞前法向速度（m/s）'] = rounded(run.contact.normalBefore);
        metrics[name + '碰撞后法向速度（m/s）'] = rounded(run.contact.normalAfter);
        metrics[name + '碰撞前切向速度（m/s）'] = rounded(run.contact.tangentBefore);
        metrics[name + '碰撞后切向速度（m/s）'] = rounded(run.contact.tangentAfter);
        metrics[name + '接触次数'] = run.contacts;
      }
      metrics['有限值状态（1是/0否）'] = state.finite ? 1 : 0;
      metrics['模拟时间（s）'] = rounded(state.time);
      return metrics;
    },
  });
  controller.setRunning(false);
  controller.canvas.setAttribute('aria-label', '地面与静态圆碰撞：显示真实共享函数的穿透修正、法线、法向和切向速度以及接近接触次数');

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
      restitution: Number(ranges[0].value),
      friction: Number(ranges[1].value),
      vx: Number(ranges[2].value),
      vy: Number(ranges[3].value),
    }));
    clearWarning();
    controller.draw();
    controller.report();
  });
})();
