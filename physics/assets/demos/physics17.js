(() => {
  'use strict';
  const container = document.getElementById('rigid-contact');
  if (!container) return;
  const {collideSpherePlane} = window.RTPhysics;
  const ui = window.RTPhysicsUI;
  const defaults = Object.freeze({radius: 0.35, restitution: 0.58, friction: 0.18, initialVx: 1.8});
  const mass = 1;
  const gravity = -9.81;
  const planeY = 0;
  const dropHeight = 2;
  const outerDt = 1 / 60;
  const substepDt = 1 / 120;
  const totalDuration = 6;
  const duration = Math.round(totalDuration / outerDt);

  function createState(values = defaults) {
    const radius = values.radius;
    return {
      radius, restitution: values.restitution, friction: values.friction,
      initialVx: values.initialVx, mass, gravity, planeY, dropHeight,
      outerDt, substepDt, totalDuration, duration,
      particle: {x: 0, y: radius + dropHeight, vx: values.initialVx, vy: 0, w: 1 / mass},
      contactY: planeY + radius,
      contacts: 0, inContact: false, maxCenterY: null,
      bounceHeight: 0, slidingDistance: 0, slidingReferenceX: null, lastContact: null, coreCalls: 0,
      time: 0, frame: 0, finite: true, invalid: false, finished: false,
    };
  }

  function valid(values) {
    return Number.isFinite(values.radius) && values.radius > 0 &&
      Number.isFinite(values.restitution) && values.restitution >= 0 && values.restitution <= 1 &&
      Number.isFinite(values.friction) && values.friction >= 0 && values.friction <= 1 &&
      Number.isFinite(values.initialVx);
  }

  function validParticle(particle) {
    return particle && ['x', 'y', 'vx', 'vy', 'w'].every(key => Number.isFinite(particle[key])) &&
      particle.w === 1 / mass;
  }

  function pauseWithGuidance(state, message) {
    state.invalid = true;
    state.finished = true;
    state.finite = false;
    controller.setRunning(false);
    warning.hidden = false;
    warning.textContent = message;
  }

  function finish(state) {
    state.finished = true;
    controller.setRunning(false);
    warning.hidden = false;
    warning.textContent = '已完成 6 s 物理时间并暂停。点击重置可复现同一初始状态。';
  }

  function update(state, dt, values) {
    if (state.finished || state.invalid) {
      controller.setRunning(false);
      return state;
    }
    if (dt !== outerDt || !valid(values) || !valid({
      radius: state.radius, restitution: state.restitution,
      friction: state.friction, initialVx: state.initialVx,
    }) || !validParticle(state.particle) || !(state.radius > 0)) {
      pauseWithGuidance(state, '控件、时间步或球状态无效，已暂停。请重置实验。');
      return state;
    }
    state.contactY = planeY + state.radius;
    for (let substep = 0; substep < 2; substep += 1) {
      state.particle.vy += gravity * substepDt;
      state.particle.x += state.particle.vx * substepDt;
      state.particle.y += state.particle.vy * substepDt;
      const before = Object.assign({}, state.particle);
      const approaching = before.vy < 0 && before.y <= state.contactY;
      const response = collideSpherePlane(before, state.radius, planeY,
        state.restitution, state.friction);
      state.coreCalls += 1;
      state.particle = response;
      if (!validParticle(response)) {
        pauseWithGuidance(state, '共享接触核心产生非有限状态，已暂停。请重置，不要裁剪数值错误。');
        return state;
      }
      if (approaching && !state.inContact) {
        state.contacts += 1;
        state.lastContact = {
          before, after: Object.assign({}, response),
          radius: state.radius, planeY,
          restitution: state.restitution, friction: state.friction,
        };
      }
      if (approaching && state.maxCenterY === null) {
        state.maxCenterY = response.y;
        state.slidingReferenceX = response.x;
      }
      // An episode ends only after the ball actually leaves the contact height.
      // A positive substep velocity that is immediately projected back to the
      // floor is numerical micro-bounce, not a new physical separation.
      state.inContact = response.y <= state.contactY;
      if (state.maxCenterY !== null) {
        state.maxCenterY = Math.max(state.maxCenterY, response.y);
        state.bounceHeight = state.maxCenterY - state.radius;
        // The first response only establishes the baseline; later responses add travel.
        state.slidingDistance += Math.abs(response.x - state.slidingReferenceX);
        state.slidingReferenceX = response.x;
      }
    }
    state.time += dt;
    state.frame += 1;
    state.finite = validParticle(state.particle) &&
      [state.time, state.bounceHeight, state.slidingDistance, state.contactY,
        state.coreCalls].every(Number.isFinite);
    if (!state.finite) {
      pauseWithGuidance(state, '状态出现非有限值，已暂停。请重置，不要用裁剪隐藏数值错误。');
      return state;
    }
    if (state.frame >= state.duration) finish(state);
    return state;
  }

  function draw(ctx, state, width, height) {
    ctx.fillStyle = '#10242c';
    ctx.fillRect(0, 0, width, height);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.font = '11px system-ui';
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('平移球接触：x右 y上', 8, 12);
    ctx.fillText('地面 y=0；真实共享核心', 8, 27);
    ctx.fillText('H=1/60；h=1/120 s', 8, 42);
    ctx.fillText('R=' + state.radius.toFixed(2) + ' e=' + state.restitution.toFixed(2) +
      ' f=' + state.friction.toFixed(2), 8, 57);
    ctx.fillText('初速 vx=' + state.initialVx.toFixed(1) + '；时长 6 s', 8, 72);
    const floorY = 150;
    const scaleY = (floorY - 78) / 4.4;
    const direction = state.initialVx < 0 ? -1 : state.initialVx > 0 ? 1 : 0;
    const extent = Math.max(11, Math.abs(state.initialVx) * totalDuration + 1);
    const scaleX = (width - 48) / extent;
    const originX = direction < 0 ? width - 24 : direction > 0 ? 24 : width / 2;
    ctx.strokeStyle = '#547079';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(12, floorY);
    ctx.lineTo(width - 12, floorY);
    ctx.stroke();
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('地面 y=0', 8, 166);
    ctx.fillStyle = '#dce8df';
    ctx.fillText('接触 ' + state.contacts + '；反弹 H=' + state.bounceHeight.toFixed(3), 8, 176);
    if (!state.finite) {
      ctx.fillStyle = '#ef9a67';
      ctx.fillText('状态异常：请重置', 24, 116);
      return;
    }
    const x = originX + state.particle.x * scaleX;
    const y = floorY - state.particle.y * scaleY;
    const radius = state.radius * scaleY;
    ctx.strokeStyle = '#ef9a67';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x, floorY);
    ctx.stroke();
    ctx.fillStyle = '#8fb7ff';
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#dce8df';
    ctx.stroke();
    ctx.fillStyle = '#dce8df';
    ctx.beginPath();
    ctx.arc(x, y, Math.max(1.5, radius * 0.22), 0, Math.PI * 2);
    ctx.fill();
  }

  const controller = ui.mount(container, {
    createState,
    controls: [
      {key: 'radius', label: '球半径 R（m）', min: 0.2, max: 0.8, step: 0.05, value: defaults.radius},
      {key: 'restitution', label: '恢复系数 e（0..1）', min: 0, max: 1, step: 0.01, value: defaults.restitution},
      {key: 'friction', label: '摩擦阻尼 f（0..1）', min: 0, max: 1, step: 0.01, value: defaults.friction},
      {key: 'initialVx', label: '初始水平速度 vx（m/s）', min: -3, max: 3, step: 0.1, value: defaults.initialVx},
    ],
    update,
    draw,
    metrics(state) {
      const rounded = value => Number(value.toFixed(8));
      return {
        '球半径 R（m）': state.radius,
        '恢复系数 e': state.restitution,
        '摩擦阻尼 f': state.friction,
        '初始水平速度 vx（m/s）': state.initialVx,
        '质量（kg）': mass,
        '重力 g（m/s²）': gravity,
        '接触高度 y=R（m）': rounded(state.contactY),
        '接触次数': state.contacts,
        '反弹高度（m）': rounded(state.bounceHeight),
        '滑行距离（m）': rounded(state.slidingDistance),
        '当前 vx（m/s）': rounded(state.particle.vx),
        '当前 vy（m/s）': rounded(state.particle.vy),
        '物理时间（s）': rounded(state.time),
        '共享核心调用次数': state.coreCalls,
        '有限值状态（1是/0否）': state.finite ? 1 : 0,
      };
    },
  });
  controller.setRunning(false);
  controller.canvas.setAttribute('aria-label', '平移球地面接触实验：x 向右、y 向上，真实共享碰撞核心，显示球心、地面、反弹高度和滑行距离');

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
    const values = {
      radius: Number(ranges[0].value), restitution: Number(ranges[1].value),
      friction: Number(ranges[2].value), initialVx: Number(ranges[3].value),
    };
    if (!valid(values)) {
      pauseWithGuidance(controller.getState(), '控件含非有限值、越界值或非正半径，已暂停。请重置实验。');
      return;
    }
    Object.assign(controller.getState(), createState(values));
    clearWarning();
    controller.draw();
    controller.report();
  });
})();
