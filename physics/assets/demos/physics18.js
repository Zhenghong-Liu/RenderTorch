(() => {
  'use strict';
  const container = document.getElementById('rotation-intuition');
  if (!container) return;
  const ui = window.RTPhysicsUI;
  const defaults = Object.freeze({torque: 1.2, inertia: 0.125, horizon: 2});
  const mass = 1;
  const radius = 0.5;
  const solidDiskInertia = 0.5 * mass * radius * radius;
  const ringInertia = mass * radius * radius;
  const outerDt = 1 / 60;
  const allowedHorizons = [0.5, 1, 2, 3];

  function distribution(inertia) {
    if (inertia === solidDiskInertia) return '均匀实心圆盘';
    if (inertia === ringInertia) return '均匀薄圆环';
    const ringFraction = (inertia - solidDiskInertia) / (ringInertia - solidDiskInertia);
    return '圆盘/外环质量混合，外环质量分数 ' + ringFraction.toFixed(3);
  }

  function createState(values = defaults) {
    const inertia = values.inertia;
    const effectiveInertia = inertia + mass * radius * radius;
    const alpha = values.torque / effectiveInertia;
    const duration = Math.round(values.horizon / outerDt);
    return {
      torque: values.torque, inertia, effectiveInertia,
      mass, radius, solidDiskInertia, ringInertia, distribution: distribution(inertia),
      alpha, omega: 0, theta: 0, rimArc: 0, x: 0, v: 0,
      horizon: values.horizon, duration, outerDt,
      time: 0, frame: 0, finite: true, invalid: false, finished: false,
    };
  }

  function valid(values) {
    return Number.isFinite(values.torque) && values.torque >= 0 && values.torque <= 4 &&
      Number.isFinite(values.inertia) && values.inertia >= solidDiskInertia &&
      values.inertia <= ringInertia && allowedHorizons.includes(values.horizon);
  }

  function reconstruct(state) {
    state.effectiveInertia = state.inertia + mass * radius * radius;
    state.alpha = state.torque / state.effectiveInertia;
    state.omega = state.alpha * state.time;
    state.theta = 0.5 * state.alpha * state.time * state.time;
    state.rimArc = radius * state.theta;
    state.x = radius * state.theta;
    state.v = radius * state.omega;
    state.distribution = distribution(state.inertia);
  }

  function pauseWithGuidance(state, message) {
    state.invalid = true;
    state.finished = true;
    state.finite = false;
    controller.setRunning(false);
    warning.hidden = false;
    warning.textContent = message;
  }

  function update(state, dt, values) {
    if (state.finished || state.invalid) {
      controller.setRunning(false);
      return state;
    }
    if (dt !== outerDt || !valid(values) || !valid({
      torque: state.torque, inertia: state.inertia, horizon: state.horizon,
    }) || !(state.effectiveInertia > 0)) {
      pauseWithGuidance(state, '控件、时间步或转动惯量无效，已暂停。请重置实验。');
      return state;
    }
    state.time = Math.min(state.horizon, state.time + dt);
    state.frame += 1;
    if (state.frame >= state.duration) state.time = state.horizon;
    reconstruct(state);
    state.finite = [state.time, state.alpha, state.omega, state.theta, state.rimArc,
      state.x, state.v, state.effectiveInertia].every(Number.isFinite);
    if (!state.finite) {
      pauseWithGuidance(state, '状态出现非有限值，已暂停。请重置，不要裁剪数值错误。');
      return state;
    }
    if (state.frame >= state.duration) {
      state.finished = true;
      controller.setRunning(false);
      warning.hidden = false;
      warning.textContent = '已完成所选 ' + state.horizon + ' s 有限实验范围并暂停。点击重置可复现。';
    }
    return state;
  }

  function draw(ctx, state, width, height) {
    ctx.fillStyle = '#10242c';
    ctx.fillRect(0, 0, width, height);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.font = '11px system-ui';
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('理想无滑动：浏览器概念实验', 8, 12);
    ctx.fillText('固定 m=1 kg；R=0.5 m', 8, 27);
    ctx.fillText('τ=' + state.torque.toFixed(2) + '；I=' + state.inertia.toFixed(3), 8, 42);
    ctx.fillText('Ieff=I+mR²；v=Rω', 8, 57);
    ctx.fillStyle = '#dce8df';
    ctx.fillText('α=' + state.alpha.toFixed(3) + ' ω=' + state.omega.toFixed(3), 8, 72);
    ctx.fillText('θ=' + state.theta.toFixed(3) + ' rad', 8, 87);
    const floorY = 150;
    const finalTravel = radius * 0.5 * state.alpha * state.horizon * state.horizon;
    const scale = Math.min(64, (width - 80) / Math.max(1, finalTravel), floorY - 86);
    const diskRadius = radius * scale;
    const x = 40 + state.x * scale;
    const y = floorY - diskRadius;
    const angle = state.theta % (Math.PI * 2);
    ctx.strokeStyle = '#547079';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(12, floorY);
    ctx.lineTo(width - 12, floorY);
    ctx.stroke();
    ctx.fillStyle = '#8fb7ff';
    ctx.beginPath();
    ctx.arc(x, y, diskRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#dce8df';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#ef9a67';
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + diskRadius * Math.cos(angle), y + diskRadius * Math.sin(angle));
    ctx.stroke();
    ctx.fillStyle = '#ef9a67';
    ctx.beginPath();
    ctx.arc(x + diskRadius * Math.cos(angle), y + diskRadius * Math.sin(angle),
      Math.max(1.5, diskRadius * 0.1), 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#8fd1b0';
    ctx.beginPath();
    ctx.arc(x, y, diskRadius + 7, 0, angle);
    ctx.stroke();
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('地面 y=0', 8, 166);
    ctx.fillStyle = '#dce8df';
    ctx.fillText('s=Rθ；x=Rθ；圆弧为当前圈', 8, 176);
    if (!state.finite) {
      ctx.fillStyle = '#ef9a67';
      ctx.fillText('状态异常：请重置', 24, 120);
    }
  }

  const controller = ui.mount(container, {
    createState,
    controls: [
      {key: 'torque', label: '切向力矩 τ（N·m）', min: 0, max: 4, step: 0.1, value: defaults.torque},
      {key: 'inertia', label: '正转动惯量 I（kg·m²）', min: solidDiskInertia, max: ringInertia,
        step: 0.015625, value: defaults.inertia},
      {key: 'horizon', label: '有限实验时间 T（s）', min: 0.5, max: 3, step: 0.5, value: defaults.horizon},
    ],
    update,
    draw,
    metrics(state) {
      const rounded = value => Number(value.toFixed(8));
      return {
        '固定质量 m（kg）': mass,
        '固定半径 R（m）': radius,
        '质量分布': state.distribution,
        '转动惯量 I（kg·m²）': state.inertia,
        '有效惯量 I+mR²（kg·m²）': rounded(state.effectiveInertia),
        '切向力矩 τ（N·m）': state.torque,
        '角加速度 α（rad/s²）': rounded(state.alpha),
        '角速度 ω（rad/s）': rounded(state.omega),
        '转角 θ（rad）': rounded(state.theta),
        '轮缘弧长 s=Rθ（m）': rounded(state.rimArc),
        '平移 x=Rθ（m）': rounded(state.x),
        '轮心速度 v=Rω（m/s）': rounded(state.v),
        '实验时间范围 T（s）': state.horizon,
        '物理时间（s）': rounded(state.time),
        '有限值状态（1是/0否）': state.finite ? 1 : 0,
      };
    },
  });
  controller.setRunning(false);
  controller.canvas.setAttribute('aria-label', '理想无滑动圆盘滚动概念实验：固定质量和半径，径向标记、地面线与转角圆弧显示角速度、弧长和平移');

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
      torque: Number(ranges[0].value), inertia: Number(ranges[1].value),
      horizon: Number(ranges[2].value),
    };
    if (!valid(values)) {
      pauseWithGuidance(controller.getState(), '控件含非有限值、越界值或非正惯量，已暂停。请重置实验。');
      return;
    }
    Object.assign(controller.getState(), createState(values));
    clearWarning();
    controller.draw();
    controller.report();
  });
})();
