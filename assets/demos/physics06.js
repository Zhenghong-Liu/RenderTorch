(() => {
  'use strict';
  const container = document.getElementById('energy-ledger');
  if (!container) return;
  const {springForce, energy1d, stableSpringStep} = window.RTPhysics;
  const ui = window.RTPhysicsUI;
  const defaults = Object.freeze({mass: 1, k: 10, damping: 0.5});
  const duration = 12;
  const frameLimit = duration * 60;

  function createState(values = defaults) {
    const initial = energy1d(values.mass, 0, values.k, 1);
    const initialEnergy = initial.kinetic + initial.potential;
    return {
      mass: values.mass,
      k: values.k,
      damping: values.damping,
      w: 1 / values.mass,
      x: 1,
      v: 0,
      force: springForce(1, 0, values.k, values.damping),
      kinetic: initial.kinetic,
      potential: initial.potential,
      totalEnergy: initialEnergy,
      initialEnergy,
      relativeEnergyLoss: 0,
      stepHint: stableSpringStep(values.k, values.mass),
      time: 0,
      frame: 0,
      finished: false,
    };
  }

  function draw(ctx, state, width, height) {
    ctx.fillStyle = '#10242c';
    ctx.fillRect(0, 0, width, height);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.font = '11px system-ui';
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('一维弹簧振子', 10, 16);
    ctx.fillText('x = ' + state.x.toFixed(3) + ' m', 10, 34);

    const equilibriumX = width / 2;
    const ballX = equilibriumX + Math.max(-1, Math.min(1, state.x)) * Math.min(42, width * 0.14);
    const anchorX = 18;
    const springY = 58;
    ctx.strokeStyle = '#8fb7ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(anchorX, springY);
    for (let index = 1; index < 8; index += 1) {
      const ratio = index / 8;
      ctx.lineTo(anchorX + (ballX - anchorX) * ratio, springY + (index % 2 ? -6 : 6));
    }
    ctx.lineTo(ballX, springY);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(ballX, springY, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#f4f1e8';
    ctx.fill();
    ctx.strokeStyle = '#10242c';
    ctx.stroke();
    ctx.strokeStyle = 'rgba(140,164,170,.45)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(equilibriumX, springY - 10);
    ctx.lineTo(equilibriumX, springY + 10);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('平衡位置 x=0', equilibriumX - 40, springY - 14);

    const rows = [
      {label: '动能 K', value: state.kinetic, color: '#ef9a67'},
      {label: '势能 U', value: state.potential, color: '#58c7bd'},
      {label: '总能量 E', value: state.totalEnergy, color: '#ffd48a'},
    ];
    const maximum = Math.max(1, state.initialEnergy, ...rows.map(row => row.value));
    const barX = 74;
    const barWidth = Math.max(1, width - 154);
    rows.forEach((row, index) => {
      const baseline = 82 + index * 22;
      ctx.fillStyle = '#b7c9cb';
      ctx.fillText(row.label, 10, baseline);
      ctx.fillStyle = 'rgba(140,164,170,.18)';
      ctx.fillRect(barX, baseline - 11, barWidth, 12);
      ctx.fillStyle = row.color;
      ctx.fillRect(barX, baseline - 11, Math.max(0, Math.min(1, row.value / maximum)) * barWidth, 12);
      ctx.fillStyle = '#b7c9cb';
      ctx.fillText(row.value.toFixed(3), width - 70, baseline);
    });
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('h_hint = ' + state.stepHint.toFixed(6) + ' s', 10, 150);
    ctx.fillText('仅诊断，不保证稳定', 10, 168);
  }

  const controller = ui.mount(container, {
    createState,
    controls: [
      {key: 'mass', label: '质量 m（kg）', min: 0.5, max: 2, step: 0.1, value: defaults.mass},
      {key: 'k', label: '刚度 k（N/m）', min: 1, max: 40, step: 1, value: defaults.k},
      {key: 'damping', label: '阻尼 c（N·s/m）', min: 0, max: 5, step: 0.1, value: defaults.damping},
    ],
    update(state, dt, values) {
      if (state.finished) return state;
      state.force = springForce(state.x, state.v, state.k, state.damping);
      state.v += state.force / state.mass * dt;
      state.x += state.v * dt;
      const energy = energy1d(state.mass, state.v, state.k, state.x);
      state.kinetic = energy.kinetic;
      state.potential = energy.potential;
      state.totalEnergy = state.kinetic + state.potential;
      state.relativeEnergyLoss = (state.initialEnergy - state.totalEnergy) / state.initialEnergy;
      state.time += dt;
      state.frame += 1;
      if (state.frame >= frameLimit) {
        state.time = duration;
        state.finished = true;
        controller.setRunning(false);
        warning.hidden = false;
        warning.textContent = '已完成 12 s 能量观察并暂停。稳定步长提示只用于诊断，不保证任意积分稳定；点击重置可复现。';
      }
      return state;
    },
    draw,
    metrics(state) {
      const rounded = value => Number(value.toFixed(6));
      return {
        '质量（kg）': state.mass,
        '刚度（N/m）': state.k,
        '阻尼（N·s/m）': state.damping,
        '位移 x（m）': rounded(state.x),
        '速度 v（m/s）': rounded(state.v),
        '弹簧力（N）': rounded(state.force),
        '动能（J）': rounded(state.kinetic),
        '势能（J）': rounded(state.potential),
        '总能量（J）': rounded(state.totalEnergy),
        '初始能量（J）': rounded(state.initialEnergy),
        '相对能量损失': rounded(state.relativeEnergyLoss),
        '稳定步长提示（s，非保证）': rounded(state.stepHint),
        '时间（s）': rounded(state.time),
      };
    },
  });
  controller.setRunning(false);
  controller.canvas.setAttribute('aria-label', '阻尼弹簧振子能量账本：绘制动能、势能与总能量三段能量条');

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
      mass: Number(ranges[0].value),
      k: Number(ranges[1].value),
      damping: Number(ranges[2].value),
    }));
    clearWarning();
    controller.draw();
    controller.report();
  });
})();
