(() => {
  'use strict';
  const container = document.getElementById('constraint-order');
  if (!container) return;
  const {projectDistance} = window.RTPhysics;
  const ui = window.RTPhysicsUI;
  const defaults = Object.freeze({offset: 0.25, iterations: 4});
  const names = Object.freeze(['forward', 'reverse', 'alternating']);
  const labels = Object.freeze(['正序', '逆序', '交替']);
  const colors = Object.freeze(['#8fb7ff', '#ef9a67', '#9bd6ae']);
  const restLength = 1;

  function initialParticles(offset) {
    return [
      {x: 0, y: 0, vx: 0, vy: 0, w: 0},
      {x: 1 + offset, y: 0.3, vx: 0, vy: 0, w: 1},
      {x: 2, y: 0, vx: 0, vy: 0, w: 0},
    ];
  }

  function cloneParticles(particles) {
    return particles.map(particle => Object.assign({}, particle));
  }

  function relativeError(particles) {
    return Math.max(...[0, 1].map(index => {
      const a = particles[index];
      const b = particles[index + 1];
      return Math.abs(Math.hypot(b.x - a.x, b.y - a.y) - restLength) / restLength;
    }));
  }

  function createState(values = defaults) {
    const particles = initialParticles(values.offset);
    return {
      offset: values.offset,
      iterations: values.iterations,
      restLength,
      runs: names.map(name => ({name, particles: cloneParticles(particles), error: 0})),
      initialError: relativeError(particles),
      iteration: 0,
      finite: true,
      invalid: false,
      time: 0,
      frame: 0,
      finished: false,
    };
  }

  function sweep(run, order) {
    for (const index of order) {
      const projected = projectDistance(run.particles[index], run.particles[index + 1], restLength, 1);
      run.particles[index] = projected.a;
      run.particles[index + 1] = projected.b;
    }
    // Fixed endpoints are an explicit boundary, not merely an inverse-mass hint.
    run.particles[0].x = 0;
    run.particles[0].y = 0;
    run.particles[2].x = 2;
    run.particles[2].y = 0;
  }

  function update(state, dt, values) {
    if (state.finished || state.invalid) {
      controller.setRunning(false);
      return state;
    }
    sweep(state.runs[0], [0, 1]);
    sweep(state.runs[1], [1, 0]);
    sweep(state.runs[2], [0, 1]);
    sweep(state.runs[2], [1, 0]);
    state.runs.forEach(run => { run.error = relativeError(run.particles); });
    state.iteration += 1;
    state.frame = state.iteration;
    state.time += dt;
    state.finite = Number.isFinite(state.initialError) && Number.isFinite(state.time) &&
      state.runs.every(run => Number.isFinite(run.error) && run.particles.every(particle =>
        [particle.x, particle.y, particle.vx, particle.vy, particle.w].every(Number.isFinite)));
    if (!state.finite) {
      state.invalid = true;
      state.finished = true;
      controller.setRunning(false);
      warning.hidden = false;
      warning.textContent = '状态出现非有限值，已暂停。请重置，不要用裁剪隐藏数值错误。';
      return state;
    }
    if (state.iteration >= state.iterations) {
      state.finished = true;
      controller.setRunning(false);
      warning.hidden = false;
      warning.textContent = '已完成所选 ' + state.iterations + ' 轮投影并暂停。三种顺序从相同初始状态出发；点击重置可复现。';
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
    ctx.fillText('两端固定；中点逆质量=1', 8, 14);
    ctx.fillText('正序 0,1；逆序 1,0', 8, 30);
    ctx.fillText('交替每轮 0,1 再 1,0', 8, 46);

    const centers = [62, 104, 146];
    const map = particle => ({x: 84 + particle.x * 110, y: 0});
    state.runs.forEach((run, row) => {
      const points = run.particles.map(particle => {
        const point = map(particle);
        point.y = centers[row] - particle.y * 30;
        return point;
      });
      ctx.fillStyle = '#9db6b8';
      ctx.fillText(labels[row], 8, centers[row] + 4);
      ctx.strokeStyle = colors[row];
      ctx.lineWidth = 2;
      for (let index = 0; index < 2; index += 1) {
        ctx.beginPath();
        ctx.moveTo(points[index].x, points[index].y);
        ctx.lineTo(points[index + 1].x, points[index + 1].y);
        ctx.stroke();
      }
      ctx.lineWidth = 1;
      points.forEach((point, index) => {
        if (index !== 1) {
          ctx.fillStyle = '#ffd48a';
          ctx.fillRect(point.x - 4, point.y - 4, 8, 8);
        }
        ctx.beginPath();
        ctx.arc(point.x, point.y, index === 1 ? 4 : 2.5, 0, Math.PI * 2);
        ctx.fillStyle = index === 1 ? '#f4f1e8' : '#ffd48a';
        ctx.fill();
        ctx.strokeStyle = '#10242c';
        ctx.stroke();
      });
    });
  }

  const controller = ui.mount(container, {
    createState,
    controls: [
      {key: 'offset', label: '初始水平偏移（m）', min: 0.1, max: 0.8, step: 0.05, value: defaults.offset},
      {key: 'iterations', label: '迭代数', min: 1, max: 80, step: 1, value: defaults.iterations},
    ],
    update,
    draw,
    metrics(state) {
      const rounded = value => Number(value.toFixed(8));
      const metrics = {
        '初始水平偏移（m）': rounded(state.offset),
        '所选迭代数': state.iterations,
        '已完成迭代轮数': state.iteration,
        '初始最大相对约束误差': rounded(state.initialError),
      };
      names.forEach((name, index) => {
        const prefix = labels[index];
        metrics[prefix + '最大相对约束误差'] = rounded(state.runs[index].error);
        metrics[prefix + '中点 x（m）'] = rounded(state.runs[index].particles[1].x);
        metrics[prefix + '中点 y（m）'] = rounded(state.runs[index].particles[1].y);
      });
      metrics['有限值状态（1是/0否）'] = state.finite ? 1 : 0;
      metrics['界面时间（s）'] = rounded(state.time);
      return metrics;
    },
  });
  controller.setRunning(false);
  controller.canvas.setAttribute('aria-label', '三质点约束顺序比较：两端固定，正序逆序与交替 Gauss–Seidel 投影并列显示');

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
      offset: Number(ranges[0].value),
      iterations: Number(ranges[1].value),
    }));
    clearWarning();
    controller.draw();
    controller.report();
  });
})();
