(() => {
  'use strict';
  const container = document.getElementById('continuous-collision');
  if (!container) return;
  const ui = window.RTPhysicsUI;
  const defaults = Object.freeze({speed: 18, wallThickness: 0.04, dt: 0.08});
  const halfSize = 0.2;
  const initialX = 0.1;
  const wallLeft = 2;
  const duration = 2;
  const postCrossingWindow = 2;

  function createBox(speed) {
    return {x: initialX, y: 0, vx: speed, vy: 0};
  }

  function createState(values = defaults) {
    return {
      speed: values.speed,
      wallThickness: values.wallThickness,
      dt: values.dt,
      wall: {left: wallLeft, right: wallLeft + values.wallThickness, minY: -0.65, maxY: 0.65},
      discrete: Object.assign(createBox(values.speed), {hit: false, missed: false, status: '未检测'}),
      swept: Object.assign(createBox(values.speed), {
        hit: false, crossed: false, entryTime: 0, exitTime: 0, hitX: 0, hitY: 0, hitFace: '',
      }),
      postCrossingSteps: 0,
      time: 0,
      frame: 0,
      finite: true,
      invalid: false,
      finished: false,
    };
  }

  function endpointOverlap(box, wall) {
    return box.x + halfSize >= wall.left && box.x - halfSize <= wall.right &&
      box.y + halfSize >= wall.minY && box.y - halfSize <= wall.maxY;
  }

  function sweptInterval(start, end, wall, speed, dt) {
    const yOverlaps = start.y + halfSize >= wall.minY && start.y - halfSize <= wall.maxY;
    if (!yOverlaps) return null;
    if (speed === 0) {
      if (end.x + halfSize < wall.left || end.x - halfSize > wall.right) return null;
      return {entry: 0, exit: dt};
    }
    const first = (wall.left - (start.x + halfSize)) / speed;
    const second = (wall.right - (start.x - halfSize)) / speed;
    const entry = Math.max(0, Math.min(first, second));
    const exit = Math.min(dt, Math.max(first, second));
    return entry <= exit + 1e-12 ? {entry, exit} : null;
  }

  function update(state, transportDt, values) {
    if (state.finished || state.invalid) {
      controller.setRunning(false);
      return state;
    }
    if (!Number.isFinite(state.dt) || state.dt <= 0 || !Number.isFinite(state.speed) ||
        state.speed < 0 || !Number.isFinite(state.wallThickness) || state.wallThickness <= 0) {
      state.invalid = true;
      state.finite = false;
      state.finished = true;
      controller.setRunning(false);
      warning.hidden = false;
      warning.textContent = '速度、墙厚或物理时间步无效，已暂停。请重置实验。';
      return state;
    }

    const stepStart = state.time;
    const oldDiscrete = Object.assign({}, state.discrete);
    const oldSwept = Object.assign({}, state.swept);
    // Detection only: both boxes have exactly the same constant-velocity path.
    state.discrete.x += state.speed * state.dt;
    state.swept.x += state.speed * state.dt;
    state.discrete.hit = state.discrete.hit || endpointOverlap(state.discrete, state.wall);
    const interval = sweptInterval(oldSwept, state.swept, state.wall, state.speed, state.dt);
    if (interval && !state.swept.hit) {
      state.swept.hit = true;
      state.swept.entryTime = stepStart + interval.entry;
      state.swept.exitTime = stepStart + interval.exit;
      state.swept.hitX = state.wall.left;
      state.swept.hitY = state.swept.y;
      state.swept.hitFace = '左面';
    }
    state.swept.crossed = state.swept.x - halfSize > state.wall.right;
    if (oldSwept.crossed && state.swept.crossed) state.postCrossingSteps += 1;
    state.discrete.missed = state.swept.crossed && !state.discrete.hit &&
      state.postCrossingSteps >= postCrossingWindow;
    if (state.discrete.hit) {
      state.discrete.status = '命中';
    } else if (state.swept.crossed && state.discrete.missed) {
      state.discrete.status = '漏碰';
    } else if (state.swept.crossed) {
      state.discrete.status = '观察中';
    } else if (state.speed === 0) {
      state.discrete.status = '未穿越';
    } else {
      state.discrete.status = '未命中';
    }

    state.time += state.dt;
    state.frame += 1;
    state.finite = [state.time, state.speed, state.wallThickness, state.dt,
      state.wall.left, state.wall.right, state.discrete.x, state.swept.x,
      state.swept.entryTime, state.swept.exitTime, state.swept.hitX, state.swept.hitY]
      .every(Number.isFinite) && oldDiscrete.x === oldDiscrete.x && oldSwept.x === oldSwept.x;
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
      warning.textContent = '已完成 2 s 模拟时间观察并暂停。位移和检测都使用所选 dt；共享 UI 调度仍为 1/60 s。';
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
    ctx.fillText('只检测；不反弹 / 不重力 / 不裁剪', 8, 14);
    ctx.fillText('相同速度与轨迹；dt 驱动模拟时间', 8, 30);
    ctx.fillText('墙 x=2 m；上下比较端点与 swept', 8, 46);

    const minX = -0.4;
    const maxX = Math.max(state.wall.right + 0.5, initialX + halfSize + state.speed * duration + 0.5);
    const scaleY = 34;
    const panels = [
      {name: '离散：端点', box: state.discrete, x: 8, centerY: 96, color: '#8fb7ff'},
      {name: '连续：swept', box: state.swept, x: width / 2 + 4, centerY: 150, color: '#9bd6ae'},
    ];
    panels.forEach(panel => {
      const panelWidth = width / 2 - 12;
      const mapX = x => panel.x + 8 + (x - minX) / (maxX - minX) * (panelWidth - 16);
      const mapY = y => panel.centerY - y * scaleY;
      ctx.fillStyle = '#9db6b8';
      ctx.fillText(panel.name, panel.x, panel.centerY - 34);
      ctx.strokeStyle = '#718f94';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(mapX(state.wall.left), mapY(state.wall.minY));
      ctx.lineTo(mapX(state.wall.left), mapY(state.wall.maxY));
      ctx.moveTo(mapX(state.wall.right), mapY(state.wall.minY));
      ctx.lineTo(mapX(state.wall.right), mapY(state.wall.maxY));
      ctx.moveTo(mapX(state.wall.left), mapY(state.wall.minY));
      ctx.lineTo(mapX(state.wall.right), mapY(state.wall.minY));
      ctx.moveTo(mapX(state.wall.left), mapY(state.wall.maxY));
      ctx.lineTo(mapX(state.wall.right), mapY(state.wall.maxY));
      ctx.stroke();

      if (panel.box === state.swept && state.swept.hit) {
        const entryX = initialX + state.speed * state.swept.entryTime;
        const exitX = initialX + state.speed * state.swept.exitTime;
        ctx.strokeStyle = '#ef9a67';
        ctx.beginPath();
        ctx.moveTo(mapX(entryX), mapY(0.5));
        ctx.lineTo(mapX(entryX), mapY(-0.5));
        ctx.moveTo(mapX(exitX), mapY(0.5));
        ctx.lineTo(mapX(exitX), mapY(-0.5));
        ctx.stroke();
        ctx.fillStyle = '#ef9a67';
        ctx.fillText('入', mapX(entryX) - 6, mapY(-0.5) - 3);
        ctx.fillText('离', mapX(exitX) - 6, mapY(-0.5) - 3);
      }

      const x = mapX(panel.box.x);
      const y = mapY(panel.box.y);
      const boxWidth = 2 * halfSize / (maxX - minX) * (panelWidth - 16);
      const boxHeight = 2 * halfSize * scaleY;
      ctx.strokeStyle = panel.color;
      ctx.strokeRect(x - boxWidth / 2, y - boxHeight / 2, boxWidth, boxHeight);
      if (panel.box === state.swept && state.swept.hit) {
        ctx.beginPath();
        ctx.arc(mapX(state.swept.hitX), mapY(state.swept.hitY), 3, 0, Math.PI * 2);
        ctx.fillStyle = '#ef9a67';
        ctx.fill();
      }
    });
    ctx.lineWidth = 1;
    ctx.fillStyle = '#9db6b8';
    ctx.fillText('离散：' + state.discrete.status + '；连续：' + (state.swept.hit ? '命中' : '未命中'), 8, 178);
  }

  const controller = ui.mount(container, {
    createState,
    controls: [
      {key: 'speed', label: '水平速度（m/s）', min: 0, max: 30, step: 0.5, value: defaults.speed},
      {key: 'wallThickness', label: '墙厚（m）', min: 0.04, max: 0.6, step: 0.02, value: defaults.wallThickness},
      {key: 'dt', label: '实际物理时间步 dt（s）', min: 0.005, max: 0.2, step: 0.005, value: defaults.dt},
    ],
    update,
    draw,
    metrics(state) {
      const rounded = value => Number(value.toFixed(8));
      return {
        '水平速度（m/s）': state.speed,
        '墙厚（m）': state.wallThickness,
        '物理时间步（s）': rounded(state.dt),
        '离散是否命中（1是/0否）': state.discrete.hit ? 1 : 0,
        '离散是否漏碰（1是/0否）': state.discrete.missed ? 1 : 0,
        '离散穿越后观察步数': state.postCrossingSteps,
        '连续是否命中（1是/0否）': state.swept.hit ? 1 : 0,
        '连续进入时间（s，未命中为0）': state.swept.hit ? rounded(state.swept.entryTime) : 0,
        '连续离开时间（s，未命中为0）': state.swept.hit ? rounded(state.swept.exitTime) : 0,
        '连续命中位置 x（m，未命中为0）': state.swept.hit ? rounded(state.swept.hitX) : 0,
        '连续命中位置 y（m，未命中为0）': state.swept.hit ? rounded(state.swept.hitY) : 0,
        '连续命中面（1左面/0未命中）': state.swept.hit ? 1 : 0,
        '有限值状态（1是/0否）': state.finite ? 1 : 0,
        '模拟时间（s）': rounded(state.time),
      };
    },
  });
  controller.setRunning(false);
  controller.canvas.setAttribute('aria-label', '高速方块与薄墙：上排离散端点检测，下排 swept AABB 连续进入离开检测；两者只检测不反弹');

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
      speed: Number(ranges[0].value),
      wallThickness: Number(ranges[1].value),
      dt: Number(ranges[2].value),
    }));
    clearWarning();
    controller.draw();
    controller.report();
  });
})();
