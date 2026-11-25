(() => {
  const menuButton = document.querySelector("[data-menu-toggle]");
  const sidebar = document.querySelector("[data-sidebar]");
  if (menuButton && sidebar) {
    menuButton.addEventListener("click", () => {
      const open = sidebar.dataset.open !== "true";
      sidebar.dataset.open = String(open);
      menuButton.setAttribute("aria-expanded", String(open));
    });
    sidebar.addEventListener("click", (event) => {
      if (event.target.closest("a") && window.matchMedia("(max-width: 720px)").matches) {
        sidebar.dataset.open = "false";
        menuButton.setAttribute("aria-expanded", "false");
      }
    });
  }

  document.querySelectorAll("[data-copy]").forEach((button) => {
    button.addEventListener("click", async () => {
      const code = button.closest(".code-card").querySelector("pre");
      try {
        await navigator.clipboard.writeText(code.textContent);
        button.textContent = "已复制";
      } catch {
        button.textContent = "请手动复制";
      }
      window.setTimeout(() => { button.textContent = "复制代码"; }, 1600);
    });
  });

  const demo = document.getElementById("comparison-demo");
  if (!demo) return;

  const canvases = [...demo.querySelectorAll("[data-comparison-canvas]")];
  const playButton = document.getElementById("play-comparison");
  const resetButton = document.getElementById("reset-comparison");
  const iterationInput = document.getElementById("comparison-iterations");
  const complianceInput = document.getElementById("comparison-compliance");
  const iterationOutput = document.getElementById("iterations-output");
  const complianceOutput = document.getElementById("compliance-output");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const columns = 8;
  const rows = 7;
  const spacing = 1;
  const gravity = 7.5;
  const step = 1 / 120;
  let playing = !reducedMotion;
  let previousTime = 0;
  let accumulator = 0;

  const constraints = [];
  const addConstraint = (a, b, stiffness = 1, length = spacing) => constraints.push({ a, b, length, stiffness });
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const index = row * columns + column;
      if (column + 1 < columns) addConstraint(index, index + 1);
      if (row + 1 < rows) addConstraint(index, index + columns);
      if (row + 1 < rows && column + 1 < columns) {
        addConstraint(index, index + columns + 1, 0.45, Math.SQRT2);
        addConstraint(index + 1, index + columns, 0.45, Math.SQRT2);
      }
      if (column + 2 < columns) addConstraint(index, index + 2, 0.12, spacing * 2);
    }
  }

  class ClothDemo {
    constructor(canvas, algorithm) {
      this.canvas = canvas;
      this.context = canvas.getContext("2d");
      this.algorithm = algorithm;
      this.output = canvas.closest("figure").querySelector("[data-error]");
      this.reset();
    }

    reset() {
      this.positions = [];
      this.previous = [];
      this.lambdas = new Float64Array(constraints.length);
      for (let row = 0; row < rows; row += 1) {
        for (let column = 0; column < columns; column += 1) {
          const x = column * spacing - (columns - 1) * spacing / 2;
          const y = row * spacing + 0.35;
          this.positions.push({ x, y });
          this.previous.push({ x, y });
        }
      }
      this.draw();
      this.output.value = "0.0000";
    }

    predict() {
      const damping = this.algorithm === "mass_spring" ? 0.996 : 0.992;
      for (let index = columns; index < this.positions.length; index += 1) {
        const position = this.positions[index];
        const previous = this.previous[index];
        const next = {
          x: position.x + (position.x - previous.x) * damping,
          y: position.y + (position.y - previous.y) * damping + gravity * step * step,
        };
        previous.x = position.x;
        previous.y = position.y;
        position.x = next.x;
        position.y = next.y;
      }
    }

    springForces() {
      const forces = this.positions.map(() => ({ x: 0, y: gravity }));
      for (const constraint of constraints) {
        const a = this.positions[constraint.a];
        const b = this.positions[constraint.b];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const distance = Math.max(Math.hypot(dx, dy), 1e-6);
        const stiffness = 520 * constraint.stiffness;
        const magnitude = stiffness * (distance - constraint.length);
        const fx = dx / distance * magnitude;
        const fy = dy / distance * magnitude;
        forces[constraint.a].x += fx;
        forces[constraint.a].y += fy;
        forces[constraint.b].x -= fx;
        forces[constraint.b].y -= fy;
      }
      for (let index = columns; index < this.positions.length; index += 1) {
        const position = this.positions[index];
        const previous = this.previous[index];
        const velocityX = (position.x - previous.x) / step;
        const velocityY = (position.y - previous.y) / step;
        previous.x = position.x;
        previous.y = position.y;
        position.x += (velocityX + forces[index].x * step) * step;
        position.y += (velocityY + forces[index].y * step) * step;
      }
    }

    projectDistance(index, stiffness, alpha = 0) {
      const constraint = constraints[index];
      const a = this.positions[constraint.a];
      const b = this.positions[constraint.b];
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const distance = Math.max(Math.hypot(dx, dy), 1e-6);
      const error = distance - constraint.length;
      const weightA = constraint.a < columns ? 0 : 1;
      const weightB = constraint.b < columns ? 0 : 1;
      const weight = weightA + weightB;
      if (!weight) return;
      if (alpha > 0) {
        const deltaLambda = (-error - alpha * this.lambdas[index]) / (weight + alpha);
        this.lambdas[index] += deltaLambda;
        a.x -= weightA * deltaLambda * dx / distance;
        a.y -= weightA * deltaLambda * dy / distance;
        b.x += weightB * deltaLambda * dx / distance;
        b.y += weightB * deltaLambda * dy / distance;
      } else {
        const correction = error * stiffness * constraint.stiffness / (weight * distance);
        a.x += weightA * correction * dx;
        a.y += weightA * correction * dy;
        b.x -= weightB * correction * dx;
        b.y -= weightB * correction * dy;
      }
    }

    solve() {
      const iterations = Number(iterationInput.value);
      if (this.algorithm === "mass_spring") {
        this.springForces();
        return;
      }
      this.lambdas.fill(0);
      const alpha = 10 ** Number(complianceInput.value) / (step * step);
      for (let iteration = 0; iteration < iterations; iteration += 1) {
        for (let index = 0; index < constraints.length; index += 1) {
          this.projectDistance(index, 0.95, this.algorithm === "xpbd" ? alpha : 0);
        }
      }
    }

    collide() {
      const floor = 6.2;
      for (let index = columns; index < this.positions.length; index += 1) {
        const position = this.positions[index];
        const previous = this.previous[index];
        if (position.y > floor) {
          position.y = floor;
          previous.x += (position.x - previous.x) * 0.35;
          previous.y = position.y;
        }
      }
    }

    advance() {
      if (this.algorithm === "mass_spring") {
        this.springForces();
      } else {
        this.predict();
        this.solve();
      }
      this.collide();
    }

    measure() {
      let error = 0;
      for (const constraint of constraints) {
        const a = this.positions[constraint.a];
        const b = this.positions[constraint.b];
        error += Math.abs(Math.hypot(b.x - a.x, b.y - a.y) - constraint.length) / constraint.length;
      }
      this.output.value = (error / constraints.length).toFixed(4);
    }

    draw() {
      const context = this.context;
      const width = this.canvas.width;
      const height = this.canvas.height;
      const scale = 36;
      const originX = width / 2;
      const originY = 24;
      context.clearRect(0, 0, width, height);
      context.fillStyle = "#132832";
      context.fillRect(0, 0, width, height);
      context.strokeStyle = "#35525a";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(18, originY + floorY());
      context.lineTo(width - 18, originY + floorY());
      context.stroke();
      context.lineWidth = this.algorithm === "mass_spring" ? 1.4 : 1.8;
      context.strokeStyle = this.algorithm === "xpbd" ? "#e4a16f" : "#72c9c0";
      for (const constraint of constraints) {
        if (constraint.stiffness < 0.2) continue;
        const a = this.positions[constraint.a];
        const b = this.positions[constraint.b];
        context.beginPath();
        context.moveTo(originX + a.x * scale, originY + a.y * scale);
        context.lineTo(originX + b.x * scale, originY + b.y * scale);
        context.stroke();
      }
      context.fillStyle = "#f2dfcf";
      for (let index = 0; index < columns; index += 1) {
        const position = this.positions[index];
        context.beginPath();
        context.arc(originX + position.x * scale, originY + position.y * scale, 2.4, 0, Math.PI * 2);
        context.fill();
      }
    }
  }

  function floorY() {
    return 6.2 * 36;
  }

  const simulations = canvases.map((canvas) => new ClothDemo(canvas, canvas.dataset.algorithm));

  function frame(time) {
    const elapsed = previousTime ? Math.min((time - previousTime) / 1000, 0.05) : 0;
    previousTime = time;
    if (playing) {
      accumulator += elapsed;
      while (accumulator >= step * 4) {
        for (const simulation of simulations) simulation.advance();
        accumulator -= step * 4;
      }
    }
    for (const simulation of simulations) {
      simulation.measure();
      simulation.draw();
    }
    window.requestAnimationFrame(frame);
  }

  function updatePlayButton() {
    playButton.textContent = playing ? "暂停" : "开始";
    playButton.setAttribute("aria-pressed", String(playing));
  }

  playButton.addEventListener("click", () => {
    playing = !playing;
    updatePlayButton();
  });
  resetButton.addEventListener("click", () => simulations.forEach((simulation) => simulation.reset()));
  iterationInput.addEventListener("input", () => { iterationOutput.value = iterationInput.value; });
  complianceInput.addEventListener("input", () => {
    complianceOutput.value = `1.0e${Number(complianceInput.value)}`;
  });
  updatePlayButton();
  window.requestAnimationFrame(frame);
})();
