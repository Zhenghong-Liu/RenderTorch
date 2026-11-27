/* Shared, dependency-free numerical core for the RenderTorch physics book. */
(function (root, factory) {
  'use strict';
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.RTPhysics = factory();
  }
}(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';

  const EPSILON = 1e-12;
  const vec2 = {
    add(a, b) {
      return {x: a.x + b.x, y: a.y + b.y};
    },
    sub(a, b) {
      return {x: a.x - b.x, y: a.y - b.y};
    },
    scale(a, scalar) {
      return {x: a.x * scalar, y: a.y * scalar};
    },
    dot(a, b) {
      return a.x * b.x + a.y * b.y;
    },
    length(a) {
      return Math.hypot(a.x, a.y);
    },
    normalize(a) {
      const magnitude = Math.hypot(a.x, a.y);
      if (magnitude <= EPSILON) {
        return {x: 0, y: 0};
      }
      return {x: a.x / magnitude, y: a.y / magnitude};
    }
  };

  function integrateProjectile(state, gravity, dt, method) {
    const vx = state.vx + gravity.x * dt;
    const vy = state.vy + gravity.y * dt;
    if (method === 'exact') {
      return Object.assign({}, state, {
        x: state.x + state.vx * dt + 0.5 * gravity.x * dt * dt,
        y: state.y + state.vy * dt + 0.5 * gravity.y * dt * dt,
        vx,
        vy
      });
    }
    if (method === 'forward') {
      return Object.assign({}, state, {
        x: state.x + state.vx * dt,
        y: state.y + state.vy * dt,
        vx,
        vy
      });
    }
    if (method === 'semi-implicit') {
      return Object.assign({}, state, {
        x: state.x + vx * dt,
        y: state.y + vy * dt,
        vx,
        vy
      });
    }
    throw new RangeError('Unknown integration method: ' + method);
  }

  function springForce(extension, relativeVelocity, k, c) {
    return -k * extension - c * relativeVelocity;
  }

  function energy1d(m, v, k, x) {
    return {kinetic: 0.5 * m * v * v, potential: 0.5 * k * x * x};
  }

  function stableSpringStep(k, m) {
    return 2 / Math.sqrt(k / m);
  }

  function distanceGeometry(a, b, restLength) {
    const wa = a.w;
    const wb = b.w;
    const denom = wa + wb;
    const difference = vec2.sub(b, a);
    const distance = vec2.length(difference);
    if (denom <= 0 || distance <= EPSILON) {
      return null;
    }
    return {
      wa,
      wb,
      denom,
      n: vec2.scale(difference, 1 / distance),
      C: distance - restLength
    };
  }

  function projectDistance(a, b, restLength, stiffness = 1) {
    const geometry = distanceGeometry(a, b, restLength);
    const nextA = Object.assign({}, a);
    const nextB = Object.assign({}, b);
    if (geometry === null) {
      return {a: nextA, b: nextB};
    }
    const correction = vec2.scale(geometry.n, stiffness * geometry.C / geometry.denom);
    nextA.x += geometry.wa * correction.x;
    nextA.y += geometry.wa * correction.y;
    nextB.x -= geometry.wb * correction.x;
    nextB.y -= geometry.wb * correction.y;
    return {a: nextA, b: nextB};
  }

  function projectDistanceXPBD(a, b, restLength, compliance, dt, lambda) {
    const nextA = Object.assign({}, a);
    const nextB = Object.assign({}, b);
    if (dt === 0) {
      return {a: nextA, b: nextB, lambda};
    }
    const geometry = distanceGeometry(a, b, restLength);
    if (geometry === null) {
      return {a: nextA, b: nextB, lambda};
    }
    const alphaTilde = compliance / (dt * dt);
    const deltaLambda = (-geometry.C - alphaTilde * lambda) /
      (geometry.denom + alphaTilde);
    const correction = vec2.scale(geometry.n, -deltaLambda);
    nextA.x += geometry.wa * correction.x;
    nextA.y += geometry.wa * correction.y;
    nextB.x -= geometry.wb * correction.x;
    nextB.y -= geometry.wb * correction.y;
    return {a: nextA, b: nextB, lambda: lambda + deltaLambda};
  }

  function collideSpherePlane(particle, radius, planeY, restitution, friction) {
    const result = Object.assign({}, particle);
    const contactY = planeY + radius;
    if (result.w === 0 || result.y > contactY) {
      return result;
    }

    result.y = contactY;
    if (result.vy < 0) {
      result.vy *= -restitution;
      // Teaching simplification only: multiplicative tangential damping,
      // not a complete Coulomb friction impulse model.
      result.vx *= Math.max(0, 1 - friction);
    }
    return result;
  }

  function collideParticleSphere(particle, center, circleRadius, particleRadius,
                                 restitution, friction) {
    const result = Object.assign({}, particle);
    if (result.w === 0) {
      return result;
    }
    const offset = vec2.sub(result, center);
    const distance = vec2.length(offset);
    const contactDistance = circleRadius + particleRadius;
    if (distance <= EPSILON || distance > contactDistance) {
      return result;
    }

    const normal = vec2.scale(offset, 1 / distance);
    result.x = center.x + normal.x * contactDistance;
    result.y = center.y + normal.y * contactDistance;
    const velocity = {x: result.vx, y: result.vy};
    const normalVelocity = vec2.dot(velocity, normal);
    if (normalVelocity < 0) {
      const tangentVelocity = vec2.sub(velocity, vec2.scale(normal, normalVelocity));
      const response = vec2.add(
        vec2.scale(tangentVelocity, Math.max(0, 1 - friction)),
        vec2.scale(normal, -restitution * normalVelocity)
      );
      result.vx = response.x;
      result.vy = response.y;
    }
    return result;
  }

  function projectAreaXPBD(p0, p1, p2, p3, restArea, compliance, dt, lambda) {
    const particles = [p0, p1, p2, p3].map(p => Object.assign({}, p));
    if (dt === 0) {
      return {particles, lambda};
    }
    const gradients = [
      {x: (p1.y - p3.y) / 2, y: (p3.x - p1.x) / 2},
      {x: (p2.y - p0.y) / 2, y: (p0.x - p2.x) / 2},
      {x: (p3.y - p1.y) / 2, y: (p1.x - p3.x) / 2},
      {x: (p0.y - p2.y) / 2, y: (p2.x - p0.x) / 2}
    ];
    const denom = particles.reduce((sum, p, i) =>
      sum + p.w * vec2.dot(gradients[i], gradients[i]), 0);
    if (denom <= EPSILON) {
      return {particles, lambda};
    }

    let area = 0;
    for (let i = 0; i < particles.length; i += 1) {
      const next = particles[(i + 1) % particles.length];
      area += particles[i].x * next.y - next.x * particles[i].y;
    }
    const orientation = area < 0 ? -1 : 1;
    const C = area / 2 - orientation * restArea;
    const alphaTilde = compliance / (dt * dt);
    const deltaLambda = (-C - alphaTilde * lambda) / (denom + alphaTilde);
    particles.forEach((p, i) => {
      p.x += p.w * deltaLambda * gradients[i].x;
      p.y += p.w * deltaLambda * gradients[i].y;
    });
    return {particles, lambda: lambda + deltaLambda};
  }

  return {
    vec2,
    integrateProjectile,
    springForce,
    energy1d,
    stableSpringStep,
    projectDistance,
    projectDistanceXPBD,
    collideSpherePlane,
    collideParticleSphere,
    projectAreaXPBD
  };
}));
