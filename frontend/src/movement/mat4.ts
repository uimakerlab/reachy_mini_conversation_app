/**
 * 4x4 homogeneous transformation matrix utilities.
 *
 * Row-major layout matching numpy's default:
 *   [R00 R01 R02 tx]
 *   [R10 R11 R12 ty]
 *   [R20 R21 R22 tz]
 *   [ 0   0   0   1]
 *
 * Index mapping: M[row][col] = data[row * 4 + col]
 *
 * Rotation convention: intrinsic XYZ Euler angles (roll, pitch, yaw)
 * matching scipy.spatial.transform.Rotation.from_euler("xyz", [r, p, y]).
 * Resulting matrix: R = Rz(yaw) * Ry(pitch) * Rx(roll)
 */

export type Mat4 = Float64Array;

// ── Construction ──

export function mat4Identity(): Mat4 {
  const m = new Float64Array(16);
  m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1;
  return m;
}

export function mat4Clone(src: Mat4): Mat4 {
  return new Float64Array(src);
}

/**
 * Create a 4x4 pose matrix from position + intrinsic XYZ Euler angles.
 * Exact equivalent of Python's create_head_pose().
 *
 * @param x  meters (or mm if mm=true)
 * @param y  meters
 * @param z  meters
 * @param roll   radians (or degrees if degrees=true)
 * @param pitch  radians
 * @param yaw    radians
 */
export function createHeadPoseMat4(
  x = 0, y = 0, z = 0,
  roll = 0, pitch = 0, yaw = 0,
  opts: { degrees?: boolean; mm?: boolean } = {},
): Mat4 {
  const d = opts.degrees ? Math.PI / 180 : 1;
  const s = opts.mm ? 0.001 : 1;

  const r = roll * d;
  const p = pitch * d;
  const w = yaw * d;

  // R = Rz(yaw) * Ry(pitch) * Rx(roll)
  const cr = Math.cos(r), sr = Math.sin(r);
  const cp = Math.cos(p), sp = Math.sin(p);
  const cw = Math.cos(w), sw = Math.sin(w);

  const m = new Float64Array(16);

  // Row 0
  m[0]  = cw * cp;
  m[1]  = cw * sp * sr - sw * cr;
  m[2]  = cw * sp * cr + sw * sr;
  m[3]  = x * s;

  // Row 1
  m[4]  = sw * cp;
  m[5]  = sw * sp * sr + cw * cr;
  m[6]  = sw * sp * cr - cw * sr;
  m[7]  = y * s;

  // Row 2
  m[8]  = -sp;
  m[9]  = cp * sr;
  m[10] = cp * cr;
  m[11] = z * s;

  // Row 3
  m[15] = 1;

  return m;
}

// ── Composition ──

/**
 * Compose an absolute world-frame pose with a world-frame offset.
 * Exact equivalent of Python's compose_world_offset().
 *
 *   translations: t_final = t_abs + t_off
 *   rotations:    R_final = R_off @ R_abs
 */
export function composeWorldOffset(
  T_abs: Mat4,
  T_off: Mat4,
  reorthonormalize = false,
): Mat4 {
  const out = new Float64Array(16);

  // R_final = R_off @ R_abs  (3x3 matrix multiply)
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let sum = 0;
      for (let k = 0; k < 3; k++) {
        sum += T_off[i * 4 + k] * T_abs[k * 4 + j];
      }
      out[i * 4 + j] = sum;
    }
  }

  if (reorthonormalize) {
    svdReorthonormalize3x3(out);
  }

  // t_final = t_abs + t_off
  out[3]  = T_abs[3]  + T_off[3];
  out[7]  = T_abs[7]  + T_off[7];
  out[11] = T_abs[11] + T_off[11];
  out[15] = 1;

  return out;
}

// ── Interpolation ──

/**
 * Linearly interpolate between two 4x4 poses (SLERP for rotation, lerp for translation).
 * Exact equivalent of Python's linear_pose_interpolation().
 *
 * Uses axis-angle representation for smooth rotation interpolation:
 *   q_rel = rot_start^-1 * rot_end
 *   rotvec = q_rel.as_rotvec() * t
 *   rot_interp = rot_start * from_rotvec(rotvec)
 */
export function linearPoseInterpolation(start: Mat4, end: Mat4, t: number): Mat4 {
  const out = new Float64Array(16);

  // Extract 3x3 rotations
  const Rs = extractRot3(start);
  const Re = extractRot3(end);

  // R_rel = Rs^T * Re  (Rs^-1 for orthonormal = Rs^T)
  const Rrel = mat3Multiply(mat3Transpose(Rs), Re);

  // Convert relative rotation to axis-angle (rotation vector)
  const rv = mat3ToRotvec(Rrel);

  // Scale by t
  const rvScaled: [number, number, number] = [rv[0] * t, rv[1] * t, rv[2] * t];

  // R_interp = Rs * from_rotvec(rv_scaled)
  const Rscaled = rotvecToMat3(rvScaled);
  const Rinterp = mat3Multiply(Rs, Rscaled);

  // Write rotation to output
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      out[i * 4 + j] = Rinterp[i * 3 + j];
    }
  }

  // Lerp translation
  out[3]  = start[3]  + (end[3]  - start[3])  * t;
  out[7]  = start[7]  + (end[7]  - start[7])  * t;
  out[11] = start[11] + (end[11] - start[11]) * t;
  out[15] = 1;

  return out;
}

// ── Serialization ──

/** Flatten to 16-element array for the daemon's Matrix4x4Pose {m: [...]} format. */
export function mat4Flatten(m: Mat4): number[] {
  return Array.from(m);
}

// ── Euler conversions ──

/** Convert a HeadPose-like object to a Mat4. */
export function mat4FromEuler(
  p: { x: number; y: number; z: number; roll: number; pitch: number; yaw: number },
): Mat4 {
  return createHeadPoseMat4(p.x, p.y, p.z, p.roll, p.pitch, p.yaw);
}

/**
 * Extract intrinsic XYZ Euler angles (roll, pitch, yaw) + translation from a Mat4.
 * Inverse of createHeadPoseMat4.
 *
 * From R = Rz(yaw) * Ry(pitch) * Rx(roll):
 *   R[2][0] = -sin(pitch)
 *   R[2][1] = cos(pitch)*sin(roll)
 *   R[2][2] = cos(pitch)*cos(roll)
 *   R[1][0] = sin(yaw)*cos(pitch)
 *   R[0][0] = cos(yaw)*cos(pitch)
 */
export function mat4ToEuler(m: Mat4): {
  x: number; y: number; z: number;
  roll: number; pitch: number; yaw: number;
} {
  const r20 = m[8];
  const r21 = m[9];
  const r22 = m[10];
  const r10 = m[4];
  const r00 = m[0];

  const pitch = -Math.asin(clamp(r20, -1, 1));
  const cp = Math.cos(pitch);

  let roll: number, yaw: number;
  if (Math.abs(cp) > 1e-6) {
    roll = Math.atan2(r21, r22);
    yaw  = Math.atan2(r10, r00);
  } else {
    // Gimbal lock fallback
    roll = Math.atan2(-m[6], m[5]);
    yaw  = 0;
  }

  return { x: m[3], y: m[7], z: m[11], roll, pitch, yaw };
}

// ── Internal 3x3 helpers ──

type Mat3 = Float64Array; // 9 elements, row-major

function extractRot3(m: Mat4): Mat3 {
  const r = new Float64Array(9);
  r[0] = m[0]; r[1] = m[1]; r[2] = m[2];
  r[3] = m[4]; r[4] = m[5]; r[5] = m[6];
  r[6] = m[8]; r[7] = m[9]; r[8] = m[10];
  return r;
}

function mat3Multiply(a: Mat3, b: Mat3): Mat3 {
  const out = new Float64Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      out[i * 3 + j] =
        a[i * 3 + 0] * b[0 * 3 + j] +
        a[i * 3 + 1] * b[1 * 3 + j] +
        a[i * 3 + 2] * b[2 * 3 + j];
    }
  }
  return out;
}

function mat3Transpose(m: Mat3): Mat3 {
  const out = new Float64Array(9);
  out[0] = m[0]; out[1] = m[3]; out[2] = m[6];
  out[3] = m[1]; out[4] = m[4]; out[5] = m[7];
  out[6] = m[2]; out[7] = m[5]; out[8] = m[8];
  return out;
}

/**
 * Convert a 3x3 rotation matrix to axis-angle (rotation vector).
 * Matches scipy Rotation.from_matrix(R).as_rotvec().
 */
function mat3ToRotvec(R: Mat3): [number, number, number] {
  // angle = acos((trace - 1) / 2)
  const trace = R[0] + R[4] + R[8];
  const cosAngle = clamp((trace - 1) / 2, -1, 1);
  const angle = Math.acos(cosAngle);

  if (angle < 1e-10) {
    return [0, 0, 0];
  }

  if (Math.PI - angle < 1e-6) {
    // Near 180 degrees: extract axis from the symmetric part
    // Find the column of (R + I) with largest norm
    const Rp = [R[0] + 1, R[1], R[2], R[3], R[4] + 1, R[5], R[6], R[7], R[8] + 1];
    let bestCol = 0;
    let bestNorm = 0;
    for (let c = 0; c < 3; c++) {
      const n = Rp[c] ** 2 + Rp[3 + c] ** 2 + Rp[6 + c] ** 2;
      if (n > bestNorm) { bestNorm = n; bestCol = c; }
    }
    const invN = 1 / Math.sqrt(bestNorm);
    const ax = Rp[bestCol] * invN;
    const ay = Rp[3 + bestCol] * invN;
    const az = Rp[6 + bestCol] * invN;
    return [ax * angle, ay * angle, az * angle];
  }

  // General case: axis from antisymmetric part
  const s = 1 / (2 * Math.sin(angle));
  const ax = (R[7] - R[5]) * s; // R[2][1] - R[1][2]
  const ay = (R[2] - R[6]) * s; // R[0][2] - R[2][0]
  const az = (R[3] - R[1]) * s; // R[1][0] - R[0][1]
  return [ax * angle, ay * angle, az * angle];
}

/** Convert an axis-angle rotation vector to a 3x3 rotation matrix (Rodrigues). */
function rotvecToMat3(rv: [number, number, number]): Mat3 {
  const angle = Math.sqrt(rv[0] ** 2 + rv[1] ** 2 + rv[2] ** 2);

  if (angle < 1e-10) {
    const id = new Float64Array(9);
    id[0] = 1; id[4] = 1; id[8] = 1;
    return id;
  }

  const ax = rv[0] / angle;
  const ay = rv[1] / angle;
  const az = rv[2] / angle;
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const t = 1 - c;

  const out = new Float64Array(9);
  out[0] = t * ax * ax + c;
  out[1] = t * ax * ay - s * az;
  out[2] = t * ax * az + s * ay;
  out[3] = t * ay * ax + s * az;
  out[4] = t * ay * ay + c;
  out[5] = t * ay * az - s * ax;
  out[6] = t * az * ax - s * ay;
  out[7] = t * az * ay + s * ax;
  out[8] = t * az * az + c;
  return out;
}

/**
 * SVD re-orthonormalize the 3x3 rotation part of a Mat4 (in-place).
 * Uses a fast iterative polar decomposition (3 iterations suffice for small drift).
 */
function svdReorthonormalize3x3(m: Mat4): void {
  // Fast polar decomposition: R_new = 0.5 * (R + R^-T)
  // Converges quickly for near-orthonormal input.
  for (let iter = 0; iter < 3; iter++) {
    // Compute inverse-transpose (= adjugate / det for 3x3)
    const a = m[0], b = m[1], c = m[2];
    const d = m[4], e = m[5], f = m[6];
    const g = m[8], h = m[9], k = m[10];

    const det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
    if (Math.abs(det) < 1e-12) return;
    const invDet = 1 / det;

    // cofactor matrix transposed = adjugate, then divide by det
    const it00 = (e * k - f * h) * invDet;
    const it01 = (c * h - b * k) * invDet;
    const it02 = (b * f - c * e) * invDet;
    const it10 = (f * g - d * k) * invDet;
    const it11 = (a * k - c * g) * invDet;
    const it12 = (c * d - a * f) * invDet;
    const it20 = (d * h - e * g) * invDet;
    const it21 = (b * g - a * h) * invDet;
    const it22 = (a * e - b * d) * invDet;

    m[0]  = 0.5 * (a + it00);
    m[1]  = 0.5 * (b + it01);
    m[2]  = 0.5 * (c + it02);
    m[4]  = 0.5 * (d + it10);
    m[5]  = 0.5 * (e + it11);
    m[6]  = 0.5 * (f + it12);
    m[8]  = 0.5 * (g + it20);
    m[9]  = 0.5 * (h + it21);
    m[10] = 0.5 * (k + it22);
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
