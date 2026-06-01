export const gpuDepthSurfaceShader = /* wgsl */ `
const INF_U: u32 = 4294967295u;
const INF_F: f32 = 1.0e20;
const DEPTH_SCALE: f32 = 10000.0;
const DEPTH_INV_SCALE: f32 = 0.0001;
const EDGE_EMPTY: u32 = 0u;
const EDGE_T_SCALE: f32 = 4095.0;
const EDGE_T_MASK: u32 = 4095u;
const EDGE_DEPTH_SHIFT: u32 = 12u;
const EDGE_PRIORITY_SHIFT: u32 = 24u;
const EDGE_DEPTH_MAX: f32 = 32.0;
const TAU: f32 = 6.28318530718;
const MAX_SMOOTHING_HALF_SIZE: u32 = 8u;
const MAX_SPLAT_RADIUS_PIXELS: f32 = 192.0;
const MAX_BOUNDARY_RELAX: f32 = 10.0;

struct Uniforms {
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
  invView: mat4x4<f32>,
  captureView: mat4x4<f32>,
  captureProj: mat4x4<f32>,
  captureInvView: mat4x4<f32>,
  viewportGrid: vec4<f32>,
  params0: vec4<f32>,
  params1: vec4<f32>,
  params2: vec4<f32>,
}

struct SurfaceVertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) local: vec2<f32>,
  @location(3) depth: f32,
  @location(4) valid: f32,
  @location(5) barycentric: vec3<f32>,
}

struct ParticleVertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
  @location(1) alpha: f32,
}

struct FloorVertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) local: vec2<f32>,
}

struct MeshVertex {
  world: vec4<f32>,
  normalDepth: vec4<f32>,
  local: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> particlesCompute: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> rawDepth: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> tempDepth: array<u32>;
@group(0) @binding(4) var<storage, read_write> filteredDepthCompute: array<u32>;
@group(0) @binding(7) var<storage, read_write> horizontalEdgesCompute: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> verticalEdgesCompute: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> meshVerticesCompute: array<MeshVertex>;
@group(0) @binding(10) var<storage, read_write> indirectArgsCompute: array<atomic<u32>>;

@group(0) @binding(5) var<storage, read> particlesRender: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> filteredDepth: array<u32>;
@group(0) @binding(12) var<storage, read> meshVerticesRender: array<MeshVertex>;

fn viewportSize() -> vec2<f32> {
  return uniforms.viewportGrid.xy;
}

fn gridColumns() -> u32 {
  return u32(uniforms.viewportGrid.z);
}

fn gridRows() -> u32 {
  return u32(uniforms.viewportGrid.w);
}

fn gridCount() -> u32 {
  return gridColumns() * gridRows();
}

fn cellColumns() -> u32 {
  return max(1u, gridColumns() - 1u);
}

fn cellRows() -> u32 {
  return max(1u, gridRows() - 1u);
}

fn cellCount() -> u32 {
  return cellColumns() * cellRows();
}

fn horizontalEdgeCount() -> u32 {
  return cellColumns() * gridRows();
}

fn verticalEdgeCount() -> u32 {
  return gridColumns() * cellRows();
}

fn particleCount() -> u32 {
  return u32(uniforms.params1.x);
}

fn presetIndex() -> u32 {
  return u32(uniforms.params1.y);
}

fn smoothingHalfSize() -> u32 {
  return min(u32(uniforms.params1.z), MAX_SMOOTHING_HALF_SIZE);
}

fn renderMode() -> u32 {
  return u32(uniforms.params1.w);
}

fn frozenSurface() -> bool {
  return uniforms.params2.w > 0.5;
}

fn maxDepthJumpPacked() -> u32 {
  if (uniforms.params2.z > 0.5) {
    return max(1u, u32(max(uniforms.params2.y, DEPTH_INV_SCALE) * DEPTH_SCALE));
  }

  let threshold = uniforms.params0.y * 2.7;
  return max(1u, u32(threshold * DEPTH_SCALE));
}

fn packDepth(depth: f32) -> u32 {
  return u32(clamp(depth * DEPTH_SCALE, 0.0, 4294967040.0));
}

fn unpackDepth(depth: u32) -> f32 {
  if (depth >= INF_U) {
    return INF_F;
  }

  return f32(depth) * DEPTH_INV_SCALE;
}

fn edgeHasCut(edge: u32) -> bool {
  return edge != EDGE_EMPTY;
}

fn packEdgeCut(depth: f32, t: f32, priority: f32) -> u32 {
  let priorityBucket = u32(clamp(priority * 255.0, 1.0, 255.0));
  let depthBucket = u32(clamp(depth / EDGE_DEPTH_MAX, 0.0, 1.0) * EDGE_T_SCALE + 0.5);
  let depthRank = EDGE_T_MASK - depthBucket;
  let packedT = u32(clamp(t, 0.0, 1.0) * EDGE_T_SCALE + 0.5);
  return (priorityBucket << EDGE_PRIORITY_SHIFT) | (depthRank << EDGE_DEPTH_SHIFT) | packedT;
}

fn unpackEdgeT(edge: u32) -> f32 {
  if (!edgeHasCut(edge)) {
    return 0.5;
  }

  return f32(edge & EDGE_T_MASK) / EDGE_T_SCALE;
}

fn unpackEdgeDepth(edge: u32) -> f32 {
  if (!edgeHasCut(edge)) {
    return INF_F;
  }

  let depthRank = (edge >> EDGE_DEPTH_SHIFT) & EDGE_T_MASK;
  return f32(EDGE_T_MASK - depthRank) * EDGE_DEPTH_MAX / EDGE_T_SCALE;
}

fn closeDepth(a: u32, b: u32, threshold: u32) -> bool {
  let hi = max(a, b);
  let lo = min(a, b);
  return hi - lo <= threshold;
}

fn random01(index: u32, salt: f32) -> f32 {
  let value = sin(f32(index) * salt + salt * 17.17) * 43758.5453;
  return fract(value);
}

fn vortexPoint(index: u32, time: f32) -> vec3<f32> {
  let u = random01(index, 12.9898);
  let v = random01(index, 78.233);
  let w = random01(index, 37.719);
  let phase = random01(index, 91.423) * TAU;
  let twist = time * 0.48 + phase;
  let strand = u * TAU * 3.4 + twist;
  let lift = (v - 0.5) * 2.45;
  let pulse = sin(time * 1.7 + w * TAU) * 0.16;
  let radius = 0.78 + w * 0.42 + pulse;

  return vec3<f32>(
    cos(strand) * radius + sin(lift * 2.1 + time) * 0.16,
    lift + sin(strand * 1.7 + time * 0.8) * 0.22,
    sin(strand) * radius * 0.72 + cos(u * TAU + time) * 0.18
  );
}

fn sheetPoint(index: u32, time: f32) -> vec3<f32> {
  let u = random01(index, 12.9898);
  let v = random01(index, 78.233);
  let w = random01(index, 37.719);
  let phase = random01(index, 91.423) * TAU;
  let band = select(-1.0, 1.0, random01(index, 11.135) > 0.55);
  let xBase = (u - 0.5) * 2.85;
  let yBase = 1.35 - v * 2.75;
  let wake = sin(u * 11.0 + time * 1.8) * 0.12;
  let fold = cos(v * 8.5 + time * 1.15 + phase) * 0.2;
  let pinch = exp(-abs(xBase) * 1.8) * sin(time * 2.0 + v * 5.0);

  return vec3<f32>(
    xBase + wake * band,
    yBase + sin(u * TAU + time + phase) * 0.12,
    fold + pinch * 0.35 + (w - 0.5) * 0.18
  );
}

fn crownPoint(index: u32, time: f32) -> vec3<f32> {
  let u = random01(index, 12.9898);
  let v = random01(index, 78.233);
  let w = random01(index, 37.719);
  let phase = random01(index, 91.423) * TAU;
  let wave = fract(time * 0.33 + v);
  let angle = u * TAU + sin(time * 0.8 + phase) * 0.45;
  let burst = sin(wave * 3.14159265359);
  let ring = 0.22 + wave * 1.55 + w * 0.18;
  let column = max(0.0, 1.0 - wave * 1.2);

  return vec3<f32>(
    cos(angle) * ring + cos(phase + time) * 0.08,
    -0.8 + burst * 1.95 + column * 1.1 + (w - 0.5) * 0.16,
    sin(angle) * ring * 0.72 + sin(phase * 1.7) * 0.12
  );
}

fn particlePosition(index: u32, time: f32) -> vec3<f32> {
  if (presetIndex() == 1u) {
    return sheetPoint(index, time);
  }

  if (presetIndex() == 2u) {
    return crownPoint(index, time);
  }

  return vortexPoint(index, time);
}

fn weightFor(halfSize: u32, offset: u32) -> f32 {
  if (halfSize == 1u) {
    return select(2.0, 1.0, offset == 1u);
  }

  if (halfSize == 2u) {
    if (offset == 0u) { return 6.0; }
    if (offset == 1u) { return 4.0; }
    return 1.0;
  }

  if (halfSize == 3u) {
    if (offset == 0u) { return 20.0; }
    if (offset == 1u) { return 15.0; }
    if (offset == 2u) { return 6.0; }
    return 1.0;
  }

  if (halfSize == 4u) {
    if (offset == 0u) { return 70.0; }
    if (offset == 1u) { return 56.0; }
    if (offset == 2u) { return 28.0; }
    if (offset == 3u) { return 8.0; }
    return 1.0;
  }

  if (halfSize == 5u) {
    if (offset == 0u) { return 252.0; }
    if (offset == 1u) { return 210.0; }
    if (offset == 2u) { return 120.0; }
    if (offset == 3u) { return 45.0; }
    if (offset == 4u) { return 10.0; }
    return 1.0;
  }

  if (halfSize == 6u) {
    if (offset == 0u) { return 924.0; }
    if (offset == 1u) { return 792.0; }
    if (offset == 2u) { return 495.0; }
    if (offset == 3u) { return 220.0; }
    if (offset == 4u) { return 66.0; }
    if (offset == 5u) { return 12.0; }
    return 1.0;
  }

  if (halfSize == 7u) {
    if (offset == 0u) { return 3432.0; }
    if (offset == 1u) { return 3003.0; }
    if (offset == 2u) { return 2002.0; }
    if (offset == 3u) { return 1001.0; }
    if (offset == 4u) { return 364.0; }
    if (offset == 5u) { return 91.0; }
    if (offset == 6u) { return 14.0; }
    return 1.0;
  }

  if (offset == 0u) { return 12870.0; }
  if (offset == 1u) { return 11440.0; }
  if (offset == 2u) { return 8008.0; }
  if (offset == 3u) { return 4368.0; }
  if (offset == 4u) { return 1820.0; }
  if (offset == 5u) { return 560.0; }
  if (offset == 6u) { return 120.0; }
  if (offset == 7u) { return 16.0; }
  return 1.0;
}

@compute @workgroup_size(256)
fn clearDepth(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  let maxCount = max(gridCount(), max(horizontalEdgeCount(), verticalEdgeCount()));

  if (index >= maxCount) {
    return;
  }

  if (index < gridCount()) {
    atomicStore(&rawDepth[index], INF_U);
    tempDepth[index] = INF_U;
    filteredDepthCompute[index] = INF_U;
  }

  if (index < horizontalEdgeCount()) {
    atomicStore(&horizontalEdgesCompute[index], EDGE_EMPTY);
  }

  if (index < verticalEdgeCount()) {
    atomicStore(&verticalEdgesCompute[index], EDGE_EMPTY);
  }

  if (index == 0u) {
    atomicStore(&indirectArgsCompute[0], 0u);
    atomicStore(&indirectArgsCompute[1], 1u);
    atomicStore(&indirectArgsCompute[2], 0u);
    atomicStore(&indirectArgsCompute[3], 0u);
  }
}

fn candidateBelongsToFront(a: u32, b: u32, candidateDepth: f32) -> bool {
  let aValid = a < INF_U;
  let bValid = b < INF_U;

  if (aValid && bValid) {
    return candidateDepth < (unpackDepth(a) + unpackDepth(b)) * 0.5;
  }

  if (aValid != bValid) {
    let validDepth = unpackDepth(select(b, a, aValid));
    return candidateDepth <= validDepth + f32(maxDepthJumpPacked()) * DEPTH_INV_SCALE;
  }

  return false;
}

fn edgeCandidatePriority(a: u32, b: u32, t: f32) -> f32 {
  let aValid = a < INF_U;
  let bValid = b < INF_U;
  let frontIsA = aValid && (!bValid || a <= b);
  return select(1.0 - t, t, frontIsA);
}

fn storeHorizontalEdgeCut(edgeCellX: i32, row: i32, t: f32, depth: f32) {
  if (row < 0 || row >= i32(gridRows()) || edgeCellX < 0 || edgeCellX >= i32(cellColumns())) {
    return;
  }

  let index = u32(row) * cellColumns() + u32(edgeCellX);
  let a = atomicLoad(&rawDepth[u32(row) * gridColumns() + u32(edgeCellX)]);
  let b = atomicLoad(&rawDepth[u32(row) * gridColumns() + u32(edgeCellX + 1)]);

  if (edgeNeedsCut(a, b) && candidateBelongsToFront(a, b, depth)) {
    atomicMax(&horizontalEdgesCompute[index], packEdgeCut(depth, t, edgeCandidatePriority(a, b, t)));
  }
}

fn storeVerticalEdgeCut(column: i32, edgeCellY: i32, t: f32, depth: f32) {
  if (column < 0 || column >= i32(gridColumns()) || edgeCellY < 0 || edgeCellY >= i32(cellRows())) {
    return;
  }

  let index = u32(edgeCellY) * gridColumns() + u32(column);
  let a = atomicLoad(&rawDepth[u32(edgeCellY) * gridColumns() + u32(column)]);
  let b = atomicLoad(&rawDepth[u32(edgeCellY + 1) * gridColumns() + u32(column)]);

  if (edgeNeedsCut(a, b) && candidateBelongsToFront(a, b, depth)) {
    atomicMax(&verticalEdgesCompute[index], packEdgeCut(depth, t, edgeCandidatePriority(a, b, t)));
  }
}

@compute @workgroup_size(64)
fn projectParticles(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;

  if (index >= particleCount()) {
    return;
  }

  let time = uniforms.params0.x;
  let world = particlePosition(index, time);
  particlesCompute[index] = vec4<f32>(world, 1.0);

  let viewPosition = uniforms.captureView * vec4<f32>(world, 1.0);
  let cameraDepth = -viewPosition.z;

  if (cameraDepth <= 0.1 || cameraDepth >= 30.0) {
    return;
  }

  let clip = uniforms.captureProj * viewPosition;

  if (clip.w <= 0.0) {
    return;
  }

  let ndc = clip.xyz / clip.w;

  if (ndc.x < -1.15 || ndc.x > 1.15 || ndc.y < -1.15 || ndc.y > 1.15 || ndc.z < -1.2 || ndc.z > 1.2) {
    return;
  }

  let viewport = viewportSize();
  let screen = vec2<f32>(
    (ndc.x * 0.5 + 0.5) * viewport.x,
    (1.0 - (ndc.y * 0.5 + 0.5)) * viewport.y
  );
  let radiusWorld = uniforms.params0.y;
  let gridSpacing = uniforms.params0.z;
  let radiusPixels = clamp((radiusWorld * viewport.y * uniforms.captureProj[1][1]) / (2.0 * cameraDepth), 4.0, MAX_SPLAT_RADIUS_PIXELS);

  if (
    screen.x < -radiusPixels ||
    screen.x > viewport.x + radiusPixels ||
    screen.y < -radiusPixels ||
    screen.y > viewport.y + radiusPixels
  ) {
    return;
  }

  let columns = i32(gridColumns());
  let rows = i32(gridRows());
  let minX = max(0, i32(floor((screen.x - radiusPixels) / gridSpacing)));
  let maxX = min(columns - 1, i32(ceil((screen.x + radiusPixels) / gridSpacing)));
  let minY = max(0, i32(floor((screen.y - radiusPixels) / gridSpacing)));
  let maxY = min(rows - 1, i32(ceil((screen.y + radiusPixels) / gridSpacing)));
  let radiusSquared = radiusPixels * radiusPixels;

  for (var y = minY; y <= maxY; y = y + 1) {
    let nodeY = f32(y) * gridSpacing;
    let dy = nodeY - screen.y;

    for (var x = minX; x <= maxX; x = x + 1) {
      let nodeX = f32(x) * gridSpacing;
      let dx = nodeX - screen.x;
      let distanceSquared = dx * dx + dy * dy;

      if (distanceSquared <= radiusSquared) {
        let bulge = sqrt(max(0.0, 1.0 - distanceSquared / radiusSquared));
        let frontDepth = cameraDepth - radiusWorld * 0.92 * bulge;
        let nodeIndex = u32(y) * gridColumns() + u32(x);
        atomicMin(&rawDepth[nodeIndex], packDepth(frontDepth));
      }
    }
  }

}

@compute @workgroup_size(64)
fn selectSilhouetteCandidates(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;

  if (index >= particleCount()) {
    return;
  }

  let world = particlesCompute[index].xyz;
  let viewPosition = uniforms.captureView * vec4<f32>(world, 1.0);
  let cameraDepth = -viewPosition.z;

  if (cameraDepth <= 0.1 || cameraDepth >= 30.0) {
    return;
  }

  let clip = uniforms.captureProj * viewPosition;

  if (clip.w <= 0.0) {
    return;
  }

  let ndc = clip.xyz / clip.w;

  if (ndc.x < -1.15 || ndc.x > 1.15 || ndc.y < -1.15 || ndc.y > 1.15 || ndc.z < -1.2 || ndc.z > 1.2) {
    return;
  }

  let viewport = viewportSize();
  let screen = vec2<f32>(
    (ndc.x * 0.5 + 0.5) * viewport.x,
    (1.0 - (ndc.y * 0.5 + 0.5)) * viewport.y
  );
  let gridSpacing = uniforms.params0.z;
  let radiusPixels = clamp((uniforms.params0.y * viewport.y * uniforms.captureProj[1][1]) / (2.0 * cameraDepth), 4.0, MAX_SPLAT_RADIUS_PIXELS);

  if (
    screen.x < -radiusPixels ||
    screen.x > viewport.x + radiusPixels ||
    screen.y < -radiusPixels ||
    screen.y > viewport.y + radiusPixels
  ) {
    return;
  }

  let columns = i32(gridColumns());
  let rows = i32(gridRows());
  let minX = max(0, i32(floor((screen.x - radiusPixels) / gridSpacing)));
  let maxX = min(columns - 1, i32(ceil((screen.x + radiusPixels) / gridSpacing)));
  let minY = max(0, i32(floor((screen.y - radiusPixels) / gridSpacing)));
  let maxY = min(rows - 1, i32(ceil((screen.y + radiusPixels) / gridSpacing)));
  let radiusSquared = radiusPixels * radiusPixels;

  for (var y = minY; y <= maxY; y = y + 1) {
    let nodeY = f32(y) * gridSpacing;
    let dy = nodeY - screen.y;
    let remaining = radiusSquared - dy * dy;

    if (remaining >= 0.0) {
      let root = sqrt(remaining);
      let leftX = screen.x - root;
      let rightX = screen.x + root;
      let leftCell = i32(floor(leftX / gridSpacing));
      let rightCell = i32(floor(rightX / gridSpacing));
      let leftT = (leftX - f32(leftCell) * gridSpacing) / gridSpacing;
      let rightT = (rightX - f32(rightCell) * gridSpacing) / gridSpacing;
      storeHorizontalEdgeCut(leftCell, y, leftT, cameraDepth);

      if (rightCell != leftCell) {
        storeHorizontalEdgeCut(rightCell, y, rightT, cameraDepth);
      }
    }
  }

  for (var x = minX; x <= maxX; x = x + 1) {
    let nodeX = f32(x) * gridSpacing;
    let dx = nodeX - screen.x;
    let remaining = radiusSquared - dx * dx;

    if (remaining >= 0.0) {
      let root = sqrt(remaining);
      let topY = screen.y - root;
      let bottomY = screen.y + root;
      let topCell = i32(floor(topY / gridSpacing));
      let bottomCell = i32(floor(bottomY / gridSpacing));
      let topT = (topY - f32(topCell) * gridSpacing) / gridSpacing;
      let bottomT = (bottomY - f32(bottomCell) * gridSpacing) / gridSpacing;
      storeVerticalEdgeCut(x, topCell, topT, cameraDepth);

      if (bottomCell != topCell) {
        storeVerticalEdgeCut(x, bottomCell, bottomT, cameraDepth);
      }
    }
  }
}

@compute @workgroup_size(256)
fn smoothHorizontal(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;

  if (index >= gridCount()) {
    return;
  }

  let columns = gridColumns();
  let rows = gridRows();
  let x = index % columns;
  let y = index / columns;
  let halfSize = smoothingHalfSize();
  let center = atomicLoad(&rawDepth[index]);

  if (center >= INF_U) {
    tempDepth[index] = INF_U;
    return;
  }

  if (halfSize == 0u) {
    tempDepth[index] = center;
    return;
  }

  var total = f32(center) * weightFor(halfSize, 0u);
  var totalWeight = weightFor(halfSize, 0u);
  let threshold = maxDepthJumpPacked();

  for (var step = 1u; step <= MAX_SMOOTHING_HALF_SIZE; step = step + 1u) {
    if (step > halfSize || x < step || x + step >= columns) {
      continue;
    }

    let a = atomicLoad(&rawDepth[y * columns + x - step]);
    let b = atomicLoad(&rawDepth[y * columns + x + step]);

    if (a < INF_U && b < INF_U && closeDepth(a, center, threshold) && closeDepth(b, center, threshold)) {
      let weight = weightFor(halfSize, step);
      total = total + f32(a + b) * weight;
      totalWeight = totalWeight + weight * 2.0;
    }
  }

  tempDepth[index] = u32(total / totalWeight + 0.5);
}

@compute @workgroup_size(256)
fn smoothVertical(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;

  if (index >= gridCount()) {
    return;
  }

  let columns = gridColumns();
  let rows = gridRows();
  let x = index % columns;
  let y = index / columns;
  let halfSize = smoothingHalfSize();
  let center = tempDepth[index];

  if (center >= INF_U) {
    filteredDepthCompute[index] = INF_U;
    return;
  }

  if (halfSize == 0u) {
    filteredDepthCompute[index] = center;
    return;
  }

  var total = f32(center) * weightFor(halfSize, 0u);
  var totalWeight = weightFor(halfSize, 0u);
  let threshold = maxDepthJumpPacked();

  for (var step = 1u; step <= MAX_SMOOTHING_HALF_SIZE; step = step + 1u) {
    if (step > halfSize || y < step || y + step >= rows) {
      continue;
    }

    let a = tempDepth[(y - step) * columns + x];
    let b = tempDepth[(y + step) * columns + x];

    if (a < INF_U && b < INF_U && closeDepth(a, center, threshold) && closeDepth(b, center, threshold)) {
      let weight = weightFor(halfSize, step);
      total = total + f32(a + b) * weight;
      totalWeight = totalWeight + weight * 2.0;
    }
  }

  filteredDepthCompute[index] = u32(total / totalWeight + 0.5);
}

@compute @workgroup_size(256)
fn closeDepthGaps(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;

  if (index >= gridCount()) {
    return;
  }

  let strength = clamp(uniforms.params2.x, 0.0, MAX_BOUNDARY_RELAX);
  let center = filteredDepthCompute[index];

  if (strength < 0.5 || center >= INF_U) {
    tempDepth[index] = center;
    return;
  }

  let columns = gridColumns();
  let rows = gridRows();
  let x = i32(index % columns);
  let y = i32(index / columns);
  let radius = 1;
  let threshold = maxDepthJumpPacked();
  var total = 0.0;
  var totalWeight = 0.0;
  var count = 0u;

  for (var oy = -2; oy <= 2; oy = oy + 1) {
    if (abs(oy) > radius) {
      continue;
    }

    let ny = y + oy;

    if (ny < 0 || ny >= i32(rows)) {
      continue;
    }

    for (var ox = -2; ox <= 2; ox = ox + 1) {
      if (abs(ox) > radius || (ox == 0 && oy == 0)) {
        continue;
      }

      let nx = x + ox;

      if (nx < 0 || nx >= i32(columns)) {
        continue;
      }

      let sample = filteredDepthCompute[u32(ny) * columns + u32(nx)];

      if (sample >= INF_U) {
        continue;
      }

      if (!closeDepth(center, sample, threshold)) {
        continue;
      }

      let distanceWeight = 1.0 / (1.0 + f32(abs(ox) + abs(oy)));
      total = total + f32(sample) * distanceWeight;
      totalWeight = totalWeight + distanceWeight;
      count = count + 1u;
    }
  }

  if (count >= 2u && totalWeight > 0.0) {
    let blend = min(0.14, strength * 0.025);
    tempDepth[index] = u32(mix(f32(center), total / totalWeight, blend) + 0.5);
  } else {
    tempDepth[index] = center;
  }
}

@compute @workgroup_size(256)
fn commitClosedDepth(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;

  if (index >= gridCount()) {
    return;
  }

  filteredDepthCompute[index] = tempDepth[index];
}

fn edgeNeedsCut(a: u32, b: u32) -> bool {
  let aValid = a < INF_U;
  let bValid = b < INF_U;

  if (!aValid && !bValid) {
    return false;
  }

  return (aValid != bValid) || (aValid && bValid && !closeDepth(a, b, maxDepthJumpPacked()));
}

@compute @workgroup_size(256)
fn detectSilhouettes(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;

  if (index < horizontalEdgeCount()) {
    let x = index % cellColumns();
    let y = index / cellColumns();
    let a = filteredDepthCompute[y * gridColumns() + x];
    let b = filteredDepthCompute[y * gridColumns() + x + 1u];

    if (edgeNeedsCut(a, b) && !edgeHasCut(atomicLoad(&horizontalEdgesCompute[index]))) {
      atomicStore(&horizontalEdgesCompute[index], packEdgeCut(unpackDepth(min(a, b)), 0.5, 0.001));
    }
  }

  if (index < verticalEdgeCount()) {
    let x = index % gridColumns();
    let y = index / gridColumns();
    let a = filteredDepthCompute[y * gridColumns() + x];
    let b = filteredDepthCompute[(y + 1u) * gridColumns() + x];

    if (edgeNeedsCut(a, b) && !edgeHasCut(atomicLoad(&verticalEdgesCompute[index]))) {
      atomicStore(&verticalEdgesCompute[index], packEdgeCut(unpackDepth(min(a, b)), 0.5, 0.001));
    }
  }
}

fn horizontalEdge(cellX: u32, y: u32) -> u32 {
  if (cellX >= cellColumns() || y >= gridRows()) {
    return INF_U;
  }

  return atomicLoad(&horizontalEdgesCompute[y * cellColumns() + cellX]);
}

fn verticalEdge(x: u32, cellY: u32) -> u32 {
  if (x >= gridColumns() || cellY >= cellRows()) {
    return INF_U;
  }

  return atomicLoad(&verticalEdgesCompute[cellY * gridColumns() + x]);
}

fn normalFromScreen(screen: vec2<f32>, depth: f32) -> vec3<f32> {
  let spacing = uniforms.params0.z;
  let centerNode = vec2<u32>(
    min(gridColumns() - 1u, u32(max(0.0, round(screen.x / spacing)))),
    min(gridRows() - 1u, u32(max(0.0, round(screen.y / spacing))))
  );
  let leftScreen = screen + vec2<f32>(-spacing, 0.0);
  let rightScreen = screen + vec2<f32>(spacing, 0.0);
  let upScreen = screen + vec2<f32>(0.0, -spacing);
  let downScreen = screen + vec2<f32>(0.0, spacing);
  let leftDepth = unpackDepth(gridDepth(select(centerNode.x, centerNode.x - 1u, centerNode.x > 0u), centerNode.y));
  let rightDepth = unpackDepth(gridDepth(select(centerNode.x, centerNode.x + 1u, centerNode.x + 1u < gridColumns()), centerNode.y));
  let upDepth = unpackDepth(gridDepth(centerNode.x, select(centerNode.y, centerNode.y - 1u, centerNode.y > 0u)));
  let downDepth = unpackDepth(gridDepth(centerNode.x, select(centerNode.y, centerNode.y + 1u, centerNode.y + 1u < gridRows())));
  let px0 = unproject(leftScreen, select(depth, leftDepth, leftDepth < INF_F));
  let px1 = unproject(rightScreen, select(depth, rightDepth, rightDepth < INF_F));
  let py0 = unproject(upScreen, select(depth, upDepth, upDepth < INF_F));
  let py1 = unproject(downScreen, select(depth, downDepth, downDepth < INF_F));
  var normal = normalize(cross(px1 - px0, py1 - py0));
  let cameraPosition = uniforms.captureInvView[3].xyz;
  let centerWorld = unproject(screen, depth);

  if (dot(normal, normalize(cameraPosition - centerWorld)) < 0.0) {
    normal = -normal;
  }

  return normal;
}

fn writeMeshVertex(index: u32, screen: vec2<f32>, depth: f32, local: vec2<f32>, barycentric: vec3<f32>) {
  if (index >= arrayLength(&meshVerticesCompute)) {
    return;
  }

  let world = unproject(screen, depth);
  meshVerticesCompute[index] = MeshVertex(
    vec4<f32>(world, 1.0),
    vec4<f32>(normalFromScreen(screen, depth), depth),
    vec4<f32>(local, barycentric.xy)
  );
}

fn emitTriangle(
  screenA: vec2<f32>,
  depthA: f32,
  localA: vec2<f32>,
  screenB: vec2<f32>,
  depthB: f32,
  localB: vec2<f32>,
  screenC: vec2<f32>,
  depthC: f32,
  localC: vec2<f32>
) {
  let base = atomicAdd(&indirectArgsCompute[0], 3u);
  writeMeshVertex(base, screenA, depthA, localA, vec3<f32>(1.0, 0.0, 0.0));
  writeMeshVertex(base + 1u, screenB, depthB, localB, vec3<f32>(0.0, 1.0, 0.0));
  writeMeshVertex(base + 2u, screenC, depthC, localC, vec3<f32>(0.0, 0.0, 1.0));
}

fn addPolygonPoint(
  count: ptr<function, u32>,
  screens: ptr<function, array<vec2<f32>, 8>>,
  depths: ptr<function, array<f32, 8>>,
  locals: ptr<function, array<vec2<f32>, 8>>,
  screen: vec2<f32>,
  depth: f32,
  local: vec2<f32>
) {
  let index = *count;

  if (index >= 8u) {
    return;
  }

  (*screens)[index] = screen;
  (*depths)[index] = depth;
  (*locals)[index] = local;
  *count = index + 1u;
}

fn edgeCutPoint(
  edgeIndex: u32,
  cellX: u32,
  cellY: u32,
  corners: array<vec2<f32>, 4>,
  depths: array<f32, 4>,
  edge: u32
) -> vec4<f32> {
  let t = mix(unpackEdgeT(edge), 0.5, clamp(uniforms.params2.x, 0.0, MAX_BOUNDARY_RELAX) * 0.025);
  let candidateDepth = unpackEdgeDepth(edge);

  if (edgeIndex == 0u) {
    let screen = mix(corners[0], corners[1], t);
    return vec4<f32>(screen, select(min(depths[0], depths[1]), candidateDepth, candidateDepth < INF_F), t);
  }

  if (edgeIndex == 1u) {
    let screen = mix(corners[1], corners[2], t);
    return vec4<f32>(screen, select(min(depths[1], depths[2]), candidateDepth, candidateDepth < INF_F), t);
  }

  if (edgeIndex == 2u) {
    let screen = mix(corners[3], corners[2], t);
    return vec4<f32>(screen, select(min(depths[3], depths[2]), candidateDepth, candidateDepth < INF_F), t);
  }

  let screen = mix(corners[0], corners[3], t);
  return vec4<f32>(screen, select(min(depths[0], depths[3]), candidateDepth, candidateDepth < INF_F), t);
}

fn edgeLocal(edgeIndex: u32, t: f32) -> vec2<f32> {
  if (edgeIndex == 0u) { return vec2<f32>(t, 0.0); }
  if (edgeIndex == 1u) { return vec2<f32>(1.0, t); }
  if (edgeIndex == 2u) { return vec2<f32>(t, 1.0); }
  return vec2<f32>(0.0, t);
}

fn maskHasCorner(mask: u32, corner: u32) -> bool {
  return (mask & (1u << corner)) != 0u;
}

fn edgeStartCorner(edgeIndex: u32) -> u32 {
  if (edgeIndex == 0u) { return 0u; }
  if (edgeIndex == 1u) { return 1u; }
  if (edgeIndex == 2u) { return 3u; }
  return 0u;
}

fn edgeEndCorner(edgeIndex: u32) -> u32 {
  if (edgeIndex == 0u) { return 1u; }
  if (edgeIndex == 1u) { return 2u; }
  if (edgeIndex == 2u) { return 2u; }
  return 3u;
}

fn layerCutPoint(
  edgeIndex: u32,
  mask: u32,
  frontLayer: bool,
  corners: array<vec2<f32>, 4>,
  depths: array<f32, 4>,
  edge: u32
) -> vec4<f32> {
  let t = mix(unpackEdgeT(edge), 0.5, clamp(uniforms.params2.x, 0.0, MAX_BOUNDARY_RELAX) * 0.025);
  let candidateDepth = unpackEdgeDepth(edge);
  let startCorner = edgeStartCorner(edgeIndex);
  let endCorner = edgeEndCorner(edgeIndex);
  let startInside = maskHasCorner(mask, startCorner);
  let layerCorner = select(endCorner, startCorner, startInside);
  let otherCorner = select(startCorner, endCorner, startInside);
  let layerDepth = depths[layerCorner];
  let otherDepth = depths[otherCorner];
  let cutDepth = select(layerDepth, candidateDepth, frontLayer && candidateDepth < INF_F && layerDepth <= otherDepth);

  if (edgeIndex == 0u) {
    let screen = mix(corners[0], corners[1], t);
    return vec4<f32>(screen, cutDepth, t);
  }

  if (edgeIndex == 1u) {
    let screen = mix(corners[1], corners[2], t);
    return vec4<f32>(screen, cutDepth, t);
  }

  if (edgeIndex == 2u) {
    let screen = mix(corners[3], corners[2], t);
    return vec4<f32>(screen, cutDepth, t);
  }

  let screen = mix(corners[0], corners[3], t);
  return vec4<f32>(screen, cutDepth, t);
}

fn emitLayerMask(
  mask: u32,
  frontLayer: bool,
  corners: array<vec2<f32>, 4>,
  depths: array<f32, 4>,
  locals: array<vec2<f32>, 4>,
  edges: array<u32, 4>
) {
  if (mask == 0u) {
    return;
  }

  if (mask == 15u) {
    emitTriangle(corners[0], depths[0], locals[0], corners[1], depths[1], locals[1], corners[2], depths[2], locals[2]);
    emitTriangle(corners[0], depths[0], locals[0], corners[2], depths[2], locals[2], corners[3], depths[3], locals[3]);
    return;
  }

  if (mask == 5u || mask == 10u) {
    for (var corner = 0u; corner < 4u; corner = corner + 1u) {
      if (!maskHasCorner(mask, corner)) {
        continue;
      }

      let prevEdge = select(3u, corner - 1u, corner > 0u);
      let nextEdge = corner;
      let prevCut = layerCutPoint(prevEdge, mask, frontLayer, corners, depths, edges[prevEdge]);
      let nextCut = layerCutPoint(nextEdge, mask, frontLayer, corners, depths, edges[nextEdge]);
      emitTriangle(prevCut.xy, prevCut.z, edgeLocal(prevEdge, prevCut.w), corners[corner], depths[corner], locals[corner], nextCut.xy, nextCut.z, edgeLocal(nextEdge, nextCut.w));
    }
    return;
  }

  var polyScreens: array<vec2<f32>, 8>;
  var polyDepths: array<f32, 8>;
  var polyLocals: array<vec2<f32>, 8>;
  var count = 0u;

  for (var corner = 0u; corner < 4u; corner = corner + 1u) {
    let nextCorner = (corner + 1u) & 3u;
    let inside = maskHasCorner(mask, corner);
    let nextInside = maskHasCorner(mask, nextCorner);

    if (inside) {
      addPolygonPoint(&count, &polyScreens, &polyDepths, &polyLocals, corners[corner], depths[corner], locals[corner]);
    }

    if (inside != nextInside) {
      let cut = layerCutPoint(corner, mask, frontLayer, corners, depths, edges[corner]);
      addPolygonPoint(&count, &polyScreens, &polyDepths, &polyLocals, cut.xy, cut.z, edgeLocal(corner, cut.w));
    }
  }

  if (count < 3u) {
    return;
  }

  for (var point = 1u; point + 1u < count; point = point + 1u) {
    emitTriangle(
      polyScreens[0], polyDepths[0], polyLocals[0],
      polyScreens[point], polyDepths[point], polyLocals[point],
      polyScreens[point + 1u], polyDepths[point + 1u], polyLocals[point + 1u]
    );
  }
}

@compute @workgroup_size(128)
fn emitMesh(@builtin(global_invocation_id) id: vec3<u32>) {
  let cellIndex = id.x;

  if (cellIndex >= cellCount()) {
    return;
  }

  let cx = cellIndex % cellColumns();
  let cy = cellIndex / cellColumns();
  let spacing = uniforms.params0.z;
  let baseScreen = vec2<f32>(f32(cx), f32(cy)) * spacing;
  let corners = array<vec2<f32>, 4>(
    baseScreen,
    baseScreen + vec2<f32>(spacing, 0.0),
    baseScreen + vec2<f32>(spacing, spacing),
    baseScreen + vec2<f32>(0.0, spacing)
  );
  let locals = array<vec2<f32>, 4>(
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 1.0)
  );
  let d0u = gridDepth(cx, cy);
  let d1u = gridDepth(cx + 1u, cy);
  let d2u = gridDepth(cx + 1u, cy + 1u);
  let d3u = gridDepth(cx, cy + 1u);
  let valid = array<bool, 4>(d0u < INF_U, d1u < INF_U, d2u < INF_U, d3u < INF_U);

  if (!valid[0] && !valid[1] && !valid[2] && !valid[3]) {
    return;
  }

  let depths = array<f32, 4>(unpackDepth(d0u), unpackDepth(d1u), unpackDepth(d2u), unpackDepth(d3u));
  let minDepth = min(min(select(INF_F, depths[0], valid[0]), select(INF_F, depths[1], valid[1])), min(select(INF_F, depths[2], valid[2]), select(INF_F, depths[3], valid[3])));
  let edges = array<u32, 4>(
    horizontalEdge(cx, cy),
    verticalEdge(cx + 1u, cy),
    horizontalEdge(cx, cy + 1u),
    verticalEdge(cx, cy)
  );
  let cuts = array<bool, 4>(
    edgeNeedsCut(d0u, d1u),
    edgeNeedsCut(d1u, d2u),
    edgeNeedsCut(d3u, d2u),
    edgeNeedsCut(d0u, d3u)
  );
  let validMask =
    select(0u, 1u, valid[0]) |
    select(0u, 2u, valid[1]) |
    select(0u, 4u, valid[2]) |
    select(0u, 8u, valid[3]);

  if (validMask == 15u && !cuts[0] && !cuts[1] && !cuts[2] && !cuts[3]) {
    if (((cx + cy) & 1u) == 0u) {
      emitTriangle(corners[0], depths[0], locals[0], corners[1], depths[1], locals[1], corners[2], depths[2], locals[2]);
      emitTriangle(corners[0], depths[0], locals[0], corners[2], depths[2], locals[2], corners[3], depths[3], locals[3]);
    } else {
      emitTriangle(corners[0], depths[0], locals[0], corners[1], depths[1], locals[1], corners[3], depths[3], locals[3]);
      emitTriangle(corners[1], depths[1], locals[1], corners[2], depths[2], locals[2], corners[3], depths[3], locals[3]);
    }
    return;
  }

  let threshold = f32(maxDepthJumpPacked()) * DEPTH_INV_SCALE;
  let frontMask =
    select(0u, 1u, valid[0] && depths[0] <= minDepth + threshold) |
    select(0u, 2u, valid[1] && depths[1] <= minDepth + threshold) |
    select(0u, 4u, valid[2] && depths[2] <= minDepth + threshold) |
    select(0u, 8u, valid[3] && depths[3] <= minDepth + threshold);
  let backMask = validMask & (~frontMask);

  emitLayerMask(frontMask, true, corners, depths, locals, edges);

  if (validMask == 15u) {
    emitLayerMask(backMask, false, corners, depths, locals, edges);
  }
}

@compute @workgroup_size(1)
fn finalizeMesh() {
  let capacity = (arrayLength(&meshVerticesCompute) / 3u) * 3u;
  let vertexCount = (min(atomicLoad(&indirectArgsCompute[0]), capacity) / 3u) * 3u;
  atomicStore(&indirectArgsCompute[0], vertexCount);
  atomicStore(&indirectArgsCompute[1], 1u);
  atomicStore(&indirectArgsCompute[2], 0u);
  atomicStore(&indirectArgsCompute[3], 0u);
}

fn quadCorner(corner: u32) -> vec2<f32> {
  if (corner == 0u) { return vec2<f32>(0.0, 0.0); }
  if (corner == 1u) { return vec2<f32>(1.0, 0.0); }
  if (corner == 2u) { return vec2<f32>(1.0, 1.0); }
  if (corner == 3u) { return vec2<f32>(0.0, 0.0); }
  if (corner == 4u) { return vec2<f32>(1.0, 1.0); }
  return vec2<f32>(0.0, 1.0);
}

fn quadBillboardCorner(corner: u32) -> vec2<f32> {
  if (corner == 0u) { return vec2<f32>(-1.0, -1.0); }
  if (corner == 1u) { return vec2<f32>(1.0, -1.0); }
  if (corner == 2u) { return vec2<f32>(1.0, 1.0); }
  if (corner == 3u) { return vec2<f32>(-1.0, -1.0); }
  if (corner == 4u) { return vec2<f32>(1.0, 1.0); }
  return vec2<f32>(-1.0, 1.0);
}

fn screenToNdc(screen: vec2<f32>) -> vec2<f32> {
  let viewport = viewportSize();
  return vec2<f32>(
    screen.x / viewport.x * 2.0 - 1.0,
    1.0 - screen.y / viewport.y * 2.0
  );
}

fn unproject(screen: vec2<f32>, depth: f32) -> vec3<f32> {
  let ndc = screenToNdc(screen);
  let camera = vec4<f32>(
    ndc.x * depth / uniforms.captureProj[0][0],
    ndc.y * depth / uniforms.captureProj[1][1],
    -depth,
    1.0
  );

  return (uniforms.captureInvView * camera).xyz;
}

fn gridDepth(x: u32, y: u32) -> u32 {
  if (x >= gridColumns() || y >= gridRows()) {
    return INF_U;
  }

  return filteredDepthCompute[y * gridColumns() + x];
}

fn surfaceCellValid(d0: u32, d1: u32, d2: u32, d3: u32) -> bool {
  if (d0 >= INF_U || d1 >= INF_U || d2 >= INF_U || d3 >= INF_U) {
    return false;
  }

  let minDepth = min(min(d0, d1), min(d2, d3));
  let maxDepth = max(max(d0, d1), max(d2, d3));
  return maxDepth - minDepth <= maxDepthJumpPacked();
}

fn surfaceNormal(nodeX: u32, nodeY: u32, screen: vec2<f32>, depth: f32) -> vec3<f32> {
  let spacing = uniforms.params0.z;
  let columns = gridColumns();
  let rows = gridRows();
  let leftX = select(nodeX, nodeX - 1u, nodeX > 0u);
  let rightX = select(nodeX, nodeX + 1u, nodeX + 1u < columns);
  let upY = select(nodeY, nodeY - 1u, nodeY > 0u);
  let downY = select(nodeY, nodeY + 1u, nodeY + 1u < rows);
  let leftDepth = unpackDepth(gridDepth(leftX, nodeY));
  let rightDepth = unpackDepth(gridDepth(rightX, nodeY));
  let upDepth = unpackDepth(gridDepth(nodeX, upY));
  let downDepth = unpackDepth(gridDepth(nodeX, downY));
  let leftScreen = screen + vec2<f32>(-spacing, 0.0);
  let rightScreen = screen + vec2<f32>(spacing, 0.0);
  let upScreen = screen + vec2<f32>(0.0, -spacing);
  let downScreen = screen + vec2<f32>(0.0, spacing);
  let px0 = unproject(leftScreen, select(depth, leftDepth, leftDepth < INF_F));
  let px1 = unproject(rightScreen, select(depth, rightDepth, rightDepth < INF_F));
  let py0 = unproject(upScreen, select(depth, upDepth, upDepth < INF_F));
  let py1 = unproject(downScreen, select(depth, downDepth, downDepth < INF_F));
  var normal = normalize(cross(px1 - px0, py1 - py0));
  let cameraPosition = uniforms.captureInvView[3].xyz;
  let centerWorld = unproject(screen, depth);

  if (dot(normal, normalize(cameraPosition - centerWorld)) < 0.0) {
    normal = -normal;
  }

  return normal;
}

@vertex
fn backgroundVertex(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4<f32> {
  if (vertexIndex == 0u) { return vec4<f32>(-1.0, -1.0, 0.0, 1.0); }
  if (vertexIndex == 1u) { return vec4<f32>(3.0, -1.0, 0.0, 1.0); }
  return vec4<f32>(-1.0, 3.0, 0.0, 1.0);
}

fn backgroundColor(uv: vec2<f32>) -> vec3<f32> {
  let horizon = smoothstep(0.5, 0.88, uv.y);
  let base = mix(vec3<f32>(0.018, 0.03, 0.048), vec3<f32>(0.02, 0.09, 0.12), horizon);
  let sideGlow = smoothstep(0.15, 0.95, uv.x) * smoothstep(1.0, 0.3, uv.y) * 0.06;
  return base + vec3<f32>(0.0, 0.25, 0.32) * sideGlow;
}

@fragment
fn backgroundFragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
  return vec4<f32>(backgroundColor(position.xy / viewportSize()), 1.0);
}

fn floorLocalPoint(vertexIndex: u32) -> vec2<f32> {
  if (vertexIndex == 0u) { return vec2<f32>(-4.5, -4.5); }
  if (vertexIndex == 1u) { return vec2<f32>(4.5, -4.5); }
  if (vertexIndex == 2u) { return vec2<f32>(-4.5, 4.5); }
  if (vertexIndex == 3u) { return vec2<f32>(-4.5, 4.5); }
  if (vertexIndex == 4u) { return vec2<f32>(4.5, -4.5); }
  return vec2<f32>(4.5, 4.5);
}

@vertex
fn floorVertex(@builtin(vertex_index) vertexIndex: u32) -> FloorVertexOut {
  var output: FloorVertexOut;
  let local = floorLocalPoint(vertexIndex);
  let world = vec3<f32>(local.x, -1.82, local.y);
  output.position = uniforms.proj * uniforms.view * vec4<f32>(world, 1.0);
  output.world = world;
  output.local = local;
  return output;
}

@fragment
fn floorFragment(input: FloorVertexOut) -> @location(0) vec4<f32> {
  let cameraPosition = uniforms.invView[3].xyz;
  let distanceToCamera = length(cameraPosition - input.world);
  let fog = smoothstep(6.5, 13.0, distanceToCamera);
  let radius = length(input.local);
  let ring = 1.0 - smoothstep(0.0, 0.035, abs(radius - 1.9));
  let broadGlow = smoothstep(3.8, 0.2, radius) * 0.18;
  let gridX = 1.0 - smoothstep(0.0, max(fwidth(input.local.x) * 1.2, 0.012), abs(fract(input.local.x * 0.5 + 0.5) - 0.5));
  let gridZ = 1.0 - smoothstep(0.0, max(fwidth(input.local.y) * 1.2, 0.012), abs(fract(input.local.y * 0.5 + 0.5) - 0.5));
  let grid = max(gridX, gridZ) * 0.045;
  var color = vec3<f32>(0.018, 0.043, 0.062);
  color = color + vec3<f32>(0.0, 0.45, 0.58) * (ring * 0.34 + broadGlow + grid);
  color = mix(color, vec3<f32>(0.005, 0.008, 0.013), fog);
  return vec4<f32>(color, 1.0);
}

@vertex
fn surfaceVertex(@builtin(vertex_index) vertexIndex: u32) -> SurfaceVertexOut {
  var output: SurfaceVertexOut;
  let vertex = meshVerticesRender[vertexIndex];

  let world = vertex.world.xyz;
  output.position = uniforms.proj * uniforms.view * vec4<f32>(world, 1.0);
  output.world = world;
  output.normal = vertex.normalDepth.xyz;
  output.local = vertex.local.xy;
  output.depth = vertex.normalDepth.w;
  output.valid = 1.0;
  output.barycentric = vec3<f32>(vertex.local.zw, max(0.0, 1.0 - vertex.local.z - vertex.local.w));
  return output;
}

@fragment
fn surfaceFragment(input: SurfaceVertexOut) -> @location(0) vec4<f32> {
  if (input.valid < 0.5) {
    discard;
  }

  let mode = renderMode();
  let depthTint = clamp((input.depth - 2.0) / 4.0, 0.0, 1.0);

  if (mode == 3u) {
    let color = mix(vec3<f32>(0.88, 0.98, 1.0), vec3<f32>(0.03, 0.22, 0.35), depthTint);
    return vec4<f32>(color, 0.96);
  }

  let cameraPosition = uniforms.invView[3].xyz;
  let viewDirection = normalize(cameraPosition - input.world);
  let sourceNormal = normalize(input.normal);
  let backSide = dot(sourceNormal, viewDirection) < 0.0;
  let normal = select(sourceNormal, -sourceNormal, backSide);
  let keyLight = normalize(vec3<f32>(-0.52, 0.74, 0.42));
  let rimLight = normalize(vec3<f32>(0.62, 0.16, -0.77));
  let diffuse = max(dot(normal, keyLight), 0.0);
  let rim = pow(max(1.0 - dot(normal, viewDirection), 0.0), 2.8);
  let fresnel = pow(max(1.0 - dot(normal, viewDirection), 0.0), 3.2);
  let specular = pow(max(dot(reflect(-keyLight, normal), viewDirection), 0.0), 62.0);
  let sideSpecular = pow(max(dot(reflect(-rimLight, normal), viewDirection), 0.0), 38.0);
  let base = mix(vec3<f32>(0.26, 0.78, 0.94), vec3<f32>(0.72, 0.94, 0.99), 1.0 - depthTint);
  var color = base * (0.28 + diffuse * 0.84) + vec3<f32>(1.0) * specular * 0.85 + vec3<f32>(0.45, 0.94, 1.0) * sideSpecular * 0.55;
  color = color + vec3<f32>(0.58, 0.95, 1.0) * rim * 0.56;

  if (mode == 1u) {
    let cellEdge = min(min(input.local.x, 1.0 - input.local.x), min(input.local.y, 1.0 - input.local.y));
    let triangleEdge = min(input.barycentric.x, min(input.barycentric.y, input.barycentric.z));
    let cellWidth = max(fwidth(cellEdge) * 1.75, 0.018);
    let triangleWidth = max(fwidth(triangleEdge) * 1.25, 0.01);
    let cellLine = 1.0 - smoothstep(0.0, cellWidth, cellEdge);
    let triangleLine = 1.0 - smoothstep(0.0, triangleWidth, triangleEdge);
    let line = max(cellLine, triangleLine * 0.78);
    let baseMesh = select(vec3<f32>(0.05, 0.23, 0.3), vec3<f32>(0.28, 0.16, 0.1), frozenSurface() && backSide);
    let meshColor = mix(baseMesh, vec3<f32>(0.9, 0.99, 1.0), line);
    return vec4<f32>(meshColor, 0.96);
  }

  let thickness = clamp(uniforms.params0.w, 0.05, 1.2);
  let viewNormal = normalize((uniforms.view * vec4<f32>(normal, 0.0)).xyz);
  let refractOffset = viewNormal.xy * (0.012 + thickness * 0.034) * (0.45 + depthTint * 0.7);
  let refracted = backgroundColor(clamp(input.position.xy / viewportSize() + refractOffset, vec2<f32>(0.0), vec2<f32>(1.0)));
  let absorption = mix(vec3<f32>(0.62, 0.94, 1.0), vec3<f32>(0.16, 0.72, 0.82), clamp(thickness * 0.72 + depthTint * 0.22, 0.0, 1.0));
  color = mix(refracted * absorption, color, 0.66 + fresnel * 0.14);
  color = color + vec3<f32>(0.82, 0.98, 1.0) * fresnel * (0.28 + thickness * 0.22);

  if (frozenSurface() && backSide) {
    color = mix(color, vec3<f32>(0.94, 0.53, 0.24), 0.34);
  }

  return vec4<f32>(color, clamp(0.52 + thickness * 0.28 + fresnel * 0.15, 0.48, 0.9));
}

@vertex
fn particleVertex(@builtin(vertex_index) vertexIndex: u32) -> ParticleVertexOut {
  var output: ParticleVertexOut;
  let particleIndex = vertexIndex / 6u;
  let corner = quadBillboardCorner(vertexIndex % 6u);

  if (particleIndex >= particleCount()) {
    output.position = vec4<f32>(2.0, 2.0, 0.0, 1.0);
    output.local = corner;
    output.alpha = 0.0;
    return output;
  }

  let world = particlesRender[particleIndex].xyz;
  let viewPosition = uniforms.view * vec4<f32>(world, 1.0);
  let depth = -viewPosition.z;
  let clip = uniforms.proj * viewPosition;

  if (clip.w <= 0.0 || depth <= 0.1) {
    output.position = vec4<f32>(2.0, 2.0, 0.0, 1.0);
    output.local = corner;
    output.alpha = 0.0;
    return output;
  }

  let viewport = viewportSize();
  let ndc = clip.xy / clip.w;
  let size = clamp((uniforms.params0.y * viewport.y * uniforms.proj[1][1]) / (depth * 9.0), 3.0, 9.0);
  let offset = corner * vec2<f32>(size * 2.0 / viewport.x, size * 2.0 / viewport.y);
  output.position = vec4<f32>(ndc + offset, 0.35, 1.0);
  output.local = corner;
  output.alpha = clamp(1.3 - depth * 0.16, 0.28, 0.85);
  return output;
}

@fragment
fn particleFragment(input: ParticleVertexOut) -> @location(0) vec4<f32> {
  let distanceFromCenter = length(input.local);

  if (distanceFromCenter > 1.0 || input.alpha <= 0.0) {
    discard;
  }

  let core = 1.0 - smoothstep(0.05, 1.0, distanceFromCenter);
  let color = mix(vec3<f32>(1.0, 0.42, 0.16), vec3<f32>(1.0, 0.78, 0.42), core);
  return vec4<f32>(color, input.alpha * core);
}
`
