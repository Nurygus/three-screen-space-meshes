import * as THREE from 'three'

export interface ScreenSpaceMeshOptions {
  width: number
  height: number
  gridSpacing: number
  particleRadius: number
  maxDepthJump: number
  depthSmoothing: number
  silhouetteSmoothing: number
}

export interface ScreenSpaceMeshResult {
  positions: Float32Array
  normals: Float32Array
  colors: Float32Array
  indices: Uint32Array
  wireIndices: Uint32Array
  vertices: number
  triangles: number
  gridColumns: number
  gridRows: number
  buildMs: number
}

interface EdgeCandidate {
  t: number
  depth: number
  priority: number
  synthetic: boolean
}

interface TopologyCase {
  mask: number
  label: 'regular' | 'turn' | 'source' | 'pathological'
  missingEdge?: number
  extraEdges?: number[]
}

const INF = 1e20
const scratchWorld = new THREE.Vector3()
const scratchCamera = new THREE.Vector3()
const scratchProjected = new THREE.Vector3()
const scratchUnprojected = new THREE.Vector3()
const EDGE_ORIENTED_CORNERS = [
  [0, 1],
  [1, 2],
  [3, 2],
  [0, 3],
] as const
const EDGE_LOOP_CORNERS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 0],
] as const
const TOPOLOGY_CASES: TopologyCase[] = [
  { mask: 0, label: 'regular' },
  { mask: 1, label: 'source' },
  { mask: 2, label: 'source' },
  { mask: 3, label: 'turn' },
  { mask: 4, label: 'source' },
  { mask: 5, label: 'turn' },
  { mask: 6, label: 'turn' },
  { mask: 7, label: 'pathological', missingEdge: 3 },
  { mask: 8, label: 'source' },
  { mask: 9, label: 'turn' },
  { mask: 10, label: 'turn' },
  { mask: 11, label: 'pathological', missingEdge: 2 },
  { mask: 12, label: 'turn' },
  { mask: 13, label: 'pathological', missingEdge: 1 },
  { mask: 14, label: 'pathological', missingEdge: 0 },
  { mask: 15, label: 'pathological', extraEdges: [0, 1] },
]

export function buildScreenSpaceMesh(
  particles: Float32Array,
  particleCount: number,
  camera: THREE.PerspectiveCamera,
  options: ScreenSpaceMeshOptions,
): ScreenSpaceMeshResult {
  const startedAt = performance.now()
  const normalizedOptions = { ...options, gridSpacing: Math.max(1, options.gridSpacing) }
  const gridColumns = Math.ceil(normalizedOptions.width / normalizedOptions.gridSpacing) + 1
  const gridRows = Math.ceil(normalizedOptions.height / normalizedOptions.gridSpacing) + 1
  const nodeCount = gridColumns * gridRows
  const depth = new Float32Array(nodeCount)
  const horizontalEdges = createEdgeCandidates(Math.max(0, (gridColumns - 1) * gridRows))
  const verticalEdges = createEdgeCandidates(Math.max(0, gridColumns * (gridRows - 1)))

  depth.fill(INF)
  rasterizeParticles(particles, particleCount, camera, normalizedOptions, gridColumns, gridRows, depth)
  selectSilhouetteCandidates(
    particles,
    particleCount,
    camera,
    normalizedOptions,
    gridColumns,
    gridRows,
    depth,
    horizontalEdges,
    verticalEdges,
  )

  const filteredDepth = smoothDepthMap(
    depth,
    gridColumns,
    gridRows,
    Math.max(0, Math.round(normalizedOptions.depthSmoothing)),
    normalizedOptions.maxDepthJump,
  )
  const mesh = triangulateDepthMap(
    depth,
    filteredDepth,
    horizontalEdges,
    verticalEdges,
    gridColumns,
    gridRows,
    camera,
    normalizedOptions,
  )

  return {
    ...mesh,
    gridColumns,
    gridRows,
    buildMs: performance.now() - startedAt,
  }
}

function createEdgeCandidates(count: number): Array<EdgeCandidate | null> {
  return Array.from({ length: count }, () => null)
}

function rasterizeParticles(
  particles: Float32Array,
  particleCount: number,
  camera: THREE.PerspectiveCamera,
  options: ScreenSpaceMeshOptions,
  gridColumns: number,
  gridRows: number,
  depth: Float32Array,
): void {
  const focalY = camera.projectionMatrix.elements[5]
  const radiusWorld = options.particleRadius
  const count = Math.min(particleCount, particles.length / 3)

  camera.updateMatrixWorld()
  camera.updateProjectionMatrix()

  for (let index = 0; index < count; index += 1) {
    const projection = projectParticle(particles, index, camera, options, focalY, radiusWorld)

    if (!projection) {
      continue
    }

    const { screenX, screenY, cameraDepth, radiusPixels } = projection
    const minX = clampInt(Math.floor((screenX - radiusPixels) / options.gridSpacing), 0, gridColumns - 1)
    const maxX = clampInt(Math.ceil((screenX + radiusPixels) / options.gridSpacing), 0, gridColumns - 1)
    const minY = clampInt(Math.floor((screenY - radiusPixels) / options.gridSpacing), 0, gridRows - 1)
    const maxY = clampInt(Math.ceil((screenY + radiusPixels) / options.gridSpacing), 0, gridRows - 1)
    const radiusPixelsSquared = radiusPixels * radiusPixels

    for (let y = minY; y <= maxY; y += 1) {
      const nodeY = y * options.gridSpacing
      const dy = nodeY - screenY

      for (let x = minX; x <= maxX; x += 1) {
        const nodeX = x * options.gridSpacing
        const dx = nodeX - screenX
        const distanceSquared = dx * dx + dy * dy

        if (distanceSquared > radiusPixelsSquared) {
          continue
        }

        const bulge = Math.sqrt(Math.max(0, 1 - distanceSquared / radiusPixelsSquared))
        const frontDepth = cameraDepth - radiusWorld * 0.92 * bulge
        const nodeIndex = y * gridColumns + x

        if (frontDepth < depth[nodeIndex]) {
          depth[nodeIndex] = frontDepth
        }
      }
    }
  }
}

function selectSilhouetteCandidates(
  particles: Float32Array,
  particleCount: number,
  camera: THREE.PerspectiveCamera,
  options: ScreenSpaceMeshOptions,
  gridColumns: number,
  gridRows: number,
  depth: Float32Array,
  horizontalEdges: Array<EdgeCandidate | null>,
  verticalEdges: Array<EdgeCandidate | null>,
): void {
  const focalY = camera.projectionMatrix.elements[5]
  const radiusWorld = options.particleRadius
  const count = Math.min(particleCount, particles.length / 3)

  for (let index = 0; index < count; index += 1) {
    const projection = projectParticle(particles, index, camera, options, focalY, radiusWorld)

    if (!projection) {
      continue
    }

    const { screenX, screenY, cameraDepth, radiusPixels } = projection
    const minX = clampInt(Math.floor((screenX - radiusPixels) / options.gridSpacing), 0, gridColumns - 1)
    const maxX = clampInt(Math.ceil((screenX + radiusPixels) / options.gridSpacing), 0, gridColumns - 1)
    const minY = clampInt(Math.floor((screenY - radiusPixels) / options.gridSpacing), 0, gridRows - 1)
    const maxY = clampInt(Math.ceil((screenY + radiusPixels) / options.gridSpacing), 0, gridRows - 1)
    const radiusPixelsSquared = radiusPixels * radiusPixels

    for (let y = minY; y <= maxY; y += 1) {
      const nodeY = y * options.gridSpacing
      const dy = nodeY - screenY
      const remaining = radiusPixelsSquared - dy * dy

      if (remaining < 0) {
        continue
      }

      const root = Math.sqrt(remaining)
      storeHorizontalCandidate(
        Math.floor((screenX - root) / options.gridSpacing),
        y,
        screenX - root,
        cameraDepth,
        options,
        gridColumns,
        gridRows,
        depth,
        horizontalEdges,
      )
      storeHorizontalCandidate(
        Math.floor((screenX + root) / options.gridSpacing),
        y,
        screenX + root,
        cameraDepth,
        options,
        gridColumns,
        gridRows,
        depth,
        horizontalEdges,
      )
    }

    for (let x = minX; x <= maxX; x += 1) {
      const nodeX = x * options.gridSpacing
      const dx = nodeX - screenX
      const remaining = radiusPixelsSquared - dx * dx

      if (remaining < 0) {
        continue
      }

      const root = Math.sqrt(remaining)
      storeVerticalCandidate(
        x,
        Math.floor((screenY - root) / options.gridSpacing),
        screenY - root,
        cameraDepth,
        options,
        gridColumns,
        gridRows,
        depth,
        verticalEdges,
      )
      storeVerticalCandidate(
        x,
        Math.floor((screenY + root) / options.gridSpacing),
        screenY + root,
        cameraDepth,
        options,
        gridColumns,
        gridRows,
        depth,
        verticalEdges,
      )
    }
  }
}

function projectParticle(
  particles: Float32Array,
  index: number,
  camera: THREE.PerspectiveCamera,
  options: ScreenSpaceMeshOptions,
  focalY: number,
  radiusWorld: number,
) {
  const offset = index * 3
  scratchWorld.set(particles[offset], particles[offset + 1], particles[offset + 2])
  scratchCamera.copy(scratchWorld).applyMatrix4(camera.matrixWorldInverse)

  const cameraDepth = -scratchCamera.z
  if (cameraDepth <= camera.near || cameraDepth >= camera.far) {
    return null
  }

  scratchProjected.copy(scratchWorld).project(camera)
  if (scratchProjected.z < -1 || scratchProjected.z > 1) {
    return null
  }

  const screenX = (scratchProjected.x * 0.5 + 0.5) * options.width
  const screenY = (-scratchProjected.y * 0.5 + 0.5) * options.height
  const radiusPixels = clamp((radiusWorld * options.height * focalY) / (2 * cameraDepth), 4, 192)

  if (
    screenX < -radiusPixels ||
    screenX > options.width + radiusPixels ||
    screenY < -radiusPixels ||
    screenY > options.height + radiusPixels
  ) {
    return null
  }

  return { screenX, screenY, cameraDepth, radiusPixels }
}

function storeHorizontalCandidate(
  edgeCellX: number,
  row: number,
  screenX: number,
  cameraDepth: number,
  options: ScreenSpaceMeshOptions,
  gridColumns: number,
  gridRows: number,
  depth: Float32Array,
  edges: Array<EdgeCandidate | null>,
): void {
  if (row < 0 || row >= gridRows || edgeCellX < 0 || edgeCellX >= gridColumns - 1) {
    return
  }

  const t = (screenX - edgeCellX * options.gridSpacing) / options.gridSpacing
  const a = row * gridColumns + edgeCellX
  const b = a + 1
  storeCandidate(row * (gridColumns - 1) + edgeCellX, t, cameraDepth, depth[a], depth[b], options, edges)
}

function storeVerticalCandidate(
  column: number,
  edgeCellY: number,
  screenY: number,
  cameraDepth: number,
  options: ScreenSpaceMeshOptions,
  gridColumns: number,
  gridRows: number,
  depth: Float32Array,
  edges: Array<EdgeCandidate | null>,
): void {
  if (column < 0 || column >= gridColumns || edgeCellY < 0 || edgeCellY >= gridRows - 1) {
    return
  }

  const t = (screenY - edgeCellY * options.gridSpacing) / options.gridSpacing
  const a = edgeCellY * gridColumns + column
  const b = (edgeCellY + 1) * gridColumns + column
  storeCandidate(edgeCellY * gridColumns + column, t, cameraDepth, depth[a], depth[b], options, edges)
}

function storeCandidate(
  edgeIndex: number,
  t: number,
  candidateDepth: number,
  aDepth: number,
  bDepth: number,
  options: ScreenSpaceMeshOptions,
  edges: Array<EdgeCandidate | null>,
): void {
  const clampedT = clamp(t, 0, 1)
  if (!edgeNeedsCutDepths(aDepth, bDepth, options.maxDepthJump)) {
    return
  }

  const aValid = isFiniteDepth(aDepth)
  const bValid = isFiniteDepth(bDepth)

  if (aValid && bValid && candidateDepth >= (aDepth + bDepth) * 0.5) {
    return
  }

  if (aValid !== bValid) {
    const validDepth = aValid ? aDepth : bDepth
    if (candidateDepth > validDepth + options.maxDepthJump) {
      return
    }
  }

  const frontIsA = aValid && (!bValid || aDepth <= bDepth)
  const priority = frontIsA ? clampedT : 1 - clampedT
  const previous = edges[edgeIndex]

  if (!previous || priority > previous.priority) {
    edges[edgeIndex] = { t: clampedT, depth: candidateDepth, priority, synthetic: false }
  }
}

function smoothDepthMap(
  source: Float32Array,
  columns: number,
  rows: number,
  halfSize: number,
  maxDepthJump: number,
): Float32Array {
  if (halfSize === 0) {
    return source
  }

  const weights = binomialWeights(halfSize)
  const horizontal = new Float32Array(source.length)
  const vertical = new Float32Array(source.length)

  filterDepth(source, horizontal, columns, rows, halfSize, weights, maxDepthJump, true)
  filterDepth(horizontal, vertical, columns, rows, halfSize, weights, maxDepthJump, false)

  return vertical
}

function filterDepth(
  source: Float32Array,
  target: Float32Array,
  columns: number,
  rows: number,
  halfSize: number,
  weights: number[],
  maxDepthJump: number,
  horizontal: boolean,
): void {
  target.fill(INF)

  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < columns; x += 1) {
      const centerIndex = y * columns + x
      const centerDepth = source[centerIndex]

      if (!isFiniteDepth(centerDepth)) {
        continue
      }

      let total = centerDepth * weights[halfSize]
      let weightTotal = weights[halfSize]

      for (let step = 1; step <= halfSize; step += 1) {
        const ax = horizontal ? x - step : x
        const ay = horizontal ? y : y - step
        const bx = horizontal ? x + step : x
        const by = horizontal ? y : y + step

        if (ax < 0 || ay < 0 || bx >= columns || by >= rows) {
          continue
        }

        const depthA = source[ay * columns + ax]
        const depthB = source[by * columns + bx]
        const validPair =
          isFiniteDepth(depthA) &&
          isFiniteDepth(depthB) &&
          Math.abs(depthA - centerDepth) <= maxDepthJump &&
          Math.abs(depthB - centerDepth) <= maxDepthJump

        if (validPair) {
          const weight = weights[halfSize - step]
          total += (depthA + depthB) * weight
          weightTotal += weight * 2
        }
      }

      target[centerIndex] = total / weightTotal
    }
  }
}

function triangulateDepthMap(
  rawDepth: Float32Array,
  depth: Float32Array,
  horizontalEdges: Array<EdgeCandidate | null>,
  verticalEdges: Array<EdgeCandidate | null>,
  columns: number,
  rows: number,
  camera: THREE.PerspectiveCamera,
  options: ScreenSpaceMeshOptions,
) {
  const screenX: number[] = []
  const screenY: number[] = []
  const vertexDepth: number[] = []
  const boundary: boolean[] = []
  const glueKeys: string[] = []
  const indices: number[] = []
  const wireIndices: number[] = []
  const vertexByKey = new Map<string, number>()

  const addVertex = (key: string, x: number, y: number, z: number, isBoundary: boolean, glueKey = key) => {
    const existing = vertexByKey.get(key)

    if (existing !== undefined) {
      boundary[existing] = boundary[existing] || isBoundary
      return existing
    }

    const vertex = screenX.length
    vertexByKey.set(key, vertex)
    screenX.push(x)
    screenY.push(y)
    vertexDepth.push(z)
    boundary.push(isBoundary)
    glueKeys.push(glueKey)
    return vertex
  }

  for (let row = 0; row < rows - 1; row += 1) {
    for (let column = 0; column < columns - 1; column += 1) {
      addCellTriangles(column, row, columns, rawDepth, depth, horizontalEdges, verticalEdges, options, addVertex, indices, wireIndices)
    }
  }

  smoothSilhouettes(
    screenX,
    screenY,
    vertexDepth,
    boundary,
    glueKeys,
    indices,
    options.silhouetteSmoothing,
    options.maxDepthJump,
  )

  const positions = unprojectVertices(screenX, screenY, vertexDepth, camera, options.width, options.height)
  const typedIndices = new Uint32Array(indices)
  const normals = computeNormals(positions, typedIndices)
  const colors = computeDepthColors(vertexDepth)

  return {
    positions,
    normals,
    colors,
    indices: typedIndices,
    wireIndices: new Uint32Array(wireIndices),
    vertices: screenX.length,
    triangles: typedIndices.length / 3,
  }
}

type AddVertex = (key: string, x: number, y: number, depth: number, isBoundary: boolean, glueKey?: string) => number

function addCellTriangles(
  column: number,
  row: number,
  columns: number,
  rawDepth: Float32Array,
  depth: Float32Array,
  horizontalEdges: Array<EdgeCandidate | null>,
  verticalEdges: Array<EdgeCandidate | null>,
  options: ScreenSpaceMeshOptions,
  addVertex: AddVertex,
  indices: number[],
  wireIndices: number[],
): void {
  const spacing = options.gridSpacing
  const nodes = [
    row * columns + column,
    row * columns + column + 1,
    (row + 1) * columns + column + 1,
    (row + 1) * columns + column,
  ]
  const cornerX = [column * spacing, (column + 1) * spacing, (column + 1) * spacing, column * spacing]
  const cornerY = [row * spacing, row * spacing, (row + 1) * spacing, (row + 1) * spacing]
  const valid = nodes.map((node) => isFiniteDepth(depth[node]))
  const edgeCandidates = [
    getHorizontalEdge(column, row, columns, rawDepth, depth, horizontalEdges, options),
    getVerticalEdge(column + 1, row, columns, rawDepth, depth, verticalEdges, options),
    getHorizontalEdge(column, row + 1, columns, rawDepth, depth, horizontalEdges, options),
    getVerticalEdge(column, row, columns, rawDepth, depth, verticalEdges, options),
  ]
  const edgeMask = edgeCandidates.reduce((mask, edge, index) => mask | (edge ? 1 << index : 0), 0)
  const topology = TOPOLOGY_CASES[edgeMask]
  const connected = [
    !edgeCandidates[0] && connectedEdge(depth, nodes[0], nodes[1], options.maxDepthJump),
    !edgeCandidates[1] && connectedEdge(depth, nodes[1], nodes[2], options.maxDepthJump),
    !edgeCandidates[2] && connectedEdge(depth, nodes[2], nodes[3], options.maxDepthJump),
    !edgeCandidates[3] && connectedEdge(depth, nodes[3], nodes[0], options.maxDepthJump),
  ]
  const components = findComponents(valid, connected)

  for (const mask of components) {
    const polygon: number[] = []
    const extraEdges = extraEdgesForTopology(topology, mask)

    for (let corner = 0; corner < 4; corner += 1) {
      const next = (corner + 1) % 4
      const active = hasCorner(mask, corner)
      const nextActive = hasCorner(mask, next)

      if (active) {
        const previous = (corner + 3) % 4
        const isBoundary = !hasCorner(mask, previous) || !nextActive || components.length > 1
        polygon.push(addVertex(`g:${nodes[corner]}`, cornerX[corner], cornerY[corner], depth[nodes[corner]], isBoundary))
      }

      if (active !== nextActive) {
        const sourceCorner = active ? corner : next
        polygon.push(
          addEdgeVertex(
            column,
            row,
            corner,
            sourceCorner,
            mask,
            false,
            nodes,
            cornerX,
            cornerY,
            rawDepth,
            depth,
            edgeCandidates[corner],
            addVertex,
          ),
        )
      } else if (!active && !nextActive && extraEdges.includes(corner) && edgeCandidates[corner]) {
        polygon.push(
          addEdgeVertex(
            column,
            row,
            corner,
            firstCornerInMask(mask) ?? corner,
            mask,
            true,
            nodes,
            cornerX,
            cornerY,
            rawDepth,
            depth,
            edgeCandidates[corner],
            addVertex,
          ),
        )
      }
    }

    addPolygonEdges(polygon, wireIndices)
    addFanTriangles(polygon, indices)
  }
}

function getHorizontalEdge(
  cellX: number,
  row: number,
  columns: number,
  rawDepth: Float32Array,
  depth: Float32Array,
  edges: Array<EdgeCandidate | null>,
  options: ScreenSpaceMeshOptions,
): EdgeCandidate | null {
  if (cellX < 0 || row < 0 || cellX >= columns - 1) {
    return null
  }

  const edge = edges[row * (columns - 1) + cellX]
  if (edge) {
    return edge
  }

  const a = row * columns + cellX
  const b = a + 1
  return fallbackCandidate(rawDepth[a], rawDepth[b], depth[a], depth[b], options.maxDepthJump)
}

function getVerticalEdge(
  column: number,
  cellY: number,
  columns: number,
  rawDepth: Float32Array,
  depth: Float32Array,
  edges: Array<EdgeCandidate | null>,
  options: ScreenSpaceMeshOptions,
): EdgeCandidate | null {
  if (column < 0 || cellY < 0 || column >= columns) {
    return null
  }

  const edge = edges[cellY * columns + column]
  if (edge) {
    return edge
  }

  const a = cellY * columns + column
  const b = (cellY + 1) * columns + column
  return fallbackCandidate(rawDepth[a], rawDepth[b], depth[a], depth[b], options.maxDepthJump)
}

function fallbackCandidate(
  rawA: number,
  rawB: number,
  filteredA: number,
  filteredB: number,
  maxDepthJump: number,
): EdgeCandidate | null {
  if (!edgeNeedsCutDepths(filteredA, filteredB, maxDepthJump) && !edgeNeedsCutDepths(rawA, rawB, maxDepthJump)) {
    return null
  }

  const depth = Math.min(isFiniteDepth(filteredA) ? filteredA : INF, isFiniteDepth(filteredB) ? filteredB : INF)
  return isFiniteDepth(depth) ? { t: 0.5, depth, priority: -1, synthetic: true } : null
}

function addEdgeVertex(
  column: number,
  row: number,
  edgeIndex: number,
  sourceCorner: number,
  componentMask: number,
  extra: boolean,
  nodes: number[],
  cornerX: number[],
  cornerY: number[],
  rawDepth: Float32Array,
  depth: Float32Array,
  candidate: EdgeCandidate | null,
  addVertex: AddVertex,
): number {
  if (!candidate) {
    return addVertex(`missing:${column}:${row}:${edgeIndex}:${componentMask}`, cornerX[sourceCorner], cornerY[sourceCorner], depth[nodes[sourceCorner]], true)
  }

  const [x, y] = edgePoint(edgeIndex, candidate.t, cornerX, cornerY)
  const oriented = EDGE_ORIENTED_CORNERS[edgeIndex]
  const edgeNodeA = nodes[oriented[0]]
  const edgeNodeB = nodes[oriented[1]]
  const edgeKey = edgeVertexKey(edgeNodeA, edgeNodeB)
  const vertexDepth = extra
    ? extrapolateExtraDepth(componentMask, nodes, depth)
    : silhouetteDepthForComponent(edgeIndex, sourceCorner, nodes, rawDepth, depth, candidate)
  const layerKey = extra ? `extra:${column}:${row}:${componentMask}` : `${nodes[sourceCorner]}`

  return addVertex(`e:${edgeKey}:${layerKey}`, x, y, vertexDepth, true, `e:${edgeKey}`)
}

function silhouetteDepthForComponent(
  edgeIndex: number,
  sourceCorner: number,
  nodes: number[],
  rawDepth: Float32Array,
  depth: Float32Array,
  candidate: EdgeCandidate,
): number {
  if (candidate.synthetic) {
    return candidate.depth
  }

  const [aCorner, bCorner] = EDGE_ORIENTED_CORNERS[edgeIndex]
  const aRaw = rawDepth[nodes[aCorner]]
  const bRaw = rawDepth[nodes[bCorner]]
  const aValid = isFiniteDepth(aRaw)
  const bValid = isFiniteDepth(bRaw)

  if (aValid !== bValid) {
    return adjustedFrontDepth(aValid ? aCorner : bCorner, nodes, rawDepth, depth, candidate)
  }

  if (!aValid || !bValid) {
    return candidate.depth
  }

  const frontCorner = aRaw <= bRaw ? aCorner : bCorner
  if (sourceCorner === frontCorner) {
    return adjustedFrontDepth(frontCorner, nodes, rawDepth, depth, candidate)
  }

  return interpolateBackDepth(edgeIndex, nodes, depth, candidate.t)
}

function adjustedFrontDepth(
  frontCorner: number,
  nodes: number[],
  rawDepth: Float32Array,
  depth: Float32Array,
  candidate: EdgeCandidate,
): number {
  const raw = rawDepth[nodes[frontCorner]]
  const filtered = depth[nodes[frontCorner]]
  return isFiniteDepth(raw) && isFiniteDepth(filtered) ? candidate.depth + filtered - raw : candidate.depth
}

function interpolateBackDepth(edgeIndex: number, nodes: number[], depth: Float32Array, t: number): number {
  const [aCorner, bCorner] = EDGE_ORIENTED_CORNERS[edgeIndex]
  const a = depth[nodes[aCorner]]
  const b = depth[nodes[bCorner]]

  if (isFiniteDepth(a) && isFiniteDepth(b)) {
    return a + (b - a) * t
  }

  if (isFiniteDepth(a)) {
    return a
  }

  return isFiniteDepth(b) ? b : INF
}

function extrapolateExtraDepth(componentMask: number, nodes: number[], depth: Float32Array): number {
  let total = 0
  let count = 0

  for (let corner = 0; corner < 4; corner += 1) {
    if (hasCorner(componentMask, corner) && isFiniteDepth(depth[nodes[corner]])) {
      total += depth[nodes[corner]]
      count += 1
    }
  }

  return count > 0 ? total / count : INF
}

function edgePoint(edgeIndex: number, t: number, cornerX: number[], cornerY: number[]): [number, number] {
  const [a, b] = EDGE_ORIENTED_CORNERS[edgeIndex]
  return [cornerX[a] + (cornerX[b] - cornerX[a]) * t, cornerY[a] + (cornerY[b] - cornerY[a]) * t]
}

function extraEdgesForTopology(topology: TopologyCase, componentMask: number): number[] {
  if (topology.mask === 15) {
    return hasCorner(componentMask, 3) ? topology.extraEdges ?? [] : []
  }

  if (topology.label !== 'pathological' || topology.missingEdge === undefined) {
    return []
  }

  const [a, b] = EDGE_LOOP_CORNERS[topology.missingEdge]
  return hasCorner(componentMask, a) && hasCorner(componentMask, b) ? [oppositeEdge(topology.missingEdge)] : []
}

function oppositeEdge(edge: number): number {
  return (edge + 2) % 4
}

function findComponents(valid: boolean[], connected: boolean[]): number[] {
  const components: number[] = []
  const visited = [false, false, false, false]

  for (let start = 0; start < 4; start += 1) {
    if (!valid[start] || visited[start]) {
      continue
    }

    const stack = [start]
    let mask = 0
    visited[start] = true

    while (stack.length > 0) {
      const corner = stack.pop()
      if (corner === undefined) {
        continue
      }

      mask |= 1 << corner
      const next = (corner + 1) % 4
      const previous = (corner + 3) % 4

      if (connected[corner] && !visited[next]) {
        visited[next] = true
        stack.push(next)
      }

      if (connected[previous] && !visited[previous]) {
        visited[previous] = true
        stack.push(previous)
      }
    }

    components.push(mask)
  }

  return components
}

function addFanTriangles(polygon: number[], indices: number[]): void {
  if (polygon.length < 3) {
    return
  }

  for (let index = 1; index < polygon.length - 1; index += 1) {
    const a = polygon[0]
    const b = polygon[index]
    const c = polygon[index + 1]

    if (a !== b && b !== c && c !== a) {
      indices.push(a, b, c)
    }
  }
}

function addPolygonEdges(polygon: number[], wireIndices: number[]): void {
  if (polygon.length < 2) {
    return
  }

  for (let index = 0; index < polygon.length; index += 1) {
    const a = polygon[index]
    const b = polygon[(index + 1) % polygon.length]

    if (a !== b) {
      wireIndices.push(a, b)
    }
  }
}

function smoothSilhouettes(
  screenX: number[],
  screenY: number[],
  depth: number[],
  boundary: boolean[],
  glueKeys: string[],
  indices: number[],
  iterations: number,
  maxDepthJump: number,
): void {
  const passes = Math.max(0, Math.round(iterations))
  if (passes === 0 || screenX.length === 0) {
    return
  }

  const neighbors = Array.from({ length: screenX.length }, () => new Set<number>())
  const glueGroups = new Map<string, number[]>()

  for (let vertex = 0; vertex < glueKeys.length; vertex += 1) {
    if (!boundary[vertex]) {
      continue
    }

    const group = glueGroups.get(glueKeys[vertex])
    if (group) {
      group.push(vertex)
    } else {
      glueGroups.set(glueKeys[vertex], [vertex])
    }
  }

  for (let index = 0; index < indices.length; index += 3) {
    linkNeighbors(neighbors, indices[index], indices[index + 1])
    linkNeighbors(neighbors, indices[index + 1], indices[index + 2])
    linkNeighbors(neighbors, indices[index + 2], indices[index])
  }

  for (let pass = 0; pass < passes; pass += 1) {
    const nextX = screenX.slice()
    const nextY = screenY.slice()

    for (let vertex = 0; vertex < screenX.length; vertex += 1) {
      if (!boundary[vertex]) {
        continue
      }

      let x = screenX[vertex]
      let y = screenY[vertex]
      let count = 1

      for (const neighbor of neighbors[vertex]) {
        if (Math.abs(depth[neighbor] - depth[vertex]) > maxDepthJump) {
          continue
        }

        x += screenX[neighbor]
        y += screenY[neighbor]
        count += 1
      }

      nextX[vertex] = x / count
      nextY[vertex] = y / count
    }

    for (const group of glueGroups.values()) {
      if (group.length < 2) {
        continue
      }

      const averageX = group.reduce((total, vertex) => total + nextX[vertex], 0) / group.length
      const averageY = group.reduce((total, vertex) => total + nextY[vertex], 0) / group.length

      for (const vertex of group) {
        nextX[vertex] = averageX
        nextY[vertex] = averageY
      }
    }

    for (let vertex = 0; vertex < screenX.length; vertex += 1) {
      screenX[vertex] = nextX[vertex]
      screenY[vertex] = nextY[vertex]
    }
  }
}

function unprojectVertices(
  screenX: number[],
  screenY: number[],
  depth: number[],
  camera: THREE.PerspectiveCamera,
  width: number,
  height: number,
): Float32Array {
  const positions = new Float32Array(screenX.length * 3)
  const projection = camera.projectionMatrix.elements

  for (let vertex = 0; vertex < screenX.length; vertex += 1) {
    const cameraDepth = depth[vertex]
    const ndcX = (screenX[vertex] / width) * 2 - 1
    const ndcY = 1 - (screenY[vertex] / height) * 2
    const cameraX = (ndcX * cameraDepth) / projection[0]
    const cameraY = (ndcY * cameraDepth) / projection[5]

    scratchUnprojected.set(cameraX, cameraY, -cameraDepth).applyMatrix4(camera.matrixWorld)

    const offset = vertex * 3
    positions[offset] = scratchUnprojected.x
    positions[offset + 1] = scratchUnprojected.y
    positions[offset + 2] = scratchUnprojected.z
  }

  return positions
}

function computeNormals(positions: Float32Array, indices: Uint32Array): Float32Array {
  const normals = new Float32Array(positions.length)

  for (let index = 0; index < indices.length; index += 3) {
    const a = indices[index] * 3
    const b = indices[index + 1] * 3
    const c = indices[index + 2] * 3
    const ab = vectorBetween(positions, a, b)
    const ac = vectorBetween(positions, a, c)
    const ba = vectorBetween(positions, b, a)
    const bc = vectorBetween(positions, b, c)
    const ca = vectorBetween(positions, c, a)
    const cb = vectorBetween(positions, c, b)
    const normal = normalizeVector(cross(ab, ac))

    if (normal[0] === 0 && normal[1] === 0 && normal[2] === 0) {
      continue
    }

    addNormal(normals, a, normal, angleBetween(ab, ac))
    addNormal(normals, b, normal, angleBetween(ba, bc))
    addNormal(normals, c, normal, angleBetween(ca, cb))
  }

  for (let index = 0; index < normals.length; index += 3) {
    const length = Math.hypot(normals[index], normals[index + 1], normals[index + 2]) || 1
    normals[index] /= length
    normals[index + 1] /= length
    normals[index + 2] /= length
  }

  return normals
}

function vectorBetween(values: Float32Array, from: number, to: number): [number, number, number] {
  return [values[to] - values[from], values[to + 1] - values[from + 1], values[to + 2] - values[from + 2]]
}

function cross(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

function normalizeVector(value: [number, number, number]): [number, number, number] {
  const length = Math.hypot(value[0], value[1], value[2])
  return length > 0 ? [value[0] / length, value[1] / length, value[2] / length] : [0, 0, 0]
}

function angleBetween(a: [number, number, number], b: [number, number, number]): number {
  const aLength = Math.hypot(a[0], a[1], a[2])
  const bLength = Math.hypot(b[0], b[1], b[2])

  if (aLength === 0 || bLength === 0) {
    return 0
  }

  return Math.acos(clamp((a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (aLength * bLength), -1, 1))
}

function addNormal(normals: Float32Array, offset: number, normal: [number, number, number], weight: number): void {
  normals[offset] += normal[0] * weight
  normals[offset + 1] += normal[1] * weight
  normals[offset + 2] += normal[2] * weight
}

function computeDepthColors(depth: number[]): Float32Array {
  const colors = new Float32Array(depth.length * 3)
  let minDepth = INF
  let maxDepth = -INF

  for (const value of depth) {
    minDepth = Math.min(minDepth, value)
    maxDepth = Math.max(maxDepth, value)
  }

  const range = Math.max(0.0001, maxDepth - minDepth)

  for (let vertex = 0; vertex < depth.length; vertex += 1) {
    const t = clamp((depth[vertex] - minDepth) / range, 0, 1)
    const offset = vertex * 3

    colors[offset] = 0.12 + (1 - t) * 0.78
    colors[offset + 1] = 0.35 + (1 - t) * 0.55
    colors[offset + 2] = 0.58 + t * 0.32
  }

  return colors
}

function binomialWeights(halfSize: number): number[] {
  const order = halfSize * 2
  const weights = [1]

  for (let row = 0; row < order; row += 1) {
    for (let index = weights.length - 1; index > 0; index -= 1) {
      weights[index] += weights[index - 1]
    }
    weights.push(1)
  }

  return weights
}

function connectedEdge(depth: Float32Array, a: number, b: number, maxDepthJump: number): boolean {
  return isFiniteDepth(depth[a]) && isFiniteDepth(depth[b]) && Math.abs(depth[a] - depth[b]) <= maxDepthJump
}

function edgeNeedsCutDepths(a: number, b: number, maxDepthJump: number): boolean {
  const aValid = isFiniteDepth(a)
  const bValid = isFiniteDepth(b)

  if (!aValid && !bValid) {
    return false
  }

  return aValid !== bValid || Math.abs(a - b) > maxDepthJump
}

function edgeVertexKey(a: number, b: number): string {
  return a < b ? `${a}:${b}` : `${b}:${a}`
}

function firstCornerInMask(mask: number): number | null {
  for (let corner = 0; corner < 4; corner += 1) {
    if (hasCorner(mask, corner)) {
      return corner
    }
  }

  return null
}

function hasCorner(mask: number, corner: number): boolean {
  return (mask & (1 << corner)) !== 0
}

function isFiniteDepth(value: number): boolean {
  return value < INF * 0.5
}

function linkNeighbors(neighbors: Array<Set<number>>, a: number, b: number): void {
  neighbors[a].add(b)
  neighbors[b].add(a)
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function clampInt(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}
