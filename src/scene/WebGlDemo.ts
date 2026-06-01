import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { ParticleField } from '../particles/ParticleField'
import { effectiveMaxDepthJump, type DemoSettings, type DemoStats, type RenderMode } from '../types/demo'

type StatsHandler = (stats: DemoStats) => void

const MAX_PARTICLES = 22000
const FAR_DEPTH = 30
const INVALID_DEPTH = 0.999
const MAX_GRID_CELLS = 120000
const MAX_SMOOTHING_HALF_SIZE = 8
const MAX_SPLAT_RADIUS_TEXELS = 192
const SURFACE_VERTICES_PER_CELL = 9

export class WebGlDemo {
  private readonly field = new ParticleField(MAX_PARTICLES)

  private readonly scene = new THREE.Scene()

  private readonly depthScene = new THREE.Scene()

  private readonly passScene = new THREE.Scene()

  private readonly camera = new THREE.PerspectiveCamera(46, 1, 0.1, FAR_DEPTH)

  private readonly passCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)

  private readonly canvas = document.createElement('canvas')

  private readonly renderer = new THREE.WebGLRenderer({
    canvas: this.canvas,
    antialias: true,
    alpha: true,
    powerPreference: 'high-performance',
  })

  private readonly controls: OrbitControls

  private readonly particleGeometry = new THREE.BufferGeometry()

  private readonly visiblePointMaterial = new THREE.PointsMaterial({
    color: 0xffb86b,
    size: 0.036,
    transparent: true,
    opacity: 0.78,
    depthWrite: false,
  })

  private readonly depthSplatMaterial = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: {
      captureViewMatrix: { value: new THREE.Matrix4() },
      captureProjectionMatrix: { value: new THREE.Matrix4() },
      targetSize: { value: new THREE.Vector2(1, 1) },
      radiusWorld: { value: 0.18 },
      farDepth: { value: FAR_DEPTH },
    },
    vertexShader: depthSplatVertexShader,
    fragmentShader: depthSplatFragmentShader,
    depthTest: true,
    depthWrite: true,
    transparent: false,
    toneMapped: false,
  })

  private readonly smoothMaterial = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: {
      depthTexture: { value: null as THREE.Texture | null },
      depthSize: { value: new THREE.Vector2(1, 1) },
      direction: { value: new THREE.Vector2(1, 0) },
      halfSize: { value: 2 },
      maxDepthJump: { value: 0.02 },
    },
    vertexShader: fullscreenVertexShader,
    fragmentShader: smoothFragmentShader,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
  })

  private readonly surfaceMaterial = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: {
      depthTexture: { value: null as THREE.Texture | null },
      captureProjectionMatrix: { value: new THREE.Matrix4() },
      captureInvViewMatrix: { value: new THREE.Matrix4() },
      viewportSize: { value: new THREE.Vector2(1, 1) },
      depthSize: { value: new THREE.Vector2(1, 1) },
      gridSpacing: { value: 6 },
      maxDepthJump: { value: 0.02 },
      materialThickness: { value: 0.46 },
      renderMode: { value: 0 },
      wireMode: { value: 0 },
      farDepth: { value: FAR_DEPTH },
    },
    vertexShader: surfaceVertexShader,
    fragmentShader: surfaceFragmentShader,
    side: THREE.DoubleSide,
    transparent: true,
    depthWrite: true,
  })

  private readonly wireMaterial = this.surfaceMaterial.clone()

  private readonly passMesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.smoothMaterial)

  private readonly points = new THREE.Points(this.particleGeometry, this.visiblePointMaterial)

  private readonly depthPoints = new THREE.Points(this.particleGeometry, this.depthSplatMaterial)

  private readonly surfaceMesh = new THREE.Mesh(new THREE.BufferGeometry(), this.surfaceMaterial)

  private readonly wireMesh = new THREE.Mesh(this.surfaceMesh.geometry, this.wireMaterial)

  private readonly captureViewMatrix = new THREE.Matrix4()

  private readonly captureProjectionMatrix = new THREE.Matrix4()

  private readonly captureInverseViewMatrix = new THREE.Matrix4()

  private readonly resizeObserver: ResizeObserver

  private settings: DemoSettings

  private viewportWidth = 1

  private viewportHeight = 1

  private gridColumns = 1

  private gridRows = 1

  private gridSpacingPixels = 6

  private depthTarget: THREE.WebGLRenderTarget | null = null

  private pingTarget: THREE.WebGLRenderTarget | null = null

  private pongTarget: THREE.WebGLRenderTarget | null = null

  private filteredDepthTexture: THREE.Texture | null = null

  private disposed = false

  private frames = 0

  private lastStatsAt = 0

  private lastRenderAt = 0

  private simulationTime = 0

  private depthInitialized = false

  private freezeCapturePending = false

  private readonly container: HTMLElement

  private readonly onStats: StatsHandler

  constructor(
    container: HTMLElement,
    initialSettings: DemoSettings,
    onStats: StatsHandler,
  ) {
    this.container = container
    this.onStats = onStats
    this.settings = { ...initialSettings }
    this.controls = new OrbitControls(this.camera, this.canvas)
    this.resizeObserver = new ResizeObserver(() => this.resize())
    this.wireMaterial.wireframe = true
    this.wireMaterial.uniforms.wireMode.value = 1
  }

  async start(): Promise<void> {
    this.configureScene()
    this.container.append(this.canvas)
    this.resizeObserver.observe(this.container)
    this.resize()
    this.renderer.setAnimationLoop((time) => this.render(time))
  }

  updateSettings(settings: DemoSettings): void {
    const gridChanged = settings.gridSpacing !== this.settings.gridSpacing
    const wasFrozen = this.settings.freezeSurface
    this.settings = { ...settings }
    this.controls.autoRotate = settings.animate && settings.mode === 'beauty' && !settings.freezeSurface
    this.updateMode(settings.mode)

    if (!wasFrozen && settings.freezeSurface) {
      this.freezeCapturePending = true
    }

    if (gridChanged) {
      this.ensureTargets(true)
    }
  }

  dispose(): void {
    this.disposed = true
    this.resizeObserver.disconnect()
    this.renderer.setAnimationLoop(null)
    this.controls.dispose()
    this.particleGeometry.dispose()
    this.visiblePointMaterial.dispose()
    this.depthSplatMaterial.dispose()
    this.smoothMaterial.dispose()
    this.surfaceMaterial.dispose()
    this.wireMaterial.dispose()
    this.passMesh.geometry.dispose()
    this.surfaceMesh.geometry.dispose()
    this.depthTarget?.dispose()
    this.pingTarget?.dispose()
    this.pongTarget?.dispose()
    this.renderer.dispose()
    this.canvas.remove()
  }

  private configureScene(): void {
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.65))
    this.renderer.setClearColor(0x05080d, 1)
    this.renderer.outputColorSpace = THREE.SRGBColorSpace
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping
    this.renderer.toneMappingExposure = 1.05

    this.scene.background = new THREE.Color(0x05080d)
    this.scene.fog = new THREE.Fog(0x05080d, 6.5, 13)
    this.camera.position.set(0.35, 1.1, 5.2)
    this.camera.lookAt(0, 0.05, 0)

    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.08
    this.controls.autoRotate = true
    this.controls.autoRotateSpeed = 0.24
    this.controls.enablePan = false
    this.controls.minDistance = 3.3
    this.controls.maxDistance = 8
    this.controls.target.set(0, 0.05, 0)

    this.particleGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(this.field.positions, 3).setUsage(THREE.DynamicDrawUsage),
    )

    this.points.frustumCulled = false
    this.depthPoints.frustumCulled = false
    this.surfaceMesh.frustumCulled = false
    this.wireMesh.frustumCulled = false

    this.scene.add(this.surfaceMesh, this.wireMesh, this.points)
    this.depthScene.add(this.depthPoints)
    this.passScene.add(this.passMesh)
    this.addStudio()
    this.updateMode(this.settings.mode)
  }

  private addStudio(): void {
    this.scene.add(new THREE.HemisphereLight(0xd9f6ff, 0x0d1422, 1.5))

    const key = new THREE.DirectionalLight(0xffffff, 4)
    key.position.set(-3.2, 4.6, 3.4)
    this.scene.add(key)

    const rim = new THREE.DirectionalLight(0x74e5ff, 2.6)
    rim.position.set(3.8, 1.9, -2.5)
    this.scene.add(rim)

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(9, 9),
      new THREE.MeshStandardMaterial({ color: 0x071019, roughness: 0.62, metalness: 0.18 }),
    )
    floor.rotation.x = -Math.PI / 2
    floor.position.y = -1.82
    this.scene.add(floor)

    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(1.9, 0.008, 8, 160),
      new THREE.MeshBasicMaterial({ color: 0x4fdcff, transparent: true, opacity: 0.2 }),
    )
    ring.rotation.x = Math.PI / 2
    ring.position.y = -1.79
    this.scene.add(ring)
  }

  private resize(): void {
    const bounds = this.container.getBoundingClientRect()
    this.viewportWidth = Math.max(1, Math.floor(bounds.width))
    this.viewportHeight = Math.max(1, Math.floor(bounds.height))
    this.camera.aspect = this.viewportWidth / this.viewportHeight
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(this.viewportWidth, this.viewportHeight, false)
    this.ensureTargets(true)
  }

  private render(timeMs: number): void {
    if (this.disposed) {
      return
    }

    if (this.lastRenderAt === 0) {
      this.lastRenderAt = timeMs
    }

    const deltaSeconds = Math.min(0.05, Math.max(0, (timeMs - this.lastRenderAt) / 1000))
    this.lastRenderAt = timeMs
    const frameStartedAt = performance.now()
    const particleCount = Math.min(this.settings.particleCount, MAX_PARTICLES)

    if (this.settings.animate && !this.settings.freezeSurface) {
      this.simulationTime += deltaSeconds
    }

    this.controls.update()
    this.ensureTargets(false)

    const refreshDepth = !this.settings.freezeSurface || this.freezeCapturePending || !this.depthInitialized

    if (refreshDepth) {
      this.field.update(this.simulationTime, this.settings.preset, particleCount)
      this.updateParticleGeometry(particleCount)
      this.captureCurrentCamera()
      this.renderDepthAndSmoothing()
      this.depthInitialized = true
      this.freezeCapturePending = false
    }

    this.updateSurfaceUniforms()
    this.renderMode()
    this.publishStats(timeMs, performance.now() - frameStartedAt, particleCount)
  }

  private updateParticleGeometry(particleCount: number): void {
    const position = this.particleGeometry.getAttribute('position')

    if (position) {
      position.needsUpdate = true
      this.particleGeometry.setDrawRange(0, particleCount)
    }
  }

  private captureCurrentCamera(): void {
    this.camera.updateMatrixWorld()
    this.camera.updateProjectionMatrix()
    this.captureViewMatrix.copy(this.camera.matrixWorldInverse)
    this.captureProjectionMatrix.copy(this.camera.projectionMatrix)
    this.captureInverseViewMatrix.copy(this.camera.matrixWorld)
  }

  private renderDepthAndSmoothing(): void {
    const depthTarget = this.requireDepthTarget()
    this.depthSplatMaterial.uniforms.captureViewMatrix.value.copy(this.captureViewMatrix)
    this.depthSplatMaterial.uniforms.captureProjectionMatrix.value.copy(this.captureProjectionMatrix)
    this.depthSplatMaterial.uniforms.targetSize.value.set(this.gridColumns, this.gridRows)
    this.depthSplatMaterial.uniforms.radiusWorld.value = this.settings.particleRadius

    this.renderer.setRenderTarget(depthTarget)
    // Render targets already use texture pixels. A manual setViewport here would be scaled by DPR.
    this.renderer.setScissorTest(false)
    this.renderer.setClearColor(0xffffff, 1)
    this.renderer.clear(true, true, true)
    this.renderer.render(this.depthScene, this.camera)

    const halfSize = Math.min(MAX_SMOOTHING_HALF_SIZE, Math.max(0, Math.round(this.settings.depthSmoothing)))
    if (halfSize === 0) {
      this.filteredDepthTexture = depthTarget.texture
      this.renderer.setRenderTarget(null)
      this.renderer.setClearColor(0x05080d, 1)
      return
    }

    const pingTarget = this.requirePingTarget()
    const pongTarget = this.requirePongTarget()

    this.smoothMaterial.uniforms.depthSize.value.set(this.gridColumns, this.gridRows)
    this.smoothMaterial.uniforms.halfSize.value = halfSize
    this.smoothMaterial.uniforms.maxDepthJump.value = this.effectiveNormalizedDepthJump()

    this.smoothMaterial.uniforms.depthTexture.value = depthTarget.texture
    this.smoothMaterial.uniforms.direction.value.set(1, 0)
    this.renderer.setRenderTarget(pingTarget)
    this.renderer.render(this.passScene, this.passCamera)

    this.smoothMaterial.uniforms.depthTexture.value = pingTarget.texture
    this.smoothMaterial.uniforms.direction.value.set(0, 1)
    this.renderer.setRenderTarget(pongTarget)
    this.renderer.render(this.passScene, this.passCamera)

    this.filteredDepthTexture = pongTarget.texture
    this.renderer.setRenderTarget(null)
    this.renderer.setClearColor(0x05080d, 1)
  }

  private updateSurfaceUniforms(): void {
    const uniforms = this.surfaceMaterial.uniforms
    const wireUniforms = this.wireMaterial.uniforms

    for (const target of [uniforms, wireUniforms]) {
      target.depthTexture.value = this.filteredDepthTexture ?? this.requireDepthTarget().texture
      target.captureProjectionMatrix.value.copy(this.captureProjectionMatrix)
      target.captureInvViewMatrix.value.copy(this.captureInverseViewMatrix)
      target.viewportSize.value.set(this.viewportWidth, this.viewportHeight)
      target.depthSize.value.set(this.gridColumns, this.gridRows)
      target.gridSpacing.value = this.gridSpacingPixels
      target.maxDepthJump.value = this.effectiveNormalizedDepthJump()
      target.materialThickness.value = this.settings.materialThickness
      target.renderMode.value = this.renderModeIndex(this.settings.mode)
    }
  }

  private effectiveNormalizedDepthJump(): number {
    return effectiveMaxDepthJump(this.settings) / FAR_DEPTH
  }

  private updateMode(mode: RenderMode): void {
    const surfaceMode = mode === 'beauty' || mode === 'mesh' || mode === 'depth'
    this.surfaceMesh.visible = surfaceMode
    this.wireMesh.visible = mode === 'mesh'
    this.points.visible = mode === 'particles'
  }

  private renderMode(): void {
    if (this.settings.mode === 'compare') {
      this.renderCompare()
      return
    }

    this.renderer.setScissorTest(false)
    this.updateMode(this.settings.mode)
    this.renderer.setViewport(0, 0, this.viewportWidth, this.viewportHeight)
    this.renderer.render(this.scene, this.camera)
  }

  private renderCompare(): void {
    const splitX = Math.round(this.viewportWidth * this.settings.split)

    this.renderer.setScissorTest(true)
    this.surfaceMesh.visible = false
    this.wireMesh.visible = false
    this.points.visible = true
    this.renderer.setViewport(0, 0, splitX, this.viewportHeight)
    this.renderer.setScissor(0, 0, splitX, this.viewportHeight)
    this.renderer.render(this.scene, this.camera)

    this.surfaceMesh.visible = true
    this.wireMesh.visible = false
    this.points.visible = false
    this.renderer.setViewport(splitX, 0, this.viewportWidth - splitX, this.viewportHeight)
    this.renderer.setScissor(splitX, 0, this.viewportWidth - splitX, this.viewportHeight)
    this.renderer.render(this.scene, this.camera)
    this.renderer.setScissorTest(false)
  }

  private ensureTargets(force: boolean): void {
    const spacing = this.effectiveGridSpacing()
    const columns = Math.ceil(this.viewportWidth / spacing) + 1
    const rows = Math.ceil(this.viewportHeight / spacing) + 1

    if (!force && columns === this.gridColumns && rows === this.gridRows && spacing === this.gridSpacingPixels) {
      return
    }

    this.gridColumns = columns
    this.gridRows = rows
    this.gridSpacingPixels = spacing
    this.depthTarget?.dispose()
    this.pingTarget?.dispose()
    this.pongTarget?.dispose()
    this.depthTarget = this.createDepthTarget(columns, rows)
    this.pingTarget = this.createDepthTarget(columns, rows)
    this.pongTarget = this.createDepthTarget(columns, rows)
    this.filteredDepthTexture = this.depthTarget.texture
    this.rebuildSurfaceGeometry(columns, rows)
    this.depthInitialized = false

    if (this.settings.freezeSurface) {
      this.freezeCapturePending = true
    }
  }

  private createDepthTarget(width: number, height: number): THREE.WebGLRenderTarget {
    const target = new THREE.WebGLRenderTarget(width, height, {
      format: THREE.RGBAFormat,
      type: THREE.HalfFloatType,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthBuffer: true,
      stencilBuffer: false,
      generateMipmaps: false,
    })
    target.texture.colorSpace = THREE.NoColorSpace
    target.texture.name = 'webgl-depth-field'
    return target
  }

  private rebuildSurfaceGeometry(columns: number, rows: number): void {
    const cellsX = Math.max(0, columns - 1)
    const cellsY = Math.max(0, rows - 1)
    const vertexCount = cellsX * cellsY * SURFACE_VERTICES_PER_CELL
    const positions = new Float32Array(vertexCount * 3)
    const cells = new Float32Array(vertexCount * 2)
    const slots = new Float32Array(vertexCount)
    let vertex = 0

    for (let y = 0; y < cellsY; y += 1) {
      for (let x = 0; x < cellsX; x += 1) {
        for (let slot = 0; slot < SURFACE_VERTICES_PER_CELL; slot += 1) {
          const cellOffset = vertex * 2
          cells[cellOffset] = x
          cells[cellOffset + 1] = y
          slots[vertex] = slot
          vertex += 1
        }
      }
    }

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.setAttribute('cell', new THREE.BufferAttribute(cells, 2))
    geometry.setAttribute('slot', new THREE.BufferAttribute(slots, 1))
    geometry.setDrawRange(0, vertexCount)
    geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 100)
    this.surfaceMesh.geometry.dispose()
    this.surfaceMesh.geometry = geometry
    this.wireMesh.geometry = geometry
  }

  private effectiveGridSpacing(): number {
    const requested = Math.max(1, Math.round(this.settings.gridSpacing))
    let spacing = requested

    while (Math.ceil(this.viewportWidth / spacing) * Math.ceil(this.viewportHeight / spacing) > MAX_GRID_CELLS) {
      spacing += 1
    }

    return spacing
  }

  private renderModeIndex(mode: RenderMode): number {
    if (mode === 'depth') {
      return 3
    }

    if (mode === 'mesh') {
      return 1
    }

    return 0
  }

  private publishStats(timeMs: number, frameMs: number, particleCount: number): void {
    this.frames += 1

    if (timeMs - this.lastStatsAt < 250) {
      return
    }

    const seconds = (timeMs - this.lastStatsAt) / 1000 || 1
    const fps = Math.round(this.frames / seconds)
    this.frames = 0
    this.lastStatsAt = timeMs

    const cells = Math.max(0, (this.gridColumns - 1) * (this.gridRows - 1))
    const vertices = cells * SURFACE_VERTICES_PER_CELL
    const smoothingPasses = this.settings.depthSmoothing > 0 ? 2 : 0
    const drawCalls = this.settings.freezeSurface ? 1 : 2 + smoothingPasses + (this.settings.mode === 'compare' ? 2 : 1)

    this.onStats({
      backend: 'WebGL',
      fps,
      particles: particleCount,
      vertices,
      triangles: cells * (SURFACE_VERTICES_PER_CELL / 3),
      grid: `${this.gridColumns} x ${this.gridRows}`,
      frameMs,
      cameraMs: 0,
      buffersMs: 0,
      uniformsMs: 0,
      commandsMs: frameMs,
      gpuComputeMs: null,
      gpuRenderMs: null,
      gpuTotalMs: null,
      gpuTimerAvailable: false,
      gpuTimingMode: 'unavailable',
      drawCalls,
      computeDispatches: 0,
      computeWorkgroups: 0,
      gridCells: this.gridColumns * this.gridRows,
      splatCells: this.estimateSplatCells(),
      canvas: `${this.canvas.width} x ${this.canvas.height}`,
      pixelRatio: Math.min(window.devicePixelRatio, 1.65),
      depthMemoryMb: this.estimateTextureMemoryMb(vertices),
      projectionWork: particleCount * this.estimateSplatCells(),
      smoothingWork: smoothingPasses * this.gridColumns * this.gridRows,
      meshWork: cells,
      surfaceWork: vertices,
      hotspot: this.settings.freezeSurface ? 'surface draw' : 'texture splat',
    })
  }

  private estimateSplatCells(): number {
    const averageDepth = 4.6
    const radiusTexels = Math.min(
      MAX_SPLAT_RADIUS_TEXELS,
      Math.max(
        1,
        (this.settings.particleRadius * this.viewportHeight * this.camera.projectionMatrix.elements[5]) /
          (2 * averageDepth * this.gridSpacingPixels),
      ),
    )

    return Math.max(1, Math.round(Math.PI * radiusTexels * radiusTexels))
  }

  private estimateTextureMemoryMb(surfaceVertices: number): number {
    const textureBytes = this.gridColumns * this.gridRows * 4 * 2 * 3
    const geometryBytes = surfaceVertices * 6 * Float32Array.BYTES_PER_ELEMENT
    return (textureBytes + geometryBytes) / (1024 * 1024)
  }

  private requireDepthTarget(): THREE.WebGLRenderTarget {
    if (!this.depthTarget) {
      throw new Error('Depth render target is not initialized')
    }

    return this.depthTarget
  }

  private requirePingTarget(): THREE.WebGLRenderTarget {
    if (!this.pingTarget) {
      throw new Error('Ping render target is not initialized')
    }

    return this.pingTarget
  }

  private requirePongTarget(): THREE.WebGLRenderTarget {
    if (!this.pongTarget) {
      throw new Error('Pong render target is not initialized')
    }

    return this.pongTarget
  }
}

const depthSplatVertexShader = /* glsl */ `
uniform mat4 captureViewMatrix;
uniform mat4 captureProjectionMatrix;
uniform vec2 targetSize;
uniform float radiusWorld;

out float vDepth;
out float vRadiusWorld;
out float vValid;

void main() {
  vec4 viewPosition = captureViewMatrix * vec4(position, 1.0);
  float cameraDepth = -viewPosition.z;
  vec4 clip = captureProjectionMatrix * viewPosition;
  vDepth = cameraDepth;
  vRadiusWorld = radiusWorld;
  vValid = cameraDepth > 0.1 && cameraDepth < ${FAR_DEPTH.toFixed(1)} && clip.w > 0.0 ? 1.0 : 0.0;

  if (vValid < 0.5) {
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
    gl_PointSize = 1.0;
    return;
  }

  float radiusTexels = clamp((radiusWorld * targetSize.y * captureProjectionMatrix[1][1]) / (2.0 * cameraDepth), 1.0, ${MAX_SPLAT_RADIUS_TEXELS.toFixed(1)});
  gl_Position = clip;
  gl_PointSize = radiusTexels * 2.0;
}
`

const depthSplatFragmentShader = /* glsl */ `
precision highp float;

uniform float farDepth;

in float vDepth;
in float vRadiusWorld;
in float vValid;

out vec4 outColor;

void main() {
  if (vValid < 0.5) {
    discard;
  }

  vec2 local = gl_PointCoord * 2.0 - 1.0;
  float distanceSquared = dot(local, local);

  if (distanceSquared > 1.0) {
    discard;
  }

  float bulge = sqrt(max(0.0, 1.0 - distanceSquared));
  float depth = vDepth - vRadiusWorld * 0.92 * bulge;
  float encodedDepth = clamp(depth / farDepth, 0.0, 1.0);
  gl_FragDepth = encodedDepth;
  outColor = vec4(encodedDepth, encodedDepth, encodedDepth, 1.0);
}
`

const fullscreenVertexShader = /* glsl */ `
out vec2 vUv;

void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`

const smoothFragmentShader = /* glsl */ `
precision highp float;

uniform sampler2D depthTexture;
uniform vec2 depthSize;
uniform vec2 direction;
uniform int halfSize;
uniform float maxDepthJump;

in vec2 vUv;
out vec4 outColor;

float weightFor(int halfSizeValue, int offset) {
  if (halfSizeValue == 1) {
    return offset == 0 ? 2.0 : 1.0;
  }

  if (halfSizeValue == 2) {
    if (offset == 0) { return 6.0; }
    if (offset == 1) { return 4.0; }
    return 1.0;
  }

  if (halfSizeValue == 3) {
    if (offset == 0) { return 20.0; }
    if (offset == 1) { return 15.0; }
    if (offset == 2) { return 6.0; }
    return 1.0;
  }

  if (halfSizeValue == 4) {
    if (offset == 0) { return 70.0; }
    if (offset == 1) { return 56.0; }
    if (offset == 2) { return 28.0; }
    if (offset == 3) { return 8.0; }
    return 1.0;
  }

  if (halfSizeValue == 5) {
    if (offset == 0) { return 252.0; }
    if (offset == 1) { return 210.0; }
    if (offset == 2) { return 120.0; }
    if (offset == 3) { return 45.0; }
    if (offset == 4) { return 10.0; }
    return 1.0;
  }

  if (halfSizeValue == 6) {
    if (offset == 0) { return 924.0; }
    if (offset == 1) { return 792.0; }
    if (offset == 2) { return 495.0; }
    if (offset == 3) { return 220.0; }
    if (offset == 4) { return 66.0; }
    if (offset == 5) { return 12.0; }
    return 1.0;
  }

  if (halfSizeValue == 7) {
    if (offset == 0) { return 3432.0; }
    if (offset == 1) { return 3003.0; }
    if (offset == 2) { return 2002.0; }
    if (offset == 3) { return 1001.0; }
    if (offset == 4) { return 364.0; }
    if (offset == 5) { return 91.0; }
    if (offset == 6) { return 14.0; }
    return 1.0;
  }

  if (offset == 0) { return 12870.0; }
  if (offset == 1) { return 11440.0; }
  if (offset == 2) { return 8008.0; }
  if (offset == 3) { return 4368.0; }
  if (offset == 4) { return 1820.0; }
  if (offset == 5) { return 560.0; }
  if (offset == 6) { return 120.0; }
  if (offset == 7) { return 16.0; }
  return 1.0;
}

float sampleDepth(vec2 uv) {
  return texture(depthTexture, clamp(uv, vec2(0.0), vec2(1.0))).r;
}

void main() {
  float center = sampleDepth(vUv);

  if (center >= ${INVALID_DEPTH.toFixed(3)}) {
    outColor = vec4(1.0);
    return;
  }

  int activeHalfSize = clamp(halfSize, 0, ${MAX_SMOOTHING_HALF_SIZE});
  float total = center * weightFor(activeHalfSize, 0);
  float totalWeight = weightFor(activeHalfSize, 0);
  vec2 texel = direction / depthSize;

  for (int step = 1; step <= ${MAX_SMOOTHING_HALF_SIZE}; step += 1) {
    if (step > activeHalfSize) {
      continue;
    }

    float a = sampleDepth(vUv - texel * float(step));
    float b = sampleDepth(vUv + texel * float(step));

    if (a < ${INVALID_DEPTH.toFixed(3)} && b < ${INVALID_DEPTH.toFixed(3)} && abs(a - center) <= maxDepthJump && abs(b - center) <= maxDepthJump) {
      float weight = weightFor(activeHalfSize, step);
      total += (a + b) * weight;
      totalWeight += weight * 2.0;
    }
  }

  float depth = total / totalWeight;
  outColor = vec4(depth, depth, depth, 1.0);
}
`

const surfaceVertexShader = /* glsl */ `
precision highp float;

uniform sampler2D depthTexture;
uniform mat4 captureProjectionMatrix;
uniform mat4 captureInvViewMatrix;
uniform vec2 viewportSize;
uniform vec2 depthSize;
uniform float gridSpacing;
uniform float maxDepthJump;
uniform float farDepth;

in vec2 cell;
in float slot;

out vec3 vWorld;
out vec3 vNormal;
out vec2 vLocal;
out float vDepth;
out float vValid;

struct SurfacePoint {
  vec2 screen;
  vec2 local;
  float depth;
  float valid;
};

float readDepth(vec2 node) {
  vec2 clampedNode = clamp(node, vec2(0.0), depthSize - vec2(1.0));
  // Render target UVs are bottom-left; grid nodes use top-left screen coordinates.
  vec2 textureNode = vec2(clampedNode.x, depthSize.y - 1.0 - clampedNode.y);
  return texture(depthTexture, (textureNode + vec2(0.5)) / depthSize).r;
}

bool validDepth(float depthValue) {
  return depthValue < ${INVALID_DEPTH.toFixed(3)};
}

vec3 unprojectScreen(vec2 screen, float depthValue) {
  vec2 ndc = vec2(screen.x / viewportSize.x * 2.0 - 1.0, 1.0 - screen.y / viewportSize.y * 2.0);
  float cameraDepth = depthValue * farDepth;
  vec4 camera = vec4(
    ndc.x * cameraDepth / captureProjectionMatrix[0][0],
    ndc.y * cameraDepth / captureProjectionMatrix[1][1],
    -cameraDepth,
    1.0
  );
  return (captureInvViewMatrix * camera).xyz;
}

vec3 unprojectNode(vec2 node, float depthValue) {
  return unprojectScreen(node * gridSpacing, depthValue);
}

bool closeDepth(float a, float b) {
  return abs(a - b) <= maxDepthJump;
}

bool maskHasCorner(int mask, int cornerIndex) {
  return (mask & (1 << cornerIndex)) != 0;
}

SurfacePoint invalidPoint() {
  return SurfacePoint(vec2(0.0), vec2(0.0), 1.0, 0.0);
}

SurfacePoint cornerPoint(int cornerIndex, vec2 corners[4], vec2 locals[4], float depths[4]) {
  return SurfacePoint(corners[cornerIndex], locals[cornerIndex], depths[cornerIndex], 1.0);
}

vec2 edgeScreen(int edgeIndex, vec2 corners[4], float t) {
  if (edgeIndex == 0) {
    return mix(corners[0], corners[1], t);
  }

  if (edgeIndex == 1) {
    return mix(corners[1], corners[2], t);
  }

  if (edgeIndex == 2) {
    return mix(corners[3], corners[2], t);
  }

  return mix(corners[0], corners[3], t);
}

vec2 edgeLocal(int edgeIndex, float t) {
  if (edgeIndex == 0) {
    return vec2(t, 0.0);
  }

  if (edgeIndex == 1) {
    return vec2(1.0, t);
  }

  if (edgeIndex == 2) {
    return vec2(t, 1.0);
  }

  return vec2(0.0, t);
}

int edgeStartCorner(int edgeIndex) {
  if (edgeIndex == 0) {
    return 0;
  }

  if (edgeIndex == 1) {
    return 1;
  }

  if (edgeIndex == 2) {
    return 3;
  }

  return 0;
}

int edgeEndCorner(int edgeIndex) {
  if (edgeIndex == 0) {
    return 1;
  }

  if (edgeIndex == 1) {
    return 2;
  }

  if (edgeIndex == 2) {
    return 2;
  }

  return 3;
}

SurfacePoint layerCutPoint(int edgeIndex, int mask, vec2 corners[4], float depths[4]) {
  float t = 0.5;
  int startCorner = edgeStartCorner(edgeIndex);
  int endCorner = edgeEndCorner(edgeIndex);
  int layerCorner = maskHasCorner(mask, startCorner) ? startCorner : endCorner;
  return SurfacePoint(edgeScreen(edgeIndex, corners, t), edgeLocal(edgeIndex, t), depths[layerCorner], 1.0);
}

SurfacePoint fullCellPoint(int slotIndex, vec2 corners[4], vec2 locals[4], float depths[4]) {
  if (slotIndex == 0) {
    return cornerPoint(0, corners, locals, depths);
  }

  if (slotIndex == 1) {
    return cornerPoint(1, corners, locals, depths);
  }

  if (slotIndex == 2) {
    return cornerPoint(2, corners, locals, depths);
  }

  if (slotIndex == 3) {
    return cornerPoint(0, corners, locals, depths);
  }

  if (slotIndex == 4) {
    return cornerPoint(2, corners, locals, depths);
  }

  if (slotIndex == 5) {
    return cornerPoint(3, corners, locals, depths);
  }

  return invalidPoint();
}

SurfacePoint oppositeMaskPoint(int slotIndex, int mask, vec2 corners[4], vec2 locals[4], float depths[4]) {
  if (slotIndex >= 6) {
    return invalidPoint();
  }

  int triangleIndex = slotIndex / 3;
  int pointIndex = slotIndex - triangleIndex * 3;
  int cornerIndex = 0;

  if (mask == 5) {
    cornerIndex = triangleIndex == 0 ? 0 : 2;
  } else {
    cornerIndex = triangleIndex == 0 ? 1 : 3;
  }

  int prevEdge = cornerIndex == 0 ? 3 : cornerIndex - 1;
  int nextEdge = cornerIndex;

  if (pointIndex == 0) {
    return layerCutPoint(prevEdge, mask, corners, depths);
  }

  if (pointIndex == 1) {
    return cornerPoint(cornerIndex, corners, locals, depths);
  }

  return layerCutPoint(nextEdge, mask, corners, depths);
}

SurfacePoint polygonMaskPoint(int slotIndex, int mask, vec2 corners[4], vec2 locals[4], float depths[4]) {
  SurfacePoint points[8];
  int count = 0;

  for (int cornerIndex = 0; cornerIndex < 4; cornerIndex += 1) {
    int nextCorner = (cornerIndex + 1) & 3;
    bool inside = maskHasCorner(mask, cornerIndex);
    bool nextInside = maskHasCorner(mask, nextCorner);

    if (inside) {
      points[count] = cornerPoint(cornerIndex, corners, locals, depths);
      count += 1;
    }

    if (inside != nextInside) {
      points[count] = layerCutPoint(cornerIndex, mask, corners, depths);
      count += 1;
    }
  }

  int triangleIndex = slotIndex / 3;
  int pointIndex = slotIndex - triangleIndex * 3;

  if (count < 3 || triangleIndex >= count - 2) {
    return invalidPoint();
  }

  int polygonIndex = pointIndex == 0 ? 0 : triangleIndex + pointIndex;
  return points[polygonIndex];
}

SurfacePoint pointForMask(int slotIndex, int mask, vec2 corners[4], vec2 locals[4], float depths[4]) {
  if (mask == 0) {
    return invalidPoint();
  }

  if (mask == 15) {
    return fullCellPoint(slotIndex, corners, locals, depths);
  }

  if (mask == 5 || mask == 10) {
    return oppositeMaskPoint(slotIndex, mask, corners, locals, depths);
  }

  return polygonMaskPoint(slotIndex, mask, corners, locals, depths);
}

vec3 normalAt(vec2 screen, float depthValue) {
  vec2 node = screen / gridSpacing;
  vec2 leftNode = node + vec2(-1.0, 0.0);
  vec2 rightNode = node + vec2(1.0, 0.0);
  vec2 upNode = node + vec2(0.0, -1.0);
  vec2 downNode = node + vec2(0.0, 1.0);
  float leftDepth = readDepth(leftNode);
  float rightDepth = readDepth(rightNode);
  float upDepth = readDepth(upNode);
  float downDepth = readDepth(downNode);
  leftDepth = validDepth(leftDepth) && closeDepth(leftDepth, depthValue) ? leftDepth : depthValue;
  rightDepth = validDepth(rightDepth) && closeDepth(rightDepth, depthValue) ? rightDepth : depthValue;
  upDepth = validDepth(upDepth) && closeDepth(upDepth, depthValue) ? upDepth : depthValue;
  downDepth = validDepth(downDepth) && closeDepth(downDepth, depthValue) ? downDepth : depthValue;
  vec3 px0 = unprojectScreen(screen + vec2(-gridSpacing, 0.0), leftDepth);
  vec3 px1 = unprojectScreen(screen + vec2(gridSpacing, 0.0), rightDepth);
  vec3 py0 = unprojectScreen(screen + vec2(0.0, -gridSpacing), upDepth);
  vec3 py1 = unprojectScreen(screen + vec2(0.0, gridSpacing), downDepth);
  vec3 normal = normalize(cross(px1 - px0, py1 - py0));
  return length(normal) > 0.0 ? normal : vec3(0.0, 0.0, 1.0);
}

void main() {
  vec2 baseScreen = cell * gridSpacing;
  vec2 corners[4] = vec2[4](
    baseScreen,
    baseScreen + vec2(gridSpacing, 0.0),
    baseScreen + vec2(gridSpacing, gridSpacing),
    baseScreen + vec2(0.0, gridSpacing)
  );
  vec2 locals[4] = vec2[4](
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),
    vec2(1.0, 1.0),
    vec2(0.0, 1.0)
  );
  float depths[4] = float[4](
    readDepth(cell),
    readDepth(cell + vec2(1.0, 0.0)),
    readDepth(cell + vec2(1.0, 1.0)),
    readDepth(cell + vec2(0.0, 1.0))
  );
  bool valid[4] = bool[4](
    validDepth(depths[0]),
    validDepth(depths[1]),
    validDepth(depths[2]),
    validDepth(depths[3])
  );
  int validMask = 0;
  validMask |= valid[0] ? 1 : 0;
  validMask |= valid[1] ? 2 : 0;
  validMask |= valid[2] ? 4 : 0;
  validMask |= valid[3] ? 8 : 0;

  float minDepth = min(
    min(valid[0] ? depths[0] : 1.0, valid[1] ? depths[1] : 1.0),
    min(valid[2] ? depths[2] : 1.0, valid[3] ? depths[3] : 1.0)
  );

  int frontMask = 0;
  frontMask |= valid[0] && depths[0] <= minDepth + maxDepthJump ? 1 : 0;
  frontMask |= valid[1] && depths[1] <= minDepth + maxDepthJump ? 2 : 0;
  frontMask |= valid[2] && depths[2] <= minDepth + maxDepthJump ? 4 : 0;
  frontMask |= valid[3] && depths[3] <= minDepth + maxDepthJump ? 8 : 0;

  SurfacePoint point = pointForMask(int(slot + 0.5), frontMask, corners, locals, depths);

  if (point.valid < 0.5) {
    vWorld = vec3(0.0);
    vNormal = vec3(0.0, 0.0, 1.0);
    vLocal = vec2(0.0);
    vDepth = 0.0;
    vValid = 0.0;
    gl_Position = vec4(0.0, -1000.0, 0.0, 1.0);
    return;
  }

  vec3 world = unprojectScreen(point.screen, point.depth);
  vec3 normal = normalAt(point.screen, point.depth);

  vWorld = world;
  vNormal = normal;
  vLocal = point.local;
  vDepth = point.depth * farDepth;
  vValid = 1.0;
  gl_Position = projectionMatrix * viewMatrix * vec4(world, 1.0);
}
`

const surfaceFragmentShader = /* glsl */ `
precision highp float;

uniform float materialThickness;
uniform int renderMode;
uniform int wireMode;

in vec3 vWorld;
in vec3 vNormal;
in vec2 vLocal;
in float vDepth;
in float vValid;

out vec4 outColor;

void main() {
  if (vValid < 0.5) {
    discard;
  }

  float depthTint = clamp((vDepth - 2.0) / 4.0, 0.0, 1.0);

  if (wireMode == 1) {
    outColor = vec4(0.9, 0.99, 1.0, 0.42);
    return;
  }

  if (renderMode == 3) {
    vec3 color = mix(vec3(0.88, 0.98, 1.0), vec3(0.03, 0.22, 0.35), depthTint);
    outColor = vec4(color, 0.96);
    return;
  }

  vec3 eyePosition = cameraPosition;
  vec3 viewDirection = normalize(eyePosition - vWorld);
  vec3 normal = normalize(vNormal);
  if (dot(normal, viewDirection) < 0.0) {
    normal = -normal;
  }

  vec3 keyLight = normalize(vec3(-0.52, 0.74, 0.42));
  vec3 rimLight = normalize(vec3(0.62, 0.16, -0.77));
  float diffuse = max(dot(normal, keyLight), 0.0);
  float rim = pow(max(1.0 - dot(normal, viewDirection), 0.0), 2.8);
  float fresnel = pow(max(1.0 - dot(normal, viewDirection), 0.0), 3.2);
  float specular = pow(max(dot(reflect(-keyLight, normal), viewDirection), 0.0), 62.0);
  float sideSpecular = pow(max(dot(reflect(-rimLight, normal), viewDirection), 0.0), 38.0);
  vec3 base = mix(vec3(0.20, 0.72, 0.9), vec3(0.72, 0.94, 0.99), 1.0 - depthTint);
  vec3 color = base * (0.28 + diffuse * 0.84) + vec3(1.0) * specular * 0.85 + vec3(0.45, 0.94, 1.0) * sideSpecular * 0.55;
  color += vec3(0.58, 0.95, 1.0) * rim * 0.56;

  if (renderMode == 1) {
    outColor = vec4(mix(vec3(0.05, 0.23, 0.3), color, 0.35), 0.94);
    return;
  }

  color += vec3(0.82, 0.98, 1.0) * fresnel * (0.28 + materialThickness * 0.22);
  outColor = vec4(color, clamp(0.52 + materialThickness * 0.28 + fresnel * 0.15, 0.48, 0.9));
}
`
