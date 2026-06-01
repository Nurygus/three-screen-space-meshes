import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { gpuDepthSurfaceShader } from '../gpu/shaders'
import type { DemoSettings, DemoStats, FlowPreset, GpuTimingMode, RenderMode } from '../types/demo'

type StatsHandler = (stats: DemoStats) => void

const MAX_PARTICLES = 40960
const UNIFORM_FLOATS = 112
const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus'
const TIMESTAMP_QUERY_COUNT = 3
const MESH_VERTEX_BYTES = 48
const EDGE_BYTES = 4
const MAX_MESH_VERTICES_PER_CELL = 24
const MESH_BUFFER_SAFETY_BYTES = 1024 * 1024
const BUFFER_USAGE = {
  MAP_READ: 0x1,
  COPY_SRC: 0x4,
  COPY_DST: 0x8,
  INDIRECT: 0x100,
  UNIFORM: 0x40,
  STORAGE: 0x80,
  QUERY_RESOLVE: 0x200,
} as const
const SHADER_STAGE = {
  VERTEX: 0x1,
  FRAGMENT: 0x2,
  COMPUTE: 0x4,
} as const
const TEXTURE_USAGE = {
  RENDER_ATTACHMENT: 0x10,
} as const
const MAP_MODE = {
  READ: 0x1,
} as const

interface FrameProfile {
  frameMs: number
  cameraMs: number
  buffersMs: number
  uniformsMs: number
  commandsMs: number
  refreshDepth: boolean
  drawCalls: number
  computeDispatches: number
  computeWorkgroups: number
  splatCells: number
  projectionWork: number
  smoothingWork: number
  meshWork: number
  surfaceWork: number
  hotspot: string
}

const presetIndex: Record<FlowPreset, number> = {
  vortex: 0,
  sheet: 1,
  crown: 2,
}

const modeIndex: Record<RenderMode, number> = {
  beauty: 0,
  mesh: 1,
  particles: 2,
  depth: 3,
  compare: 4,
}

export class WebGpuDemo {
  private readonly canvas = document.createElement('canvas')

  private readonly camera = new THREE.PerspectiveCamera(46, 1, 0.1, 30)

  private readonly controls: OrbitControls

  private readonly resizeObserver: ResizeObserver

  private adapter: GPUAdapter | null = null

  private device: GPUDevice | null = null

  private context: GPUCanvasContext | null = null

  private format: GPUTextureFormat | null = null

  private computeBindGroupLayout: GPUBindGroupLayout | null = null

  private renderBindGroupLayout: GPUBindGroupLayout | null = null

  private computeBindGroup: GPUBindGroup | null = null

  private renderBindGroup: GPUBindGroup | null = null

  private clearPipeline: GPUComputePipeline | null = null

  private projectPipeline: GPUComputePipeline | null = null

  private candidatePipeline: GPUComputePipeline | null = null

  private horizontalPipeline: GPUComputePipeline | null = null

  private verticalPipeline: GPUComputePipeline | null = null

  private closeGapPipeline: GPUComputePipeline | null = null

  private commitDepthPipeline: GPUComputePipeline | null = null

  private silhouettePipeline: GPUComputePipeline | null = null

  private emitMeshPipeline: GPUComputePipeline | null = null

  private finalizeMeshPipeline: GPUComputePipeline | null = null

  private backgroundPipeline: GPURenderPipeline | null = null

  private floorPipeline: GPURenderPipeline | null = null

  private surfacePipeline: GPURenderPipeline | null = null

  private particlePipeline: GPURenderPipeline | null = null

  private uniformBuffer: GPUBuffer | null = null

  private particleBuffer: GPUBuffer | null = null

  private rawDepthBuffer: GPUBuffer | null = null

  private tempDepthBuffer: GPUBuffer | null = null

  private filteredDepthBuffer: GPUBuffer | null = null

  private horizontalEdgeBuffer: GPUBuffer | null = null

  private verticalEdgeBuffer: GPUBuffer | null = null

  private meshVertexBuffer: GPUBuffer | null = null

  private indirectArgsBuffer: GPUBuffer | null = null

  private depthTexture: GPUTexture | null = null

  private timestampQuerySet: GPUQuerySet | null = null

  private timestampResolveBuffer: GPUBuffer | null = null

  private timestampReadBuffer: GPUBuffer | null = null

  private meshStatsReadBuffer: GPUBuffer | null = null

  private readonly uniformData = new Float32Array(UNIFORM_FLOATS)

  private readonly captureViewMatrix = new THREE.Matrix4()

  private readonly captureProjectionMatrix = new THREE.Matrix4()

  private readonly captureInverseViewMatrix = new THREE.Matrix4()

  private settings: DemoSettings

  private viewportWidth = 1

  private viewportHeight = 1

  private pixelRatio = 1

  private gridColumns = 1

  private gridRows = 1

  private gridSpacingPixels = 1

  private animationFrame = 0

  private disposed = false

  private frames = 0

  private lastStatsAt = 0

  private lastRenderAt = 0

  private simulationTime = 0

  private captureInitialized = false

  private freezeCapturePending = false

  private timestampQueriesAvailable = false

  private timestampReadPending = false

  private lastTimestampRequestAt = 0

  private queueTimingReadPending = false

  private lastQueueTimingRequestAt = 0

  private meshStatsReadPending = false

  private lastMeshStatsRequestAt = 0

  private gpuComputeMs: number | null = null

  private gpuRenderMs: number | null = null

  private gpuTotalMs: number | null = null

  private gpuTimingMode: GpuTimingMode = 'unavailable'

  private meshVertexCount = 0

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
  }

  async start(): Promise<void> {
    if (!navigator.gpu) {
      throw new Error('WebGPU is not available in this browser')
    }

    this.adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })

    if (this.disposed) {
      return
    }

    if (!this.adapter) {
      throw new Error('No WebGPU adapter was found')
    }

    this.timestampQueriesAvailable = this.adapter.features.has('timestamp-query' as GPUFeatureName)
    this.device = await this.adapter.requestDevice(
      this.timestampQueriesAvailable
        ? { requiredFeatures: ['timestamp-query' as GPUFeatureName] }
        : undefined,
    )

    if (this.disposed) {
      this.device.destroy()
      return
    }

    this.context = this.canvas.getContext('webgpu') as GPUCanvasContext | null

    if (!this.context) {
      throw new Error('Could not create a WebGPU canvas context')
    }

    this.device.addEventListener('uncapturederror', (event) => {
      console.error(`WebGPU ${event.error.constructor.name}: ${event.error.message}`)
    })
    this.device.lost.then((info) => {
      console.error(`WebGPU device lost: ${info.reason || 'unknown'} ${info.message}`)
    })

    this.format = navigator.gpu.getPreferredCanvasFormat()
    this.configureCamera()
    this.createPipelines()

    if (this.disposed) {
      return
    }

    this.container.append(this.canvas)
    this.resizeObserver.observe(this.container)
    this.resize()
    this.animationFrame = requestAnimationFrame(this.render)
  }

  updateSettings(settings: DemoSettings): void {
    const wasFrozen = this.settings.freezeSurface
    this.settings = { ...settings }
    this.controls.autoRotate = settings.animate && settings.mode === 'beauty' && !settings.freezeSurface

    if (!wasFrozen && settings.freezeSurface) {
      this.freezeCapturePending = true
    }
  }

  dispose(): void {
    this.disposed = true
    cancelAnimationFrame(this.animationFrame)
    this.resizeObserver.disconnect()
    this.controls.dispose()
    this.destroyGridBuffers()
    this.particleBuffer?.destroy()
    this.uniformBuffer?.destroy()
    this.depthTexture?.destroy()
    this.timestampQuerySet?.destroy()
    this.timestampResolveBuffer?.destroy()
    this.timestampReadBuffer?.destroy()
    this.meshStatsReadBuffer?.destroy()
    this.canvas.remove()
  }

  private configureCamera(): void {
    this.camera.position.set(0.35, 1.1, 5.2)
    this.camera.lookAt(-0.35, 0.05, 0)

    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.08
    this.controls.autoRotate = true
    this.controls.autoRotateSpeed = 0.24
    this.controls.enablePan = false
    this.controls.minDistance = 3.3
    this.controls.maxDistance = 8
    this.controls.target.set(-0.35, 0.05, 0)
  }

  private createPipelines(): void {
    const device = this.requireDevice()
    const format = this.requireFormat()
    const shaderModule = device.createShaderModule({ code: gpuDepthSurfaceShader })

    this.uniformBuffer = device.createBuffer({
      size: this.uniformData.byteLength,
      usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST,
    })
    this.particleBuffer = device.createBuffer({
      size: MAX_PARTICLES * 16,
      usage: BUFFER_USAGE.STORAGE,
    })

    if (this.timestampQueriesAvailable) {
      const timestampByteLength = TIMESTAMP_QUERY_COUNT * BigUint64Array.BYTES_PER_ELEMENT
      this.timestampQuerySet = device.createQuerySet({ type: 'timestamp', count: TIMESTAMP_QUERY_COUNT })
      this.timestampResolveBuffer = device.createBuffer({
        size: timestampByteLength,
        usage: BUFFER_USAGE.QUERY_RESOLVE | BUFFER_USAGE.COPY_SRC,
      })
      this.timestampReadBuffer = device.createBuffer({
        size: timestampByteLength,
        usage: BUFFER_USAGE.MAP_READ | BUFFER_USAGE.COPY_DST,
      })
    }

    this.meshStatsReadBuffer = device.createBuffer({
      size: Uint32Array.BYTES_PER_ELEMENT,
      usage: BUFFER_USAGE.MAP_READ | BUFFER_USAGE.COPY_DST,
    })

    this.computeBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: SHADER_STAGE.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: SHADER_STAGE.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: SHADER_STAGE.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: SHADER_STAGE.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: SHADER_STAGE.COMPUTE, buffer: { type: 'storage' } },
        { binding: 7, visibility: SHADER_STAGE.COMPUTE, buffer: { type: 'storage' } },
        { binding: 8, visibility: SHADER_STAGE.COMPUTE, buffer: { type: 'storage' } },
        { binding: 9, visibility: SHADER_STAGE.COMPUTE, buffer: { type: 'storage' } },
        { binding: 10, visibility: SHADER_STAGE.COMPUTE, buffer: { type: 'storage' } },
      ],
    })
    this.renderBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: SHADER_STAGE.VERTEX | SHADER_STAGE.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 5, visibility: SHADER_STAGE.VERTEX | SHADER_STAGE.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 6, visibility: SHADER_STAGE.VERTEX | SHADER_STAGE.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 12, visibility: SHADER_STAGE.VERTEX | SHADER_STAGE.FRAGMENT, buffer: { type: 'read-only-storage' } },
      ],
    })

    const computeLayout = device.createPipelineLayout({ bindGroupLayouts: [this.computeBindGroupLayout] })
    const renderLayout = device.createPipelineLayout({ bindGroupLayouts: [this.renderBindGroupLayout] })

    this.clearPipeline = device.createComputePipeline({
      layout: computeLayout,
      compute: { module: shaderModule, entryPoint: 'clearDepth' },
    })
    this.projectPipeline = device.createComputePipeline({
      layout: computeLayout,
      compute: { module: shaderModule, entryPoint: 'projectParticles' },
    })
    this.candidatePipeline = device.createComputePipeline({
      layout: computeLayout,
      compute: { module: shaderModule, entryPoint: 'selectSilhouetteCandidates' },
    })
    this.horizontalPipeline = device.createComputePipeline({
      layout: computeLayout,
      compute: { module: shaderModule, entryPoint: 'smoothHorizontal' },
    })
    this.verticalPipeline = device.createComputePipeline({
      layout: computeLayout,
      compute: { module: shaderModule, entryPoint: 'smoothVertical' },
    })
    this.closeGapPipeline = device.createComputePipeline({
      layout: computeLayout,
      compute: { module: shaderModule, entryPoint: 'closeDepthGaps' },
    })
    this.commitDepthPipeline = device.createComputePipeline({
      layout: computeLayout,
      compute: { module: shaderModule, entryPoint: 'commitClosedDepth' },
    })
    this.silhouettePipeline = device.createComputePipeline({
      layout: computeLayout,
      compute: { module: shaderModule, entryPoint: 'detectSilhouettes' },
    })
    this.emitMeshPipeline = device.createComputePipeline({
      layout: computeLayout,
      compute: { module: shaderModule, entryPoint: 'emitMesh' },
    })
    this.finalizeMeshPipeline = device.createComputePipeline({
      layout: computeLayout,
      compute: { module: shaderModule, entryPoint: 'finalizeMesh' },
    })

    const blend: GPUBlendState = {
      color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    }

    this.backgroundPipeline = device.createRenderPipeline({
      layout: renderLayout,
      vertex: { module: shaderModule, entryPoint: 'backgroundVertex' },
      fragment: { module: shaderModule, entryPoint: 'backgroundFragment', targets: [{ format }] },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'always' },
      primitive: { topology: 'triangle-list' },
    })
    this.floorPipeline = device.createRenderPipeline({
      layout: renderLayout,
      vertex: { module: shaderModule, entryPoint: 'floorVertex' },
      fragment: { module: shaderModule, entryPoint: 'floorFragment', targets: [{ format }] },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    })
    this.surfacePipeline = device.createRenderPipeline({
      layout: renderLayout,
      vertex: { module: shaderModule, entryPoint: 'surfaceVertex' },
      fragment: { module: shaderModule, entryPoint: 'surfaceFragment', targets: [{ format, blend }] },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    })
    this.particlePipeline = device.createRenderPipeline({
      layout: renderLayout,
      vertex: { module: shaderModule, entryPoint: 'particleVertex' },
      fragment: { module: shaderModule, entryPoint: 'particleFragment', targets: [{ format, blend }] },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'always' },
      primitive: { topology: 'triangle-list' },
    })
  }

  private resize(): void {
    const context = this.requireContext()
    const device = this.requireDevice()
    const format = this.requireFormat()
    const bounds = this.container.getBoundingClientRect()
    this.pixelRatio = Math.min(window.devicePixelRatio, 1.65)
    this.viewportWidth = Math.max(1, Math.floor(bounds.width * this.pixelRatio))
    this.viewportHeight = Math.max(1, Math.floor(bounds.height * this.pixelRatio))
    this.canvas.width = this.viewportWidth
    this.canvas.height = this.viewportHeight
    this.camera.aspect = this.viewportWidth / this.viewportHeight
    this.camera.updateProjectionMatrix()
    context.configure({ device, format, usage: TEXTURE_USAGE.RENDER_ATTACHMENT, alphaMode: 'opaque' })
    this.depthTexture?.destroy()
    this.depthTexture = device.createTexture({
      size: [this.viewportWidth, this.viewportHeight],
      format: DEPTH_FORMAT,
      usage: TEXTURE_USAGE.RENDER_ATTACHMENT,
    })
    this.ensureGridBuffers()
  }

  private render = (timeMs: number): void => {
    if (this.disposed) {
      return
    }

    if (this.lastRenderAt === 0) {
      this.lastRenderAt = timeMs
    }

    const deltaSeconds = Math.min(0.05, Math.max(0, (timeMs - this.lastRenderAt) / 1000))
    this.lastRenderAt = timeMs

    if (this.settings.animate && !this.settings.freezeSurface) {
      this.simulationTime += deltaSeconds
    }

    const frameStartedAt = performance.now()
    const cameraStartedAt = performance.now()
    this.controls.update()
    this.camera.updateMatrixWorld()
    this.camera.updateProjectionMatrix()
    const cameraMs = performance.now() - cameraStartedAt
    const buffersStartedAt = performance.now()
    this.ensureGridBuffers()
    const buffersMs = performance.now() - buffersStartedAt
    const uniformsStartedAt = performance.now()
    const refreshDepth = this.updateUniforms()
    const uniformsMs = performance.now() - uniformsStartedAt
    const commandsStartedAt = performance.now()
    const workProfile = this.encodeFrame(refreshDepth, timeMs)
    const commandsMs = performance.now() - commandsStartedAt
    this.publishStats(timeMs, {
      ...workProfile,
      frameMs: performance.now() - frameStartedAt,
      cameraMs,
      buffersMs,
      uniformsMs,
      commandsMs,
      refreshDepth,
    })
    this.animationFrame = requestAnimationFrame(this.render)
  }

  private ensureGridBuffers(): void {
    const device = this.requireDevice()
    const spacing = Math.max(1, this.settings.gridSpacing * this.pixelRatio)
    const columns = Math.ceil(this.viewportWidth / spacing) + 1
    const rows = Math.ceil(this.viewportHeight / spacing) + 1

    if (columns === this.gridColumns && rows === this.gridRows && Math.abs(spacing - this.gridSpacingPixels) < 0.01) {
      return
    }

    this.gridColumns = columns
    this.gridRows = rows
    this.gridSpacingPixels = spacing
    this.destroyGridBuffers()

    const depthByteLength = Math.max(4, this.gridColumns * this.gridRows * 4)
    const horizontalEdgeByteLength = Math.max(EDGE_BYTES, this.horizontalEdgeCount() * EDGE_BYTES)
    const verticalEdgeByteLength = Math.max(EDGE_BYTES, this.verticalEdgeCount() * EDGE_BYTES)
    const meshVertexByteLength = Math.max(MESH_VERTEX_BYTES, this.meshVertexCapacity() * MESH_VERTEX_BYTES)
    this.rawDepthBuffer = device.createBuffer({ size: depthByteLength, usage: BUFFER_USAGE.STORAGE })
    this.tempDepthBuffer = device.createBuffer({ size: depthByteLength, usage: BUFFER_USAGE.STORAGE })
    this.filteredDepthBuffer = device.createBuffer({ size: depthByteLength, usage: BUFFER_USAGE.STORAGE })
    this.horizontalEdgeBuffer = device.createBuffer({ size: horizontalEdgeByteLength, usage: BUFFER_USAGE.STORAGE })
    this.verticalEdgeBuffer = device.createBuffer({ size: verticalEdgeByteLength, usage: BUFFER_USAGE.STORAGE })
    this.meshVertexBuffer = device.createBuffer({ size: meshVertexByteLength, usage: BUFFER_USAGE.STORAGE })
    this.indirectArgsBuffer = device.createBuffer({
      size: 16,
      usage: BUFFER_USAGE.STORAGE | BUFFER_USAGE.INDIRECT | BUFFER_USAGE.COPY_SRC,
    })
    this.createBindGroups()

    if (this.settings.freezeSurface) {
      this.freezeCapturePending = true
    }
  }

  private createBindGroups(): void {
    const device = this.requireDevice()
    const uniformBuffer = this.requireUniformBuffer()
    const particleBuffer = this.requireParticleBuffer()
    const rawDepthBuffer = this.requireRawDepthBuffer()
    const tempDepthBuffer = this.requireTempDepthBuffer()
    const filteredDepthBuffer = this.requireFilteredDepthBuffer()
    const horizontalEdgeBuffer = this.requireHorizontalEdgeBuffer()
    const verticalEdgeBuffer = this.requireVerticalEdgeBuffer()
    const meshVertexBuffer = this.requireMeshVertexBuffer()
    const indirectArgsBuffer = this.requireIndirectArgsBuffer()
    const computeLayout = this.requireComputeLayout()
    const renderLayout = this.requireRenderLayout()

    this.computeBindGroup = device.createBindGroup({
      layout: computeLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: particleBuffer } },
        { binding: 2, resource: { buffer: rawDepthBuffer } },
        { binding: 3, resource: { buffer: tempDepthBuffer } },
        { binding: 4, resource: { buffer: filteredDepthBuffer } },
        { binding: 7, resource: { buffer: horizontalEdgeBuffer } },
        { binding: 8, resource: { buffer: verticalEdgeBuffer } },
        { binding: 9, resource: { buffer: meshVertexBuffer } },
        { binding: 10, resource: { buffer: indirectArgsBuffer } },
      ],
    })
    this.renderBindGroup = device.createBindGroup({
      layout: renderLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 5, resource: { buffer: particleBuffer } },
        { binding: 6, resource: { buffer: filteredDepthBuffer } },
        { binding: 12, resource: { buffer: meshVertexBuffer } },
      ],
    })
  }

  private destroyGridBuffers(): void {
    this.rawDepthBuffer?.destroy()
    this.tempDepthBuffer?.destroy()
    this.filteredDepthBuffer?.destroy()
    this.horizontalEdgeBuffer?.destroy()
    this.verticalEdgeBuffer?.destroy()
    this.meshVertexBuffer?.destroy()
    this.indirectArgsBuffer?.destroy()
    this.rawDepthBuffer = null
    this.tempDepthBuffer = null
    this.filteredDepthBuffer = null
    this.horizontalEdgeBuffer = null
    this.verticalEdgeBuffer = null
    this.meshVertexBuffer = null
    this.indirectArgsBuffer = null
    this.meshVertexCount = 0
    this.computeBindGroup = null
    this.renderBindGroup = null
  }

  private updateUniforms(): boolean {
    const device = this.requireDevice()
    const uniformBuffer = this.requireUniformBuffer()
    const refreshDepth = !this.settings.freezeSurface || this.freezeCapturePending || !this.captureInitialized

    if (refreshDepth) {
      this.captureCurrentCamera()
    }

    this.uniformData.set(this.camera.matrixWorldInverse.elements, 0)
    this.uniformData.set(this.camera.projectionMatrix.elements, 16)
    this.uniformData.set(this.camera.matrixWorld.elements, 32)
    this.uniformData.set(this.captureViewMatrix.elements, 48)
    this.uniformData.set(this.captureProjectionMatrix.elements, 64)
    this.uniformData.set(this.captureInverseViewMatrix.elements, 80)
    this.uniformData[96] = this.viewportWidth
    this.uniformData[97] = this.viewportHeight
    this.uniformData[98] = this.gridColumns
    this.uniformData[99] = this.gridRows
    this.uniformData[100] = this.simulationTime
    this.uniformData[101] = this.settings.particleRadius
    this.uniformData[102] = this.gridSpacingPixels
    this.uniformData[103] = this.settings.materialThickness
    this.uniformData[104] = Math.min(this.settings.particleCount, MAX_PARTICLES)
    this.uniformData[105] = presetIndex[this.settings.preset]
    this.uniformData[106] = this.settings.depthSmoothing
    this.uniformData[107] = modeIndex[this.settings.mode]
    this.uniformData[108] = this.settings.silhouetteSmoothing
    this.uniformData[109] = this.settings.zMax
    this.uniformData[110] = this.settings.useExplicitZMax ? 1 : 0
    this.uniformData[111] = this.settings.freezeSurface ? 1 : 0
    device.queue.writeBuffer(uniformBuffer, 0, this.uniformData)
    this.freezeCapturePending = false

    return refreshDepth
  }

  private captureCurrentCamera(): void {
    this.captureViewMatrix.copy(this.camera.matrixWorldInverse)
    this.captureProjectionMatrix.copy(this.camera.projectionMatrix)
    this.captureInverseViewMatrix.copy(this.camera.matrixWorld)
    this.captureInitialized = true
  }

  private encodeFrame(refreshDepth: boolean, timeMs: number): Omit<
    FrameProfile,
    'frameMs' | 'cameraMs' | 'buffersMs' | 'uniformsMs' | 'commandsMs' | 'refreshDepth'
  > {
    const device = this.requireDevice()
    const context = this.requireContext()
    const computeBindGroup = this.requireComputeBindGroup()
    const renderBindGroup = this.requireRenderBindGroup()
    const clearPipeline = this.requireClearPipeline()
    const projectPipeline = this.requireProjectPipeline()
    const candidatePipeline = this.requireCandidatePipeline()
    const horizontalPipeline = this.requireHorizontalPipeline()
    const verticalPipeline = this.requireVerticalPipeline()
    const closeGapPipeline = this.requireCloseGapPipeline()
    const commitDepthPipeline = this.requireCommitDepthPipeline()
    const silhouettePipeline = this.requireSilhouettePipeline()
    const emitMeshPipeline = this.requireEmitMeshPipeline()
    const finalizeMeshPipeline = this.requireFinalizeMeshPipeline()
    const backgroundPipeline = this.requireBackgroundPipeline()
    const floorPipeline = this.requireFloorPipeline()
    const surfacePipeline = this.requireSurfacePipeline()
    const particlePipeline = this.requireParticlePipeline()
    const depthTexture = this.requireDepthTexture()
    const encoder = device.createCommandEncoder()
    const gridCells = this.gridColumns * this.gridRows
    const edgeCount = Math.max(this.horizontalEdgeCount(), this.verticalEdgeCount(), 1)
    const meshCells = Math.max(this.surfaceCellCount(), 1)
    const clearWorkgroups = Math.max(1, Math.ceil(this.clearElementCount() / 256))
    const gridWorkgroups = Math.max(1, Math.ceil(gridCells / 256))
    const edgeWorkgroups = Math.max(1, Math.ceil(edgeCount / 256))
    const meshWorkgroups = Math.max(1, Math.ceil(meshCells / 128))
    const particleWorkgroups = Math.max(1, Math.ceil(Math.min(this.settings.particleCount, MAX_PARTICLES) / 64))
    const canUseGpuTimestamps = this.canUseGpuTimestamps(encoder)
    const shouldProfileGpu = canUseGpuTimestamps && this.beginGpuTimestampProfile(timeMs)
    const queueProfileStartedAt = canUseGpuTimestamps ? null : this.beginGpuQueueProfile(timeMs)
    const shouldReadMeshStats = this.beginMeshStatsRead(timeMs)

    if (shouldProfileGpu) {
      this.writeGpuTimestamp(encoder, 0)
    }

    if (refreshDepth) {
      const computePass = encoder.beginComputePass()
      computePass.setBindGroup(0, computeBindGroup)
      computePass.setPipeline(clearPipeline)
      computePass.dispatchWorkgroups(clearWorkgroups)
      computePass.setPipeline(projectPipeline)
      computePass.dispatchWorkgroups(particleWorkgroups)
      computePass.setPipeline(candidatePipeline)
      computePass.dispatchWorkgroups(particleWorkgroups)
      computePass.setPipeline(horizontalPipeline)
      computePass.dispatchWorkgroups(gridWorkgroups)
      computePass.setPipeline(verticalPipeline)
      computePass.dispatchWorkgroups(gridWorkgroups)
      computePass.setPipeline(closeGapPipeline)
      computePass.dispatchWorkgroups(gridWorkgroups)
      computePass.setPipeline(commitDepthPipeline)
      computePass.dispatchWorkgroups(gridWorkgroups)
      computePass.setPipeline(silhouettePipeline)
      computePass.dispatchWorkgroups(edgeWorkgroups)
      computePass.setPipeline(emitMeshPipeline)
      computePass.dispatchWorkgroups(meshWorkgroups)
      computePass.setPipeline(finalizeMeshPipeline)
      computePass.dispatchWorkgroups(1)
      computePass.end()
    }

    if (shouldReadMeshStats) {
      encoder.copyBufferToBuffer(this.requireIndirectArgsBuffer(), 0, this.requireMeshStatsReadBuffer(), 0, 4)
    }

    if (shouldProfileGpu) {
      this.writeGpuTimestamp(encoder, 1)
    }

    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.012, g: 0.018, b: 0.028, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'discard',
      },
    })

    renderPass.setBindGroup(0, renderBindGroup)
    renderPass.setPipeline(backgroundPipeline)
    renderPass.draw(3)
    renderPass.setPipeline(floorPipeline)
    renderPass.draw(6)

    if (this.settings.mode === 'particles') {
      this.drawParticles(renderPass, particlePipeline)
    } else if (this.settings.mode === 'compare') {
      const splitX = Math.round(this.viewportWidth * this.settings.split)
      renderPass.setScissorRect(0, 0, splitX, this.viewportHeight)
      this.drawParticles(renderPass, particlePipeline)
      renderPass.setScissorRect(splitX, 0, this.viewportWidth - splitX, this.viewportHeight)
      this.drawSurface(renderPass, surfacePipeline)
      renderPass.setScissorRect(0, 0, this.viewportWidth, this.viewportHeight)
    } else {
      this.drawSurface(renderPass, surfacePipeline)
    }

    renderPass.end()
    if (shouldProfileGpu) {
      this.writeGpuTimestamp(encoder, 2)
      this.resolveGpuTimestampProfile(encoder)
    }
    device.queue.submit([encoder.finish()])

    if (shouldProfileGpu) {
      this.readGpuTimestampProfile()
    }

    if (queueProfileStartedAt !== null) {
      this.readGpuQueueProfile(queueProfileStartedAt)
    }

    if (shouldReadMeshStats) {
      this.readMeshStats()
    }

    return this.frameWorkProfile(refreshDepth, {
      clearWorkgroups,
      gridWorkgroups,
      edgeWorkgroups,
      meshWorkgroups,
      particleWorkgroups,
    })
  }

  private canUseGpuTimestamps(encoder: GPUCommandEncoder): boolean {
    return (
      this.timestampQueriesAvailable &&
      this.encoderCanWriteTimestamp(encoder) &&
      !!this.timestampQuerySet &&
      !!this.timestampResolveBuffer &&
      !!this.timestampReadBuffer
    )
  }

  private beginGpuTimestampProfile(timeMs: number): boolean {
    if (this.timestampReadPending || timeMs - this.lastTimestampRequestAt < 500) {
      return false
    }

    this.lastTimestampRequestAt = timeMs
    this.timestampReadPending = true
    return true
  }

  private encoderCanWriteTimestamp(
    encoder: GPUCommandEncoder,
  ): encoder is GPUCommandEncoder & { writeTimestamp: (querySet: GPUQuerySet, queryIndex: number) => void } {
    return typeof (encoder as { writeTimestamp?: unknown }).writeTimestamp === 'function'
  }

  private writeGpuTimestamp(encoder: GPUCommandEncoder, queryIndex: number): void {
    if (this.encoderCanWriteTimestamp(encoder)) {
      encoder.writeTimestamp(this.requireTimestampQuerySet(), queryIndex)
    }
  }

  private resolveGpuTimestampProfile(encoder: GPUCommandEncoder): void {
    const querySet = this.requireTimestampQuerySet()
    const resolveBuffer = this.requireTimestampResolveBuffer()
    const readBuffer = this.requireTimestampReadBuffer()
    const byteLength = TIMESTAMP_QUERY_COUNT * BigUint64Array.BYTES_PER_ELEMENT

    encoder.resolveQuerySet(querySet, 0, TIMESTAMP_QUERY_COUNT, resolveBuffer, 0)
    encoder.copyBufferToBuffer(resolveBuffer, 0, readBuffer, 0, byteLength)
  }

  private readGpuTimestampProfile(): void {
    const readBuffer = this.requireTimestampReadBuffer()
    let mapped = false

    readBuffer
      .mapAsync(MAP_MODE.READ)
      .then(() => {
        mapped = true
        if (this.disposed) {
          return
        }

        const data = new BigUint64Array(readBuffer.getMappedRange().slice(0))
        const computeNs = data[1] > data[0] ? data[1] - data[0] : 0n
        const renderNs = data[2] > data[1] ? data[2] - data[1] : 0n
        const totalNs = data[2] > data[0] ? data[2] - data[0] : 0n
        this.gpuComputeMs = Number(computeNs) / 1_000_000
        this.gpuRenderMs = Number(renderNs) / 1_000_000
        this.gpuTotalMs = Number(totalNs) / 1_000_000
        this.gpuTimingMode = 'timestamp'
      })
      .catch(() => {
        this.gpuComputeMs = null
        this.gpuRenderMs = null
        this.gpuTotalMs = null
        this.gpuTimingMode = 'unavailable'
      })
      .finally(() => {
        if (mapped) {
          readBuffer.unmap()
        }
        this.timestampReadPending = false
      })
  }

  private beginGpuQueueProfile(timeMs: number): number | null {
    if (this.queueTimingReadPending || timeMs - this.lastQueueTimingRequestAt < 500) {
      return null
    }

    this.lastQueueTimingRequestAt = timeMs
    this.queueTimingReadPending = true
    return performance.now()
  }

  private readGpuQueueProfile(startedAt: number): void {
    this.requireDevice()
      .queue.onSubmittedWorkDone()
      .then(() => {
        if (this.disposed) {
          return
        }

        this.gpuComputeMs = null
        this.gpuRenderMs = null
        this.gpuTotalMs = performance.now() - startedAt
        this.gpuTimingMode = 'queue'
      })
      .catch(() => {
        this.gpuComputeMs = null
        this.gpuRenderMs = null
        this.gpuTotalMs = null
        this.gpuTimingMode = 'unavailable'
      })
      .finally(() => {
        this.queueTimingReadPending = false
      })
  }

  private beginMeshStatsRead(timeMs: number): boolean {
    if (!this.meshStatsReadBuffer || this.meshStatsReadPending || timeMs - this.lastMeshStatsRequestAt < 250) {
      return false
    }

    this.lastMeshStatsRequestAt = timeMs
    this.meshStatsReadPending = true
    return true
  }

  private readMeshStats(): void {
    const readBuffer = this.requireMeshStatsReadBuffer()
    let mapped = false

    readBuffer
      .mapAsync(MAP_MODE.READ)
      .then(() => {
        mapped = true
        if (this.disposed) {
          return
        }

        this.meshVertexCount = new Uint32Array(readBuffer.getMappedRange().slice(0))[0] ?? 0
      })
      .catch(() => {
        this.meshVertexCount = 0
      })
      .finally(() => {
        if (mapped) {
          readBuffer.unmap()
        }
        this.meshStatsReadPending = false
      })
  }

  private frameWorkProfile(
    refreshDepth: boolean,
    workgroups: {
      clearWorkgroups: number
      gridWorkgroups: number
      edgeWorkgroups: number
      meshWorkgroups: number
      particleWorkgroups: number
    },
  ): Omit<FrameProfile, 'frameMs' | 'cameraMs' | 'buffersMs' | 'uniformsMs' | 'commandsMs' | 'refreshDepth'> {
    const gridCells = this.gridColumns * this.gridRows
    const particleCount = Math.min(this.settings.particleCount, MAX_PARTICLES)
    const surfaceVertices = this.surfaceVertexCount()
    const particleVertices = particleCount * 6
    const splatCells = this.estimateSplatCells()
    const projectionWork = refreshDepth ? particleCount * splatCells * 2 : 0
    const smoothingWork = refreshDepth ? gridCells * (3 + Math.max(1, this.settings.depthSmoothing) * 4) : 0
    const meshWork = refreshDepth ? this.horizontalEdgeCount() + this.verticalEdgeCount() + this.surfaceCellCount() * 4 : 0
    const surfaceWork = this.settings.mode === 'particles' ? particleVertices : surfaceVertices

    return {
      drawCalls: this.drawCallCount(),
      computeDispatches: refreshDepth ? 10 : 0,
      computeWorkgroups: refreshDepth
        ? workgroups.clearWorkgroups +
          workgroups.gridWorkgroups * 4 +
          workgroups.edgeWorkgroups +
          workgroups.meshWorkgroups +
          workgroups.particleWorkgroups * 2 +
          1
        : 0,
      splatCells,
      projectionWork,
      smoothingWork,
      meshWork,
      surfaceWork,
      hotspot: this.hotspotFor(projectionWork, smoothingWork, meshWork, surfaceWork, refreshDepth),
    }
  }

  private estimateSplatCells(): number {
    const averageDepth = 4.6
    const radiusPixels = Math.min(
      192,
      Math.max(
        4,
        (this.settings.particleRadius * this.viewportHeight * this.camera.projectionMatrix.elements[5]) /
          (2 * averageDepth),
      ),
    )

    return Math.max(1, Math.round(Math.PI * (radiusPixels / this.gridSpacingPixels) ** 2))
  }

  private hotspotFor(
    projectionWork: number,
    smoothingWork: number,
    meshWork: number,
    surfaceWork: number,
    refreshDepth: boolean,
  ): string {
    if (!refreshDepth) {
      return 'surface draw'
    }

    if (projectionWork >= smoothingWork && projectionWork >= meshWork && projectionWork >= surfaceWork) {
      return 'particle splatting'
    }

    if (smoothingWork >= meshWork && smoothingWork >= surfaceWork) {
      return 'depth smoothing'
    }

    if (meshWork >= surfaceWork) {
      return 'mesh extraction'
    }

    return 'surface draw'
  }

  private drawCallCount(): number {
    return this.settings.mode === 'compare' ? 4 : 3
  }

  private drawSurface(pass: GPURenderPassEncoder, pipeline: GPURenderPipeline): void {
    pass.setPipeline(pipeline)
    pass.drawIndirect(this.requireIndirectArgsBuffer(), 0)
  }

  private drawParticles(pass: GPURenderPassEncoder, pipeline: GPURenderPipeline): void {
    pass.setPipeline(pipeline)
    pass.draw(Math.min(this.settings.particleCount, MAX_PARTICLES) * 6)
  }

  private surfaceVertexCount(): number {
    return this.meshVertexCount || Math.min(this.meshVertexCapacity(), this.surfaceCellCount() * 6)
  }

  private surfaceCellCount(): number {
    return Math.max(0, (this.gridColumns - 1) * (this.gridRows - 1))
  }

  private horizontalEdgeCount(): number {
    return Math.max(0, (this.gridColumns - 1) * this.gridRows)
  }

  private verticalEdgeCount(): number {
    return Math.max(0, this.gridColumns * (this.gridRows - 1))
  }

  private meshVertexCapacity(): number {
    const requestedCapacity = this.surfaceCellCount() * MAX_MESH_VERTICES_PER_CELL
    const storageLimit = Math.max(0, this.requireDevice().limits.maxStorageBufferBindingSize - MESH_BUFFER_SAFETY_BYTES)
    const limitCapacity = Math.max(3, Math.floor(storageLimit / MESH_VERTEX_BYTES))
    const capacity = Math.min(requestedCapacity, limitCapacity)

    return Math.max(3, capacity - (capacity % 3))
  }

  private clearElementCount(): number {
    return Math.max(this.gridColumns * this.gridRows, this.horizontalEdgeCount(), this.verticalEdgeCount(), 1)
  }

  private publishStats(timeMs: number, profile: FrameProfile): void {
    this.frames += 1

    if (timeMs - this.lastStatsAt < 250) {
      return
    }

    const seconds = (timeMs - this.lastStatsAt) / 1000 || 1
    const fps = Math.round(this.frames / seconds)
    this.frames = 0
    this.lastStatsAt = timeMs
    const vertices = this.surfaceVertexCount()
    const triangles = Math.floor(vertices / 3)
    const gridCells = this.gridColumns * this.gridRows
    const depthMemoryMb =
      (gridCells * 4 * 3 +
        this.horizontalEdgeCount() * EDGE_BYTES +
        this.verticalEdgeCount() * EDGE_BYTES +
        this.meshVertexCapacity() * MESH_VERTEX_BYTES) /
      (1024 * 1024)

    this.onStats({
      backend: 'WebGPU',
      fps,
      particles: Math.min(this.settings.particleCount, MAX_PARTICLES),
      vertices,
      triangles,
      grid: `${this.gridColumns} x ${this.gridRows}`,
      frameMs: profile.frameMs,
      cameraMs: profile.cameraMs,
      buffersMs: profile.buffersMs,
      uniformsMs: profile.uniformsMs,
      commandsMs: profile.commandsMs,
      gpuComputeMs: this.gpuComputeMs,
      gpuRenderMs: this.gpuRenderMs,
      gpuTotalMs: this.gpuTotalMs,
      gpuTimerAvailable: this.timestampQueriesAvailable,
      gpuTimingMode: this.gpuTimingMode,
      drawCalls: profile.drawCalls,
      computeDispatches: profile.computeDispatches,
      computeWorkgroups: profile.computeWorkgroups,
      gridCells,
      splatCells: profile.splatCells,
      canvas: `${this.viewportWidth} x ${this.viewportHeight}`,
      pixelRatio: this.pixelRatio,
      depthMemoryMb,
      projectionWork: profile.projectionWork,
      smoothingWork: profile.smoothingWork,
      meshWork: profile.meshWork,
      surfaceWork: profile.surfaceWork,
      hotspot: profile.hotspot,
    })
  }

  private requireDevice(): GPUDevice {
    if (!this.device) {
      throw new Error('WebGPU device is not initialized')
    }

    return this.device
  }

  private requireContext(): GPUCanvasContext {
    if (!this.context) {
      throw new Error('WebGPU context is not initialized')
    }

    return this.context
  }

  private requireFormat(): GPUTextureFormat {
    if (!this.format) {
      throw new Error('WebGPU format is not initialized')
    }

    return this.format
  }

  private requireUniformBuffer(): GPUBuffer {
    if (!this.uniformBuffer) {
      throw new Error('Uniform buffer is not initialized')
    }

    return this.uniformBuffer
  }

  private requireParticleBuffer(): GPUBuffer {
    if (!this.particleBuffer) {
      throw new Error('Particle buffer is not initialized')
    }

    return this.particleBuffer
  }

  private requireRawDepthBuffer(): GPUBuffer {
    if (!this.rawDepthBuffer) {
      throw new Error('Raw depth buffer is not initialized')
    }

    return this.rawDepthBuffer
  }

  private requireTempDepthBuffer(): GPUBuffer {
    if (!this.tempDepthBuffer) {
      throw new Error('Temporary depth buffer is not initialized')
    }

    return this.tempDepthBuffer
  }

  private requireFilteredDepthBuffer(): GPUBuffer {
    if (!this.filteredDepthBuffer) {
      throw new Error('Filtered depth buffer is not initialized')
    }

    return this.filteredDepthBuffer
  }

  private requireHorizontalEdgeBuffer(): GPUBuffer {
    if (!this.horizontalEdgeBuffer) {
      throw new Error('Horizontal edge buffer is not initialized')
    }

    return this.horizontalEdgeBuffer
  }

  private requireVerticalEdgeBuffer(): GPUBuffer {
    if (!this.verticalEdgeBuffer) {
      throw new Error('Vertical edge buffer is not initialized')
    }

    return this.verticalEdgeBuffer
  }

  private requireMeshVertexBuffer(): GPUBuffer {
    if (!this.meshVertexBuffer) {
      throw new Error('Mesh vertex buffer is not initialized')
    }

    return this.meshVertexBuffer
  }

  private requireIndirectArgsBuffer(): GPUBuffer {
    if (!this.indirectArgsBuffer) {
      throw new Error('Indirect argument buffer is not initialized')
    }

    return this.indirectArgsBuffer
  }

  private requireMeshStatsReadBuffer(): GPUBuffer {
    if (!this.meshStatsReadBuffer) {
      throw new Error('Mesh stats readback buffer is not initialized')
    }

    return this.meshStatsReadBuffer
  }

  private requireDepthTexture(): GPUTexture {
    if (!this.depthTexture) {
      throw new Error('Depth texture is not initialized')
    }

    return this.depthTexture
  }

  private requireTimestampQuerySet(): GPUQuerySet {
    if (!this.timestampQuerySet) {
      throw new Error('Timestamp query set is not initialized')
    }

    return this.timestampQuerySet
  }

  private requireTimestampResolveBuffer(): GPUBuffer {
    if (!this.timestampResolveBuffer) {
      throw new Error('Timestamp resolve buffer is not initialized')
    }

    return this.timestampResolveBuffer
  }

  private requireTimestampReadBuffer(): GPUBuffer {
    if (!this.timestampReadBuffer) {
      throw new Error('Timestamp read buffer is not initialized')
    }

    return this.timestampReadBuffer
  }

  private requireComputeLayout(): GPUBindGroupLayout {
    if (!this.computeBindGroupLayout) {
      throw new Error('Compute bind group layout is not initialized')
    }

    return this.computeBindGroupLayout
  }

  private requireRenderLayout(): GPUBindGroupLayout {
    if (!this.renderBindGroupLayout) {
      throw new Error('Render bind group layout is not initialized')
    }

    return this.renderBindGroupLayout
  }

  private requireComputeBindGroup(): GPUBindGroup {
    if (!this.computeBindGroup) {
      throw new Error('Compute bind group is not initialized')
    }

    return this.computeBindGroup
  }

  private requireRenderBindGroup(): GPUBindGroup {
    if (!this.renderBindGroup) {
      throw new Error('Render bind group is not initialized')
    }

    return this.renderBindGroup
  }

  private requireClearPipeline(): GPUComputePipeline {
    if (!this.clearPipeline) {
      throw new Error('Clear pipeline is not initialized')
    }

    return this.clearPipeline
  }

  private requireProjectPipeline(): GPUComputePipeline {
    if (!this.projectPipeline) {
      throw new Error('Project pipeline is not initialized')
    }

    return this.projectPipeline
  }

  private requireCandidatePipeline(): GPUComputePipeline {
    if (!this.candidatePipeline) {
      throw new Error('Silhouette candidate pipeline is not initialized')
    }

    return this.candidatePipeline
  }

  private requireHorizontalPipeline(): GPUComputePipeline {
    if (!this.horizontalPipeline) {
      throw new Error('Horizontal smoothing pipeline is not initialized')
    }

    return this.horizontalPipeline
  }

  private requireVerticalPipeline(): GPUComputePipeline {
    if (!this.verticalPipeline) {
      throw new Error('Vertical smoothing pipeline is not initialized')
    }

    return this.verticalPipeline
  }

  private requireCloseGapPipeline(): GPUComputePipeline {
    if (!this.closeGapPipeline) {
      throw new Error('Gap closing pipeline is not initialized')
    }

    return this.closeGapPipeline
  }

  private requireCommitDepthPipeline(): GPUComputePipeline {
    if (!this.commitDepthPipeline) {
      throw new Error('Depth commit pipeline is not initialized')
    }

    return this.commitDepthPipeline
  }

  private requireSilhouettePipeline(): GPUComputePipeline {
    if (!this.silhouettePipeline) {
      throw new Error('Silhouette pipeline is not initialized')
    }

    return this.silhouettePipeline
  }

  private requireEmitMeshPipeline(): GPUComputePipeline {
    if (!this.emitMeshPipeline) {
      throw new Error('Mesh emission pipeline is not initialized')
    }

    return this.emitMeshPipeline
  }

  private requireFinalizeMeshPipeline(): GPUComputePipeline {
    if (!this.finalizeMeshPipeline) {
      throw new Error('Mesh finalization pipeline is not initialized')
    }

    return this.finalizeMeshPipeline
  }

  private requireBackgroundPipeline(): GPURenderPipeline {
    if (!this.backgroundPipeline) {
      throw new Error('Background pipeline is not initialized')
    }

    return this.backgroundPipeline
  }

  private requireFloorPipeline(): GPURenderPipeline {
    if (!this.floorPipeline) {
      throw new Error('Floor pipeline is not initialized')
    }

    return this.floorPipeline
  }

  private requireSurfacePipeline(): GPURenderPipeline {
    if (!this.surfacePipeline) {
      throw new Error('Surface pipeline is not initialized')
    }

    return this.surfacePipeline
  }

  private requireParticlePipeline(): GPURenderPipeline {
    if (!this.particlePipeline) {
      throw new Error('Particle pipeline is not initialized')
    }

    return this.particlePipeline
  }
}
