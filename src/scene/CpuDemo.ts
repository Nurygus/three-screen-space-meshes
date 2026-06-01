import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { buildScreenSpaceMesh, type ScreenSpaceMeshResult } from '../mesh/ScreenSpaceMesh'
import { ParticleField } from '../particles/ParticleField'
import { effectiveMaxDepthJump, type DemoSettings, type DemoStats, type RenderMode } from '../types/demo'

type StatsHandler = (stats: DemoStats) => void

const MAX_PARTICLES = 22000
const CPU_MESH_PARTICLE_CAP = 8000
const CPU_GRID_SPACING_MIN = 2

export class CpuDemo {
  private readonly field = new ParticleField(MAX_PARTICLES)

  private readonly scene = new THREE.Scene()

  private readonly camera = new THREE.PerspectiveCamera(46, 1, 0.1, 30)

  private readonly canvas = document.createElement('canvas')

  private readonly renderer = new THREE.WebGLRenderer({
    canvas: this.canvas,
    antialias: true,
    alpha: true,
    powerPreference: 'high-performance',
  })

  private readonly controls: OrbitControls

  private readonly surfaceGeometry = new THREE.BufferGeometry()

  private readonly wireGeometry = new THREE.BufferGeometry()

  private readonly pointGeometry = new THREE.BufferGeometry()

  private readonly beautyMaterial = new THREE.ShaderMaterial({
    uniforms: {
      materialThickness: { value: 0.46 },
    },
    vertexShader: cpuBeautyVertexShader,
    fragmentShader: cpuBeautyFragmentShader,
    transparent: true,
    depthWrite: true,
    side: THREE.DoubleSide,
  })

  private readonly meshMaterial = new THREE.MeshBasicMaterial({
    color: 0x123f52,
    side: THREE.DoubleSide,
  })

  private readonly depthMaterial = new THREE.MeshBasicMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
  })

  private readonly wireMaterial = new THREE.LineBasicMaterial({
    color: 0xf4fbff,
    transparent: true,
    opacity: 0.58,
    depthWrite: false,
  })

  private readonly pointMaterial = new THREE.PointsMaterial({
    color: 0xffb86b,
    size: 0.036,
    transparent: true,
    opacity: 0.78,
    depthWrite: false,
  })

  private readonly surfaceMesh = new THREE.Mesh<THREE.BufferGeometry, THREE.Material>(
    this.surfaceGeometry,
    this.beautyMaterial,
  )

  private readonly wireMesh = new THREE.LineSegments<THREE.BufferGeometry, THREE.LineBasicMaterial>(
    this.wireGeometry,
    this.wireMaterial,
  )

  private readonly points = new THREE.Points(this.pointGeometry, this.pointMaterial)

  private readonly resizeObserver: ResizeObserver

  private settings: DemoSettings

  private viewportWidth = 1

  private viewportHeight = 1

  private backend = 'initializing'

  private disposed = false

  private frames = 0

  private lastStatsAt = 0

  private lastRenderAt = 0

  private simulationTime = 0

  private cachedMesh: ScreenSpaceMeshResult | null = null

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
    this.configureScene()
    this.container.append(this.canvas)
    this.resizeObserver.observe(this.container)
    this.resize()
    this.backend = 'CPU'
    this.renderer.setAnimationLoop((time) => this.render(time))
  }

  updateSettings(settings: DemoSettings): void {
    this.settings = { ...settings }
    this.controls.autoRotate = settings.animate && settings.mode === 'beauty' && !settings.freezeSurface
    this.beautyMaterial.uniforms.materialThickness.value = settings.materialThickness
    this.updateMode(settings.mode)
  }

  dispose(): void {
    this.disposed = true
    this.resizeObserver.disconnect()
    this.renderer.setAnimationLoop(null)
    this.controls.dispose()
    this.surfaceGeometry.dispose()
    this.wireGeometry.dispose()
    this.pointGeometry.dispose()
    this.beautyMaterial.dispose()
    this.meshMaterial.dispose()
    this.depthMaterial.dispose()
    this.wireMaterial.dispose()
    this.pointMaterial.dispose()
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

    this.pointGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(this.field.positions, 3).setUsage(THREE.DynamicDrawUsage),
    )

    this.surfaceMesh.frustumCulled = false
    this.wireMesh.frustumCulled = false
    this.points.frustumCulled = false

    this.scene.add(this.surfaceMesh, this.wireMesh, this.points)
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

    if (this.settings.animate && !this.settings.freezeSurface) {
      this.simulationTime += deltaSeconds
    }

    this.controls.update()
    const particleCount = this.effectiveParticleCount()
    let mesh = this.cachedMesh

    if (!this.settings.freezeSurface || !mesh) {
      this.field.update(this.simulationTime, this.settings.preset, particleCount)
      this.updatePointGeometry(particleCount)
      mesh = buildScreenSpaceMesh(this.field.positions, particleCount, this.camera, {
        width: this.viewportWidth,
        height: this.viewportHeight,
        gridSpacing: this.effectiveGridSpacing(),
        particleRadius: this.settings.particleRadius,
        maxDepthJump: effectiveMaxDepthJump(this.settings),
        depthSmoothing: this.settings.depthSmoothing,
        silhouetteSmoothing: this.settings.silhouetteSmoothing,
      })
      this.cachedMesh = mesh
      this.updateSurfaceGeometry(mesh)
    }

    this.renderMode()
    this.publishStats(timeMs, mesh, performance.now() - frameStartedAt, particleCount)
  }

  private updatePointGeometry(particleCount: number): void {
    const position = this.pointGeometry.getAttribute('position')

    if (position) {
      position.needsUpdate = true
      this.pointGeometry.setDrawRange(0, particleCount)
    }
  }

  private updateSurfaceGeometry(mesh: {
    positions: Float32Array
    normals: Float32Array
    colors: Float32Array
    indices: Uint32Array
    wireIndices: Uint32Array
  }): void {
    this.surfaceGeometry.setAttribute('position', new THREE.BufferAttribute(mesh.positions, 3))
    this.surfaceGeometry.setAttribute('normal', new THREE.BufferAttribute(mesh.normals, 3))
    this.surfaceGeometry.setAttribute('color', new THREE.BufferAttribute(mesh.colors, 3))
    this.surfaceGeometry.setIndex(new THREE.BufferAttribute(mesh.indices, 1))
    this.surfaceGeometry.computeBoundingSphere()

    this.wireGeometry.setAttribute('position', new THREE.BufferAttribute(mesh.positions, 3))
    this.wireGeometry.setIndex(new THREE.BufferAttribute(mesh.wireIndices, 1))
    this.wireGeometry.computeBoundingSphere()
  }

  private updateMode(mode: RenderMode): void {
    const surfaceMode = mode === 'beauty' || mode === 'mesh' || mode === 'depth'
    this.surfaceMesh.visible = surfaceMode
    this.wireMesh.visible = mode === 'mesh'
    this.points.visible = mode === 'particles'

    if (mode === 'depth') {
      this.surfaceMesh.material = this.depthMaterial
    } else if (mode === 'mesh') {
      this.surfaceMesh.material = this.meshMaterial
    } else {
      this.surfaceMesh.material = this.beautyMaterial
    }
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
    this.surfaceMesh.material = this.beautyMaterial
    this.wireMesh.visible = false
    this.points.visible = false
    this.renderer.setViewport(splitX, 0, this.viewportWidth - splitX, this.viewportHeight)
    this.renderer.setScissor(splitX, 0, this.viewportWidth - splitX, this.viewportHeight)
    this.renderer.render(this.scene, this.camera)

    this.renderer.setScissorTest(false)
  }

  private publishStats(
    timeMs: number,
    mesh: { vertices: number; triangles: number; gridColumns: number; gridRows: number; buildMs: number },
    frameMs: number,
    particleCount: number,
  ): void {
    this.frames += 1

    if (timeMs - this.lastStatsAt < 250) {
      return
    }

    const seconds = (timeMs - this.lastStatsAt) / 1000 || 1
    const fps = Math.round(this.frames / seconds)
    this.frames = 0
    this.lastStatsAt = timeMs

    this.onStats({
      backend: this.backend,
      fps,
      particles: particleCount,
      vertices: mesh.vertices,
      triangles: mesh.triangles,
      grid: `${mesh.gridColumns} x ${mesh.gridRows}`,
      frameMs,
      cameraMs: 0,
      buffersMs: 0,
      uniformsMs: 0,
      commandsMs: mesh.buildMs,
      gpuComputeMs: null,
      gpuRenderMs: null,
      gpuTotalMs: null,
      gpuTimerAvailable: false,
      gpuTimingMode: 'unavailable',
      drawCalls: this.settings.mode === 'compare' ? 2 : 1,
      computeDispatches: 0,
      computeWorkgroups: 0,
      gridCells: mesh.gridColumns * mesh.gridRows,
      splatCells: 0,
      canvas: `${this.canvas.width} x ${this.canvas.height}`,
      pixelRatio: Math.min(window.devicePixelRatio, 1.65),
      depthMemoryMb: this.estimateCpuMemoryMb(mesh),
      projectionWork: particleCount,
      smoothingWork: mesh.gridColumns * mesh.gridRows,
      meshWork: mesh.triangles,
      surfaceWork: mesh.vertices,
      hotspot: this.settings.freezeSurface ? 'surface draw' : 'CPU mesh build',
    })
  }

  private effectiveParticleCount(): number {
    return Math.min(this.settings.particleCount, MAX_PARTICLES, CPU_MESH_PARTICLE_CAP)
  }

  private effectiveGridSpacing(): number {
    return Math.max(CPU_GRID_SPACING_MIN, this.settings.gridSpacing)
  }

  private estimateCpuMemoryMb(mesh: { vertices: number; triangles: number; gridColumns: number; gridRows: number }): number {
    const meshBytes = mesh.vertices * 9 * Float32Array.BYTES_PER_ELEMENT + mesh.triangles * 3 * Uint32Array.BYTES_PER_ELEMENT
    const gridBytes = mesh.gridColumns * mesh.gridRows * Float32Array.BYTES_PER_ELEMENT

    return (meshBytes + gridBytes) / (1024 * 1024)
  }
}

const cpuBeautyVertexShader = /* glsl */ `
varying vec3 vWorld;
varying vec3 vNormal;
varying float vDepth;

void main() {
  vec4 world = modelMatrix * vec4(position, 1.0);
  vec4 view = viewMatrix * world;

  vWorld = world.xyz;
  vNormal = normalize(normalMatrix * normal);
  vDepth = -view.z;
  gl_Position = projectionMatrix * view;
}
`

const cpuBeautyFragmentShader = /* glsl */ `
uniform float materialThickness;

varying vec3 vWorld;
varying vec3 vNormal;
varying float vDepth;

void main() {
  float depthTint = clamp((vDepth - 2.0) / 4.0, 0.0, 1.0);
  vec3 viewDirection = normalize(cameraPosition - vWorld);
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
  color += vec3(0.82, 0.98, 1.0) * fresnel * (0.28 + materialThickness * 0.22);

  gl_FragColor = vec4(color, clamp(0.52 + materialThickness * 0.28 + fresnel * 0.15, 0.48, 0.9));
}
`
