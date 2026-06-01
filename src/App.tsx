import GUI from 'three/addons/libs/lil-gui.module.min.js'
import { useCallback, useEffect, useRef, useState } from 'react'
import './App.css'
import { CpuDemo } from './scene/CpuDemo'
import { WebGlDemo } from './scene/WebGlDemo'
import { WebGpuDemo } from './scene/WebGpuDemo'
import {
  automaticMaxDepthJump,
  defaultSettings,
  defaultStats,
  type DemoSettings,
  type DemoStats,
  type RenderMode,
} from './types/demo'

type DemoEngine = 'webgpu' | 'webgl' | 'webgl-cpu'

interface DemoRenderer {
  start: () => Promise<void>
  updateSettings: (settings: DemoSettings) => void
  dispose: () => void
}

interface DisplayController {
  updateDisplay: () => unknown
}

interface EnableController extends DisplayController {
  enable: (enabled?: boolean) => unknown
}

interface ShowController extends DisplayController {
  show: (show?: boolean) => unknown
}

interface GuiState {
  renderer: DemoEngine
  mode: RenderMode
  flow: DemoSettings['preset']
  particles: number
  grid: number
  radius: number
  thickness: number
  depthBlur: number
  silhouette: number
  explicitZMax: boolean
  zMax: number
  split: number
  freeze: boolean
  animate: boolean
  reset: () => void
}

const engineOptions: Record<string, DemoEngine> = {
  WebGPU: 'webgpu',
  WebGL: 'webgl',
  CPU: 'webgl-cpu',
}

const modeOptions: Record<string, RenderMode> = {
  Beauty: 'beauty',
  Mesh: 'mesh',
  Particles: 'particles',
  Depth: 'depth',
  Compare: 'compare',
}

const flowOptions: Record<string, DemoSettings['preset']> = {
  Vortex: 'vortex',
  Sheet: 'sheet',
  Crown: 'crown',
}

const controlRanges = {
  radius: { min: 0.04, max: 0.8, step: 0.01 },
  depthBlur: { min: 0, max: 8, step: 1 },
  boundaryRelax: { min: 0, max: 10, step: 1 },
  zMax: { min: 0.01, max: 4, step: 0.01 },
}

function App() {
  const stageRef = useRef<HTMLDivElement>(null)
  const demoRef = useRef<DemoRenderer | null>(null)
  const settingsRef = useRef<DemoSettings>(defaultSettings)
  const guiStateRef = useRef<GuiState | null>(null)
  const guiControllersRef = useRef<DisplayController[]>([])
  const boundaryControllerRef = useRef<EnableController | null>(null)
  const zMaxControllerRef = useRef<EnableController | null>(null)
  const splitControllerRef = useRef<ShowController | null>(null)
  const [engine, setEngine] = useState<DemoEngine>(() => engineFromLocation())
  const [settings, setSettings] = useState<DemoSettings>(defaultSettings)
  const [stats, setStats] = useState<DemoStats>(defaultStats)
  const [error, setError] = useState<string | null>(null)

  const updateSettings = useCallback((patch: Partial<DemoSettings>) => {
    setSettings((current) => ({ ...current, ...patch }))
  }, [])

  const resetSettings = useCallback(() => {
    setSettings({ ...defaultSettings })
  }, [])

  const updateEngine = useCallback((nextEngine: DemoEngine) => {
    const url = new URL(window.location.href)

    if (nextEngine === 'webgpu') {
      url.searchParams.delete('engine')
    } else {
      url.searchParams.set('engine', nextEngine)
    }

    window.history.pushState(null, '', url)
    setEngine(nextEngine)
  }, [])

  useEffect(() => {
    settingsRef.current = settings
  }, [settings])

  useEffect(() => {
    const stage = stageRef.current

    if (!stage) {
      return undefined
    }

    setError(null)
    setStats({ ...defaultStats, backend: startingBackendLabel(engine) })
    const initialSettings = settingsRef.current
    const demo =
      engine === 'webgpu'
        ? new WebGpuDemo(stage, initialSettings, setStats)
        : engine === 'webgl'
          ? new WebGlDemo(stage, initialSettings, setStats)
          : new CpuDemo(stage, initialSettings, setStats)
    demoRef.current = demo

    demo.start().catch((reason: unknown) => {
      setError(reason instanceof Error ? reason.message : 'Renderer failed to start')
    })

    return () => {
      demo.dispose()
      demoRef.current = null
    }
  }, [engine])

  useEffect(() => {
    demoRef.current?.updateSettings(settings)
  }, [settings])

  useEffect(() => {
    const guiState = createGuiState(settingsRef.current, engine, resetSettings)
    const gui = new GUI({ title: 'Controls', width: 300 })
    const controllers: DisplayController[] = []

    guiStateRef.current = guiState

    const renderer = gui.add(guiState, 'renderer', engineOptions).name('renderer')
    renderer.onChange((value) => updateEngine(value))
    controllers.push(renderer)

    const view = gui.addFolder('view')
    const mode = view.add(guiState, 'mode', modeOptions).name('mode')
    mode.onChange((value) => updateSettings({ mode: value }))
    controllers.push(mode)

    const split = view.add(guiState, 'split', 0.3, 0.7, 0.01).name('split')
    split.onChange((value) => updateSettings({ split: value }))
    split.show(guiState.mode === 'compare')
    splitControllerRef.current = split
    controllers.push(split)

    const simulation = gui.addFolder('simulation')
    const flow = simulation.add(guiState, 'flow', flowOptions).name('flow')
    flow.onChange((value) => updateSettings({ preset: value }))
    controllers.push(flow)

    const particles = simulation.add(guiState, 'particles', 700, 22000, 500).name('particles')
    particles.onChange((value) => updateSettings({ particleCount: value }))
    controllers.push(particles)

    const animate = simulation.add(guiState, 'animate').name('animate')
    animate.onChange((value) => updateSettings({ animate: value }))
    controllers.push(animate)

    const freeze = simulation.add(guiState, 'freeze').name('freeze surface')
    freeze.onChange((value) => updateSettings({ freezeSurface: value }))
    controllers.push(freeze)

    const reconstruction = gui.addFolder('reconstruction')
    const grid = reconstruction.add(guiState, 'grid', 1, 14, 1).name('grid px')
    grid.onChange((value) => updateSettings({ gridSpacing: value }))
    controllers.push(grid)

    const radius = reconstruction
      .add(guiState, 'radius', controlRanges.radius.min, controlRanges.radius.max, controlRanges.radius.step)
      .name('radius')
    radius.onChange((value) => updateSettings({ particleRadius: value }))
    controllers.push(radius)

    const depthBlur = reconstruction
      .add(guiState, 'depthBlur', controlRanges.depthBlur.min, controlRanges.depthBlur.max, controlRanges.depthBlur.step)
      .name('depth blur')
    depthBlur.onChange((value) => updateSettings({ depthSmoothing: value }))
    controllers.push(depthBlur)

    const silhouette = reconstruction
      .add(
        guiState,
        'silhouette',
        controlRanges.boundaryRelax.min,
        controlRanges.boundaryRelax.max,
        controlRanges.boundaryRelax.step,
      )
      .name('boundary relax')
    silhouette.onChange((value) => updateSettings({ silhouetteSmoothing: value }))
    silhouette.enable(engine !== 'webgl')
    boundaryControllerRef.current = silhouette
    controllers.push(silhouette)

    const explicitZMax = reconstruction.add(guiState, 'explicitZMax').name('explicit zmax')
    explicitZMax.onChange((value) => updateSettings({ useExplicitZMax: value }))
    controllers.push(explicitZMax)

    const zMax = reconstruction
      .add(guiState, 'zMax', controlRanges.zMax.min, controlRanges.zMax.max, controlRanges.zMax.step)
      .name('zmax')
    zMax.onChange((value) => updateSettings({ zMax: value }))
    zMax.enable(guiState.explicitZMax)
    zMaxControllerRef.current = zMax
    controllers.push(zMax)

    const material = gui.addFolder('material')
    const thickness = material.add(guiState, 'thickness', 0.12, 0.9, 0.02).name('thickness')
    thickness.onChange((value) => updateSettings({ materialThickness: value }))
    controllers.push(thickness)

    const reset = gui.add(guiState, 'reset').name('reset')
    controllers.push(reset)

    guiControllersRef.current = controllers

    return () => {
      gui.destroy()
      guiStateRef.current = null
      guiControllersRef.current = []
      boundaryControllerRef.current = null
      zMaxControllerRef.current = null
      splitControllerRef.current = null
    }
  }, [engine, resetSettings, updateEngine, updateSettings])

  useEffect(() => {
    const guiState = guiStateRef.current

    if (!guiState) {
      return
    }

    syncGuiState(guiState, settings, engine)
    boundaryControllerRef.current?.enable(engine !== 'webgl')
    zMaxControllerRef.current?.enable(settings.useExplicitZMax)
    splitControllerRef.current?.show(settings.mode === 'compare')
    guiControllersRef.current.forEach((controller) => controller.updateDisplay())
  }, [engine, settings])

  const workloadTotal = stats.projectionWork + stats.smoothingWork + stats.meshWork + stats.surfaceWork

  return (
    <main className="app-shell">
      <div ref={stageRef} className="stage" aria-label="Liquid Lens demo viewport" />

      <header className="status-overlay">
        <strong>Liquid Lens</strong>
        <span>{stats.backend}</span>
        <span>{stats.fps} fps</span>
        <span>{zMaxLabel(settings)}</span>
        {settings.freezeSurface && <em>frozen</em>}
      </header>

      <aside className="perf-panel" aria-label="Performance HUD">
        <div className="perf-summary">
          <span>hotspot</span>
          <strong>{stats.hotspot}</strong>
        </div>
        <div className="perf-grid">
          <PerfMetric label="CPU frame" value={`${stats.frameMs.toFixed(2)} ms`} />
          <PerfMetric label="GPU sample" value={formatGpuSample(stats.gpuTotalMs, stats.gpuTimingMode)} />
          <PerfMetric label="Draw calls" value={stats.drawCalls.toFixed(0)} />
          <PerfMetric label="Dispatches" value={stats.computeDispatches.toFixed(0)} />
          <PerfMetric label="Workgroups" value={stats.computeWorkgroups.toLocaleString()} />
          <PerfMetric label="Depth memory" value={`${stats.depthMemoryMb.toFixed(2)} MB`} />
        </div>
        <div className="perf-bars">
          <PerfBar label="CPU commands" value={stats.commandsMs} total={stats.frameMs} detail={`${stats.commandsMs.toFixed(2)} ms`} />
          <PerfBar label="GPU compute" value={stats.gpuComputeMs ?? 0} total={stats.gpuTotalMs ?? 0} detail={formatGpuSegment(stats.gpuComputeMs, stats.gpuTimingMode)} />
          <PerfBar label="GPU render" value={stats.gpuRenderMs ?? 0} total={stats.gpuTotalMs ?? 0} detail={formatGpuSegment(stats.gpuRenderMs, stats.gpuTimingMode)} />
          <PerfBar
            label="Particle splat"
            value={stats.projectionWork}
            total={workloadTotal}
            detail={`${formatCompact(stats.projectionWork)} cells`}
          />
          <PerfBar
            label="Depth smooth"
            value={stats.smoothingWork}
            total={workloadTotal}
            detail={`${formatCompact(stats.smoothingWork)} ops`}
          />
          <PerfBar
            label="Mesh extract"
            value={stats.meshWork}
            total={workloadTotal}
            detail={`${formatCompact(stats.meshWork)} ops`}
          />
          <PerfBar
            label="Surface draw"
            value={stats.surfaceWork}
            total={workloadTotal}
            detail={`${formatCompact(stats.surfaceWork)} vertices`}
          />
        </div>
        <div className="perf-foot">
          <span>{stats.canvas}</span>
          <span>{stats.gridCells.toLocaleString()} cells</span>
          <span>{stats.pixelRatio.toFixed(2)} dpr</span>
          <span>{gpuTimingLabel(stats.gpuTimingMode, stats.gpuTimerAvailable)}</span>
        </div>
      </aside>

      <footer className="stats-bar">
        <Metric label="Particles" value={stats.particles.toLocaleString()} />
        <Metric label="Triangles" value={stats.triangles.toLocaleString()} />
        <Metric label="Vertices" value={stats.vertices.toLocaleString()} />
        <Metric label="Grid" value={stats.grid} />
        <Metric label="zmax" value={zMaxLabel(settings)} />
        <Metric label="CPU frame" value={`${stats.frameMs.toFixed(1)} ms`} />
      </footer>

      {settings.mode === 'compare' && (
        <div className="split-marker" style={{ left: `${settings.split * 100}%` }}>
          <span>Particles</span>
          <span>Surface</span>
        </div>
      )}

      {error && <div className="error-banner">{error}</div>}
    </main>
  )
}

function createGuiState(settings: DemoSettings, engine: DemoEngine, reset: () => void): GuiState {
  return {
    renderer: engine,
    mode: settings.mode,
    flow: settings.preset,
    particles: settings.particleCount,
    grid: settings.gridSpacing,
    radius: settings.particleRadius,
    thickness: settings.materialThickness,
    depthBlur: settings.depthSmoothing,
    silhouette: settings.silhouetteSmoothing,
    explicitZMax: settings.useExplicitZMax,
    zMax: settings.zMax,
    split: settings.split,
    freeze: settings.freezeSurface,
    animate: settings.animate,
    reset,
  }
}

function syncGuiState(target: GuiState, settings: DemoSettings, engine: DemoEngine): void {
  target.renderer = engine
  target.mode = settings.mode
  target.flow = settings.preset
  target.particles = settings.particleCount
  target.grid = settings.gridSpacing
  target.radius = settings.particleRadius
  target.thickness = settings.materialThickness
  target.depthBlur = settings.depthSmoothing
  target.silhouette = settings.silhouetteSmoothing
  target.explicitZMax = settings.useExplicitZMax
  target.zMax = settings.zMax
  target.split = settings.split
  target.freeze = settings.freezeSurface
  target.animate = settings.animate
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function PerfMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="perf-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function PerfBar({ label, value, total, detail }: { label: string; value: number; total: number; detail: string }) {
  const percent = total > 0 ? Math.max(3, Math.min(100, (value / total) * 100)) : 0

  return (
    <div className="perf-row">
      <span>{label}</span>
      <div className="perf-track" aria-hidden="true">
        <i style={{ width: `${percent}%` }} />
      </div>
      <strong>{detail}</strong>
    </div>
  )
}

function formatGpuSample(value: number | null, mode: DemoStats['gpuTimingMode']) {
  if (value !== null) {
    return `${value.toFixed(2)} ms`
  }

  return mode === 'unavailable' ? 'off' : 'pending'
}

function formatGpuSegment(value: number | null, mode: DemoStats['gpuTimingMode']) {
  if (value !== null) {
    return `${value.toFixed(2)} ms`
  }

  return mode === 'queue' ? 'needs timestamp' : 'off'
}

function gpuTimingLabel(mode: DemoStats['gpuTimingMode'], timestampAvailable: boolean) {
  if (mode === 'timestamp') {
    return 'timestamp sample'
  }

  if (mode === 'queue') {
    return timestampAvailable ? 'queue latency sample' : 'queue fallback sample'
  }

  return timestampAvailable ? 'GPU sample pending' : 'GPU sample off'
}

function formatCompact(value: number) {
  return Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(value)
}

function zMaxLabel(settings: DemoSettings): string {
  const value = settings.useExplicitZMax ? settings.zMax : automaticZMax(settings)
  const source = settings.useExplicitZMax ? 'manual' : 'auto'

  return `${value.toFixed(2)} ${source}`
}

function automaticZMax(settings: DemoSettings): number {
  return automaticMaxDepthJump(settings)
}

function engineFromLocation(): DemoEngine {
  const value = new URLSearchParams(window.location.search).get('engine')

  if (value === 'webgl-cpu' || value === 'cpu') {
    return 'webgl-cpu'
  }

  return value === 'webgl' ? 'webgl' : 'webgpu'
}

function startingBackendLabel(engine: DemoEngine): string {
  if (engine === 'webgpu') {
    return 'starting WebGPU'
  }

  return engine === 'webgl' ? 'starting WebGL' : 'starting CPU'
}

export default App
