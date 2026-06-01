export type RenderMode = 'beauty' | 'mesh' | 'particles' | 'depth' | 'compare'

export type FlowPreset = 'vortex' | 'sheet' | 'crown'

export type GpuTimingMode = 'timestamp' | 'queue' | 'unavailable'

export interface DemoSettings {
  mode: RenderMode
  preset: FlowPreset
  particleCount: number
  gridSpacing: number
  particleRadius: number
  materialThickness: number
  depthSmoothing: number
  silhouetteSmoothing: number
  useExplicitZMax: boolean
  zMax: number
  split: number
  animate: boolean
  freezeSurface: boolean
}

export interface DemoStats {
  backend: string
  fps: number
  particles: number
  vertices: number
  triangles: number
  grid: string
  frameMs: number
  cameraMs: number
  buffersMs: number
  uniformsMs: number
  commandsMs: number
  gpuComputeMs: number | null
  gpuRenderMs: number | null
  gpuTotalMs: number | null
  gpuTimerAvailable: boolean
  gpuTimingMode: GpuTimingMode
  drawCalls: number
  computeDispatches: number
  computeWorkgroups: number
  gridCells: number
  splatCells: number
  canvas: string
  pixelRatio: number
  depthMemoryMb: number
  projectionWork: number
  smoothingWork: number
  meshWork: number
  surfaceWork: number
  hotspot: string
}

export const renderModes: Array<{ id: RenderMode; label: string }> = [
  { id: 'beauty', label: 'Beauty' },
  { id: 'mesh', label: 'Mesh' },
  { id: 'particles', label: 'Particles' },
  { id: 'depth', label: 'Depth' },
  { id: 'compare', label: 'Compare' },
]

export const flowPresets: Array<{ id: FlowPreset; label: string }> = [
  { id: 'vortex', label: 'Vortex' },
  { id: 'sheet', label: 'Sheet' },
  { id: 'crown', label: 'Crown' },
]

export const defaultSettings: DemoSettings = {
  mode: 'beauty',
  preset: 'vortex',
  particleCount: 1800,
  gridSpacing: 6,
  particleRadius: 0.18,
  materialThickness: 0.46,
  depthSmoothing: 2,
  silhouetteSmoothing: 2,
  useExplicitZMax: false,
  zMax: 0.49,
  split: 0.5,
  animate: true,
  freezeSurface: false,
}

export function automaticMaxDepthJump(settings: Pick<DemoSettings, 'particleRadius'>): number {
  return settings.particleRadius * 2.7
}

export function effectiveMaxDepthJump(
  settings: Pick<DemoSettings, 'particleRadius' | 'useExplicitZMax' | 'zMax'>,
): number {
  return settings.useExplicitZMax ? settings.zMax : automaticMaxDepthJump(settings)
}

export const defaultStats: DemoStats = {
  backend: 'initializing',
  fps: 0,
  particles: 0,
  vertices: 0,
  triangles: 0,
  grid: '0 x 0',
  frameMs: 0,
  cameraMs: 0,
  buffersMs: 0,
  uniformsMs: 0,
  commandsMs: 0,
  gpuComputeMs: null,
  gpuRenderMs: null,
  gpuTotalMs: null,
  gpuTimerAvailable: false,
  gpuTimingMode: 'unavailable',
  drawCalls: 0,
  computeDispatches: 0,
  computeWorkgroups: 0,
  gridCells: 0,
  splatCells: 0,
  canvas: '0 x 0',
  pixelRatio: 1,
  depthMemoryMb: 0,
  projectionWork: 0,
  smoothingWork: 0,
  meshWork: 0,
  surfaceWork: 0,
  hotspot: 'initializing',
}
