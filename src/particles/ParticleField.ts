import type { FlowPreset } from '../types/demo'

const TAU = Math.PI * 2

interface ParticleSeed {
  u: number
  v: number
  w: number
  phase: number
  band: number
}

export class ParticleField {
  readonly positions: Float32Array

  readonly maxCount: number

  private readonly seeds: ParticleSeed[]

  constructor(maxCount: number) {
    this.maxCount = maxCount
    this.positions = new Float32Array(maxCount * 3)
    this.seeds = Array.from({ length: maxCount }, (_, index) => makeSeed(index))
  }

  update(time: number, preset: FlowPreset, count: number): void {
    const activeCount = Math.min(count, this.maxCount)

    for (let index = 0; index < activeCount; index += 1) {
      const seed = this.seeds[index]
      const offset = index * 3
      const point =
        preset === 'sheet'
          ? sheetPoint(seed, time)
          : preset === 'crown'
            ? crownPoint(seed, time)
            : vortexPoint(seed, time)

      this.positions[offset] = point.x
      this.positions[offset + 1] = point.y
      this.positions[offset + 2] = point.z
    }
  }
}

function makeSeed(index: number): ParticleSeed {
  return {
    u: random01(index, 12.9898),
    v: random01(index, 78.233),
    w: random01(index, 37.719),
    phase: random01(index, 91.423) * TAU,
    band: random01(index, 11.135) > 0.55 ? 1 : -1,
  }
}

function vortexPoint(seed: ParticleSeed, time: number) {
  const twist = time * 0.48 + seed.phase
  const strand = seed.u * TAU * 3.4 + twist
  const lift = (seed.v - 0.5) * 2.45
  const pulse = Math.sin(time * 1.7 + seed.w * TAU) * 0.16
  const radius = 0.78 + seed.w * 0.42 + pulse

  return {
    x: Math.cos(strand) * radius + Math.sin(lift * 2.1 + time) * 0.16,
    y: lift + Math.sin(strand * 1.7 + time * 0.8) * 0.22,
    z: Math.sin(strand) * radius * 0.72 + Math.cos(seed.u * TAU + time) * 0.18,
  }
}

function sheetPoint(seed: ParticleSeed, time: number) {
  const xBase = (seed.u - 0.5) * 2.85
  const yBase = 1.35 - seed.v * 2.75
  const wake = Math.sin(seed.u * 11 + time * 1.8) * 0.12
  const fold = Math.cos(seed.v * 8.5 + time * 1.15 + seed.phase) * 0.2
  const pinch = Math.exp(-Math.abs(xBase) * 1.8) * Math.sin(time * 2 + seed.v * 5)

  return {
    x: xBase + wake * seed.band,
    y: yBase + Math.sin(seed.u * TAU + time + seed.phase) * 0.12,
    z: fold + pinch * 0.35 + (seed.w - 0.5) * 0.18,
  }
}

function crownPoint(seed: ParticleSeed, time: number) {
  const wave = (time * 0.33 + seed.v) % 1
  const angle = seed.u * TAU + Math.sin(time * 0.8 + seed.phase) * 0.45
  const burst = Math.sin(wave * Math.PI)
  const ring = 0.22 + wave * 1.55 + seed.w * 0.18
  const column = Math.max(0, 1 - wave * 1.2)

  return {
    x: Math.cos(angle) * ring + Math.cos(seed.phase + time) * 0.08,
    y: -0.8 + burst * 1.95 + column * 1.1 + (seed.w - 0.5) * 0.16,
    z: Math.sin(angle) * ring * 0.72 + Math.sin(seed.phase * 1.7) * 0.12,
  }
}

function random01(index: number, salt: number): number {
  const value = Math.sin(index * salt + salt * 17.17) * 43758.5453
  return value - Math.floor(value)
}
