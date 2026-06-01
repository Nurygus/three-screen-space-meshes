# Liquid Lens

Interactive TypeScript demo for screen-space particle meshing.

The main renderer projects animated particles into a screen-space depth grid, smooths the depth field, builds a temporary visible surface, and renders it as a liquid-like mesh. Three.js is used for camera controls and the WebGL renderers; the WebGPU path uses custom WGSL compute/render shaders.

## Run

```sh
npm install
npm run dev
```

Modes:

- `http://127.0.0.1:5173/`: WebGPU
- `http://127.0.0.1:5173/?engine=webgl`: WebGL
- `http://127.0.0.1:5173/?engine=webgl-cpu`: CPU reference

Chrome, Edge, or another browser with WebGPU enabled is required for the WebGPU page. The WebGL pages require WebGL2.

## Features

- Vortex, sheet, and crown particle presets.
- WebGPU mesh generation with compute shaders and indirect drawing.
- WebGL texture ping-pong fallback.
- CPU reference implementation for easier inspection.
- Beauty, mesh, particles, depth, and compare views.
- Freeze mode for orbiting around the current generated surface.
- HUD for grid size, triangle count, FPS, draw calls, GPU timing where available, and estimated workload.

## Notes

- WebGPU: `src/scene/WebGpuDemo.ts` and `src/gpu/shaders.ts`.
- WebGL: `src/scene/WebGlDemo.ts`.
- CPU: `src/scene/CpuDemo.ts` and `src/mesh/ScreenSpaceMesh.ts`.

The WebGPU path keeps the main reconstruction on the GPU. It writes a depth grid, stores silhouette edge candidates, smooths depth, emits triangle vertices into a storage buffer, and draws with indirect arguments.

The WebGL path is a faster visual fallback. It stores depth in render targets, smooths with fullscreen passes, and draws a fixed grid whose vertex shader samples the filtered depth texture. It does not build the same topology as the CPU/WebGPU paths.

The CPU path is slower but closest to the paper-style topology. It performs the second silhouette pass, handles the 16 cut-edge cases, adds extra vertices for pathological cases, smooths boundary vertices, and computes normals on the CPU.

`boundary relax` is exact only in the CPU renderer. WebGPU currently uses a cheaper approximation because full boundary smoothing would need adjacency data, paired boundary vertices, and extra synchronization or CPU readback. WebGL disables the control because it does not build explicit boundary vertices.

This is a rendering demo, not a full fluid solver or permanent mesh reconstruction system. The particles are procedural, and the generated surface is view-dependent.
