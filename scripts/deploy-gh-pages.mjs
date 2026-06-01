import { execFileSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { cp, mkdir, readdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'

const REPO_OWNER = 'Nurygus'
const REPO_NAME = 'three-screen-space-meshes'
const PAGES_BRANCH = 'gh-pages'

function run(command, args, options = {}) {
  execFileSync(command, args, { stdio: 'inherit', ...options })
}

function output(command, args, options = {}) {
  return execFileSync(command, args, { encoding: 'utf8', ...options }).trim()
}

function branchExists(branch, cwd) {
  try {
    output('git', ['rev-parse', '--verify', '--quiet', branch], { cwd })
    return true
  } catch {
    return false
  }
}

async function emptyDirectory(directory) {
  await mkdir(directory, { recursive: true })

  for (const entry of await readdir(directory)) {
    if (entry === '.git') {
      continue
    }

    await rm(path.join(directory, entry), { recursive: true, force: true })
  }
}

const root = output('git', ['rev-parse', '--show-toplevel'])
const sourceCommit = output('git', ['rev-parse', '--short', 'HEAD'], { cwd: root })
const dist = path.join(root, 'dist')
const deployTree = path.join(tmpdir(), `${REPO_NAME}-${PAGES_BRANCH}`)
const sourceStatus = output('git', ['status', '--porcelain'], { cwd: root })

if (sourceStatus) {
  throw new Error('Refusing to deploy with uncommitted source changes. Commit local source first.')
}

if (!existsSync(path.join(dist, 'index.html'))) {
  throw new Error('dist/index.html does not exist. Run npm run build:pages first.')
}

run('git', ['worktree', 'prune'], { cwd: root })
await rm(deployTree, { recursive: true, force: true })

const remotePagesHead = output('git', ['ls-remote', '--heads', 'origin', PAGES_BRANCH], { cwd: root })

if (remotePagesHead) {
  run('git', ['fetch', 'origin', `${PAGES_BRANCH}:${PAGES_BRANCH}`], { cwd: root })
  run('git', ['worktree', 'add', '-B', PAGES_BRANCH, deployTree, `origin/${PAGES_BRANCH}`], { cwd: root })
} else if (branchExists(PAGES_BRANCH, root)) {
  run('git', ['worktree', 'add', '-B', PAGES_BRANCH, deployTree, PAGES_BRANCH], { cwd: root })
} else {
  run('git', ['worktree', 'add', '--detach', deployTree, 'HEAD'], { cwd: root })
  run('git', ['checkout', '--orphan', PAGES_BRANCH], { cwd: deployTree })
}

await emptyDirectory(deployTree)
await cp(dist, deployTree, { recursive: true })
await writeFile(path.join(deployTree, '.nojekyll'), '')
await writeFile(
  path.join(deployTree, 'README.md'),
  [
    `# ${REPO_NAME}`,
    '',
    'This branch contains only the compiled GitHub Pages build.',
    '',
    `Source commit: ${sourceCommit}`,
    `Public demo path: https://${REPO_OWNER}.github.io/${REPO_NAME}/`,
    '',
  ].join('\n'),
)

run('git', ['add', '-A'], { cwd: deployTree })

let hasChanges = true
try {
  execFileSync('git', ['diff', '--cached', '--quiet'], { cwd: deployTree })
  hasChanges = false
} catch {
  hasChanges = true
}

if (hasChanges) {
  run('git', ['commit', '-m', `Deploy GitHub Pages build from ${sourceCommit}`], { cwd: deployTree })
} else {
  console.log('No Pages build changes to commit.')
}

run('git', ['push', 'origin', PAGES_BRANCH], { cwd: deployTree })
run('git', ['worktree', 'remove', deployTree, '--force'], { cwd: root })

console.log(`Pushed compiled build to origin/${PAGES_BRANCH}.`)
console.log(`Expected Pages URL: https://${REPO_OWNER}.github.io/${REPO_NAME}/`)
