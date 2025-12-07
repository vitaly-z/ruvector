// RuVector LLM Management Page
// Provides management interface for @ruvector/ruvllm features

import { useState, useEffect, useRef } from 'react'
import { useRouter } from 'next/router'
import { ProjectLayoutWithAuth } from 'components/layouts/ProjectLayout/ProjectLayout'
import type { NextPageWithLayout } from 'types'

// Icons as inline SVG components
const BrainIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
  </svg>
)

const CogIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
  </svg>
)

const AcademicCapIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 14l9-5-9-5-9 5 9 5z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 14l6.16-3.422a12.083 12.083 0 01.665 6.479A11.952 11.952 0 0012 20.055a11.952 11.952 0 00-6.824-2.998 12.078 12.078 0 01.665-6.479L12 14z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 14l9-5-9-5-9 5 9 5zm0 0l6.16-3.422a12.083 12.083 0 01.665 6.479A11.952 11.952 0 0012 20.055a11.952 11.952 0 00-6.824-2.998 12.078 12.078 0 01.665-6.479L12 14zm-4 6v-7.5l4-2.222" />
  </svg>
)

const ChartBarIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
  </svg>
)

const LayersIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
  </svg>
)

const UsersIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
  </svg>
)

const PlayIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
)

const ClipboardIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
  </svg>
)

const CheckIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
  </svg>
)

type TabType = 'overview' | 'engine' | 'sona' | 'training' | 'lora' | 'federated'

const tabs: { id: TabType; label: string; icon: React.ReactNode }[] = [
  { id: 'overview', label: 'Overview', icon: <BrainIcon /> },
  { id: 'engine', label: 'Engine', icon: <CogIcon /> },
  { id: 'sona', label: 'SONA Learning', icon: <AcademicCapIcon /> },
  { id: 'training', label: 'Training', icon: <ChartBarIcon /> },
  { id: 'lora', label: 'LoRA Adapters', icon: <LayersIcon /> },
  { id: 'federated', label: 'Federated', icon: <UsersIcon /> },
]

// Code block component with copy functionality
function CodeBlock({ code, language = 'typescript' }: { code: string; language?: string }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="relative group">
      <pre className="bg-surface-200 text-foreground-light p-4 rounded-lg overflow-x-auto text-sm border border-default">
        <code>{code}</code>
      </pre>
      <button
        onClick={handleCopy}
        className="absolute top-2 right-2 p-2 bg-surface-300 hover:bg-surface-400 rounded opacity-0 group-hover:opacity-100 transition-opacity"
        title="Copy to clipboard"
      >
        {copied ? <CheckIcon /> : <ClipboardIcon />}
      </button>
    </div>
  )
}

// Stats card component
function StatCard({ label, value, icon, color }: { label: string; value: string | number; icon: React.ReactNode; color: string }) {
  return (
    <div className="bg-surface-100 rounded-lg p-4 border border-default">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-foreground-light text-sm">{label}</p>
          <p className="text-2xl font-bold text-foreground mt-1">{value}</p>
        </div>
        <div className={`p-3 rounded-lg ${color}`}>
          {icon}
        </div>
      </div>
    </div>
  )
}

// Real-time metrics graph
function MetricsGraph({ data, label, color }: { data: number[]; label: string; color: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)'
    ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) {
      const y = (height / 4) * i
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }

    // Draw line
    if (data.length > 1) {
      const max = Math.max(...data, 1)
      const step = width / (data.length - 1)

      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.beginPath()
      data.forEach((val, i) => {
        const x = i * step
        const y = height - (val / max) * height * 0.9 - height * 0.05
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.stroke()

      // Fill area under line
      ctx.lineTo(width, height)
      ctx.lineTo(0, height)
      ctx.closePath()
      ctx.fillStyle = color.replace('1)', '0.1)')
      ctx.fill()
    }
  }, [data, color])

  return (
    <div className="bg-surface-100 rounded-lg p-4 border border-default">
      <p className="text-foreground-light text-sm mb-2">{label}</p>
      <canvas ref={canvasRef} width={300} height={100} className="w-full" />
    </div>
  )
}

// Overview Tab
function OverviewTab() {
  const [stats, setStats] = useState({
    models: 3,
    patterns: 1247,
    avgLatency: 45,
    cacheHitRate: 87,
    memoryNodes: 15420,
    activeAgents: 2
  })

  const [latencyData, setLatencyData] = useState<number[]>([45, 42, 48, 44, 46, 43, 45, 47, 44, 42])
  const [throughputData, setThroughputData] = useState<number[]>([120, 135, 128, 142, 138, 145, 150, 148, 155, 160])

  useEffect(() => {
    const interval = setInterval(() => {
      setLatencyData(prev => [...prev.slice(1), Math.floor(Math.random() * 20) + 35])
      setThroughputData(prev => [...prev.slice(1), Math.floor(Math.random() * 50) + 120])
    }, 2000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-pink-600 rounded-lg p-6">
        <div className="flex items-center space-x-3">
          <BrainIcon />
          <div>
            <h2 className="text-xl font-bold text-foreground">RuvLLM Engine</h2>
            <p className="text-purple-100">Self-learning LLM orchestration with SONA adaptive learning</p>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <StatCard label="Active Models" value={stats.models} icon={<BrainIcon />} color="bg-purple-500/20 text-purple-400" />
        <StatCard label="Patterns Learned" value={stats.patterns.toLocaleString()} icon={<AcademicCapIcon />} color="bg-blue-500/20 text-blue-400" />
        <StatCard label="Avg Latency" value={`${stats.avgLatency}ms`} icon={<ChartBarIcon />} color="bg-green-500/20 text-green-400" />
        <StatCard label="Cache Hit Rate" value={`${stats.cacheHitRate}%`} icon={<LayersIcon />} color="bg-yellow-500/20 text-yellow-400" />
        <StatCard label="Memory Nodes" value={stats.memoryNodes.toLocaleString()} icon={<CogIcon />} color="bg-pink-500/20 text-pink-400" />
        <StatCard label="Active Agents" value={stats.activeAgents} icon={<UsersIcon />} color="bg-cyan-500/20 text-cyan-400" />
      </div>

      {/* Real-time Graphs */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <MetricsGraph data={latencyData} label="Query Latency (ms)" color="rgba(139, 92, 246, 1)" />
        <MetricsGraph data={throughputData} label="Throughput (queries/s)" color="rgba(59, 130, 246, 1)" />
      </div>

      {/* Quick Start */}
      <div className="bg-surface-100 rounded-lg p-6 border border-default">
        <h3 className="text-lg font-semibold text-foreground mb-4">Quick Start</h3>
        <CodeBlock code={`import { RuvLLM } from '@ruvector/ruvllm';

// Initialize the engine
const llm = new RuvLLM({
  modelId: 'gpt-4',
  enableSona: true,
  memoryConfig: {
    dimensions: 1536,
    maxNodes: 100000
  }
});

// Query with automatic learning
const response = await llm.query('What is vector similarity?');
console.log(response.text);

// Provide feedback to improve
await llm.feedback(response.id, { score: 0.9, helpful: true });`} />
      </div>

      {/* Features */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div className="bg-surface-100 rounded-lg p-4 border border-default">
          <div className="flex items-center space-x-2 mb-2">
            <div className="p-2 bg-purple-500/20 rounded-lg text-purple-400">
              <BrainIcon />
            </div>
            <h4 className="font-semibold text-foreground">SONA Learning</h4>
          </div>
          <p className="text-foreground-light text-sm">Self-Organizing Neural Architecture that learns from every interaction</p>
        </div>
        <div className="bg-surface-100 rounded-lg p-4 border border-default">
          <div className="flex items-center space-x-2 mb-2">
            <div className="p-2 bg-blue-500/20 rounded-lg text-blue-400">
              <CogIcon />
            </div>
            <h4 className="font-semibold text-foreground">HNSW Memory</h4>
          </div>
          <p className="text-foreground-light text-sm">High-performance vector memory with HNSW indexing for fast retrieval</p>
        </div>
        <div className="bg-surface-100 rounded-lg p-4 border border-default">
          <div className="flex items-center space-x-2 mb-2">
            <div className="p-2 bg-green-500/20 rounded-lg text-green-400">
              <LayersIcon />
            </div>
            <h4 className="font-semibold text-foreground">LoRA Adapters</h4>
          </div>
          <p className="text-foreground-light text-sm">Parameter-efficient fine-tuning with low-rank adaptation</p>
        </div>
      </div>
    </div>
  )
}

// Engine Tab
function EngineTab() {
  const [config, setConfig] = useState({
    modelId: 'gpt-4',
    temperature: 0.7,
    maxTokens: 2048,
    enableStreaming: true,
    enableCache: true,
    cacheSize: 1000
  })

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-blue-600 to-cyan-600 rounded-lg p-6">
        <div className="flex items-center space-x-3">
          <CogIcon />
          <div>
            <h2 className="text-xl font-bold text-foreground">Engine Configuration</h2>
            <p className="text-blue-100">Configure the RuvLLM engine parameters</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Model Settings */}
        <div className="bg-surface-100 rounded-lg p-6 border border-default">
          <h3 className="text-lg font-semibold text-foreground mb-4">Model Settings</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-foreground-light mb-1">Model ID</label>
              <select
                value={config.modelId}
                onChange={(e) => setConfig({ ...config, modelId: e.target.value })}
                className="w-full bg-surface-200 border border-default rounded-lg px-3 py-2 text-foreground"
              >
                <option value="gpt-4">GPT-4</option>
                <option value="gpt-4-turbo">GPT-4 Turbo</option>
                <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                <option value="claude-3">Claude 3</option>
                <option value="custom">Custom Model</option>
              </select>
            </div>
            <div>
              <label className="block text-sm text-foreground-light mb-1">Temperature: {config.temperature}</label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={config.temperature}
                onChange={(e) => setConfig({ ...config, temperature: parseFloat(e.target.value) })}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm text-foreground-light mb-1">Max Tokens</label>
              <input
                type="number"
                value={config.maxTokens}
                onChange={(e) => setConfig({ ...config, maxTokens: parseInt(e.target.value) })}
                className="w-full bg-surface-200 border border-default rounded-lg px-3 py-2 text-foreground"
              />
            </div>
          </div>
        </div>

        {/* Performance Settings */}
        <div className="bg-surface-100 rounded-lg p-6 border border-default">
          <h3 className="text-lg font-semibold text-foreground mb-4">Performance Settings</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-foreground-light">Enable Streaming</span>
              <button
                onClick={() => setConfig({ ...config, enableStreaming: !config.enableStreaming })}
                className={`relative w-12 h-6 rounded-full transition-colors ${config.enableStreaming ? 'bg-blue-600' : 'bg-surface-300'}`}
              >
                <span className={`absolute w-4 h-4 bg-white rounded-full top-1 transition-transform ${config.enableStreaming ? 'left-7' : 'left-1'}`} />
              </button>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-foreground-light">Enable Cache</span>
              <button
                onClick={() => setConfig({ ...config, enableCache: !config.enableCache })}
                className={`relative w-12 h-6 rounded-full transition-colors ${config.enableCache ? 'bg-blue-600' : 'bg-surface-300'}`}
              >
                <span className={`absolute w-4 h-4 bg-white rounded-full top-1 transition-transform ${config.enableCache ? 'left-7' : 'left-1'}`} />
              </button>
            </div>
            <div>
              <label className="block text-sm text-foreground-light mb-1">Cache Size</label>
              <input
                type="number"
                value={config.cacheSize}
                onChange={(e) => setConfig({ ...config, cacheSize: parseInt(e.target.value) })}
                className="w-full bg-surface-200 border border-default rounded-lg px-3 py-2 text-foreground"
                disabled={!config.enableCache}
              />
            </div>
          </div>
        </div>
      </div>

      {/* API Examples */}
      <div className="bg-surface-100 rounded-lg p-6 border border-default">
        <h3 className="text-lg font-semibold text-foreground mb-4">Engine API</h3>
        <div className="space-y-4">
          <CodeBlock code={`// Initialize with configuration
const engine = new RuvLLM({
  modelId: '${config.modelId}',
  generationConfig: {
    temperature: ${config.temperature},
    maxTokens: ${config.maxTokens}
  },
  cacheConfig: {
    enabled: ${config.enableCache},
    maxSize: ${config.cacheSize}
  }
});

// Generate text
const result = await engine.generate('Explain quantum computing', {
  stream: ${config.enableStreaming}
});

// Route to best model
const decision = await engine.route('Complex math problem');
console.log('Selected model:', decision.selectedModel);

// Get engine stats
const stats = await engine.stats();
console.log('Total queries:', stats.totalQueries);`} />
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex justify-end space-x-4">
        <button className="px-4 py-2 bg-surface-200 hover:bg-surface-300 text-foreground rounded-lg transition-colors">
          Reset to Defaults
        </button>
        <button className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-foreground rounded-lg transition-colors flex items-center space-x-2">
          <span>Save Configuration</span>
        </button>
      </div>
    </div>
  )
}

// SONA Learning Tab
function SonaTab() {
  const [sonaStats, setSonaStats] = useState({
    trajectories: 3421,
    patterns: 892,
    ewcTasks: 12,
    avgReward: 0.847
  })

  const [reasoningBank, setReasoningBank] = useState([
    { id: 1, pattern: 'Technical explanation', confidence: 0.94, uses: 234 },
    { id: 2, pattern: 'Code generation', confidence: 0.91, uses: 189 },
    { id: 3, pattern: 'Problem solving', confidence: 0.88, uses: 156 },
    { id: 4, pattern: 'Creative writing', confidence: 0.85, uses: 98 },
  ])

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-green-600 to-emerald-600 rounded-lg p-6">
        <div className="flex items-center space-x-3">
          <AcademicCapIcon />
          <div>
            <h2 className="text-xl font-bold text-foreground">SONA Learning System</h2>
            <p className="text-green-100">Self-Organizing Neural Architecture with continuous adaptation</p>
          </div>
        </div>
      </div>

      {/* SONA Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Trajectories" value={sonaStats.trajectories.toLocaleString()} icon={<ChartBarIcon />} color="bg-green-500/20 text-green-400" />
        <StatCard label="Learned Patterns" value={sonaStats.patterns} icon={<AcademicCapIcon />} color="bg-blue-500/20 text-blue-400" />
        <StatCard label="EWC Tasks" value={sonaStats.ewcTasks} icon={<LayersIcon />} color="bg-purple-500/20 text-purple-400" />
        <StatCard label="Avg Reward" value={sonaStats.avgReward.toFixed(3)} icon={<BrainIcon />} color="bg-yellow-500/20 text-yellow-400" />
      </div>

      {/* Components Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Trajectory Builder */}
        <div className="bg-surface-100 rounded-lg p-6 border border-default">
          <h3 className="text-lg font-semibold text-foreground mb-4">Trajectory Builder</h3>
          <p className="text-foreground-light text-sm mb-4">Captures query-response trajectories for learning</p>
          <CodeBlock code={`const builder = new TrajectoryBuilder({
  maxTrajectories: 10000,
  rewardDecay: 0.95
});

// Record a trajectory
builder.record({
  query: 'How do vectors work?',
  response: 'Vectors are...',
  reward: 0.9,
  metadata: { topic: 'math' }
});

// Export for training
const trajectories = builder.export();`} />
        </div>

        {/* Reasoning Bank */}
        <div className="bg-surface-100 rounded-lg p-6 border border-default">
          <h3 className="text-lg font-semibold text-foreground mb-4">Reasoning Bank</h3>
          <p className="text-foreground-light text-sm mb-4">Stores and retrieves learned reasoning patterns</p>
          <div className="space-y-2">
            {reasoningBank.map((pattern) => (
              <div key={pattern.id} className="flex items-center justify-between bg-surface-200/50 rounded-lg p-3">
                <div>
                  <p className="text-foreground font-medium">{pattern.pattern}</p>
                  <p className="text-foreground-light text-xs">{pattern.uses} uses</p>
                </div>
                <div className="text-right">
                  <div className="text-green-400 font-mono">{(pattern.confidence * 100).toFixed(0)}%</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* EWC Manager */}
        <div className="bg-surface-100 rounded-lg p-6 border border-default">
          <h3 className="text-lg font-semibold text-foreground mb-4">EWC Manager</h3>
          <p className="text-foreground-light text-sm mb-4">Elastic Weight Consolidation prevents catastrophic forgetting</p>
          <CodeBlock code={`const ewc = new EwcManager({
  lambda: 1000,
  sampleSize: 200
});

// Register a learned task
await ewc.registerTask('code_generation', model);

// Compute Fisher information
await ewc.computeFisher(dataLoader);

// Get EWC penalty for training
const penalty = ewc.penalty(currentParams);`} />
        </div>

        {/* SONA Coordinator */}
        <div className="bg-surface-100 rounded-lg p-6 border border-default">
          <h3 className="text-lg font-semibold text-foreground mb-4">SONA Coordinator</h3>
          <p className="text-foreground-light text-sm mb-4">Orchestrates all SONA components</p>
          <CodeBlock code={`const sona = new SonaCoordinator({
  enableEwc: true,
  enableReasoning: true,
  batchSize: 32
});

// Process feedback signal
await sona.learn({
  trajectoryId: 'traj_123',
  reward: 0.95,
  signal: 'positive'
});

// Get learning stats
const stats = sona.getStats();
console.log('Patterns learned:', stats.patterns);`} />
        </div>
      </div>

      {/* Actions */}
      <div className="flex justify-end space-x-4">
        <button className="px-4 py-2 bg-surface-200 hover:bg-surface-300 text-foreground rounded-lg transition-colors">
          Clear Trajectories
        </button>
        <button className="px-4 py-2 bg-green-600 hover:bg-green-500 text-foreground rounded-lg transition-colors flex items-center space-x-2">
          <PlayIcon />
          <span>Train from Trajectories</span>
        </button>
      </div>
    </div>
  )
}

// Training Tab
function TrainingTab() {
  const [trainingConfig, setTrainingConfig] = useState({
    learningRate: 0.001,
    batchSize: 32,
    epochs: 10,
    scheduler: 'cosine',
    warmupSteps: 100,
    checkpointEvery: 1000
  })

  const [trainingJobs, setTrainingJobs] = useState([
    { id: 'job_001', status: 'completed', progress: 100, epoch: 10, loss: 0.0234 },
    { id: 'job_002', status: 'running', progress: 65, epoch: 7, loss: 0.0412 },
    { id: 'job_003', status: 'pending', progress: 0, epoch: 0, loss: null },
  ])

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-orange-600 to-red-600 rounded-lg p-6">
        <div className="flex items-center space-x-3">
          <ChartBarIcon />
          <div>
            <h2 className="text-xl font-bold text-foreground">Training Pipeline</h2>
            <p className="text-orange-100">Configure and run training jobs with advanced scheduling</p>
          </div>
        </div>
      </div>

      {/* Training Config */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-surface-100 rounded-lg p-6 border border-default">
          <h3 className="text-lg font-semibold text-foreground mb-4">Training Configuration</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-foreground-light mb-1">Learning Rate</label>
              <input
                type="number"
                step="0.0001"
                value={trainingConfig.learningRate}
                onChange={(e) => setTrainingConfig({ ...trainingConfig, learningRate: parseFloat(e.target.value) })}
                className="w-full bg-surface-200 border border-default rounded-lg px-3 py-2 text-foreground"
              />
            </div>
            <div>
              <label className="block text-sm text-foreground-light mb-1">Batch Size</label>
              <select
                value={trainingConfig.batchSize}
                onChange={(e) => setTrainingConfig({ ...trainingConfig, batchSize: parseInt(e.target.value) })}
                className="w-full bg-surface-200 border border-default rounded-lg px-3 py-2 text-foreground"
              >
                <option value={8}>8</option>
                <option value={16}>16</option>
                <option value={32}>32</option>
                <option value={64}>64</option>
                <option value={128}>128</option>
              </select>
            </div>
            <div>
              <label className="block text-sm text-foreground-light mb-1">Epochs</label>
              <input
                type="number"
                value={trainingConfig.epochs}
                onChange={(e) => setTrainingConfig({ ...trainingConfig, epochs: parseInt(e.target.value) })}
                className="w-full bg-surface-200 border border-default rounded-lg px-3 py-2 text-foreground"
              />
            </div>
          </div>
        </div>

        <div className="bg-surface-100 rounded-lg p-6 border border-default">
          <h3 className="text-lg font-semibold text-foreground mb-4">Scheduler Settings</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-foreground-light mb-1">LR Scheduler</label>
              <select
                value={trainingConfig.scheduler}
                onChange={(e) => setTrainingConfig({ ...trainingConfig, scheduler: e.target.value })}
                className="w-full bg-surface-200 border border-default rounded-lg px-3 py-2 text-foreground"
              >
                <option value="constant">Constant</option>
                <option value="linear">Linear Decay</option>
                <option value="cosine">Cosine Annealing</option>
                <option value="polynomial">Polynomial Decay</option>
              </select>
            </div>
            <div>
              <label className="block text-sm text-foreground-light mb-1">Warmup Steps</label>
              <input
                type="number"
                value={trainingConfig.warmupSteps}
                onChange={(e) => setTrainingConfig({ ...trainingConfig, warmupSteps: parseInt(e.target.value) })}
                className="w-full bg-surface-200 border border-default rounded-lg px-3 py-2 text-foreground"
              />
            </div>
            <div>
              <label className="block text-sm text-foreground-light mb-1">Checkpoint Every N Steps</label>
              <input
                type="number"
                value={trainingConfig.checkpointEvery}
                onChange={(e) => setTrainingConfig({ ...trainingConfig, checkpointEvery: parseInt(e.target.value) })}
                className="w-full bg-surface-200 border border-default rounded-lg px-3 py-2 text-foreground"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Training Jobs */}
      <div className="bg-surface-100 rounded-lg p-6 border border-default">
        <h3 className="text-lg font-semibold text-foreground mb-4">Training Jobs</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-foreground-light text-sm border-b border-default">
                <th className="pb-3 font-medium">Job ID</th>
                <th className="pb-3 font-medium">Status</th>
                <th className="pb-3 font-medium">Progress</th>
                <th className="pb-3 font-medium">Epoch</th>
                <th className="pb-3 font-medium">Loss</th>
                <th className="pb-3 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {trainingJobs.map((job) => (
                <tr key={job.id} className="border-b border-default/50">
                  <td className="py-3 text-foreground font-mono text-sm">{job.id}</td>
                  <td className="py-3">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      job.status === 'completed' ? 'bg-green-500/20 text-green-400' :
                      job.status === 'running' ? 'bg-blue-500/20 text-blue-400' :
                      'bg-gray-500/20 text-foreground-light'
                    }`}>
                      {job.status}
                    </span>
                  </td>
                  <td className="py-3">
                    <div className="w-32 bg-surface-200 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${job.status === 'completed' ? 'bg-green-500' : 'bg-blue-500'}`}
                        style={{ width: `${job.progress}%` }}
                      />
                    </div>
                  </td>
                  <td className="py-3 text-foreground-light">{job.epoch}/{trainingConfig.epochs}</td>
                  <td className="py-3 text-foreground-light font-mono">{job.loss?.toFixed(4) || '-'}</td>
                  <td className="py-3">
                    <button className="text-foreground-light hover:text-foreground transition-colors">
                      {job.status === 'running' ? 'Stop' : 'View'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* API Example */}
      <div className="bg-surface-100 rounded-lg p-6 border border-default">
        <h3 className="text-lg font-semibold text-foreground mb-4">Training API</h3>
        <CodeBlock code={`import { TrainingPipeline, TrainingFactory } from '@ruvector/ruvllm';

const pipeline = TrainingFactory.create({
  type: 'sona',
  config: {
    learningRate: ${trainingConfig.learningRate},
    batchSize: ${trainingConfig.batchSize},
    epochs: ${trainingConfig.epochs},
    scheduler: '${trainingConfig.scheduler}',
    warmupSteps: ${trainingConfig.warmupSteps},
    checkpointEvery: ${trainingConfig.checkpointEvery}
  }
});

// Start training
const result = await pipeline.train(trajectories, {
  onProgress: (metrics) => console.log('Loss:', metrics.loss),
  onCheckpoint: (path) => console.log('Saved:', path)
});

console.log('Final loss:', result.finalLoss);`} />
      </div>

      {/* Actions */}
      <div className="flex justify-end space-x-4">
        <button className="px-4 py-2 bg-surface-200 hover:bg-surface-300 text-foreground rounded-lg transition-colors">
          Import Dataset
        </button>
        <button className="px-4 py-2 bg-orange-600 hover:bg-orange-500 text-foreground rounded-lg transition-colors flex items-center space-x-2">
          <PlayIcon />
          <span>Start Training</span>
        </button>
      </div>
    </div>
  )
}

// LoRA Tab
function LoraTab() {
  const [adapters, setAdapters] = useState([
    { id: 'adapter_code', name: 'Code Generation', rank: 8, alpha: 16, active: true, params: '2.3M' },
    { id: 'adapter_chat', name: 'Conversational', rank: 16, alpha: 32, active: false, params: '4.6M' },
    { id: 'adapter_math', name: 'Mathematical', rank: 4, alpha: 8, active: false, params: '1.2M' },
  ])

  const [newAdapter, setNewAdapter] = useState({
    name: '',
    rank: 8,
    alpha: 16,
    targetModules: ['q_proj', 'v_proj']
  })

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-pink-600 to-purple-600 rounded-lg p-6">
        <div className="flex items-center space-x-3">
          <LayersIcon />
          <div>
            <h2 className="text-xl font-bold text-foreground">LoRA Adapters</h2>
            <p className="text-pink-100">Low-Rank Adaptation for parameter-efficient fine-tuning</p>
          </div>
        </div>
      </div>

      {/* Adapter List */}
      <div className="bg-surface-100 rounded-lg p-6 border border-default">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-foreground">Installed Adapters</h3>
          <button className="px-3 py-1 bg-purple-600 hover:bg-purple-500 text-foreground rounded-lg text-sm transition-colors">
            + New Adapter
          </button>
        </div>
        <div className="space-y-3">
          {adapters.map((adapter) => (
            <div key={adapter.id} className={`flex items-center justify-between p-4 rounded-lg border ${adapter.active ? 'bg-purple-500/10 border-purple-500/50' : 'bg-surface-200/50 border-default'}`}>
              <div className="flex items-center space-x-4">
                <button
                  onClick={() => setAdapters(adapters.map(a => ({
                    ...a,
                    active: a.id === adapter.id ? !a.active : a.active
                  })))}
                  className={`w-10 h-6 rounded-full transition-colors ${adapter.active ? 'bg-purple-600' : 'bg-surface-300'}`}
                >
                  <span className={`block w-4 h-4 bg-white rounded-full transition-transform ${adapter.active ? 'translate-x-5' : 'translate-x-1'}`} />
                </button>
                <div>
                  <p className="text-foreground font-medium">{adapter.name}</p>
                  <p className="text-foreground-light text-sm">Rank: {adapter.rank} | Alpha: {adapter.alpha} | {adapter.params} params</p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <button className="px-3 py-1 bg-surface-300 hover:bg-surface-400 text-foreground rounded text-sm transition-colors">
                  Edit
                </button>
                <button className="px-3 py-1 bg-red-600/20 hover:bg-red-600/30 text-red-400 rounded text-sm transition-colors">
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Create Adapter */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-surface-100 rounded-lg p-6 border border-default">
          <h3 className="text-lg font-semibold text-foreground mb-4">Create New Adapter</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-foreground-light mb-1">Adapter Name</label>
              <input
                type="text"
                value={newAdapter.name}
                onChange={(e) => setNewAdapter({ ...newAdapter, name: e.target.value })}
                placeholder="my_adapter"
                className="w-full bg-surface-200 border border-default rounded-lg px-3 py-2 text-foreground"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-foreground-light mb-1">Rank (r)</label>
                <select
                  value={newAdapter.rank}
                  onChange={(e) => setNewAdapter({ ...newAdapter, rank: parseInt(e.target.value) })}
                  className="w-full bg-surface-200 border border-default rounded-lg px-3 py-2 text-foreground"
                >
                  <option value={4}>4</option>
                  <option value={8}>8</option>
                  <option value={16}>16</option>
                  <option value={32}>32</option>
                  <option value={64}>64</option>
                </select>
              </div>
              <div>
                <label className="block text-sm text-foreground-light mb-1">Alpha</label>
                <select
                  value={newAdapter.alpha}
                  onChange={(e) => setNewAdapter({ ...newAdapter, alpha: parseInt(e.target.value) })}
                  className="w-full bg-surface-200 border border-default rounded-lg px-3 py-2 text-foreground"
                >
                  <option value={8}>8</option>
                  <option value={16}>16</option>
                  <option value={32}>32</option>
                  <option value={64}>64</option>
                </select>
              </div>
            </div>
            <div>
              <label className="block text-sm text-foreground-light mb-1">Target Modules</label>
              <div className="flex flex-wrap gap-2">
                {['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'].map((mod) => (
                  <button
                    key={mod}
                    onClick={() => setNewAdapter({
                      ...newAdapter,
                      targetModules: newAdapter.targetModules.includes(mod)
                        ? newAdapter.targetModules.filter(m => m !== mod)
                        : [...newAdapter.targetModules, mod]
                    })}
                    className={`px-2 py-1 rounded text-sm transition-colors ${
                      newAdapter.targetModules.includes(mod)
                        ? 'bg-purple-600 text-foreground'
                        : 'bg-surface-300 text-foreground-light'
                    }`}
                  >
                    {mod}
                  </button>
                ))}
              </div>
            </div>
            <button className="w-full px-4 py-2 bg-purple-600 hover:bg-purple-500 text-foreground rounded-lg transition-colors">
              Create Adapter
            </button>
          </div>
        </div>

        <div className="bg-surface-100 rounded-lg p-6 border border-default">
          <h3 className="text-lg font-semibold text-foreground mb-4">LoRA API</h3>
          <CodeBlock code={`import { LoraAdapter, LoraManager } from '@ruvector/ruvllm';

// Create adapter
const adapter = new LoraAdapter({
  name: '${newAdapter.name || 'my_adapter'}',
  rank: ${newAdapter.rank},
  alpha: ${newAdapter.alpha},
  targetModules: ${JSON.stringify(newAdapter.targetModules)}
});

// Register with manager
const manager = new LoraManager();
manager.register(adapter);
manager.activate('${newAdapter.name || 'my_adapter'}');

// Apply to model
const adapted = adapter.forward(hiddenStates);

// Merge adapter weights (for inference)
const merged = adapter.merge(baseWeights);`} />
        </div>
      </div>
    </div>
  )
}

// Federated Tab
function FederatedTab() {
  const [agents, setAgents] = useState([
    { id: 'agent_1', status: 'active', trajectories: 342, lastSync: '2 min ago' },
    { id: 'agent_2', status: 'active', trajectories: 287, lastSync: '5 min ago' },
    { id: 'agent_3', status: 'idle', trajectories: 156, lastSync: '1 hour ago' },
  ])

  const [coordinatorStats, setCoordinatorStats] = useState({
    totalAgents: 3,
    activeAgents: 2,
    totalTrajectories: 785,
    aggregationRounds: 24
  })

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-cyan-600 to-blue-600 rounded-lg p-6">
        <div className="flex items-center space-x-3">
          <UsersIcon />
          <div>
            <h2 className="text-xl font-bold text-foreground">Federated Learning</h2>
            <p className="text-cyan-100">Distributed learning with ephemeral agents and secure aggregation</p>
          </div>
        </div>
      </div>

      {/* Coordinator Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Total Agents" value={coordinatorStats.totalAgents} icon={<UsersIcon />} color="bg-cyan-500/20 text-cyan-400" />
        <StatCard label="Active Agents" value={coordinatorStats.activeAgents} icon={<PlayIcon />} color="bg-green-500/20 text-green-400" />
        <StatCard label="Total Trajectories" value={coordinatorStats.totalTrajectories} icon={<ChartBarIcon />} color="bg-blue-500/20 text-blue-400" />
        <StatCard label="Aggregation Rounds" value={coordinatorStats.aggregationRounds} icon={<LayersIcon />} color="bg-purple-500/20 text-purple-400" />
      </div>

      {/* Agent List */}
      <div className="bg-surface-100 rounded-lg p-6 border border-default">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-foreground">Ephemeral Agents</h3>
          <button className="px-3 py-1 bg-cyan-600 hover:bg-cyan-500 text-foreground rounded-lg text-sm transition-colors">
            + Spawn Agent
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-foreground-light text-sm border-b border-default">
                <th className="pb-3 font-medium">Agent ID</th>
                <th className="pb-3 font-medium">Status</th>
                <th className="pb-3 font-medium">Trajectories</th>
                <th className="pb-3 font-medium">Last Sync</th>
                <th className="pb-3 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {agents.map((agent) => (
                <tr key={agent.id} className="border-b border-default/50">
                  <td className="py-3 text-foreground font-mono text-sm">{agent.id}</td>
                  <td className="py-3">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      agent.status === 'active' ? 'bg-green-500/20 text-green-400' : 'bg-gray-500/20 text-foreground-light'
                    }`}>
                      {agent.status}
                    </span>
                  </td>
                  <td className="py-3 text-foreground-light">{agent.trajectories}</td>
                  <td className="py-3 text-foreground-light text-sm">{agent.lastSync}</td>
                  <td className="py-3">
                    <div className="flex space-x-2">
                      <button className="text-cyan-400 hover:text-cyan-300 text-sm transition-colors">Sync</button>
                      <button className="text-red-400 hover:text-red-300 text-sm transition-colors">Terminate</button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* API Examples */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-surface-100 rounded-lg p-6 border border-default">
          <h3 className="text-lg font-semibold text-foreground mb-4">Ephemeral Agent</h3>
          <CodeBlock code={`import { EphemeralAgent } from '@ruvector/ruvllm';

const agent = new EphemeralAgent({
  id: 'agent_edge_1',
  maxTrajectories: 1000,
  privacyBudget: 0.1
});

// Process tasks and collect trajectories
await agent.processTask({
  query: 'Summarize this document',
  context: documentText
});

// Export state for aggregation
const state = agent.exportState();
coordinator.receiveUpdate(state);`} />
        </div>

        <div className="bg-surface-100 rounded-lg p-6 border border-default">
          <h3 className="text-lg font-semibold text-foreground mb-4">Federated Coordinator</h3>
          <CodeBlock code={`import { FederatedCoordinator } from '@ruvector/ruvllm';

const coordinator = new FederatedCoordinator({
  minAgents: 3,
  aggregationStrategy: 'fedavg',
  differentialPrivacy: true
});

// Create agents
const agent = coordinator.createAgent({
  id: 'agent_1'
});

// Aggregate updates
await coordinator.aggregate([
  agent1.exportState(),
  agent2.exportState(),
  agent3.exportState()
]);

// Apply consolidated learning
await coordinator.consolidate();`} />
        </div>
      </div>

      {/* Actions */}
      <div className="flex justify-end space-x-4">
        <button className="px-4 py-2 bg-surface-200 hover:bg-surface-300 text-foreground rounded-lg transition-colors">
          Export Logs
        </button>
        <button className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-foreground rounded-lg transition-colors flex items-center space-x-2">
          <PlayIcon />
          <span>Trigger Aggregation</span>
        </button>
      </div>
    </div>
  )
}

// Main Page Component
const LLMPage: NextPageWithLayout = () => {
  const router = useRouter()
  const [activeTab, setActiveTab] = useState<TabType>('overview')

  return (
    <div className="flex flex-col h-full bg-studio">
      {/* Header */}
      <div className="bg-surface-100 border-b border-default px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-foreground">RuvLLM</h1>
            <p className="text-foreground-light text-sm">Self-learning LLM orchestration engine</p>
          </div>
          <div className="flex items-center space-x-2">
            <span className="px-2 py-1 bg-green-500/20 text-green-400 rounded text-xs font-medium">v0.2.2</span>
            <span className="px-2 py-1 bg-blue-500/20 text-blue-400 rounded text-xs font-medium">SONA Enabled</span>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="bg-surface-100 border-b border-default px-6">
        <div className="flex space-x-1 overflow-x-auto">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center space-x-2 px-4 py-3 text-sm font-medium transition-colors whitespace-nowrap ${
                activeTab === tab.id
                  ? 'text-foreground border-b-2 border-purple-500'
                  : 'text-foreground-light hover:text-foreground'
              }`}
            >
              {tab.icon}
              <span>{tab.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto bg-studio p-6">
        {activeTab === 'overview' && <OverviewTab />}
        {activeTab === 'engine' && <EngineTab />}
        {activeTab === 'sona' && <SonaTab />}
        {activeTab === 'training' && <TrainingTab />}
        {activeTab === 'lora' && <LoraTab />}
        {activeTab === 'federated' && <FederatedTab />}
      </div>
    </div>
  )
}

LLMPage.getLayout = (page) => <ProjectLayoutWithAuth>{page}</ProjectLayoutWithAuth>

export default LLMPage
