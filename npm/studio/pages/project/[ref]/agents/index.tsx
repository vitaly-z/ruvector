import { useState, useEffect, useCallback } from 'react'
import { ProjectLayoutWithAuth } from 'components/layouts/ProjectLayout/ProjectLayout'
import type { NextPageWithLayout } from 'types'
import {
  Bot,
  Zap,
  Activity,
  Plus,
  Play,
  RefreshCw,
  MessageSquare,
  CheckCircle,
  Clock,
  TrendingUp,
  Sparkles,
  Search,
  MoreVertical,
  ArrowRight,
  Users,
  Target,
  Settings,
  HelpCircle,
} from 'lucide-react'

// Simple metric card
const StatCard = ({
  label,
  value,
  sublabel,
  icon: Icon,
  color,
}: {
  label: string
  value: string | number
  sublabel?: string
  icon: any
  color: string
}) => (
  <div className="rounded-2xl border border-default bg-surface-100 p-5 hover:shadow-lg transition-shadow">
    <div className="flex items-center gap-3 mb-3">
      <div className={`p-2.5 rounded-xl ${color}`}>
        <Icon className="w-5 h-5 text-white" />
      </div>
      <span className="text-sm text-foreground-light">{label}</span>
    </div>
    <div className="text-3xl font-bold text-foreground">{value}</div>
    {sublabel && <div className="text-xs text-foreground-light mt-1">{sublabel}</div>}
  </div>
)

// Agent card - simple and clean
const AgentCard = ({
  name,
  type,
  status,
  tasksCompleted,
  successRate,
  onRun,
}: {
  name: string
  type: string
  status: 'online' | 'busy' | 'offline'
  tasksCompleted: number
  successRate: number
  onRun: () => void
}) => {
  const statusColors = {
    online: 'bg-green-500',
    busy: 'bg-yellow-500',
    offline: 'bg-gray-400',
  }
  const statusLabels = {
    online: 'Ready',
    busy: 'Working',
    offline: 'Offline',
  }

  return (
    <div className="rounded-2xl border border-default bg-surface-100 p-5 hover:border-purple-400 hover:shadow-lg transition-all group">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center shadow-lg">
            <Bot className="w-6 h-6 text-white" />
          </div>
          <div>
            <h3 className="font-semibold text-foreground">{name}</h3>
            <p className="text-sm text-foreground-light capitalize">{type}</p>
          </div>
        </div>
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-surface-200">
          <span className={`w-2 h-2 rounded-full ${statusColors[status]} ${status === 'online' ? 'animate-pulse' : ''}`} />
          <span className="text-xs text-foreground-light">{statusLabels[status]}</span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="p-3 rounded-xl bg-surface-200 text-center">
          <div className="text-lg font-semibold text-foreground">{tasksCompleted}</div>
          <div className="text-xs text-foreground-light">Tasks Done</div>
        </div>
        <div className="p-3 rounded-xl bg-surface-200 text-center">
          <div className="text-lg font-semibold text-foreground">{successRate}%</div>
          <div className="text-xs text-foreground-light">Success</div>
        </div>
      </div>

      <button
        onClick={onRun}
        disabled={status === 'offline'}
        className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
      >
        <Play className="w-4 h-4" />
        {status === 'busy' ? 'View Progress' : 'Run Agent'}
      </button>
    </div>
  )
}

// Quick action button
const QuickAction = ({
  title,
  description,
  icon: Icon,
  color,
  onClick,
}: {
  title: string
  description: string
  icon: any
  color: string
  onClick: () => void
}) => (
  <button
    onClick={onClick}
    className="flex items-center gap-4 p-4 rounded-xl border border-default bg-surface-100 hover:border-purple-400 hover:shadow-md transition-all text-left w-full group"
  >
    <div className={`p-3 rounded-xl ${color}`}>
      <Icon className="w-5 h-5 text-white" />
    </div>
    <div className="flex-1">
      <h4 className="font-medium text-foreground group-hover:text-purple-500 transition-colors">{title}</h4>
      <p className="text-sm text-foreground-light">{description}</p>
    </div>
    <ArrowRight className="w-5 h-5 text-foreground-light group-hover:text-purple-500 group-hover:translate-x-1 transition-all" />
  </button>
)

// Recent activity item
const ActivityItem = ({
  agent,
  action,
  time,
  status,
}: {
  agent: string
  action: string
  time: string
  status: 'success' | 'running' | 'failed'
}) => {
  const statusIcons = {
    success: <CheckCircle className="w-4 h-4 text-green-500" />,
    running: <Activity className="w-4 h-4 text-blue-500 animate-pulse" />,
    failed: <Clock className="w-4 h-4 text-red-500" />,
  }

  return (
    <div className="flex items-center gap-3 p-3 rounded-xl hover:bg-surface-200 transition-colors">
      <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
        <Bot className="w-4 h-4 text-white" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm text-foreground truncate">
          <span className="font-medium">{agent}</span> {action}
        </p>
        <p className="text-xs text-foreground-light">{time}</p>
      </div>
      {statusIcons[status]}
    </div>
  )
}

const AgentDashboardPage: NextPageWithLayout = () => {
  const [loading, setLoading] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [showNewAgentModal, setShowNewAgentModal] = useState(false)

  // Simple agent data
  const agents = [
    { id: 1, name: 'Support Assistant', type: 'support', status: 'online' as const, tasksCompleted: 234, successRate: 96 },
    { id: 2, name: 'Content Writer', type: 'writer', status: 'busy' as const, tasksCompleted: 189, successRate: 94 },
    { id: 3, name: 'Data Analyst', type: 'analyst', status: 'online' as const, tasksCompleted: 156, successRate: 92 },
    { id: 4, name: 'Code Helper', type: 'developer', status: 'online' as const, tasksCompleted: 312, successRate: 91 },
    { id: 5, name: 'Research Agent', type: 'researcher', status: 'offline' as const, tasksCompleted: 87, successRate: 89 },
    { id: 6, name: 'Sales Assistant', type: 'sales', status: 'online' as const, tasksCompleted: 145, successRate: 93 },
  ]

  const recentActivity = [
    { agent: 'Support Assistant', action: 'resolved 3 customer tickets', time: '2 minutes ago', status: 'success' as const },
    { agent: 'Content Writer', action: 'is generating blog post', time: 'Just now', status: 'running' as const },
    { agent: 'Code Helper', action: 'completed code review', time: '15 minutes ago', status: 'success' as const },
    { agent: 'Data Analyst', action: 'processed sales report', time: '32 minutes ago', status: 'success' as const },
  ]

  const filteredAgents = agents.filter(agent =>
    agent.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    agent.type.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const handleRefresh = async () => {
    setLoading(true)
    // Simulate refresh
    await new Promise(resolve => setTimeout(resolve, 1000))
    setLoading(false)
  }

  const handleRunAgent = (agentId: number) => {
    console.log('Running agent:', agentId)
    // TODO: Implement agent run
  }

  return (
    <div className="w-full h-full overflow-y-auto bg-studio">
      <div className="px-6 py-8">
        <div className="mx-auto max-w-6xl space-y-8">
          {/* Welcome Header */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-foreground mb-1">AI Agents</h1>
              <p className="text-foreground-light">
                Your AI team is ready to help. Select an agent to get started.
              </p>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={handleRefresh}
                className="p-2.5 rounded-xl border border-default hover:bg-surface-200 transition-colors"
                title="Refresh"
              >
                <RefreshCw className={`w-5 h-5 text-foreground-light ${loading ? 'animate-spin' : ''}`} />
              </button>
              <button
                onClick={() => setShowNewAgentModal(true)}
                className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-purple-500 to-pink-500 text-white font-medium hover:opacity-90 transition-opacity shadow-lg shadow-purple-500/25"
              >
                <Plus className="w-5 h-5" />
                Add Agent
              </button>
            </div>
          </div>

          {/* Stats Overview */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <StatCard
              label="Active Agents"
              value={agents.filter(a => a.status !== 'offline').length}
              sublabel="of 6 total"
              icon={Bot}
              color="bg-gradient-to-br from-purple-500 to-pink-500"
            />
            <StatCard
              label="Tasks Today"
              value="47"
              sublabel="+12 from yesterday"
              icon={CheckCircle}
              color="bg-gradient-to-br from-green-500 to-emerald-500"
            />
            <StatCard
              label="Avg Response"
              value="1.2s"
              sublabel="Fast performance"
              icon={Zap}
              color="bg-gradient-to-br from-yellow-500 to-orange-500"
            />
            <StatCard
              label="Success Rate"
              value="94%"
              sublabel="Last 7 days"
              icon={Target}
              color="bg-gradient-to-br from-blue-500 to-cyan-500"
            />
          </div>

          {/* Quick Actions */}
          <div>
            <h2 className="text-lg font-semibold text-foreground mb-4">Quick Actions</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <QuickAction
                title="Ask a Question"
                description="Get instant answers from AI"
                icon={MessageSquare}
                color="bg-gradient-to-br from-blue-500 to-cyan-500"
                onClick={() => {}}
              />
              <QuickAction
                title="Automate a Task"
                description="Set up recurring workflows"
                icon={Sparkles}
                color="bg-gradient-to-br from-purple-500 to-pink-500"
                onClick={() => {}}
              />
              <QuickAction
                title="View Analytics"
                description="Track agent performance"
                icon={TrendingUp}
                color="bg-gradient-to-br from-green-500 to-emerald-500"
                onClick={() => {}}
              />
            </div>
          </div>

          {/* Main Content Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Agents List - Takes 2 columns */}
            <div className="lg:col-span-2 space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold text-foreground">Your Agents</h2>
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-foreground-light" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search agents..."
                    className="pl-10 pr-4 py-2 rounded-xl bg-surface-200 border border-default text-foreground text-sm w-48 focus:w-64 focus:outline-none focus:ring-2 focus:ring-purple-500/50 transition-all"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {filteredAgents.map((agent) => (
                  <AgentCard
                    key={agent.id}
                    name={agent.name}
                    type={agent.type}
                    status={agent.status}
                    tasksCompleted={agent.tasksCompleted}
                    successRate={agent.successRate}
                    onRun={() => handleRunAgent(agent.id)}
                  />
                ))}
              </div>

              {filteredAgents.length === 0 && (
                <div className="text-center py-12">
                  <Bot className="w-12 h-12 text-foreground-light mx-auto mb-3 opacity-50" />
                  <p className="text-foreground-light">No agents found matching "{searchQuery}"</p>
                </div>
              )}
            </div>

            {/* Sidebar - Activity & Help */}
            <div className="space-y-6">
              {/* Recent Activity */}
              <div className="rounded-2xl border border-default bg-surface-100 p-5">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold text-foreground">Recent Activity</h3>
                  <button className="text-sm text-purple-500 hover:text-purple-400">View all</button>
                </div>
                <div className="space-y-1">
                  {recentActivity.map((item, i) => (
                    <ActivityItem key={i} {...item} />
                  ))}
                </div>
              </div>

              {/* Help Card */}
              <div className="rounded-2xl border border-default bg-gradient-to-br from-purple-500/10 to-pink-500/10 p-5">
                <div className="flex items-center gap-3 mb-3">
                  <div className="p-2 rounded-lg bg-purple-500/20">
                    <HelpCircle className="w-5 h-5 text-purple-500" />
                  </div>
                  <h3 className="font-semibold text-foreground">Need Help?</h3>
                </div>
                <p className="text-sm text-foreground-light mb-4">
                  Learn how to get the most out of your AI agents with our quick start guide.
                </p>
                <button className="w-full px-4 py-2.5 rounded-xl border border-purple-500/30 text-purple-500 font-medium hover:bg-purple-500/10 transition-colors">
                  View Guide
                </button>
              </div>

              {/* Team Stats */}
              <div className="rounded-2xl border border-default bg-surface-100 p-5">
                <h3 className="font-semibold text-foreground mb-4">Team Performance</h3>
                <div className="space-y-4">
                  <div>
                    <div className="flex items-center justify-between text-sm mb-1">
                      <span className="text-foreground-light">Weekly Tasks</span>
                      <span className="font-medium text-foreground">312 / 350</span>
                    </div>
                    <div className="h-2 bg-surface-200 rounded-full overflow-hidden">
                      <div className="h-full w-[89%] bg-gradient-to-r from-purple-500 to-pink-500 rounded-full" />
                    </div>
                  </div>
                  <div>
                    <div className="flex items-center justify-between text-sm mb-1">
                      <span className="text-foreground-light">Customer Satisfaction</span>
                      <span className="font-medium text-foreground">4.8 / 5.0</span>
                    </div>
                    <div className="h-2 bg-surface-200 rounded-full overflow-hidden">
                      <div className="h-full w-[96%] bg-gradient-to-r from-green-500 to-emerald-500 rounded-full" />
                    </div>
                  </div>
                  <div>
                    <div className="flex items-center justify-between text-sm mb-1">
                      <span className="text-foreground-light">Response Time Goal</span>
                      <span className="font-medium text-foreground">1.2s / 2s</span>
                    </div>
                    <div className="h-2 bg-surface-200 rounded-full overflow-hidden">
                      <div className="h-full w-[60%] bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full" />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

AgentDashboardPage.getLayout = (page) => <ProjectLayoutWithAuth>{page}</ProjectLayoutWithAuth>

export default AgentDashboardPage
