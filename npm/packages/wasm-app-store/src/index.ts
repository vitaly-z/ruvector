/**
 * @ruvector/wasm-app-store
 *
 * WASM App Store for chip-sized (≤8KB) and full WASM applications.
 * Supports browser, mobile, and embedded targets.
 */

// Re-export payment types for convenience
export * from '@ruvector/agentic-payments';

// App Store Types
export interface AppMetadata {
  id: string;
  slug: string;
  name: string;
  description: string;
  long_description?: string;
  version: Version;
  publisher_id: string;
  category: AppCategory;
  tags: string[];
  platforms: Platform[];
  license: License;
  content_rating: ContentRating;
  status: AppStatus;
  icon_url?: string;
  screenshots: string[];
  homepage_url?: string;
  repository_url?: string;
  docs_url?: string;
  wasm_size: number;
  compressed_size?: number;
  wasm_hash: string;
  featured: boolean;
  verified_publisher: boolean;
  downloads: number;
  rating: number;
  rating_count: number;
  created_at: string;
  updated_at: string;
  published_at?: string;
}

export interface Version {
  major: number;
  minor: number;
  patch: number;
  prerelease?: string;
  build?: string;
}

export type AppCategory =
  | 'utilities'
  | 'data_processing'
  | 'ai_ml'
  | 'crypto'
  | 'media'
  | 'text'
  | 'math'
  | 'games'
  | 'finance'
  | 'dev_tools'
  | 'web'
  | 'embedded'
  | 'education'
  | 'social'
  | 'health'
  | 'other';

export type Platform =
  | 'browser'
  | 'nodejs'
  | 'deno'
  | 'cloudflare_workers'
  | 'embedded'
  | 'mobile'
  | 'desktop'
  | 'universal';

export type License = 'mit' | 'apache2' | 'gpl3' | 'bsd3' | 'proprietary' | { custom: string };

export type ContentRating = 'everyone' | 'teen' | 'mature';

export type AppStatus = 'draft' | 'pending_review' | 'published' | 'suspended' | 'deprecated' | 'archived';

export type AppSize = 'chip' | 'micro' | 'small' | 'medium' | 'large' | 'full';

export interface AppPricing {
  type: 'free' | 'one_time' | 'pay_per_use' | 'subscription' | 'freemium';
  price?: number;
  price_per_use?: number;
  monthly_price?: number;
  annual_price?: number;
  free_uses_per_day?: number;
}

export interface ChipApp {
  id: string;
  metadata: AppMetadata;
  wasm: Uint8Array;
  wasm_compressed: Uint8Array;
  pricing: AppPricing;
  stats: ChipAppStats;
}

export interface ChipAppStats {
  executions: number;
  total_execution_time_ms: number;
  avg_execution_time_ms: number;
  min_execution_time_ms: number;
  max_execution_time_ms: number;
  unique_users: number;
  total_revenue: number;
}

export interface FullApp {
  id: string;
  metadata: AppMetadata;
  main_module: WasmModule;
  modules: WasmModule[];
  assets: Asset[];
  config: AppConfig;
  dependencies: AppDependency[];
  pricing: AppPricing;
  release_notes?: string;
}

export interface WasmModule {
  name: string;
  wasm: Uint8Array;
  wasm_compressed: Uint8Array;
  hash: string;
}

export interface Asset {
  path: string;
  mime_type: string;
  data: Uint8Array;
  compressed_data?: Uint8Array;
  hash: string;
}

export interface AppConfig {
  env: Record<string, string>;
  features: Record<string, boolean>;
  runtime: RuntimeConfig;
  ui?: UiConfig;
}

export interface RuntimeConfig {
  initial_memory_pages: number;
  max_memory_pages: number;
  multi_threaded: boolean;
  enable_simd: boolean;
  timeout_ms?: number;
}

export interface UiConfig {
  width: number;
  height: number;
  resizable: boolean;
  fullscreen: boolean;
  theme: string;
}

export interface AppDependency {
  app_id: string;
  version_req: string;
  optional: boolean;
}

export interface SearchResult {
  metadata: AppMetadata;
  relevance_score: number;
  is_chip_app: boolean;
}

export interface AppStoreStats {
  total_apps: number;
  chip_apps: number;
  full_apps: number;
  publishers: number;
  featured: number;
  categories: number;
}

export interface CategoryInfo {
  id: string;
  name: string;
  icon: string;
}

// App Size Constants
export const APP_SIZE_LIMITS = {
  chip: 8 * 1024,           // 8KB
  micro: 64 * 1024,         // 64KB
  small: 512 * 1024,        // 512KB
  medium: 2 * 1024 * 1024,  // 2MB
  large: 10 * 1024 * 1024,  // 10MB
  full: Infinity,
} as const;

// Category metadata
export const CATEGORIES: CategoryInfo[] = [
  { id: 'utilities', name: 'Utilities', icon: '🔧' },
  { id: 'data_processing', name: 'Data Processing', icon: '📊' },
  { id: 'ai_ml', name: 'AI & Machine Learning', icon: '🤖' },
  { id: 'crypto', name: 'Cryptography & Security', icon: '🔐' },
  { id: 'media', name: 'Media Processing', icon: '🎬' },
  { id: 'text', name: 'Text Processing', icon: '📝' },
  { id: 'math', name: 'Math & Science', icon: '🔢' },
  { id: 'games', name: 'Games', icon: '🎮' },
  { id: 'finance', name: 'Finance', icon: '💰' },
  { id: 'dev_tools', name: 'Developer Tools', icon: '👨‍💻' },
  { id: 'web', name: 'Web & APIs', icon: '🌐' },
  { id: 'embedded', name: 'Embedded & IoT', icon: '📟' },
  { id: 'education', name: 'Education', icon: '📚' },
  { id: 'social', name: 'Social', icon: '💬' },
  { id: 'health', name: 'Health & Fitness', icon: '❤️' },
  { id: 'other', name: 'Other', icon: '📦' },
];

// Utility functions
export function getAppSizeCategory(bytes: number): AppSize {
  if (bytes <= APP_SIZE_LIMITS.chip) return 'chip';
  if (bytes <= APP_SIZE_LIMITS.micro) return 'micro';
  if (bytes <= APP_SIZE_LIMITS.small) return 'small';
  if (bytes <= APP_SIZE_LIMITS.medium) return 'medium';
  if (bytes <= APP_SIZE_LIMITS.large) return 'large';
  return 'full';
}

export function isChipApp(bytes: number): boolean {
  return bytes <= APP_SIZE_LIMITS.chip;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

export function parseVersion(versionStr: string): Version {
  const cleanVersion = versionStr.replace(/^v/, '');
  const [versionPre, build] = cleanVersion.split('+');
  const [version, prerelease] = versionPre.split('-');
  const [major, minor, patch] = version.split('.').map(Number);

  return {
    major: major || 0,
    minor: minor || 0,
    patch: patch || 0,
    prerelease,
    build,
  };
}

export function formatVersion(version: Version): string {
  let str = `${version.major}.${version.minor}.${version.patch}`;
  if (version.prerelease) str += `-${version.prerelease}`;
  if (version.build) str += `+${version.build}`;
  return str;
}

export function compareVersions(a: Version, b: Version): number {
  if (a.major !== b.major) return a.major - b.major;
  if (a.minor !== b.minor) return a.minor - b.minor;
  if (a.patch !== b.patch) return a.patch - b.patch;

  // Prerelease versions have lower precedence
  if (!a.prerelease && b.prerelease) return 1;
  if (a.prerelease && !b.prerelease) return -1;
  if (a.prerelease && b.prerelease) return a.prerelease.localeCompare(b.prerelease);

  return 0;
}

// WASM validation
export function isValidWasm(bytes: Uint8Array): boolean {
  if (bytes.length < 4) return false;
  // Check WASM magic number: \0asm
  return bytes[0] === 0x00 && bytes[1] === 0x61 && bytes[2] === 0x73 && bytes[3] === 0x6d;
}

// Compression helpers
export async function compressWasm(wasm: Uint8Array): Promise<Uint8Array> {
  if (typeof CompressionStream !== 'undefined') {
    const stream = new Response(wasm).body!.pipeThrough(new CompressionStream('gzip'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }
  // Fallback: return original
  return wasm;
}

export async function decompressWasm(compressed: Uint8Array): Promise<Uint8Array> {
  if (typeof DecompressionStream !== 'undefined') {
    const stream = new Response(compressed).body!.pipeThrough(new DecompressionStream('gzip'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }
  // Fallback: return original
  return compressed;
}

// Hash calculation
export async function calculateWasmHash(wasm: Uint8Array): Promise<string> {
  const hashBuffer = await crypto.subtle.digest('SHA-256', wasm);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

// Re-export WASM module loading helper
export async function loadWasmAppStore(): Promise<typeof import('../wasm/wasm_app_store_wasm')> {
  const wasm = await import('../wasm/wasm_app_store_wasm');
  await wasm.default();
  return wasm;
}

// Template WASM modules for testing
export const TEMPLATES = {
  /** Minimal "Hello World" WASM module (~70 bytes) that returns 42 */
  helloWorld: new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, // magic number (\0asm)
    0x01, 0x00, 0x00, 0x00, // version 1
    // Type section
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
    // Function section
    0x03, 0x02, 0x01, 0x00,
    // Export section
    0x07, 0x08, 0x01, 0x04, 0x6d, 0x61, 0x69, 0x6e, 0x00, 0x00,
    // Code section
    0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x2a, 0x0b, // returns 42
  ]),

  /** Add function WASM module (~100 bytes) */
  add: new Uint8Array([
    0x00, 0x61, 0x73, 0x6d, // magic
    0x01, 0x00, 0x00, 0x00, // version
    // Type section: (i32, i32) -> i32
    0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f,
    // Function section
    0x03, 0x02, 0x01, 0x00,
    // Export section
    0x07, 0x07, 0x01, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00,
    // Code section
    0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b,
  ]),
};
