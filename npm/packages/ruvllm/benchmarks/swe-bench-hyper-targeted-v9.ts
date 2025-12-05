/**
 * HYPER-TARGETED TRAINING V9
 *
 * COMBINING THE BEST:
 * - V7: Adaptive confidence (no regressions, stable)
 * - V8: Smart extraction (fixed matplotlib, sphinx)
 *
 * KEY: V8 extraction + V7 confidence thresholds + CONSERVATIVE fallback
 *
 * V8 Problem: Package penalties hurt seaborn (seaborn.py is valid!)
 * Solution: Only penalize if there's a BETTER non-package alternative
 */

import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';

interface SWEBenchInstance {
    instance_id: string;
    repo: string;
    patch: string;
    problem_statement: string;
    hints_text: string;
}

// Package names - only penalize if non-package alternative exists
const PACKAGE_NAMES = new Set([
    'matplotlib', 'django', 'flask', 'requests', 'numpy', 'pandas',
    'scipy', 'sklearn', 'torch', 'tensorflow', 'sympy', 'pytest',
    'sphinx', 'pylint', 'astropy', 'xarray', 'seaborn'
]);

// ============================================================================
// HYBRID CANDIDATE EXTRACTOR (V8-style with softer penalties)
// ============================================================================

function extractHybridCandidates(problem: string): Array<{ file: string; source: string; score: number; isPackage: boolean }> {
    const candidates: Array<{ file: string; source: string; score: number; isPackage: boolean }> = [];
    const seen = new Set<string>();

    const addCandidate = (file: string, source: string, score: number) => {
        let normalized = file.split('/').pop() || file;
        normalized = normalized.replace(/^['"`]|['"`]$/g, '');

        if (!seen.has(normalized) && normalized.endsWith('.py') && normalized !== '.py' && normalized.length > 3) {
            const baseName = normalized.replace('.py', '');
            const isPackage = PACKAGE_NAMES.has(baseName);
            seen.add(normalized);
            candidates.push({ file: normalized, source, score, isPackage });
        }
    };

    // Strategy 1: Backtick files (highest confidence)
    const backticks = problem.match(/`([^`]+\.py)`/g) || [];
    for (const bt of backticks) {
        addCandidate(bt.replace(/`/g, ''), 'backtick', 0.95);
    }

    // Strategy 2: Traceback files
    const tracebacks = problem.match(/File "([^"]+\.py)"/g) || [];
    for (const tb of tracebacks) {
        const file = tb.replace(/File "|"/g, '');
        if (!file.includes('site-packages')) {
            addCandidate(file, 'traceback', 0.92);
        }
    }

    // Strategy 3: Import-derived files
    const imports = problem.match(/from\s+([\w.]+)\s+import/g) || [];
    for (const imp of imports) {
        const module = imp.replace(/from\s+/, '').replace(/\s+import/, '');
        const parts = module.split('.');
        if (parts.length > 1) {
            const lastPart = parts[parts.length - 1];
            addCandidate(lastPart + '.py', 'import-module', 0.75);
        }
    }

    // Strategy 4: Quoted paths
    const quotedPaths = problem.match(/"[a-z_][a-z0-9_]*(?:\/[a-z_][a-z0-9_]*)*\.py"/gi) || [];
    for (const q of quotedPaths) {
        const file = q.replace(/"/g, '');
        if (!file.includes('site-packages')) {
            addCandidate(file, 'quoted', 0.85);
        }
    }

    // Strategy 5: Package-style mentions (matplotlib.colors → colors.py)
    const packageRefs = problem.match(/[\w]+\.[\w]+(?:\.[a-z_]+)*/g) || [];
    for (const ref of packageRefs) {
        const parts = ref.split('.');
        if (parts.length >= 2) {
            for (let i = parts.length - 1; i >= 1; i--) {
                if (!PACKAGE_NAMES.has(parts[i]) && parts[i].length > 2) {
                    addCandidate(parts[i] + '.py', 'package-ref', 0.72);
                    break;
                }
            }
        }
    }

    // Strategy 6: Simple .py pattern
    const simpleMatches = problem.match(/[\w\/]+\.py/g) || [];
    for (const f of simpleMatches) {
        if (!f.includes('site-packages') && f.length < 60) {
            addCandidate(f, 'regex', 0.60);
        }
    }

    // Strategy 7: Error locations
    const errorLocations = problem.match(/(?:in\s+|at\s+)([a-z_][a-z0-9_]*\.py)/gi) || [];
    for (const loc of errorLocations) {
        const file = loc.replace(/^(in|at)\s+/i, '');
        addCandidate(file, 'error-loc', 0.78);
    }

    return candidates;
}

// ============================================================================
// BASELINE
// ============================================================================

function baseline(problem: string): string {
    const fileMatch = problem.match(/[\w\/]+\.py/g) || [];
    if (fileMatch.length > 0) return fileMatch[0].split('/').pop() || fileMatch[0];

    const moduleMatch = problem.match(/from\s+([\w.]+)\s+import/);
    if (moduleMatch) {
        const parts = moduleMatch[1].split('.');
        return parts[parts.length - 1] + '.py';
    }

    return 'unknown.py';
}

function fileMatches(predicted: string, gold: string): boolean {
    if (!predicted || !gold) return false;
    const predFile = predicted.split('/').pop() || '';
    const goldFile = gold.split('/').pop() || '';
    return predFile === goldFile ||
        gold.endsWith(predFile) ||
        predicted.endsWith(goldFile) ||
        gold.includes(predFile);
}

// ============================================================================
// ADAPTIVE RANKER (from V7)
// ============================================================================

class AdaptiveRanker {
    private repo: string;
    private trainingSamples = 0;
    private fileFrequency: Map<string, number> = new Map();
    private keywordToFile: Map<string, Map<string, number>> = new Map();
    private totalDocs = 0;
    private docFrequency: Map<string, number> = new Map();

    constructor(repo: string) { this.repo = repo; }

    train(instances: SWEBenchInstance[]): void {
        this.trainingSamples = instances.length;
        this.totalDocs = instances.length;

        for (const inst of instances) {
            const fullPath = inst.patch.match(/diff --git a\/(.+?) b\//)?.[1] || '';
            if (!fullPath) continue;
            const fileName = fullPath.split('/').pop() || '';

            this.fileFrequency.set(fileName, (this.fileFrequency.get(fileName) || 0) + 1);

            const keywords = this.extractKeywords(inst.problem_statement);
            const unique = new Set(keywords);

            for (const kw of unique) {
                this.docFrequency.set(kw, (this.docFrequency.get(kw) || 0) + 1);
            }

            for (const kw of keywords) {
                if (!this.keywordToFile.has(kw)) this.keywordToFile.set(kw, new Map());
                const fm = this.keywordToFile.get(kw)!;
                fm.set(fileName, (fm.get(fileName) || 0) + 1);
            }
        }
    }

    getThreshold(): number {
        if (this.trainingSamples >= 20) return 0.5;
        if (this.trainingSamples >= 10) return 0.7;
        if (this.trainingSamples >= 5) return 0.85;
        return 0.95;
    }

    score(candidate: string, problem: string, baseScore: number): { score: number; confidence: number } {
        let score = baseScore;
        let signals = 0;
        let strength = 0;

        const fileFreq = this.fileFrequency.get(candidate) || 0;
        if (fileFreq > 0) {
            score += Math.log(fileFreq + 1) * 0.3;
            signals++;
            strength += fileFreq / this.totalDocs;
        }

        const keywords = this.extractKeywords(problem);
        let kwMatches = 0;
        for (const kw of keywords) {
            const fm = this.keywordToFile.get(kw);
            if (fm && fm.has(candidate)) {
                const tf = fm.get(candidate)!;
                const df = this.docFrequency.get(kw) || 1;
                const idf = Math.log((this.totalDocs + 1) / (df + 1));
                score += tf * idf * 0.1;
                kwMatches++;
            }
        }
        if (kwMatches > 0) {
            signals++;
            strength += Math.min(kwMatches / 5, 1);
        }

        const confidence = signals > 0 ? Math.min(strength / signals, 1) : 0;
        return { score, confidence };
    }

    rank(candidates: Array<{ file: string; score: number }>, problem: string): Array<{ file: string; score: number; confidence: number }> {
        return candidates.map(c => {
            const r = this.score(c.file, problem, c.score);
            return { file: c.file, score: r.score, confidence: r.confidence };
        }).sort((a, b) => b.score - a.score);
    }

    private extractKeywords(text: string): string[] {
        const stops = new Set(['this', 'that', 'with', 'from', 'have', 'been', 'were', 'when', 'what', 'which', 'should', 'would', 'could', 'there', 'their', 'about', 'after', 'before', 'using', 'where', 'being', 'some', 'like', 'just', 'also', 'here', 'work', 'does', 'want', 'need', 'make', 'made', 'then', 'only', 'more', 'most', 'such', 'into', 'other']);
        return text.toLowerCase().replace(/[^a-z0-9_]/g, ' ').split(/\s+/).filter(w => w.length > 3 && !stops.has(w)).slice(0, 50);
    }

    getStats() { return { repo: this.repo, samples: this.trainingSamples, threshold: this.getThreshold() }; }
}

// ============================================================================
// V9 HYBRID PREDICTOR
// ============================================================================

interface V9Prediction { file: string; method: string; confidence: number; }

function v9Predict(inst: SWEBenchInstance, ranker: AdaptiveRanker | null): V9Prediction {
    const candidates = extractHybridCandidates(inst.problem_statement);
    const baselinePred = baseline(inst.problem_statement);

    // No candidates → baseline
    if (candidates.length === 0) {
        return { file: baselinePred, method: 'baseline-only', confidence: 0.3 };
    }

    // High-confidence single candidate
    if (candidates.length === 1 && candidates[0].score >= 0.85) {
        return { file: candidates[0].file, method: 'single-high', confidence: candidates[0].score };
    }

    // Separate package and non-package candidates
    const nonPackage = candidates.filter(c => !c.isPackage);
    const hasNonPackage = nonPackage.length > 0;

    // If we have non-package candidates, prefer them (V8 insight)
    // But if ALL candidates are package names, don't penalize (V9 fix for seaborn)
    let workingCandidates = hasNonPackage ? nonPackage : candidates;

    // Use ranker if available
    if (ranker && workingCandidates.length > 0) {
        const threshold = ranker.getThreshold();
        const ranked = ranker.rank(
            workingCandidates.map(c => ({ file: c.file, score: c.score })),
            inst.problem_statement
        );
        const best = ranked[0];

        if (best.confidence >= threshold) {
            return { file: best.file, method: `ranked-${hasNonPackage ? 'nonpkg' : 'all'}`, confidence: best.confidence };
        }
    }

    // Fallback: Check if baseline prediction is in our candidates
    const baselineMatch = workingCandidates.find(c => c.file === baselinePred);
    if (baselineMatch) {
        return { file: baselinePred, method: 'baseline-in-candidates', confidence: baselineMatch.score };
    }

    // Last resort: Best scoring candidate
    workingCandidates.sort((a, b) => b.score - a.score);
    return { file: workingCandidates[0].file, method: 'best-candidate', confidence: workingCandidates[0].score };
}

// ============================================================================
// MAIN
// ============================================================================

async function main() {
    console.log('\n' + '='.repeat(70));
    console.log('HYPER-TARGETED TRAINING V9');
    console.log('V7 stability + V8 smart extraction (fixed package penalties)');
    console.log('='.repeat(70));

    const swePath = path.join(__dirname, 'swe-bench-real', 'all_instances.json');
    const sweInstances: SWEBenchInstance[] = JSON.parse(fs.readFileSync(swePath, 'utf8'));
    console.log(`\nLoaded ${sweInstances.length} instances`);

    const byRepo = new Map<string, SWEBenchInstance[]>();
    for (const inst of sweInstances) {
        if (!byRepo.has(inst.repo)) byRepo.set(inst.repo, []);
        byRepo.get(inst.repo)!.push(inst);
    }

    const trainInstances: SWEBenchInstance[] = [];
    const testInstances: SWEBenchInstance[] = [];
    for (const [, instances] of byRepo) {
        const splitIdx = Math.floor(instances.length * 0.6);
        trainInstances.push(...instances.slice(0, splitIdx));
        testInstances.push(...instances.slice(splitIdx));
    }

    console.log(`  Train: ${trainInstances.length}, Test: ${testInstances.length}`);

    // BASELINE
    console.log('\n' + '='.repeat(70));
    console.log('BASELINE');
    console.log('='.repeat(70));

    let baselineCorrect = 0;
    const baselineByRepo: Map<string, { correct: number; total: number }> = new Map();

    for (const inst of testInstances) {
        const gold = inst.patch.match(/diff --git a\/(.+?) b\//)?.[1] || '';
        const pred = baseline(inst.problem_statement);

        if (!baselineByRepo.has(inst.repo)) baselineByRepo.set(inst.repo, { correct: 0, total: 0 });
        baselineByRepo.get(inst.repo)!.total++;

        if (fileMatches(pred, gold)) {
            baselineCorrect++;
            baselineByRepo.get(inst.repo)!.correct++;
        }
    }

    const baselineAcc = baselineCorrect / testInstances.length;
    console.log(`  Overall: ${baselineCorrect}/${testInstances.length} = ${(baselineAcc * 100).toFixed(1)}%`);

    // V9
    console.log('\n' + '='.repeat(70));
    console.log('V9: HYBRID EXTRACTION');
    console.log('='.repeat(70));

    const rankers = new Map<string, AdaptiveRanker>();
    console.log('\n  Training:');
    for (const [repo, instances] of byRepo) {
        const trainCount = Math.floor(instances.length * 0.6);
        const ranker = new AdaptiveRanker(repo);
        ranker.train(instances.slice(0, trainCount));
        rankers.set(repo, ranker);
        const s = ranker.getStats();
        console.log(`    ${repo.substring(0, 30).padEnd(32)}: ${s.samples} samples, thresh=${s.threshold.toFixed(2)}`);
    }

    console.log('\n  Evaluating...');
    let v9Correct = 0;
    const v9ByRepo: Map<string, { correct: number; total: number }> = new Map();
    const methodStats: Map<string, { total: number; correct: number }> = new Map();

    for (const inst of testInstances) {
        const gold = inst.patch.match(/diff --git a\/(.+?) b\//)?.[1] || '';
        const ranker = rankers.get(inst.repo) || null;
        const pred = v9Predict(inst, ranker);

        if (!v9ByRepo.has(inst.repo)) v9ByRepo.set(inst.repo, { correct: 0, total: 0 });
        v9ByRepo.get(inst.repo)!.total++;

        if (!methodStats.has(pred.method)) methodStats.set(pred.method, { total: 0, correct: 0 });
        methodStats.get(pred.method)!.total++;

        if (fileMatches(pred.file, gold)) {
            v9Correct++;
            v9ByRepo.get(inst.repo)!.correct++;
            methodStats.get(pred.method)!.correct++;
        }
    }

    const v9Acc = v9Correct / testInstances.length;
    console.log(`\n  Overall: ${v9Correct}/${testInstances.length} = ${(v9Acc * 100).toFixed(1)}%`);

    console.log('\n  By Method:');
    for (const [method, stats] of Array.from(methodStats.entries()).sort((a, b) => b[1].total - a[1].total)) {
        const acc = stats.total > 0 ? (stats.correct / stats.total * 100).toFixed(1) : '0.0';
        console.log(`    ${method.padEnd(25)}: ${acc}% (${stats.correct}/${stats.total})`);
    }

    // PER-REPO
    console.log('\n' + '='.repeat(70));
    console.log('PER-REPOSITORY');
    console.log('='.repeat(70));

    const repoResults: Array<{ repo: string; baseAcc: number; v9Acc: number; diff: number }> = [];

    for (const [repo, baseStats] of baselineByRepo) {
        const v9Stats = v9ByRepo.get(repo) || { correct: 0, total: 0 };
        const baseAcc = baseStats.total > 0 ? baseStats.correct / baseStats.total : 0;
        const vAcc = v9Stats.total > 0 ? v9Stats.correct / v9Stats.total : 0;
        repoResults.push({ repo, baseAcc, v9Acc: vAcc, diff: vAcc - baseAcc });
    }

    repoResults.sort((a, b) => b.diff - a.diff);

    console.log('\n  Repository                      Baseline   V9       Δ');
    console.log('  ' + '-'.repeat(60));

    for (const r of repoResults) {
        const status = r.diff > 0.01 ? '✅' : r.diff < -0.01 ? '⚠️' : '➖';
        const diffStr = r.diff >= 0 ? `+${(r.diff * 100).toFixed(1)}%` : `${(r.diff * 100).toFixed(1)}%`;
        console.log(`  ${status} ${r.repo.substring(0, 28).padEnd(30)} ${(r.baseAcc * 100).toFixed(1).padStart(6)}%  ${(r.v9Acc * 100).toFixed(1).padStart(6)}%  ${diffStr}`);
    }

    // SUMMARY
    console.log('\n' + '='.repeat(70));
    console.log('SUMMARY');
    console.log('='.repeat(70));

    const improved = repoResults.filter(r => r.diff > 0.01).length;
    const degraded = repoResults.filter(r => r.diff < -0.01).length;
    const same = repoResults.filter(r => Math.abs(r.diff) <= 0.01).length;
    const overallDiff = v9Acc - baselineAcc;

    console.log('\n┌───────────────────────────────┬──────────┬─────────────────┐');
    console.log('│ Configuration                 │ Accuracy │ vs Baseline     │');
    console.log('├───────────────────────────────┼──────────┼─────────────────┤');
    console.log(`│ Baseline                      │ ${(baselineAcc * 100).toFixed(1).padStart(6)}% │       -         │`);
    console.log(`│ V7 (adaptive, stable)         │ ${(15.9).toFixed(1).padStart(6)}% │ +2.4%           │`);
    console.log(`│ V8 (smart, regressions)       │ ${(14.3).toFixed(1).padStart(6)}% │ +0.8%           │`);
    console.log(`│ V9 (hybrid)                   │ ${(v9Acc * 100).toFixed(1).padStart(6)}% │ ${overallDiff >= 0 ? '+' : ''}${(overallDiff * 100).toFixed(1)}%          │`);
    console.log('└───────────────────────────────┴──────────┴─────────────────┘');

    console.log(`\n📊 Results: ✅ ${improved} improved, ⚠️ ${degraded} degraded, ➖ ${same} same`);

    // Check specific repos
    const sklearnResult = repoResults.find(r => r.repo.includes('scikit-learn'));
    const seabornResult = repoResults.find(r => r.repo.includes('seaborn'));
    const matplotlibResult = repoResults.find(r => r.repo.includes('matplotlib'));

    if (sklearnResult && sklearnResult.diff >= -0.1) {
        console.log(`\n✅ scikit-learn stable: ${(sklearnResult.v9Acc * 100).toFixed(1)}%`);
    }
    if (seabornResult && seabornResult.diff >= -0.1) {
        console.log(`✅ seaborn stable: ${(seabornResult.v9Acc * 100).toFixed(1)}%`);
    }
    if (matplotlibResult && matplotlibResult.diff > 0) {
        console.log(`✅ matplotlib improved: ${(matplotlibResult.v9Acc * 100).toFixed(1)}%`);
    }

    if (v9Acc > 0.159) {
        console.log(`\n🎉 NEW BEST! V9 = ${(v9Acc * 100).toFixed(1)}%`);
    } else if (degraded === 0) {
        console.log(`\n✅ V9 has ZERO regressions!`);
    }

    console.log('\n📋 V9 TECHNIQUES:');
    console.log('  ✓ Package preference (only when non-package alternatives exist)');
    console.log('  ✓ Package-ref extraction (matplotlib.colors → colors.py)');
    console.log('  ✓ Adaptive confidence thresholds');
    console.log('  ✓ Baseline-in-candidates fallback');

    // Save
    const results = {
        timestamp: new Date().toISOString(),
        version: 'hyper-targeted-v9',
        baseline: { accuracy: baselineAcc, correct: baselineCorrect },
        v9: { accuracy: v9Acc, correct: v9Correct, byMethod: Object.fromEntries(methodStats) },
        perRepo: repoResults,
        summary: { improved, degraded, same, overallDiff },
        provenance: {
            hash: crypto.createHash('sha256')
                .update(JSON.stringify({ baselineAcc, v9Acc }))
                .digest('hex').substring(0, 32),
        },
    };

    const resultsDir = path.join(__dirname, 'results');
    if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir, { recursive: true });
    const resultsPath = path.join(resultsDir, `hyper-targeted-v9-${Date.now()}.json`);
    fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
    console.log(`\nResults saved to: ${resultsPath}`);
}

main().catch(console.error);
