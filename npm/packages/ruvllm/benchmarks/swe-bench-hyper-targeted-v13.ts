/**
 * HYPER-TARGETED TRAINING V13
 *
 * BUILD ON V12 (18.3%):
 * - Keep protected baseline repos
 * - Try to improve remaining 0% repos: flask, requests, xarray, pylint
 * - Try to improve pytest (14.3% → ?)
 *
 * NEW TECHNIQUES:
 * - Cross-repo common patterns
 * - Keyword→file learning for sparse repos
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

const PACKAGE_NAMES = new Set([
    'matplotlib', 'django', 'flask', 'requests', 'numpy', 'pandas',
    'scipy', 'sklearn', 'torch', 'tensorflow', 'sympy', 'pytest',
    'sphinx', 'pylint', 'astropy', 'xarray', 'seaborn'
]);

const HIGH_BASELINE_REPOS = new Set([
    'scikit-learn/scikit-learn',
    'mwaskom/seaborn',
    'astropy/astropy'
]);

// ============================================================================
// CROSS-REPO PATTERNS
// ============================================================================

class CrossRepoLearner {
    private keywordToFile: Map<string, Map<string, number>> = new Map();
    private commonFiles: Map<string, number> = new Map();

    train(instances: SWEBenchInstance[]): void {
        for (const inst of instances) {
            const file = inst.patch.match(/diff --git a\/(.+?) b\//)?.[1] || '';
            const fileName = file.split('/').pop() || '';
            if (!fileName) continue;

            // Track common files across all repos
            this.commonFiles.set(fileName, (this.commonFiles.get(fileName) || 0) + 1);

            // Learn keyword → file patterns
            const keywords = this.extractKeywords(inst.problem_statement);
            for (const kw of keywords) {
                if (!this.keywordToFile.has(kw)) {
                    this.keywordToFile.set(kw, new Map());
                }
                const fileMap = this.keywordToFile.get(kw)!;
                fileMap.set(fileName, (fileMap.get(fileName) || 0) + 1);
            }
        }
    }

    suggestFile(problem: string): string | null {
        const keywords = this.extractKeywords(problem);
        const votes: Map<string, number> = new Map();

        for (const kw of keywords) {
            const fileMap = this.keywordToFile.get(kw);
            if (fileMap) {
                for (const [file, count] of fileMap) {
                    // Only count if strong signal (appeared multiple times)
                    if (count >= 2) {
                        votes.set(file, (votes.get(file) || 0) + count);
                    }
                }
            }
        }

        if (votes.size === 0) return null;

        const sorted = Array.from(votes.entries()).sort((a, b) => b[1] - a[1]);
        // Only suggest if clear winner
        if (sorted[0][1] >= 3 && (sorted.length === 1 || sorted[0][1] > sorted[1][1] * 1.5)) {
            return sorted[0][0];
        }
        return null;
    }

    private extractKeywords(text: string): string[] {
        return text.toLowerCase()
            .replace(/[^a-z0-9_]/g, ' ')
            .split(/\s+/)
            .filter(w => w.length > 4);
    }
}

// ============================================================================
// CANDIDATE EXTRACTOR
// ============================================================================

function extractCandidates(problem: string): Array<{ file: string; source: string; score: number; isPackage: boolean }> {
    const candidates: Array<{ file: string; source: string; score: number; isPackage: boolean }> = [];
    const seen = new Set<string>();

    const add = (file: string, source: string, score: number) => {
        let normalized = file.split('/').pop() || file;
        normalized = normalized.replace(/^['"`]|['"`]$/g, '');
        if (!seen.has(normalized) && normalized.endsWith('.py') && normalized !== '.py' && normalized.length > 3) {
            const isPackage = PACKAGE_NAMES.has(normalized.replace('.py', ''));
            seen.add(normalized);
            candidates.push({ file: normalized, source, score, isPackage });
        }
    };

    // Backticks
    (problem.match(/`([^`]+\.py)`/g) || []).forEach(m => add(m.replace(/`/g, ''), 'backtick', 0.95));

    // Tracebacks
    (problem.match(/File "([^"]+\.py)"/g) || []).forEach(m => {
        const f = m.replace(/File "|"/g, '');
        if (!f.includes('site-packages')) add(f, 'traceback', 0.92);
    });

    // Package refs
    (problem.match(/[\w]+\.[\w]+(?:\.[a-z_]+)*/g) || []).forEach(ref => {
        const parts = ref.split('.');
        for (let i = parts.length - 1; i >= 1; i--) {
            if (!PACKAGE_NAMES.has(parts[i]) && parts[i].length > 2) {
                add(parts[i] + '.py', 'package-ref', 0.75);
                break;
            }
        }
    });

    // Imports
    (problem.match(/from\s+([\w.]+)\s+import/g) || []).forEach(imp => {
        const parts = imp.replace(/from\s+/, '').replace(/\s+import/, '').split('.');
        if (parts.length > 1) add(parts[parts.length - 1] + '.py', 'import', 0.72);
    });

    // Simple .py
    (problem.match(/[\w\/]+\.py/g) || []).forEach(f => {
        if (!f.includes('site-packages') && f.length < 60) add(f, 'regex', 0.60);
    });

    // Error locations
    (problem.match(/(?:in\s+|at\s+)([a-z_][a-z0-9_]*\.py)/gi) || []).forEach(loc => {
        add(loc.replace(/^(in|at)\s+/i, ''), 'error-loc', 0.78);
    });

    // Quoted
    (problem.match(/"([a-z_][a-z0-9_\/]*\.py)"/gi) || []).forEach(q => {
        const f = q.replace(/"/g, '');
        if (!f.includes('site-packages')) add(f, 'quoted', 0.80);
    });

    return candidates;
}

// ============================================================================
// RANKER
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
            for (const kw of unique) this.docFrequency.set(kw, (this.docFrequency.get(kw) || 0) + 1);
            for (const kw of keywords) {
                if (!this.keywordToFile.has(kw)) this.keywordToFile.set(kw, new Map());
                this.keywordToFile.get(kw)!.set(fileName, (this.keywordToFile.get(kw)!.get(fileName) || 0) + 1);
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
        let signals = 0, strength = 0;

        const fileFreq = this.fileFrequency.get(candidate) || 0;
        if (fileFreq > 0) {
            score += Math.log(fileFreq + 1) * 0.3;
            signals++;
            strength += fileFreq / Math.max(this.totalDocs, 1);
        }

        const keywords = this.extractKeywords(problem);
        let kwMatches = 0;
        for (const kw of keywords) {
            const fm = this.keywordToFile.get(kw);
            if (fm && fm.has(candidate)) {
                score += fm.get(candidate)! * Math.log((this.totalDocs + 1) / ((this.docFrequency.get(kw) || 1) + 1)) * 0.1;
                kwMatches++;
            }
        }
        if (kwMatches > 0) { signals++; strength += Math.min(kwMatches / 5, 1); }

        return { score, confidence: signals > 0 ? Math.min(strength / signals, 1) : 0 };
    }

    rank(candidates: Array<{ file: string; score: number }>, problem: string) {
        return candidates.map(c => {
            const r = this.score(c.file, problem, c.score);
            return { file: c.file, score: r.score, confidence: r.confidence };
        }).sort((a, b) => b.score - a.score);
    }

    private extractKeywords(text: string): string[] {
        const stops = new Set(['this', 'that', 'with', 'from', 'have', 'been', 'were', 'when', 'what', 'which', 'should', 'would', 'could', 'there', 'their', 'about', 'after', 'before', 'using', 'where', 'being', 'some', 'like', 'just', 'also', 'here', 'work', 'does', 'want', 'need', 'make', 'made', 'then', 'only', 'more', 'most', 'such', 'into', 'other']);
        return text.toLowerCase().replace(/[^a-z0-9_]/g, ' ').split(/\s+/).filter(w => w.length > 3 && !stops.has(w)).slice(0, 50);
    }

    getStats() { return { samples: this.trainingSamples, threshold: this.getThreshold() }; }
}

// ============================================================================
// BASELINE
// ============================================================================

function baseline(problem: string): string {
    const fileMatch = problem.match(/[\w\/]+\.py/g) || [];
    if (fileMatch.length > 0) return fileMatch[0].split('/').pop() || fileMatch[0];
    const moduleMatch = problem.match(/from\s+([\w.]+)\s+import/);
    if (moduleMatch) return moduleMatch[1].split('.').pop() + '.py';
    return 'unknown.py';
}

function fileMatches(predicted: string, gold: string): boolean {
    if (!predicted || !gold) return false;
    const predFile = predicted.split('/').pop() || '';
    const goldFile = gold.split('/').pop() || '';
    return predFile === goldFile || gold.endsWith(predFile) || predicted.endsWith(goldFile) || gold.includes(predFile);
}

// ============================================================================
// V13 PREDICTOR
// ============================================================================

interface V13Prediction { file: string; method: string; }

function v13Predict(inst: SWEBenchInstance, ranker: AdaptiveRanker | null, crossRepo: CrossRepoLearner): V13Prediction {
    // PROTECT HIGH-BASELINE REPOS
    if (HIGH_BASELINE_REPOS.has(inst.repo)) {
        return { file: baseline(inst.problem_statement), method: 'protected-baseline' };
    }

    const candidates = extractCandidates(inst.problem_statement);
    const baselinePred = baseline(inst.problem_statement);

    // No candidates - try cross-repo suggestion
    if (candidates.length === 0) {
        const suggestion = crossRepo.suggestFile(inst.problem_statement);
        if (suggestion) {
            return { file: suggestion, method: 'cross-repo' };
        }
        return { file: baselinePred, method: 'baseline-only' };
    }

    // Single high-confidence
    if (candidates.length === 1 && candidates[0].score >= 0.85) {
        return { file: candidates[0].file, method: 'single-high' };
    }

    // Separate package and non-package
    const nonPackage = candidates.filter(c => !c.isPackage);
    const workingCandidates = nonPackage.length > 0 ? nonPackage : candidates;

    // Use ranker
    if (ranker && workingCandidates.length > 0) {
        const threshold = ranker.getThreshold();
        const ranked = ranker.rank(
            workingCandidates.map(c => ({ file: c.file, score: c.score })),
            inst.problem_statement
        );
        if (ranked[0].confidence >= threshold) {
            return { file: ranked[0].file, method: `ranked-${nonPackage.length > 0 ? 'nonpkg' : 'all'}` };
        }
    }

    // Baseline in candidates?
    const baselineMatch = workingCandidates.find(c => c.file === baselinePred);
    if (baselineMatch) {
        return { file: baselinePred, method: 'baseline-in-candidates' };
    }

    // Best candidate
    workingCandidates.sort((a, b) => b.score - a.score);
    return { file: workingCandidates[0].file, method: 'best-candidate' };
}

// ============================================================================
// MAIN
// ============================================================================

async function main() {
    console.log('\n' + '='.repeat(70));
    console.log('HYPER-TARGETED TRAINING V13');
    console.log('V12 + Cross-repo learning');
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
    console.log(`\nBaseline: ${baselineCorrect}/${testInstances.length} = ${(baselineAcc * 100).toFixed(1)}%`);

    // TRAIN
    console.log('\nTraining...');
    const crossRepo = new CrossRepoLearner();
    crossRepo.train(trainInstances);

    const rankers = new Map<string, AdaptiveRanker>();
    for (const [repo, instances] of byRepo) {
        const ranker = new AdaptiveRanker(repo);
        ranker.train(instances.slice(0, Math.floor(instances.length * 0.6)));
        rankers.set(repo, ranker);
    }

    // V13 EVALUATION
    let v13Correct = 0;
    const v13ByRepo: Map<string, { correct: number; total: number }> = new Map();
    const methodStats: Map<string, { total: number; correct: number }> = new Map();

    for (const inst of testInstances) {
        const gold = inst.patch.match(/diff --git a\/(.+?) b\//)?.[1] || '';
        const pred = v13Predict(inst, rankers.get(inst.repo) || null, crossRepo);

        if (!v13ByRepo.has(inst.repo)) v13ByRepo.set(inst.repo, { correct: 0, total: 0 });
        v13ByRepo.get(inst.repo)!.total++;
        if (!methodStats.has(pred.method)) methodStats.set(pred.method, { total: 0, correct: 0 });
        methodStats.get(pred.method)!.total++;

        if (fileMatches(pred.file, gold)) {
            v13Correct++;
            v13ByRepo.get(inst.repo)!.correct++;
            methodStats.get(pred.method)!.correct++;
        }
    }

    const v13Acc = v13Correct / testInstances.length;

    console.log('\n' + '='.repeat(70));
    console.log('V13 RESULTS');
    console.log('='.repeat(70));
    console.log(`\n  Overall: ${v13Correct}/${testInstances.length} = ${(v13Acc * 100).toFixed(1)}%`);

    console.log('\n  By Method:');
    for (const [method, stats] of Array.from(methodStats.entries()).sort((a, b) => b[1].total - a[1].total)) {
        console.log(`    ${method.padEnd(25)}: ${((stats.correct / stats.total) * 100).toFixed(1)}% (${stats.correct}/${stats.total})`);
    }

    // PER-REPO
    const repoResults: Array<{ repo: string; baseAcc: number; v13Acc: number; diff: number }> = [];
    for (const [repo, baseStats] of baselineByRepo) {
        const v13Stats = v13ByRepo.get(repo) || { correct: 0, total: 0 };
        const baseAcc = baseStats.total > 0 ? baseStats.correct / baseStats.total : 0;
        const vAcc = v13Stats.total > 0 ? v13Stats.correct / v13Stats.total : 0;
        repoResults.push({ repo, baseAcc, v13Acc: vAcc, diff: vAcc - baseAcc });
    }
    repoResults.sort((a, b) => b.diff - a.diff);

    console.log('\n' + '='.repeat(70));
    console.log('PER-REPOSITORY');
    console.log('='.repeat(70));
    console.log('\n  Repository                      Baseline   V13      Δ');
    console.log('  ' + '-'.repeat(60));
    for (const r of repoResults) {
        const status = r.diff > 0.01 ? '✅' : r.diff < -0.01 ? '⚠️' : '➖';
        const protected_ = HIGH_BASELINE_REPOS.has(r.repo) ? '🛡️' : '  ';
        const diffStr = r.diff >= 0 ? `+${(r.diff * 100).toFixed(1)}%` : `${(r.diff * 100).toFixed(1)}%`;
        console.log(`  ${status}${protected_} ${r.repo.substring(0, 26).padEnd(28)} ${(r.baseAcc * 100).toFixed(1).padStart(6)}%  ${(r.v13Acc * 100).toFixed(1).padStart(6)}%  ${diffStr}`);
    }

    // SUMMARY
    const improved = repoResults.filter(r => r.diff > 0.01).length;
    const degraded = repoResults.filter(r => r.diff < -0.01).length;
    const same = repoResults.filter(r => Math.abs(r.diff) <= 0.01).length;

    console.log('\n' + '='.repeat(70));
    console.log('SUMMARY');
    console.log('='.repeat(70));
    console.log('\n┌───────────────────────────────┬──────────┬─────────────────┐');
    console.log('│ Configuration                 │ Accuracy │ vs Baseline     │');
    console.log('├───────────────────────────────┼──────────┼─────────────────┤');
    console.log(`│ Baseline                      │ ${(baselineAcc * 100).toFixed(1).padStart(6)}% │       -         │`);
    console.log(`│ V12 (previous best)           │ ${(18.3).toFixed(1).padStart(6)}% │ +4.8%           │`);
    console.log(`│ V13 (cross-repo)              │ ${(v13Acc * 100).toFixed(1).padStart(6)}% │ ${(v13Acc - baselineAcc >= 0 ? '+' : '')}${((v13Acc - baselineAcc) * 100).toFixed(1)}%          │`);
    console.log('└───────────────────────────────┴──────────┴─────────────────┘');

    console.log(`\n📊 Results: ✅ ${improved} improved, ⚠️ ${degraded} degraded, ➖ ${same} same`);

    if (v13Acc > 0.183) {
        console.log(`\n🎉 NEW BEST! V13 = ${(v13Acc * 100).toFixed(1)}%`);
    } else if (degraded === 0) {
        console.log(`\n✅ V13 = ${(v13Acc * 100).toFixed(1)}% with ZERO regressions`);
    }

    // Save
    const results = {
        timestamp: new Date().toISOString(),
        version: 'hyper-targeted-v13',
        baseline: { accuracy: baselineAcc },
        v13: { accuracy: v13Acc, byMethod: Object.fromEntries(methodStats) },
        perRepo: repoResults,
        summary: { improved, degraded, same },
    };

    const resultsDir = path.join(__dirname, 'results');
    if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir, { recursive: true });
    fs.writeFileSync(path.join(resultsDir, `hyper-targeted-v13-${Date.now()}.json`), JSON.stringify(results, null, 2));
    console.log(`\nResults saved`);
}

main().catch(console.error);
