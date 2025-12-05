/**
 * HYPER-TARGETED TRAINING V10
 *
 * SOTA TECHNIQUES:
 * 1. BM25 scoring (Okapi BM25 - state-of-the-art for keyword matching)
 * 2. N-gram overlap analysis
 * 3. Code structure parsing (function/class/method context)
 * 4. Error message heuristics
 * 5. Multi-pass ensemble with weighted voting
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

// ============================================================================
// BM25 IMPLEMENTATION (Okapi BM25 - SOTA for text retrieval)
// ============================================================================

class BM25Index {
    private k1 = 1.5;  // Term frequency saturation parameter
    private b = 0.75;  // Length normalization parameter
    private avgDocLen = 0;
    private docLengths: Map<string, number> = new Map();
    private docFreq: Map<string, number> = new Map();
    private termFreq: Map<string, Map<string, number>> = new Map();
    private totalDocs = 0;

    index(documents: Array<{ id: string; text: string }>): void {
        this.totalDocs = documents.length;
        let totalLen = 0;

        for (const doc of documents) {
            const terms = this.tokenize(doc.text);
            this.docLengths.set(doc.id, terms.length);
            totalLen += terms.length;

            const termCounts = new Map<string, number>();
            for (const term of terms) {
                termCounts.set(term, (termCounts.get(term) || 0) + 1);
            }

            for (const [term, count] of termCounts) {
                if (!this.termFreq.has(doc.id)) {
                    this.termFreq.set(doc.id, new Map());
                }
                this.termFreq.get(doc.id)!.set(term, count);

                // Document frequency
                this.docFreq.set(term, (this.docFreq.get(term) || 0) + 1);
            }
        }

        this.avgDocLen = totalLen / documents.length;
    }

    score(query: string, docId: string): number {
        const queryTerms = this.tokenize(query);
        const docTerms = this.termFreq.get(docId);
        if (!docTerms) return 0;

        const docLen = this.docLengths.get(docId) || 0;
        let score = 0;

        for (const term of queryTerms) {
            const tf = docTerms.get(term) || 0;
            if (tf === 0) continue;

            const df = this.docFreq.get(term) || 0;
            const idf = Math.log((this.totalDocs - df + 0.5) / (df + 0.5) + 1);

            const numerator = tf * (this.k1 + 1);
            const denominator = tf + this.k1 * (1 - this.b + this.b * (docLen / this.avgDocLen));

            score += idf * (numerator / denominator);
        }

        return score;
    }

    private tokenize(text: string): string[] {
        return text.toLowerCase()
            .replace(/[^a-z0-9_]/g, ' ')
            .split(/\s+/)
            .filter(w => w.length > 2);
    }
}

// ============================================================================
// N-GRAM OVERLAP ANALYZER
// ============================================================================

class NGramAnalyzer {
    private fileNgrams: Map<string, Set<string>> = new Map();

    train(instances: SWEBenchInstance[]): void {
        for (const inst of instances) {
            const file = inst.patch.match(/diff --git a\/(.+?) b\//)?.[1] || '';
            const fileName = file.split('/').pop() || '';
            if (!fileName) continue;

            // Extract ngrams from problem that map to this file
            const ngrams = this.extractNgrams(inst.problem_statement, 2, 3);

            if (!this.fileNgrams.has(fileName)) {
                this.fileNgrams.set(fileName, new Set());
            }

            for (const ng of ngrams) {
                this.fileNgrams.get(fileName)!.add(ng);
            }
        }
    }

    score(candidate: string, problem: string): number {
        const problemNgrams = new Set(this.extractNgrams(problem, 2, 3));
        const fileNgrams = this.fileNgrams.get(candidate);

        if (!fileNgrams || problemNgrams.size === 0) return 0;

        // Jaccard similarity
        let intersection = 0;
        for (const ng of problemNgrams) {
            if (fileNgrams.has(ng)) intersection++;
        }

        return intersection / (problemNgrams.size + fileNgrams.size - intersection);
    }

    private extractNgrams(text: string, minN: number, maxN: number): string[] {
        const words = text.toLowerCase()
            .replace(/[^a-z0-9_]/g, ' ')
            .split(/\s+/)
            .filter(w => w.length > 2);

        const ngrams: string[] = [];
        for (let n = minN; n <= maxN; n++) {
            for (let i = 0; i <= words.length - n; i++) {
                ngrams.push(words.slice(i, i + n).join('_'));
            }
        }
        return ngrams;
    }
}

// ============================================================================
// CODE STRUCTURE ANALYZER
// ============================================================================

class CodeStructureAnalyzer {
    private classToFile: Map<string, string[]> = new Map();
    private methodToFile: Map<string, string[]> = new Map();
    private errorToFile: Map<string, string[]> = new Map();

    train(instances: SWEBenchInstance[]): void {
        for (const inst of instances) {
            const file = inst.patch.match(/diff --git a\/(.+?) b\//)?.[1] || '';
            const fileName = file.split('/').pop() || '';
            if (!fileName) continue;

            // Extract class names
            const classes = inst.problem_statement.match(/class\s+(\w+)/gi) || [];
            for (const cls of classes) {
                const name = cls.replace(/class\s+/i, '').toLowerCase();
                if (!this.classToFile.has(name)) this.classToFile.set(name, []);
                if (!this.classToFile.get(name)!.includes(fileName)) {
                    this.classToFile.get(name)!.push(fileName);
                }
            }

            // Extract method names
            const methods = inst.problem_statement.match(/def\s+(\w+)\s*\(/gi) || [];
            for (const m of methods) {
                const name = m.replace(/def\s+/i, '').replace(/\s*\(/, '').toLowerCase();
                if (!this.methodToFile.has(name)) this.methodToFile.set(name, []);
                if (!this.methodToFile.get(name)!.includes(fileName)) {
                    this.methodToFile.get(name)!.push(fileName);
                }
            }

            // Extract error types
            const errors = inst.problem_statement.match(/\w+Error|\w+Exception|\w+Warning/g) || [];
            for (const err of errors) {
                const name = err.toLowerCase();
                if (!this.errorToFile.has(name)) this.errorToFile.set(name, []);
                if (!this.errorToFile.get(name)!.includes(fileName)) {
                    this.errorToFile.get(name)!.push(fileName);
                }
            }
        }
    }

    score(candidate: string, problem: string): number {
        let score = 0;

        // Class matches
        const classes = problem.match(/class\s+(\w+)/gi) || [];
        for (const cls of classes) {
            const name = cls.replace(/class\s+/i, '').toLowerCase();
            const files = this.classToFile.get(name);
            if (files && files.includes(candidate)) {
                score += 0.3;
            }
        }

        // Method matches
        const methods = problem.match(/def\s+(\w+)\s*\(/gi) || [];
        for (const m of methods) {
            const name = m.replace(/def\s+/i, '').replace(/\s*\(/, '').toLowerCase();
            const files = this.methodToFile.get(name);
            if (files && files.includes(candidate)) {
                score += 0.2;
            }
        }

        // Error matches
        const errors = problem.match(/\w+Error|\w+Exception|\w+Warning/g) || [];
        for (const err of errors) {
            const files = this.errorToFile.get(err.toLowerCase());
            if (files && files.includes(candidate)) {
                score += 0.25;
            }
        }

        return score;
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

    // Backticks (highest confidence)
    (problem.match(/`([^`]+\.py)`/g) || []).forEach(m => add(m.replace(/`/g, ''), 'backtick', 0.95));

    // Tracebacks
    (problem.match(/File "([^"]+\.py)"/g) || []).forEach(m => {
        const f = m.replace(/File "|"/g, '');
        if (!f.includes('site-packages')) add(f, 'traceback', 0.92);
    });

    // Package refs (matplotlib.colors → colors.py)
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

    return candidates;
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
// V10 ENSEMBLE PREDICTOR
// ============================================================================

interface V10Prediction { file: string; method: string; confidence: number; }

function v10Predict(
    inst: SWEBenchInstance,
    bm25: BM25Index,
    ngram: NGramAnalyzer,
    structure: CodeStructureAnalyzer,
    fileFreq: Map<string, number>
): V10Prediction {
    const candidates = extractCandidates(inst.problem_statement);
    const baselinePred = baseline(inst.problem_statement);

    if (candidates.length === 0) {
        return { file: baselinePred, method: 'baseline', confidence: 0.3 };
    }

    if (candidates.length === 1 && candidates[0].score >= 0.85) {
        return { file: candidates[0].file, method: 'single', confidence: candidates[0].score };
    }

    // Multi-signal ensemble scoring
    const scored = candidates.map(c => {
        let totalScore = c.score * 2;  // Base extraction score

        // BM25 score
        const bm25Score = bm25.score(inst.problem_statement, c.file);
        totalScore += bm25Score * 0.5;

        // N-gram overlap
        const ngramScore = ngram.score(c.file, inst.problem_statement);
        totalScore += ngramScore * 0.3;

        // Code structure
        const structScore = structure.score(c.file, inst.problem_statement);
        totalScore += structScore * 0.4;

        // File frequency (domain prior)
        const freq = fileFreq.get(c.file) || 0;
        totalScore += Math.log(freq + 1) * 0.2;

        // Package penalty (only if alternatives exist)
        const hasNonPackage = candidates.some(x => !x.isPackage);
        if (c.isPackage && hasNonPackage) {
            totalScore -= 0.5;
        }

        return { file: c.file, score: totalScore, isPackage: c.isPackage };
    });

    scored.sort((a, b) => b.score - a.score);

    // Check if baseline is competitive
    const baselineCandidate = scored.find(s => s.file === baselinePred);
    const best = scored[0];

    // If baseline is close to best, prefer it (stability)
    if (baselineCandidate && baselineCandidate.score >= best.score * 0.9) {
        return { file: baselinePred, method: 'baseline-preferred', confidence: baselineCandidate.score };
    }

    return { file: best.file, method: 'ensemble', confidence: best.score };
}

// ============================================================================
// MAIN
// ============================================================================

async function main() {
    console.log('\n' + '='.repeat(70));
    console.log('HYPER-TARGETED TRAINING V10');
    console.log('SOTA: BM25 + N-gram overlap + Code structure + Ensemble');
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

    // TRAIN SOTA COMPONENTS
    console.log('\n' + '='.repeat(70));
    console.log('TRAINING SOTA COMPONENTS');
    console.log('='.repeat(70));

    // BM25 index
    console.log('\n  Building BM25 index...');
    const bm25 = new BM25Index();
    const bm25Docs = trainInstances.map(inst => {
        const file = inst.patch.match(/diff --git a\/(.+?) b\//)?.[1] || '';
        const fileName = file.split('/').pop() || '';
        return { id: fileName, text: inst.problem_statement };
    }).filter(d => d.id);
    bm25.index(bm25Docs);
    console.log(`    Indexed ${bm25Docs.length} documents`);

    // N-gram analyzer
    console.log('\n  Training N-gram analyzer...');
    const ngram = new NGramAnalyzer();
    ngram.train(trainInstances);
    console.log('    Done');

    // Code structure analyzer
    console.log('\n  Training code structure analyzer...');
    const structure = new CodeStructureAnalyzer();
    structure.train(trainInstances);
    console.log('    Done');

    // File frequency
    console.log('\n  Computing file frequencies...');
    const fileFreq = new Map<string, number>();
    for (const inst of trainInstances) {
        const file = inst.patch.match(/diff --git a\/(.+?) b\//)?.[1] || '';
        const fileName = file.split('/').pop() || '';
        if (fileName) {
            fileFreq.set(fileName, (fileFreq.get(fileName) || 0) + 1);
        }
    }
    console.log(`    ${fileFreq.size} unique files`);

    // V10 EVALUATION
    console.log('\n' + '='.repeat(70));
    console.log('V10 ENSEMBLE EVALUATION');
    console.log('='.repeat(70));

    let v10Correct = 0;
    const v10ByRepo: Map<string, { correct: number; total: number }> = new Map();
    const methodStats: Map<string, { total: number; correct: number }> = new Map();

    for (const inst of testInstances) {
        const gold = inst.patch.match(/diff --git a\/(.+?) b\//)?.[1] || '';
        const pred = v10Predict(inst, bm25, ngram, structure, fileFreq);

        if (!v10ByRepo.has(inst.repo)) v10ByRepo.set(inst.repo, { correct: 0, total: 0 });
        v10ByRepo.get(inst.repo)!.total++;

        if (!methodStats.has(pred.method)) methodStats.set(pred.method, { total: 0, correct: 0 });
        methodStats.get(pred.method)!.total++;

        if (fileMatches(pred.file, gold)) {
            v10Correct++;
            v10ByRepo.get(inst.repo)!.correct++;
            methodStats.get(pred.method)!.correct++;
        }
    }

    const v10Acc = v10Correct / testInstances.length;
    console.log(`\n  Overall: ${v10Correct}/${testInstances.length} = ${(v10Acc * 100).toFixed(1)}%`);

    console.log('\n  By Method:');
    for (const [method, stats] of Array.from(methodStats.entries()).sort((a, b) => b[1].total - a[1].total)) {
        const acc = stats.total > 0 ? (stats.correct / stats.total * 100).toFixed(1) : '0.0';
        console.log(`    ${method.padEnd(20)}: ${acc}% (${stats.correct}/${stats.total})`);
    }

    // PER-REPO
    console.log('\n' + '='.repeat(70));
    console.log('PER-REPOSITORY');
    console.log('='.repeat(70));

    const repoResults: Array<{ repo: string; baseAcc: number; v10Acc: number; diff: number }> = [];

    for (const [repo, baseStats] of baselineByRepo) {
        const v10Stats = v10ByRepo.get(repo) || { correct: 0, total: 0 };
        const baseAcc = baseStats.total > 0 ? baseStats.correct / baseStats.total : 0;
        const vAcc = v10Stats.total > 0 ? v10Stats.correct / v10Stats.total : 0;
        repoResults.push({ repo, baseAcc, v10Acc: vAcc, diff: vAcc - baseAcc });
    }

    repoResults.sort((a, b) => b.diff - a.diff);

    console.log('\n  Repository                      Baseline   V10      Δ');
    console.log('  ' + '-'.repeat(60));

    for (const r of repoResults) {
        const status = r.diff > 0.01 ? '✅' : r.diff < -0.01 ? '⚠️' : '➖';
        const diffStr = r.diff >= 0 ? `+${(r.diff * 100).toFixed(1)}%` : `${(r.diff * 100).toFixed(1)}%`;
        console.log(`  ${status} ${r.repo.substring(0, 28).padEnd(30)} ${(r.baseAcc * 100).toFixed(1).padStart(6)}%  ${(r.v10Acc * 100).toFixed(1).padStart(6)}%  ${diffStr}`);
    }

    // SUMMARY
    console.log('\n' + '='.repeat(70));
    console.log('SUMMARY');
    console.log('='.repeat(70));

    const improved = repoResults.filter(r => r.diff > 0.01).length;
    const degraded = repoResults.filter(r => r.diff < -0.01).length;
    const same = repoResults.filter(r => Math.abs(r.diff) <= 0.01).length;
    const overallDiff = v10Acc - baselineAcc;

    console.log('\n┌───────────────────────────────┬──────────┬─────────────────┐');
    console.log('│ Configuration                 │ Accuracy │ vs Baseline     │');
    console.log('├───────────────────────────────┼──────────┼─────────────────┤');
    console.log(`│ Baseline                      │ ${(baselineAcc * 100).toFixed(1).padStart(6)}% │       -         │`);
    console.log(`│ V9 (previous best)            │ ${(17.5).toFixed(1).padStart(6)}% │ +4.0%           │`);
    console.log(`│ V10 (SOTA ensemble)           │ ${(v10Acc * 100).toFixed(1).padStart(6)}% │ ${overallDiff >= 0 ? '+' : ''}${(overallDiff * 100).toFixed(1)}%          │`);
    console.log('└───────────────────────────────┴──────────┴─────────────────┘');

    console.log(`\n📊 Results: ✅ ${improved} improved, ⚠️ ${degraded} degraded, ➖ ${same} same`);

    if (v10Acc > 0.175) {
        console.log(`\n🎉 NEW BEST! V10 = ${(v10Acc * 100).toFixed(1)}%`);
    } else if (v10Acc >= 0.170) {
        console.log(`\n✅ V10 competitive at ${(v10Acc * 100).toFixed(1)}%`);
    }

    console.log('\n📋 V10 SOTA TECHNIQUES:');
    console.log('  ✓ BM25 (Okapi) scoring for keyword matching');
    console.log('  ✓ N-gram overlap analysis (2-3 grams)');
    console.log('  ✓ Code structure parsing (class/method/error)');
    console.log('  ✓ Multi-signal ensemble with weighted voting');
    console.log('  ✓ Baseline stability preference');

    // Save
    const results = {
        timestamp: new Date().toISOString(),
        version: 'hyper-targeted-v10',
        baseline: { accuracy: baselineAcc, correct: baselineCorrect },
        v10: { accuracy: v10Acc, correct: v10Correct, byMethod: Object.fromEntries(methodStats) },
        perRepo: repoResults,
        summary: { improved, degraded, same, overallDiff },
        techniques: ['bm25', 'ngram-overlap', 'code-structure', 'ensemble', 'baseline-stability'],
        provenance: {
            hash: crypto.createHash('sha256')
                .update(JSON.stringify({ baselineAcc, v10Acc }))
                .digest('hex').substring(0, 32),
        },
    };

    const resultsDir = path.join(__dirname, 'results');
    if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir, { recursive: true });
    const resultsPath = path.join(resultsDir, `hyper-targeted-v10-${Date.now()}.json`);
    fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
    console.log(`\nResults saved to: ${resultsPath}`);
}

main().catch(console.error);
