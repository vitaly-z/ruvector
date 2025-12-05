/**
 * Analyze failure patterns for 0% accuracy repos
 */

import * as fs from 'fs';
import * as path from 'path';

interface SWEBenchInstance {
    instance_id: string;
    repo: string;
    patch: string;
    problem_statement: string;
    hints_text: string;
}

const swePath = path.join(__dirname, 'swe-bench-real', 'all_instances.json');
const sweInstances: SWEBenchInstance[] = JSON.parse(fs.readFileSync(swePath, 'utf8'));

// Focus on 0% repos
const zeroRepos = [
    'matplotlib/matplotlib',
    'pallets/flask',
    'psf/requests',
    'pydata/xarray',
    'pylint-dev/pylint',
    'sphinx-doc/sphinx'
];

console.log('='.repeat(70));
console.log('FAILURE ANALYSIS FOR 0% REPOS');
console.log('='.repeat(70));

for (const repo of zeroRepos) {
    const instances = sweInstances.filter(i => i.repo === repo);
    const testInstances = instances.slice(Math.floor(instances.length * 0.6));

    console.log(`\n${'='.repeat(70)}`);
    console.log(`REPO: ${repo} (${testInstances.length} test instances)`);
    console.log('='.repeat(70));

    for (const inst of testInstances.slice(0, 3)) {
        const goldFile = inst.patch.match(/diff --git a\/(.+?) b\//)?.[1] || '';
        const goldFileName = goldFile.split('/').pop() || '';

        // Extract what baseline would predict
        const fileMatches = inst.problem_statement.match(/[\w\/]+\.py/g) || [];
        const baselinePred = fileMatches.length > 0 ? fileMatches[0].split('/').pop() : 'unknown.py';

        // Check for other patterns
        const backticks = inst.problem_statement.match(/`([^`]+\.py)`/g) || [];
        const quoted = inst.problem_statement.match(/"([^"]+\.py)"/g) || [];
        const tracebacks = inst.problem_statement.match(/File "([^"]+\.py)"/g) || [];
        const imports = inst.problem_statement.match(/from\s+([\w.]+)\s+import/g) || [];

        console.log(`\n  Instance: ${inst.instance_id}`);
        console.log(`  Gold file: ${goldFile}`);
        console.log(`  Gold filename: ${goldFileName}`);
        console.log(`  Baseline would predict: ${baselinePred}`);
        console.log(`  Match: ${baselinePred === goldFileName ? '✅' : '❌'}`);
        console.log(`  Patterns found:`);
        console.log(`    - .py matches: ${fileMatches.slice(0, 3).join(', ') || 'none'}`);
        console.log(`    - backticks: ${backticks.slice(0, 2).join(', ') || 'none'}`);
        console.log(`    - quoted: ${quoted.slice(0, 2).join(', ') || 'none'}`);
        console.log(`    - tracebacks: ${tracebacks.slice(0, 2).join(', ') || 'none'}`);
        console.log(`    - imports: ${imports.slice(0, 2).join(', ') || 'none'}`);

        // Check if gold file appears anywhere in problem
        if (inst.problem_statement.includes(goldFileName)) {
            console.log(`    ➡️ Gold filename IS in problem statement!`);
        } else if (inst.problem_statement.toLowerCase().includes(goldFileName.replace('.py', '').toLowerCase())) {
            console.log(`    ➡️ Gold filename (without .py) IS in problem statement!`);
        } else {
            console.log(`    ⚠️ Gold filename NOT directly mentioned`);
        }

        // Show first 300 chars of problem
        console.log(`  Problem (first 300 chars):`);
        console.log(`    "${inst.problem_statement.substring(0, 300).replace(/\n/g, ' ')}..."`);
    }
}

// Also check what the gold files look like across all 0% repos
console.log(`\n\n${'='.repeat(70)}`);
console.log('GOLD FILE PATTERNS FOR 0% REPOS');
console.log('='.repeat(70));

for (const repo of zeroRepos) {
    const instances = sweInstances.filter(i => i.repo === repo);
    const goldFiles = instances.map(i => {
        const match = i.patch.match(/diff --git a\/(.+?) b\//);
        return match ? match[1] : '';
    }).filter(f => f);

    console.log(`\n${repo}:`);
    const uniqueFiles = [...new Set(goldFiles)];
    console.log(`  Unique gold files (${uniqueFiles.length}):`);
    for (const f of uniqueFiles.slice(0, 10)) {
        console.log(`    - ${f}`);
    }
}
