#!/usr/bin/env node

/**
 * Agentic Synth CLI
 * Production-ready CLI for synthetic data generation
 */

import { Command } from 'commander';
import { AgenticSynth } from '../dist/index.js';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const program = new Command();

// Helper to load JSON config file
function loadConfig(configPath) {
  try {
    if (!existsSync(configPath)) {
      throw new Error(`Config file not found: ${configPath}`);
    }
    const content = readFileSync(configPath, 'utf8');
    return JSON.parse(content);
  } catch (error) {
    if (error.message.includes('not found')) {
      throw error;
    }
    throw new Error(`Invalid JSON in config file: ${error.message}`);
  }
}

// Helper to load schema file
function loadSchema(schemaPath) {
  try {
    if (!existsSync(schemaPath)) {
      throw new Error(`Schema file not found: ${schemaPath}`);
    }
    const content = readFileSync(schemaPath, 'utf8');
    return JSON.parse(content);
  } catch (error) {
    if (error.message.includes('not found')) {
      throw error;
    }
    throw new Error(`Invalid JSON in schema file: ${error.message}`);
  }
}

program
  .name('agentic-synth')
  .description('AI-powered synthetic data generation for agentic systems')
  .version('0.1.0');

program
  .command('generate')
  .description('Generate synthetic structured data')
  .option('-c, --count <number>', 'Number of records to generate', '10')
  .option('-s, --schema <path>', 'Path to JSON schema file')
  .option('-o, --output <path>', 'Output file path (JSON format)')
  .option('--seed <value>', 'Random seed for reproducibility')
  .option('-p, --provider <provider>', 'Model provider (gemini, openrouter)', 'gemini')
  .option('-m, --model <model>', 'Model name to use')
  .option('--format <format>', 'Output format (json, csv, array)', 'json')
  .option('--config <path>', 'Path to config file with provider settings')
  .action(async (options) => {
    try {
      // Load configuration
      let config = {
        provider: options.provider,
        model: options.model
      };

      // Load config file if provided
      if (options.config) {
        const fileConfig = loadConfig(resolve(options.config));
        config = { ...config, ...fileConfig };
      }

      // Ensure API key is set
      if (!config.apiKey && !process.env.GEMINI_API_KEY && !process.env.OPENROUTER_API_KEY) {
        console.error('Error: API key not found. Set GEMINI_API_KEY or OPENROUTER_API_KEY environment variable, or provide --config file.');
        process.exit(1);
      }

      // Initialize AgenticSynth
      const synth = new AgenticSynth(config);

      // Load schema if provided
      let schema = undefined;
      if (options.schema) {
        schema = loadSchema(resolve(options.schema));
      }

      // Parse count
      const count = parseInt(options.count, 10);
      if (isNaN(count) || count < 1) {
        throw new Error('Count must be a positive integer');
      }

      // Parse seed
      let seed = options.seed;
      if (seed) {
        const seedNum = parseInt(seed, 10);
        seed = isNaN(seedNum) ? seed : seedNum;
      }

      console.log(`Generating ${count} records...`);
      const startTime = Date.now();

      // Generate data using AgenticSynth
      const result = await synth.generateStructured({
        count,
        schema,
        seed,
        format: options.format
      });

      const duration = Date.now() - startTime;

      // Output results
      if (options.output) {
        const outputPath = resolve(options.output);
        writeFileSync(outputPath, JSON.stringify(result.data, null, 2));
        console.log(`✓ Generated ${result.metadata.count} records to ${outputPath}`);
      } else {
        console.log(JSON.stringify(result.data, null, 2));
      }

      // Display metadata
      console.error(`\nMetadata:`);
      console.error(`  Provider: ${result.metadata.provider}`);
      console.error(`  Model: ${result.metadata.model}`);
      console.error(`  Cached: ${result.metadata.cached}`);
      console.error(`  Duration: ${duration}ms`);
      console.error(`  Generated: ${result.metadata.generatedAt}`);

    } catch (error) {
      console.error('Error:', error.message);
      if (error.stack && process.env.DEBUG) {
        console.error('\nStack trace:');
        console.error(error.stack);
      }
      process.exit(1);
    }
  });

program
  .command('config')
  .description('Display or test configuration')
  .option('-f, --file <path>', 'Config file path to load')
  .option('-t, --test', 'Test configuration by initializing AgenticSynth')
  .action(async (options) => {
    try {
      let config = {};

      // Load config file if provided
      if (options.file) {
        config = loadConfig(resolve(options.file));
      }

      // Create instance to validate config
      const synth = new AgenticSynth(config);
      const currentConfig = synth.getConfig();

      console.log('Current Configuration:');
      console.log(JSON.stringify(currentConfig, null, 2));

      if (options.test) {
        console.log('\n✓ Configuration is valid and AgenticSynth initialized successfully');
      }

      // Check for API keys
      console.log('\nEnvironment Variables:');
      console.log(`  GEMINI_API_KEY: ${process.env.GEMINI_API_KEY ? '✓ Set' : '✗ Not set'}`);
      console.log(`  OPENROUTER_API_KEY: ${process.env.OPENROUTER_API_KEY ? '✓ Set' : '✗ Not set'}`);

    } catch (error) {
      console.error('Configuration error:', error.message);
      if (error.stack && process.env.DEBUG) {
        console.error('\nStack trace:');
        console.error(error.stack);
      }
      process.exit(1);
    }
  });

program
  .command('validate')
  .description('Validate configuration and dependencies')
  .option('-f, --file <path>', 'Config file path to validate')
  .action(async (options) => {
    try {
      let config = {};

      // Load config file if provided
      if (options.file) {
        config = loadConfig(resolve(options.file));
        console.log('✓ Config file is valid JSON');
      }

      // Validate by creating instance
      const synth = new AgenticSynth(config);
      console.log('✓ Configuration schema is valid');

      // Check provider settings
      const currentConfig = synth.getConfig();
      console.log(`✓ Provider: ${currentConfig.provider}`);
      console.log(`✓ Model: ${currentConfig.model || 'default'}`);
      console.log(`✓ Cache strategy: ${currentConfig.cacheStrategy}`);
      console.log(`✓ Max retries: ${currentConfig.maxRetries}`);
      console.log(`✓ Timeout: ${currentConfig.timeout}ms`);

      // Validate API key
      if (!currentConfig.apiKey && !process.env.GEMINI_API_KEY && !process.env.OPENROUTER_API_KEY) {
        console.warn('⚠ Warning: No API key found. Set GEMINI_API_KEY or OPENROUTER_API_KEY environment variable.');
      } else {
        console.log('✓ API key is configured');
      }

      console.log('\n✓ All validations passed');

    } catch (error) {
      console.error('Validation error:', error.message);
      if (error.stack && process.env.DEBUG) {
        console.error('\nStack trace:');
        console.error(error.stack);
      }
      process.exit(1);
    }
  });

// Error handler for unknown commands
program.on('command:*', function () {
  console.error('Invalid command: %s\nSee --help for a list of available commands.', program.args.join(' '));
  process.exit(1);
});

// Show help if no command provided
if (process.argv.length === 2) {
  program.help();
}

program.parse();
