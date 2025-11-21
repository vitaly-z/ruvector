#!/usr/bin/env node

/**
 * Agentic Synth CLI
 */

import { Command } from 'commander';
import { DataGenerator } from '../src/generators/data-generator.js';
import { Config } from '../src/config/config.js';
import { readFileSync, writeFileSync } from 'fs';

const program = new Command();

program
  .name('agentic-synth')
  .description('Synthetic data generation for agentic AI systems')
  .version('0.1.0');

program
  .command('generate')
  .description('Generate synthetic data')
  .option('-c, --count <number>', 'Number of records', '10')
  .option('-s, --schema <path>', 'Schema file path')
  .option('-o, --output <path>', 'Output file path')
  .option('--seed <number>', 'Random seed for reproducibility')
  .action(async (options) => {
    try {
      let schema = {};
      if (options.schema) {
        const content = readFileSync(options.schema, 'utf8');
        schema = JSON.parse(content);
      }

      const generator = new DataGenerator({
        schema,
        seed: options.seed ? parseInt(options.seed) : undefined
      });

      const count = parseInt(options.count);
      const data = generator.generate(count);

      if (options.output) {
        writeFileSync(options.output, JSON.stringify(data, null, 2));
        console.log(`Generated ${count} records to ${options.output}`);
      } else {
        console.log(JSON.stringify(data, null, 2));
      }
    } catch (error) {
      console.error('Error:', error.message);
      process.exit(1);
    }
  });

program
  .command('config')
  .description('Display configuration')
  .option('-f, --file <path>', 'Config file path')
  .action(async (options) => {
    try {
      const config = new Config(options.file ? { configPath: options.file } : {});
      console.log(JSON.stringify(config.getAll(), null, 2));
    } catch (error) {
      console.error('Error:', error.message);
      process.exit(1);
    }
  });

program
  .command('validate')
  .description('Validate configuration')
  .option('-f, --file <path>', 'Config file path')
  .action(async (options) => {
    try {
      const config = new Config(options.file ? { configPath: options.file } : {});
      const required = ['api.baseUrl', 'cache.maxSize'];
      config.validate(required);
      console.log('Configuration is valid');
    } catch (error) {
      console.error('Validation error:', error.message);
      process.exit(1);
    }
  });

program.parse();
