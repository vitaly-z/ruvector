/**
 * Structured data generator
 */

import { BaseGenerator } from './base.js';
import { GeneratorOptions, ValidationError } from '../types.js';

export class StructuredGenerator extends BaseGenerator<GeneratorOptions> {
  protected generatePrompt(options: GeneratorOptions): string {
    const { count = 10, schema, constraints, format = 'json' } = options;

    if (!schema) {
      throw new ValidationError('Schema is required for structured data generation', {
        options
      });
    }

    let prompt = `Generate ${count} realistic data records matching the following schema:

Schema:
${JSON.stringify(schema, null, 2)}

`;

    if (constraints) {
      prompt += `\nConstraints:\n${JSON.stringify(constraints, null, 2)}\n`;
    }

    prompt += `
Requirements:
1. Generate realistic, diverse data that fits the schema
2. Ensure all required fields are present
3. Follow data type constraints strictly
4. Make data internally consistent and realistic
5. Include varied but plausible values

Return ONLY a JSON array of ${count} objects, no additional text.`;

    return prompt;
  }

  protected parseResult(response: string, options: GeneratorOptions): any[] {
    try {
      // Extract JSON from response
      const jsonMatch = response.match(/\[[\s\S]*\]/);
      if (!jsonMatch) {
        throw new Error('No JSON array found in response');
      }

      const data = JSON.parse(jsonMatch[0]);

      if (!Array.isArray(data)) {
        throw new Error('Response is not an array');
      }

      // Validate against schema if provided
      if (options.schema) {
        return data.map((item, index) => {
          this.validateAgainstSchema(item, options.schema!, index);
          return item;
        });
      }

      return data;
    } catch (error: any) {
      throw new ValidationError(`Failed to parse structured data: ${error.message}`, {
        response: response.substring(0, 200),
        error
      });
    }
  }

  private validateAgainstSchema(
    item: any,
    schema: Record<string, any>,
    index: number
  ): void {
    for (const [key, schemaValue] of Object.entries(schema)) {
      // Check required fields
      if (schemaValue.required && !(key in item)) {
        throw new ValidationError(`Missing required field '${key}' at index ${index}`, {
          item,
          schema
        });
      }

      // Check types
      if (key in item && schemaValue.type) {
        const actualType = typeof item[key];
        const expectedType = schemaValue.type;

        if (expectedType === 'array' && !Array.isArray(item[key])) {
          throw new ValidationError(
            `Field '${key}' should be array at index ${index}`,
            { item, schema }
          );
        } else if (expectedType !== 'array' && actualType !== expectedType) {
          throw new ValidationError(
            `Field '${key}' has wrong type at index ${index}. Expected ${expectedType}, got ${actualType}`,
            { item, schema }
          );
        }
      }

      // Check nested objects
      if (schemaValue.properties && typeof item[key] === 'object') {
        this.validateAgainstSchema(item[key], schemaValue.properties, index);
      }
    }
  }

  /**
   * Generate structured data with specific domain
   */
  async generateDomain(domain: string, options: GeneratorOptions): Promise<any[]> {
    const domainSchemas: Record<string, any> = {
      users: {
        id: { type: 'string', required: true },
        name: { type: 'string', required: true },
        email: { type: 'string', required: true },
        age: { type: 'number', required: true },
        role: { type: 'string', required: false },
        createdAt: { type: 'string', required: true }
      },
      products: {
        id: { type: 'string', required: true },
        name: { type: 'string', required: true },
        price: { type: 'number', required: true },
        category: { type: 'string', required: true },
        inStock: { type: 'boolean', required: true },
        description: { type: 'string', required: false }
      },
      transactions: {
        id: { type: 'string', required: true },
        userId: { type: 'string', required: true },
        amount: { type: 'number', required: true },
        currency: { type: 'string', required: true },
        status: { type: 'string', required: true },
        timestamp: { type: 'string', required: true }
      }
    };

    const schema = domainSchemas[domain.toLowerCase()];
    if (!schema) {
      throw new ValidationError(`Unknown domain: ${domain}`, {
        availableDomains: Object.keys(domainSchemas)
      });
    }

    return this.generate({
      ...options,
      schema
    }).then(result => result.data);
  }

  /**
   * Generate data from JSON schema
   */
  async generateFromJSONSchema(jsonSchema: any, options: GeneratorOptions): Promise<any[]> {
    // Convert JSON Schema to internal schema format
    const schema = this.convertJSONSchema(jsonSchema);

    return this.generate({
      ...options,
      schema
    }).then(result => result.data);
  }

  private convertJSONSchema(jsonSchema: any): Record<string, any> {
    const schema: Record<string, any> = {};

    if (jsonSchema.properties) {
      for (const [key, value] of Object.entries(jsonSchema.properties)) {
        const prop: any = value;
        schema[key] = {
          type: prop.type,
          required: jsonSchema.required?.includes(key) || false
        };

        if (prop.properties) {
          schema[key].properties = this.convertJSONSchema(prop);
        }
      }
    }

    return schema;
  }
}
