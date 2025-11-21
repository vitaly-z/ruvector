/**
 * Event data generator
 */

import { BaseGenerator } from './base.js';
import { EventOptions, ValidationError } from '../types.js';

export class EventGenerator extends BaseGenerator<EventOptions> {
  protected generatePrompt(options: EventOptions): string {
    const {
      count = 100,
      eventTypes = ['click', 'view', 'purchase'],
      distribution = 'uniform',
      timeRange,
      userCount = 50,
      schema,
      constraints
    } = options;

    const start = timeRange?.start || new Date(Date.now() - 24 * 60 * 60 * 1000);
    const end = timeRange?.end || new Date();

    let prompt = `Generate ${count} event log entries with the following specifications:

Event Configuration:
- Event types: ${eventTypes.join(', ')}
- Distribution: ${distribution}
- Time range: ${start} to ${end}
- Unique users: ${userCount}

`;

    if (schema) {
      prompt += `\nSchema:\n${JSON.stringify(schema, null, 2)}\n`;
    }

    if (constraints) {
      prompt += `\nConstraints:\n${JSON.stringify(constraints, null, 2)}\n`;
    }

    prompt += `
Generate realistic event data where each event has:
- eventId: unique identifier
- eventType: one of the specified types
- timestamp: ISO 8601 formatted date within the time range
- userId: user identifier (1 to ${userCount})
- metadata: relevant event-specific data

Distribution patterns:
- uniform: events evenly distributed over time
- poisson: random but clustered events (realistic web traffic)
- normal: events concentrated around mean time

Ensure:
1. Events are chronologically ordered
2. Event types follow realistic usage patterns
3. User behavior is consistent and realistic
4. Metadata is relevant to event type
5. Timestamps fall within the specified range

Return ONLY a JSON array of events, no additional text.`;

    return prompt;
  }

  protected parseResult(response: string, options: EventOptions): any[] {
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

      // Validate event structure
      return data.map((event, index) => {
        if (!event.eventId) {
          event.eventId = `evt_${Date.now()}_${index}`;
        }

        if (!event.eventType) {
          throw new ValidationError(`Missing eventType at index ${index}`, { event });
        }

        if (!event.timestamp) {
          throw new ValidationError(`Missing timestamp at index ${index}`, { event });
        }

        if (!event.userId) {
          throw new ValidationError(`Missing userId at index ${index}`, { event });
        }

        return {
          eventId: event.eventId,
          eventType: event.eventType,
          timestamp: new Date(event.timestamp).toISOString(),
          userId: event.userId,
          metadata: event.metadata || {}
        };
      });
    } catch (error: any) {
      throw new ValidationError(`Failed to parse event data: ${error.message}`, {
        response: response.substring(0, 200),
        error
      });
    }
  }

  /**
   * Generate synthetic events with local computation
   */
  async generateLocal(options: EventOptions): Promise<any[]> {
    const {
      count = 100,
      eventTypes = ['click', 'view', 'purchase'],
      distribution = 'uniform',
      timeRange,
      userCount = 50
    } = options;

    const start = timeRange?.start
      ? new Date(timeRange.start).getTime()
      : Date.now() - 24 * 60 * 60 * 1000;
    const end = timeRange?.end ? new Date(timeRange.end).getTime() : Date.now();

    const events: any[] = [];
    const timestamps = this.generateTimestamps(count, start, end, distribution);

    for (let i = 0; i < count; i++) {
      const eventType = eventTypes[Math.floor(Math.random() * eventTypes.length)];
      const userId = `user_${Math.floor(Math.random() * userCount) + 1}`;

      events.push({
        eventId: `evt_${Date.now()}_${i}`,
        eventType,
        timestamp: new Date(timestamps[i]).toISOString(),
        userId,
        metadata: this.generateMetadata(eventType)
      });
    }

    // Sort by timestamp
    events.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

    return events;
  }

  private generateTimestamps(
    count: number,
    start: number,
    end: number,
    distribution: 'uniform' | 'poisson' | 'normal'
  ): number[] {
    const timestamps: number[] = [];
    const range = end - start;

    switch (distribution) {
      case 'uniform':
        for (let i = 0; i < count; i++) {
          timestamps.push(start + Math.random() * range);
        }
        break;

      case 'poisson':
        // Exponential inter-arrival times
        let time = start;
        const lambda = count / range; // events per ms
        for (let i = 0; i < count && time < end; i++) {
          const interval = -Math.log(1 - Math.random()) / lambda;
          time += interval;
          timestamps.push(Math.min(time, end));
        }
        break;

      case 'normal':
        // Normal distribution around midpoint
        const mean = start + range / 2;
        const stdDev = range / 6; // 99.7% within range
        for (let i = 0; i < count; i++) {
          const u1 = Math.random();
          const u2 = Math.random();
          const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
          const timestamp = mean + z * stdDev;
          timestamps.push(Math.max(start, Math.min(end, timestamp)));
        }
        break;
    }

    return timestamps.sort((a, b) => a - b);
  }

  private generateMetadata(eventType: string): Record<string, any> {
    const metadata: Record<string, any> = {};

    switch (eventType.toLowerCase()) {
      case 'click':
        metadata.element = ['button', 'link', 'image'][Math.floor(Math.random() * 3)];
        metadata.position = { x: Math.floor(Math.random() * 1920), y: Math.floor(Math.random() * 1080) };
        break;

      case 'view':
        metadata.page = `/page${Math.floor(Math.random() * 10)}`;
        metadata.duration = Math.floor(Math.random() * 300); // seconds
        break;

      case 'purchase':
        metadata.amount = Math.floor(Math.random() * 1000) / 10;
        metadata.currency = 'USD';
        metadata.items = Math.floor(Math.random() * 5) + 1;
        break;

      default:
        metadata.type = eventType;
        break;
    }

    return metadata;
  }
}
