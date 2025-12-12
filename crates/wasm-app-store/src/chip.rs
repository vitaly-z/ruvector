//! Chip Apps - Ultra-small WASM modules (≤8KB)
//!
//! Chip apps are designed for:
//! - Instant loading and execution
//! - Micropayment support (pay-per-use)
//! - Embedded and IoT devices
//! - Browser extensions and quick utilities

use serde::{Deserialize, Serialize};
use flate2::Compression;
use flate2::write::GzEncoder;
use flate2::read::GzDecoder;
use std::io::{Read, Write};

use crate::apps::{AppMetadata, AppPricing, AppInterface, MemoryRequirements, generate_short_id};
use crate::category::{AppCategory, Platform};
use crate::version::Version;
use crate::error::{AppStoreError, AppStoreResult};
use crate::MAX_CHIP_APP_SIZE;

/// Chip App - Ultra-small WASM module (≤8KB)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChipApp {
    /// Short unique identifier (8 chars)
    pub id: String,
    /// App metadata
    pub metadata: AppMetadata,
    /// Raw WASM bytes
    #[serde(with = "serde_bytes")]
    pub wasm: Vec<u8>,
    /// Compressed WASM bytes (for storage/transfer)
    #[serde(with = "serde_bytes")]
    pub wasm_compressed: Vec<u8>,
    /// App interface (exports/imports)
    pub interface: AppInterface,
    /// Pricing model
    pub pricing: AppPricing,
    /// Execution statistics
    pub stats: ChipAppStats,
}

impl ChipApp {
    /// Create a new chip app from WASM bytes
    pub fn new(
        name: String,
        description: String,
        publisher_id: String,
        wasm: Vec<u8>,
    ) -> AppStoreResult<Self> {
        // Validate size
        if wasm.len() > MAX_CHIP_APP_SIZE {
            return Err(AppStoreError::SizeExceeded {
                size: wasm.len(),
                limit: MAX_CHIP_APP_SIZE,
            });
        }

        // Validate WASM magic number
        if !Self::is_valid_wasm(&wasm) {
            return Err(AppStoreError::InvalidWasm("Invalid WASM magic number".to_string()));
        }

        // Compress WASM
        let wasm_compressed = Self::compress(&wasm)?;

        // Calculate hash
        let wasm_hash = Self::calculate_hash(&wasm);

        // Create metadata
        let mut metadata = AppMetadata::new(
            name,
            description,
            publisher_id,
            wasm.len(),
            wasm_hash,
        );
        metadata.compressed_size = Some(wasm_compressed.len());

        // Generate short ID for chip apps
        let id = generate_short_id();

        // Default interface for minimal chip apps
        let interface = AppInterface {
            exports: Vec::new(),
            imports: Vec::new(),
            memory: MemoryRequirements {
                initial_pages: 1,
                max_pages: Some(4), // 256KB max for chip apps
                shared: false,
            },
        };

        Ok(ChipApp {
            id,
            metadata,
            wasm,
            wasm_compressed,
            interface,
            pricing: AppPricing::PayPerUse { price_per_use: 1 }, // Default 1 credit per use
            stats: ChipAppStats::default(),
        })
    }

    /// Create from compressed bytes
    pub fn from_compressed(
        id: String,
        metadata: AppMetadata,
        wasm_compressed: Vec<u8>,
        interface: AppInterface,
        pricing: AppPricing,
    ) -> AppStoreResult<Self> {
        let wasm = Self::decompress(&wasm_compressed)?;

        if wasm.len() > MAX_CHIP_APP_SIZE {
            return Err(AppStoreError::SizeExceeded {
                size: wasm.len(),
                limit: MAX_CHIP_APP_SIZE,
            });
        }

        Ok(ChipApp {
            id,
            metadata,
            wasm,
            wasm_compressed,
            interface,
            pricing,
            stats: ChipAppStats::default(),
        })
    }

    /// Validate WASM magic number
    fn is_valid_wasm(bytes: &[u8]) -> bool {
        bytes.len() >= 4 && bytes[0..4] == [0x00, 0x61, 0x73, 0x6D] // \0asm
    }

    /// Compress WASM bytes
    fn compress(data: &[u8]) -> AppStoreResult<Vec<u8>> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
        encoder.write_all(data)
            .map_err(|e| AppStoreError::CompressionError(e.to_string()))?;
        encoder.finish()
            .map_err(|e| AppStoreError::CompressionError(e.to_string()))
    }

    /// Decompress WASM bytes
    fn decompress(data: &[u8]) -> AppStoreResult<Vec<u8>> {
        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)
            .map_err(|e| AppStoreError::CompressionError(e.to_string()))?;
        Ok(decompressed)
    }

    /// Calculate SHA-256 hash of WASM bytes
    fn calculate_hash(data: &[u8]) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(data);
        hex::encode(hasher.finalize())
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f64 {
        if self.wasm.is_empty() {
            return 1.0;
        }
        self.wasm_compressed.len() as f64 / self.wasm.len() as f64
    }

    /// Record an execution
    pub fn record_execution(&mut self, duration_ms: u64) {
        self.stats.executions += 1;
        self.stats.total_execution_time_ms += duration_ms;
        self.stats.avg_execution_time_ms =
            self.stats.total_execution_time_ms as f64 / self.stats.executions as f64;

        if duration_ms < self.stats.min_execution_time_ms || self.stats.min_execution_time_ms == 0 {
            self.stats.min_execution_time_ms = duration_ms;
        }
        if duration_ms > self.stats.max_execution_time_ms {
            self.stats.max_execution_time_ms = duration_ms;
        }
    }

    /// Set pricing model
    pub fn with_pricing(mut self, pricing: AppPricing) -> Self {
        self.pricing = pricing;
        self
    }

    /// Set category
    pub fn with_category(mut self, category: AppCategory) -> Self {
        self.metadata.category = category;
        self
    }

    /// Set platforms
    pub fn with_platforms(mut self, platforms: Vec<Platform>) -> Self {
        self.metadata.platforms = platforms;
        self
    }

    /// Add tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.metadata.tags = tags;
        self
    }

    /// Serialize to compact binary format for distribution
    pub fn to_distribution_bytes(&self) -> AppStoreResult<Vec<u8>> {
        // Format: [1 byte version][8 byte id][4 byte wasm_len][wasm_compressed][metadata_json]
        let mut bytes = Vec::new();

        // Version byte
        bytes.push(1);

        // ID (8 bytes, padded)
        let id_bytes = self.id.as_bytes();
        bytes.extend_from_slice(&id_bytes[..8.min(id_bytes.len())]);
        for _ in id_bytes.len()..8 {
            bytes.push(0);
        }

        // WASM compressed length (4 bytes, little-endian)
        let wasm_len = self.wasm_compressed.len() as u32;
        bytes.extend_from_slice(&wasm_len.to_le_bytes());

        // WASM compressed bytes
        bytes.extend_from_slice(&self.wasm_compressed);

        // Metadata JSON
        let metadata_json = serde_json::to_vec(&self.metadata)?;
        let meta_len = metadata_json.len() as u32;
        bytes.extend_from_slice(&meta_len.to_le_bytes());
        bytes.extend_from_slice(&metadata_json);

        Ok(bytes)
    }

    /// Deserialize from compact binary format
    pub fn from_distribution_bytes(bytes: &[u8]) -> AppStoreResult<Self> {
        if bytes.len() < 17 {
            return Err(AppStoreError::InvalidApp("Bytes too short".to_string()));
        }

        let version = bytes[0];
        if version != 1 {
            return Err(AppStoreError::InvalidApp(format!("Unsupported version: {}", version)));
        }

        // Parse ID
        let id = String::from_utf8_lossy(&bytes[1..9])
            .trim_end_matches('\0')
            .to_string();

        // Parse WASM length and bytes
        let wasm_len = u32::from_le_bytes([bytes[9], bytes[10], bytes[11], bytes[12]]) as usize;
        if bytes.len() < 13 + wasm_len + 4 {
            return Err(AppStoreError::InvalidApp("Invalid WASM length".to_string()));
        }
        let wasm_compressed = bytes[13..13 + wasm_len].to_vec();

        // Parse metadata length and JSON
        let meta_offset = 13 + wasm_len;
        let meta_len = u32::from_le_bytes([
            bytes[meta_offset],
            bytes[meta_offset + 1],
            bytes[meta_offset + 2],
            bytes[meta_offset + 3],
        ]) as usize;
        let metadata: AppMetadata = serde_json::from_slice(
            &bytes[meta_offset + 4..meta_offset + 4 + meta_len]
        )?;

        // Decompress WASM
        let wasm = Self::decompress(&wasm_compressed)?;

        Ok(ChipApp {
            id,
            metadata,
            wasm,
            wasm_compressed,
            interface: AppInterface {
                exports: Vec::new(),
                imports: Vec::new(),
                memory: MemoryRequirements::default(),
            },
            pricing: AppPricing::Free,
            stats: ChipAppStats::default(),
        })
    }
}

/// Chip app execution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChipAppStats {
    /// Total executions
    pub executions: u64,
    /// Total execution time in milliseconds
    pub total_execution_time_ms: u64,
    /// Average execution time in milliseconds
    pub avg_execution_time_ms: f64,
    /// Minimum execution time in milliseconds
    pub min_execution_time_ms: u64,
    /// Maximum execution time in milliseconds
    pub max_execution_time_ms: u64,
    /// Unique users
    pub unique_users: u64,
    /// Revenue generated (in credits)
    pub total_revenue: u64,
}

/// Builder for creating chip apps
#[derive(Default)]
pub struct ChipAppBuilder {
    name: Option<String>,
    description: Option<String>,
    publisher_id: Option<String>,
    wasm: Option<Vec<u8>>,
    category: Option<AppCategory>,
    platforms: Vec<Platform>,
    tags: Vec<String>,
    pricing: Option<AppPricing>,
}

impl ChipAppBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn publisher(mut self, publisher_id: impl Into<String>) -> Self {
        self.publisher_id = Some(publisher_id.into());
        self
    }

    pub fn wasm(mut self, wasm: Vec<u8>) -> Self {
        self.wasm = Some(wasm);
        self
    }

    pub fn category(mut self, category: AppCategory) -> Self {
        self.category = Some(category);
        self
    }

    pub fn platform(mut self, platform: Platform) -> Self {
        self.platforms.push(platform);
        self
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    pub fn pricing(mut self, pricing: AppPricing) -> Self {
        self.pricing = Some(pricing);
        self
    }

    pub fn build(self) -> AppStoreResult<ChipApp> {
        let name = self.name.ok_or_else(|| AppStoreError::InvalidApp("Name required".to_string()))?;
        let description = self.description.ok_or_else(|| AppStoreError::InvalidApp("Description required".to_string()))?;
        let publisher_id = self.publisher_id.ok_or_else(|| AppStoreError::InvalidApp("Publisher ID required".to_string()))?;
        let wasm = self.wasm.ok_or_else(|| AppStoreError::InvalidApp("WASM bytes required".to_string()))?;

        let mut app = ChipApp::new(name, description, publisher_id, wasm)?;

        if let Some(category) = self.category {
            app.metadata.category = category;
        }
        if !self.platforms.is_empty() {
            app.metadata.platforms = self.platforms;
        }
        if !self.tags.is_empty() {
            app.metadata.tags = self.tags;
        }
        if let Some(pricing) = self.pricing {
            app.pricing = pricing;
        }

        Ok(app)
    }
}

/// Example chip app templates
pub mod templates {
    use super::*;

    /// Minimal "Hello World" WASM module (~70 bytes)
    pub fn hello_world_wasm() -> Vec<u8> {
        // Minimal valid WASM module with a simple function
        vec![
            0x00, 0x61, 0x73, 0x6D, // magic number (\0asm)
            0x01, 0x00, 0x00, 0x00, // version 1
            // Type section
            0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7F,
            // Function section
            0x03, 0x02, 0x01, 0x00,
            // Export section
            0x07, 0x08, 0x01, 0x04, 0x6D, 0x61, 0x69, 0x6E, 0x00, 0x00,
            // Code section
            0x0A, 0x06, 0x01, 0x04, 0x00, 0x41, 0x2A, 0x0B, // returns 42
        ]
    }

    /// Add function WASM module (~100 bytes)
    pub fn add_wasm() -> Vec<u8> {
        vec![
            0x00, 0x61, 0x73, 0x6D, // magic
            0x01, 0x00, 0x00, 0x00, // version
            // Type section: (i32, i32) -> i32
            0x01, 0x07, 0x01, 0x60, 0x02, 0x7F, 0x7F, 0x01, 0x7F,
            // Function section
            0x03, 0x02, 0x01, 0x00,
            // Export section
            0x07, 0x07, 0x01, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00,
            // Code section
            0x0A, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6A, 0x0B,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chip_app_creation() {
        let wasm = templates::hello_world_wasm();
        let app = ChipApp::new(
            "Hello World".to_string(),
            "A minimal WASM module".to_string(),
            "pub_123".to_string(),
            wasm.clone(),
        ).unwrap();

        assert_eq!(app.id.len(), 8);
        assert!(app.wasm.len() <= MAX_CHIP_APP_SIZE);
        // Note: gzip can make small files larger due to header overhead
        assert!(app.compression_ratio() > 0.0);
        assert!(app.metadata.is_chip_app());
    }

    #[test]
    fn test_chip_app_too_large() {
        let wasm = vec![0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00];
        let mut large_wasm = wasm.clone();
        large_wasm.extend(vec![0u8; MAX_CHIP_APP_SIZE]); // Exceed limit

        let result = ChipApp::new(
            "Too Large".to_string(),
            "This app is too large".to_string(),
            "pub".to_string(),
            large_wasm,
        );

        assert!(matches!(result, Err(AppStoreError::SizeExceeded { .. })));
    }

    #[test]
    fn test_chip_app_invalid_wasm() {
        let invalid_wasm = vec![0x00, 0x00, 0x00, 0x00]; // Wrong magic

        let result = ChipApp::new(
            "Invalid".to_string(),
            "Invalid WASM".to_string(),
            "pub".to_string(),
            invalid_wasm,
        );

        assert!(matches!(result, Err(AppStoreError::InvalidWasm(_))));
    }

    #[test]
    fn test_chip_app_builder() {
        let wasm = templates::add_wasm();
        let app = ChipAppBuilder::new()
            .name("Calculator")
            .description("Basic calculator")
            .publisher("pub_123")
            .wasm(wasm)
            .category(AppCategory::Utilities)
            .platform(Platform::Browser)
            .tag("math")
            .tag("calculator")
            .pricing(AppPricing::Free)
            .build()
            .unwrap();

        assert_eq!(app.metadata.name, "Calculator");
        assert_eq!(app.metadata.category, AppCategory::Utilities);
        assert_eq!(app.metadata.tags.len(), 2);
    }

    #[test]
    fn test_chip_app_stats() {
        let wasm = templates::hello_world_wasm();
        let mut app = ChipApp::new(
            "Test".to_string(),
            "Test app".to_string(),
            "pub".to_string(),
            wasm,
        ).unwrap();

        app.record_execution(10);
        app.record_execution(20);
        app.record_execution(15);

        assert_eq!(app.stats.executions, 3);
        assert_eq!(app.stats.total_execution_time_ms, 45);
        assert_eq!(app.stats.avg_execution_time_ms, 15.0);
        assert_eq!(app.stats.min_execution_time_ms, 10);
        assert_eq!(app.stats.max_execution_time_ms, 20);
    }

    #[test]
    fn test_distribution_format() {
        let wasm = templates::hello_world_wasm();
        let app = ChipApp::new(
            "Test App".to_string(),
            "A test application".to_string(),
            "pub_123".to_string(),
            wasm,
        ).unwrap();

        let bytes = app.to_distribution_bytes().unwrap();
        let restored = ChipApp::from_distribution_bytes(&bytes).unwrap();

        assert_eq!(app.id, restored.id);
        assert_eq!(app.wasm, restored.wasm);
        assert_eq!(app.metadata.name, restored.metadata.name);
    }
}
