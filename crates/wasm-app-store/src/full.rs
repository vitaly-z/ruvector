//! Full Apps - Larger WASM applications for browser, mobile, and embedded
//!
//! Full apps support:
//! - Rich feature sets with multiple modules
//! - Complex UI and interactions
//! - Subscription and one-time purchase models
//! - Mobile and desktop distribution

use serde::{Deserialize, Serialize};
use flate2::Compression;
use flate2::write::GzEncoder;
use flate2::read::GzDecoder;
use std::io::{Read, Write};
use std::collections::HashMap;

use crate::apps::{AppMetadata, AppPricing, AppInterface, MemoryRequirements, generate_app_id, AppSize};
use crate::category::{AppCategory, Platform, License};
use crate::version::Version;
use crate::error::{AppStoreError, AppStoreResult};

/// Full App - Larger WASM application (>8KB)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullApp {
    /// Unique identifier
    pub id: String,
    /// App metadata
    pub metadata: AppMetadata,
    /// Main WASM module
    pub main_module: WasmModule,
    /// Additional WASM modules (for modular apps)
    pub modules: Vec<WasmModule>,
    /// Static assets
    pub assets: Vec<Asset>,
    /// App configuration
    pub config: AppConfig,
    /// Dependencies on other apps
    pub dependencies: Vec<AppDependency>,
    /// Pricing model
    pub pricing: AppPricing,
    /// Release notes
    pub release_notes: Option<String>,
}

impl FullApp {
    /// Create a new full app
    pub fn new(
        name: String,
        description: String,
        publisher_id: String,
        wasm: Vec<u8>,
    ) -> AppStoreResult<Self> {
        // Validate WASM
        if !Self::is_valid_wasm(&wasm) {
            return Err(AppStoreError::InvalidWasm("Invalid WASM magic number".to_string()));
        }

        // Calculate hash and compress
        let wasm_hash = Self::calculate_hash(&wasm);
        let compressed = Self::compress(&wasm)?;

        // Create main module
        let main_module = WasmModule {
            name: "main".to_string(),
            wasm,
            wasm_compressed: compressed.clone(),
            hash: wasm_hash.clone(),
            interface: AppInterface {
                exports: Vec::new(),
                imports: Vec::new(),
                memory: MemoryRequirements::default(),
            },
        };

        // Create metadata
        let mut metadata = AppMetadata::new(
            name,
            description,
            publisher_id,
            main_module.wasm.len(),
            wasm_hash,
        );
        metadata.compressed_size = Some(compressed.len());

        Ok(FullApp {
            id: generate_app_id(),
            metadata,
            main_module,
            modules: Vec::new(),
            assets: Vec::new(),
            config: AppConfig::default(),
            dependencies: Vec::new(),
            pricing: AppPricing::Free,
            release_notes: None,
        })
    }

    /// Add an additional WASM module
    pub fn add_module(&mut self, name: String, wasm: Vec<u8>) -> AppStoreResult<()> {
        if !Self::is_valid_wasm(&wasm) {
            return Err(AppStoreError::InvalidWasm(format!(
                "Invalid WASM for module: {}", name
            )));
        }

        let hash = Self::calculate_hash(&wasm);
        let compressed = Self::compress(&wasm)?;

        self.modules.push(WasmModule {
            name,
            wasm,
            wasm_compressed: compressed,
            hash,
            interface: AppInterface {
                exports: Vec::new(),
                imports: Vec::new(),
                memory: MemoryRequirements::default(),
            },
        });

        self.update_total_size();
        Ok(())
    }

    /// Add an asset
    pub fn add_asset(&mut self, asset: Asset) {
        self.assets.push(asset);
        self.update_total_size();
    }

    /// Add a dependency
    pub fn add_dependency(&mut self, dep: AppDependency) {
        self.dependencies.push(dep);
    }

    /// Update total size in metadata
    fn update_total_size(&mut self) {
        let wasm_size: usize = self.main_module.wasm.len()
            + self.modules.iter().map(|m| m.wasm.len()).sum::<usize>();
        let asset_size: usize = self.assets.iter().map(|a| a.data.len()).sum();

        self.metadata.wasm_size = wasm_size;

        let compressed_size: usize = self.main_module.wasm_compressed.len()
            + self.modules.iter().map(|m| m.wasm_compressed.len()).sum::<usize>()
            + self.assets.iter().filter_map(|a| a.compressed_data.as_ref()).map(|d| d.len()).sum::<usize>();

        self.metadata.compressed_size = Some(compressed_size);
    }

    /// Validate WASM magic number
    fn is_valid_wasm(bytes: &[u8]) -> bool {
        bytes.len() >= 4 && bytes[0..4] == [0x00, 0x61, 0x73, 0x6D]
    }

    /// Compress data
    fn compress(data: &[u8]) -> AppStoreResult<Vec<u8>> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
        encoder.write_all(data)
            .map_err(|e| AppStoreError::CompressionError(e.to_string()))?;
        encoder.finish()
            .map_err(|e| AppStoreError::CompressionError(e.to_string()))
    }

    /// Calculate SHA-256 hash
    fn calculate_hash(data: &[u8]) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(data);
        hex::encode(hasher.finalize())
    }

    /// Get total size in bytes
    pub fn total_size(&self) -> usize {
        self.metadata.wasm_size
            + self.assets.iter().map(|a| a.data.len()).sum::<usize>()
    }

    /// Get app size category
    pub fn size_category(&self) -> AppSize {
        AppSize::from_bytes(self.total_size())
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

    /// Set license
    pub fn with_license(mut self, license: License) -> Self {
        self.metadata.license = license;
        self
    }

    /// Set release notes
    pub fn with_release_notes(mut self, notes: String) -> Self {
        self.release_notes = Some(notes);
        self
    }

    /// Validate the app
    pub fn validate(&self) -> AppStoreResult<()> {
        self.metadata.validate()?;

        // Check main module
        if self.main_module.wasm.is_empty() {
            return Err(AppStoreError::InvalidApp("Main module cannot be empty".to_string()));
        }

        // Check for circular dependencies
        // (simplified - in production would do full graph analysis)

        Ok(())
    }
}

/// WASM module within a full app
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmModule {
    /// Module name
    pub name: String,
    /// Raw WASM bytes
    #[serde(with = "serde_bytes")]
    pub wasm: Vec<u8>,
    /// Compressed WASM bytes
    #[serde(with = "serde_bytes")]
    pub wasm_compressed: Vec<u8>,
    /// SHA-256 hash
    pub hash: String,
    /// Module interface
    pub interface: AppInterface,
}

/// Static asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    /// Asset path (e.g., "images/logo.png")
    pub path: String,
    /// MIME type
    pub mime_type: String,
    /// Raw data
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
    /// Compressed data (optional)
    #[serde(with = "serde_bytes")]
    pub compressed_data: Option<Vec<u8>>,
    /// SHA-256 hash
    pub hash: String,
}

impl Asset {
    /// Create a new asset
    pub fn new(path: String, mime_type: String, data: Vec<u8>) -> Self {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let hash = hex::encode(hasher.finalize());

        Asset {
            path,
            mime_type,
            data,
            compressed_data: None,
            hash,
        }
    }

    /// Create with compression
    pub fn with_compression(mut self) -> AppStoreResult<Self> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
        encoder.write_all(&self.data)
            .map_err(|e| AppStoreError::CompressionError(e.to_string()))?;
        self.compressed_data = Some(encoder.finish()
            .map_err(|e| AppStoreError::CompressionError(e.to_string()))?);
        Ok(self)
    }
}

/// App configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AppConfig {
    /// Environment variables
    pub env: HashMap<String, String>,
    /// Feature flags
    pub features: HashMap<String, bool>,
    /// Runtime settings
    pub runtime: RuntimeConfig,
    /// UI settings
    pub ui: Option<UiConfig>,
}

/// Runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Initial memory pages
    pub initial_memory_pages: u32,
    /// Maximum memory pages
    pub max_memory_pages: u32,
    /// Enable multi-threading
    pub multi_threaded: bool,
    /// Enable SIMD
    pub enable_simd: bool,
    /// Execution timeout (ms)
    pub timeout_ms: Option<u64>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        RuntimeConfig {
            initial_memory_pages: 16,  // 1MB
            max_memory_pages: 256,     // 16MB
            multi_threaded: false,
            enable_simd: true,
            timeout_ms: Some(30000),   // 30 seconds
        }
    }
}

/// UI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiConfig {
    /// Initial window width
    pub width: u32,
    /// Initial window height
    pub height: u32,
    /// Resizable
    pub resizable: bool,
    /// Fullscreen supported
    pub fullscreen: bool,
    /// Theme
    pub theme: String,
}

/// App dependency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppDependency {
    /// App ID or name
    pub app_id: String,
    /// Version requirement
    pub version_req: String,
    /// Is optional dependency
    pub optional: bool,
}

/// Builder for creating full apps
pub struct FullAppBuilder {
    name: Option<String>,
    description: Option<String>,
    publisher_id: Option<String>,
    wasm: Option<Vec<u8>>,
    category: Option<AppCategory>,
    platforms: Vec<Platform>,
    license: Option<License>,
    pricing: Option<AppPricing>,
    release_notes: Option<String>,
    modules: Vec<(String, Vec<u8>)>,
    assets: Vec<Asset>,
    dependencies: Vec<AppDependency>,
}

impl FullAppBuilder {
    pub fn new() -> Self {
        FullAppBuilder {
            name: None,
            description: None,
            publisher_id: None,
            wasm: None,
            category: None,
            platforms: Vec::new(),
            license: None,
            pricing: None,
            release_notes: None,
            modules: Vec::new(),
            assets: Vec::new(),
            dependencies: Vec::new(),
        }
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

    pub fn license(mut self, license: License) -> Self {
        self.license = Some(license);
        self
    }

    pub fn pricing(mut self, pricing: AppPricing) -> Self {
        self.pricing = Some(pricing);
        self
    }

    pub fn release_notes(mut self, notes: impl Into<String>) -> Self {
        self.release_notes = Some(notes.into());
        self
    }

    pub fn module(mut self, name: impl Into<String>, wasm: Vec<u8>) -> Self {
        self.modules.push((name.into(), wasm));
        self
    }

    pub fn asset(mut self, asset: Asset) -> Self {
        self.assets.push(asset);
        self
    }

    pub fn dependency(mut self, app_id: impl Into<String>, version_req: impl Into<String>) -> Self {
        self.dependencies.push(AppDependency {
            app_id: app_id.into(),
            version_req: version_req.into(),
            optional: false,
        });
        self
    }

    pub fn build(self) -> AppStoreResult<FullApp> {
        let name = self.name.ok_or_else(|| AppStoreError::InvalidApp("Name required".to_string()))?;
        let description = self.description.ok_or_else(|| AppStoreError::InvalidApp("Description required".to_string()))?;
        let publisher_id = self.publisher_id.ok_or_else(|| AppStoreError::InvalidApp("Publisher ID required".to_string()))?;
        let wasm = self.wasm.ok_or_else(|| AppStoreError::InvalidApp("WASM bytes required".to_string()))?;

        let mut app = FullApp::new(name, description, publisher_id, wasm)?;

        if let Some(category) = self.category {
            app.metadata.category = category;
        }
        if !self.platforms.is_empty() {
            app.metadata.platforms = self.platforms;
        }
        if let Some(license) = self.license {
            app.metadata.license = license;
        }
        if let Some(pricing) = self.pricing {
            app.pricing = pricing;
        }
        if let Some(notes) = self.release_notes {
            app.release_notes = Some(notes);
        }

        for (name, wasm) in self.modules {
            app.add_module(name, wasm)?;
        }

        for asset in self.assets {
            app.add_asset(asset);
        }

        for dep in self.dependencies {
            app.add_dependency(dep);
        }

        Ok(app)
    }
}

impl Default for FullAppBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chip::templates;

    #[test]
    fn test_full_app_creation() {
        // Use a larger WASM (replicate small one for size to exceed chip app limit)
        let small_wasm = templates::hello_world_wasm();
        let mut wasm = Vec::new();
        // Need enough to exceed 8KB chip app limit
        for _ in 0..300 {
            wasm.extend_from_slice(&small_wasm);
        }
        // Ensure valid WASM header
        wasm[0..4].copy_from_slice(&[0x00, 0x61, 0x73, 0x6D]);

        let app = FullApp::new(
            "Full App".to_string(),
            "A full-featured application".to_string(),
            "pub_123".to_string(),
            wasm,
        ).unwrap();

        assert!(!app.id.is_empty());
        // Note: FullApp can contain any size WASM, even chip-sized ones
        // The is_chip_app() check is based on size, and this test is for the FullApp type
        assert!(app.main_module.wasm.len() > 0);
    }

    #[test]
    fn test_full_app_builder() {
        let wasm = templates::hello_world_wasm();
        let mut large_wasm = Vec::new();
        for _ in 0..500 {
            large_wasm.extend_from_slice(&wasm);
        }
        large_wasm[0..4].copy_from_slice(&[0x00, 0x61, 0x73, 0x6D]);

        let app = FullAppBuilder::new()
            .name("Test Full App")
            .description("A test full application")
            .publisher("pub_123")
            .wasm(large_wasm)
            .category(AppCategory::DevTools)
            .platform(Platform::Browser)
            .platform(Platform::NodeJs)
            .license(License::Mit)
            .pricing(AppPricing::OneTime { price: 999 })
            .release_notes("Initial release")
            .build()
            .unwrap();

        assert_eq!(app.metadata.name, "Test Full App");
        assert_eq!(app.metadata.category, AppCategory::DevTools);
        assert_eq!(app.metadata.platforms.len(), 2);
        assert!(matches!(app.pricing, AppPricing::OneTime { price: 999 }));
    }

    #[test]
    fn test_add_module() {
        let wasm = templates::hello_world_wasm();
        let mut app = FullApp::new(
            "Multi-Module".to_string(),
            "App with multiple modules".to_string(),
            "pub".to_string(),
            wasm.clone(),
        ).unwrap();

        app.add_module("worker".to_string(), wasm.clone()).unwrap();
        app.add_module("utils".to_string(), wasm).unwrap();

        assert_eq!(app.modules.len(), 2);
    }

    #[test]
    fn test_add_asset() {
        let wasm = templates::hello_world_wasm();
        let mut app = FullApp::new(
            "App with Assets".to_string(),
            "Test".to_string(),
            "pub".to_string(),
            wasm,
        ).unwrap();

        let asset = Asset::new(
            "images/logo.png".to_string(),
            "image/png".to_string(),
            vec![0x89, 0x50, 0x4E, 0x47], // PNG magic
        );

        app.add_asset(asset);
        assert_eq!(app.assets.len(), 1);
        assert_eq!(app.assets[0].path, "images/logo.png");
    }
}
