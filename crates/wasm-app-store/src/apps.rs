//! App base types and metadata

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use crate::category::{AppCategory, Platform, ContentRating, License};
use crate::version::Version;
use crate::error::{AppStoreError, AppStoreResult};

/// App identifier
pub type AppId = String;

/// Publisher identifier
pub type PublisherId = String;

/// Generate a new app ID
pub fn generate_app_id() -> AppId {
    Uuid::new_v4().to_string()
}

/// Generate a short ID for chip apps (8 characters)
pub fn generate_short_id() -> String {
    let uuid = Uuid::new_v4();
    let bytes = uuid.as_bytes();
    hex::encode(&bytes[0..4])
}

/// App size category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppSize {
    /// Chip: ≤8KB (ultra-compact)
    Chip,
    /// Micro: ≤64KB
    Micro,
    /// Small: ≤512KB
    Small,
    /// Medium: ≤2MB
    Medium,
    /// Large: ≤10MB
    Large,
    /// Full: >10MB
    Full,
}

impl AppSize {
    /// Get maximum bytes for this size category
    pub fn max_bytes(&self) -> usize {
        match self {
            AppSize::Chip => 8 * 1024,
            AppSize::Micro => 64 * 1024,
            AppSize::Small => 512 * 1024,
            AppSize::Medium => 2 * 1024 * 1024,
            AppSize::Large => 10 * 1024 * 1024,
            AppSize::Full => usize::MAX,
        }
    }

    /// Determine size category from byte count
    pub fn from_bytes(bytes: usize) -> Self {
        if bytes <= 8 * 1024 {
            AppSize::Chip
        } else if bytes <= 64 * 1024 {
            AppSize::Micro
        } else if bytes <= 512 * 1024 {
            AppSize::Small
        } else if bytes <= 2 * 1024 * 1024 {
            AppSize::Medium
        } else if bytes <= 10 * 1024 * 1024 {
            AppSize::Large
        } else {
            AppSize::Full
        }
    }

    /// Get display name
    pub fn display_name(&self) -> &'static str {
        match self {
            AppSize::Chip => "Chip (≤8KB)",
            AppSize::Micro => "Micro (≤64KB)",
            AppSize::Small => "Small (≤512KB)",
            AppSize::Medium => "Medium (≤2MB)",
            AppSize::Large => "Large (≤10MB)",
            AppSize::Full => "Full (>10MB)",
        }
    }
}

/// App status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppStatus {
    /// Draft - not yet published
    Draft,
    /// Pending review
    PendingReview,
    /// Published and available
    Published,
    /// Suspended by admin
    Suspended,
    /// Deprecated - still available but not recommended
    Deprecated,
    /// Archived - no longer available
    Archived,
}

impl Default for AppStatus {
    fn default() -> Self {
        AppStatus::Draft
    }
}

/// App metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppMetadata {
    /// Unique app identifier
    pub id: AppId,
    /// URL-friendly slug
    pub slug: String,
    /// Display name
    pub name: String,
    /// Short description (max 160 chars)
    pub description: String,
    /// Long description (markdown)
    pub long_description: Option<String>,
    /// Version
    pub version: Version,
    /// Publisher ID
    pub publisher_id: PublisherId,
    /// Category
    pub category: AppCategory,
    /// Tags for discovery
    pub tags: Vec<String>,
    /// Target platforms
    pub platforms: Vec<Platform>,
    /// License
    pub license: License,
    /// Content rating
    pub content_rating: ContentRating,
    /// Status
    pub status: AppStatus,
    /// Icon URL
    pub icon_url: Option<String>,
    /// Screenshot URLs
    pub screenshots: Vec<String>,
    /// Homepage URL
    pub homepage_url: Option<String>,
    /// Repository URL
    pub repository_url: Option<String>,
    /// Documentation URL
    pub docs_url: Option<String>,
    /// WASM size in bytes
    pub wasm_size: usize,
    /// Compressed size in bytes
    pub compressed_size: Option<usize>,
    /// SHA-256 hash of WASM binary
    pub wasm_hash: String,
    /// Is featured app
    pub featured: bool,
    /// Is verified publisher
    pub verified_publisher: bool,
    /// Download count
    pub downloads: u64,
    /// Average rating (0-5)
    pub rating: f32,
    /// Rating count
    pub rating_count: u32,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Updated timestamp
    pub updated_at: DateTime<Utc>,
    /// Published timestamp
    pub published_at: Option<DateTime<Utc>>,
}

impl AppMetadata {
    /// Create new app metadata
    pub fn new(
        name: String,
        description: String,
        publisher_id: PublisherId,
        wasm_size: usize,
        wasm_hash: String,
    ) -> Self {
        let id = generate_app_id();
        let slug = Self::generate_slug(&name);
        let now = Utc::now();

        AppMetadata {
            id,
            slug,
            name,
            description,
            long_description: None,
            version: Version::default(),
            publisher_id,
            category: AppCategory::default(),
            tags: Vec::new(),
            platforms: vec![Platform::Universal],
            license: License::default(),
            content_rating: ContentRating::default(),
            status: AppStatus::Draft,
            icon_url: None,
            screenshots: Vec::new(),
            homepage_url: None,
            repository_url: None,
            docs_url: None,
            wasm_size,
            compressed_size: None,
            wasm_hash,
            featured: false,
            verified_publisher: false,
            downloads: 0,
            rating: 0.0,
            rating_count: 0,
            created_at: now,
            updated_at: now,
            published_at: None,
        }
    }

    /// Generate URL-friendly slug from name
    fn generate_slug(name: &str) -> String {
        name.to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '-' })
            .collect::<String>()
            .split('-')
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("-")
    }

    /// Get app size category
    pub fn size_category(&self) -> AppSize {
        AppSize::from_bytes(self.wasm_size)
    }

    /// Check if app is a chip app (≤8KB)
    pub fn is_chip_app(&self) -> bool {
        self.wasm_size <= 8 * 1024
    }

    /// Update rating with new value
    pub fn update_rating(&mut self, new_rating: f32) {
        if self.rating_count == 0 {
            self.rating = new_rating;
        } else {
            // Weighted average
            self.rating = (self.rating * self.rating_count as f32 + new_rating)
                / (self.rating_count + 1) as f32;
        }
        self.rating_count += 1;
        self.updated_at = Utc::now();
    }

    /// Increment download count
    pub fn record_download(&mut self) {
        self.downloads = self.downloads.saturating_add(1);
    }

    /// Publish the app
    pub fn publish(&mut self) -> AppStoreResult<()> {
        if self.status == AppStatus::Suspended {
            return Err(AppStoreError::Unauthorized("App is suspended".to_string()));
        }
        self.status = AppStatus::Published;
        self.published_at = Some(Utc::now());
        self.updated_at = Utc::now();
        Ok(())
    }

    /// Suspend the app
    pub fn suspend(&mut self, reason: &str) {
        self.status = AppStatus::Suspended;
        self.updated_at = Utc::now();
    }

    /// Deprecate the app
    pub fn deprecate(&mut self) {
        self.status = AppStatus::Deprecated;
        self.updated_at = Utc::now();
    }

    /// Archive the app
    pub fn archive(&mut self) {
        self.status = AppStatus::Archived;
        self.updated_at = Utc::now();
    }

    /// Validate metadata
    pub fn validate(&self) -> AppStoreResult<()> {
        if self.name.is_empty() || self.name.len() > 100 {
            return Err(AppStoreError::InvalidApp("Name must be 1-100 characters".to_string()));
        }
        if self.description.is_empty() || self.description.len() > 160 {
            return Err(AppStoreError::InvalidApp("Description must be 1-160 characters".to_string()));
        }
        if self.slug.is_empty() {
            return Err(AppStoreError::InvalidApp("Slug cannot be empty".to_string()));
        }
        if self.tags.len() > 10 {
            return Err(AppStoreError::InvalidApp("Maximum 10 tags allowed".to_string()));
        }
        Ok(())
    }
}

/// App pricing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppPricing {
    /// Free app
    Free,
    /// One-time purchase (credits)
    OneTime { price: u64 },
    /// Pay per use (credits per execution)
    PayPerUse { price_per_use: u64 },
    /// Subscription (monthly credits)
    Subscription { monthly_price: u64 },
    /// Freemium with paid features
    Freemium {
        free_uses_per_day: u32,
        price_per_use: u64,
    },
}

impl Default for AppPricing {
    fn default() -> Self {
        AppPricing::Free
    }
}

/// App exports/imports for WASM module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppInterface {
    /// Exported functions
    pub exports: Vec<FunctionSignature>,
    /// Required imports
    pub imports: Vec<FunctionSignature>,
    /// Memory requirements
    pub memory: MemoryRequirements,
}

/// Function signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionSignature {
    /// Function name
    pub name: String,
    /// Parameter types
    pub params: Vec<WasmType>,
    /// Return types
    pub results: Vec<WasmType>,
}

/// WASM value types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WasmType {
    I32,
    I64,
    F32,
    F64,
    V128,
    FuncRef,
    ExternRef,
}

/// Memory requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRequirements {
    /// Initial memory pages (64KB each)
    pub initial_pages: u32,
    /// Maximum memory pages
    pub max_pages: Option<u32>,
    /// Shared memory
    pub shared: bool,
}

impl Default for MemoryRequirements {
    fn default() -> Self {
        MemoryRequirements {
            initial_pages: 1,
            max_pages: Some(16),
            shared: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_size_from_bytes() {
        assert_eq!(AppSize::from_bytes(1000), AppSize::Chip);
        assert_eq!(AppSize::from_bytes(8192), AppSize::Chip);
        assert_eq!(AppSize::from_bytes(8193), AppSize::Micro);
        assert_eq!(AppSize::from_bytes(100_000), AppSize::Small);
        assert_eq!(AppSize::from_bytes(1_000_000), AppSize::Medium);
        assert_eq!(AppSize::from_bytes(5_000_000), AppSize::Large);
        assert_eq!(AppSize::from_bytes(20_000_000), AppSize::Full);
    }

    #[test]
    fn test_metadata_creation() {
        let meta = AppMetadata::new(
            "Test App".to_string(),
            "A test application".to_string(),
            "pub_123".to_string(),
            4096,
            "abc123".to_string(),
        );

        assert!(!meta.id.is_empty());
        assert_eq!(meta.slug, "test-app");
        assert!(meta.is_chip_app());
        assert_eq!(meta.size_category(), AppSize::Chip);
    }

    #[test]
    fn test_slug_generation() {
        let meta = AppMetadata::new(
            "My Cool  App!!!".to_string(),
            "Description".to_string(),
            "pub".to_string(),
            1000,
            "hash".to_string(),
        );
        assert_eq!(meta.slug, "my-cool-app");
    }

    #[test]
    fn test_rating_update() {
        let mut meta = AppMetadata::new(
            "Test".to_string(),
            "Test".to_string(),
            "pub".to_string(),
            1000,
            "hash".to_string(),
        );

        meta.update_rating(5.0);
        assert_eq!(meta.rating, 5.0);
        assert_eq!(meta.rating_count, 1);

        meta.update_rating(3.0);
        assert_eq!(meta.rating, 4.0); // (5 + 3) / 2
        assert_eq!(meta.rating_count, 2);
    }
}
