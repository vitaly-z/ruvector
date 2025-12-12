//! App Store - Main interface for app discovery, purchase, and execution

use std::sync::Arc;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use crate::apps::{AppId, AppMetadata, AppStatus, AppPricing, PublisherId};
use crate::chip::ChipApp;
use crate::full::FullApp;
use crate::registry::{AppRegistry, FeaturedInfo};
use crate::category::AppCategory;
use crate::version::Version;
use crate::error::{AppStoreError, AppStoreResult};

/// App Store configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppStoreConfig {
    /// Maximum chip app size (bytes)
    pub max_chip_app_size: usize,
    /// Maximum full app size (bytes)
    pub max_full_app_size: usize,
    /// Enable payment integration
    pub payments_enabled: bool,
    /// Enable app verification
    pub verification_enabled: bool,
    /// Default platform fee percentage
    pub platform_fee_percentage: u8,
    /// Auto-approve chip apps
    pub auto_approve_chip_apps: bool,
}

impl Default for AppStoreConfig {
    fn default() -> Self {
        AppStoreConfig {
            max_chip_app_size: 8 * 1024,        // 8KB
            max_full_app_size: 100 * 1024 * 1024, // 100MB
            payments_enabled: true,
            verification_enabled: true,
            platform_fee_percentage: 25,
            auto_approve_chip_apps: true,
        }
    }
}

/// The main App Store
pub struct AppStore {
    /// Configuration
    config: AppStoreConfig,
    /// App registry
    registry: Arc<AppRegistry>,
    /// Search index (tag -> app IDs)
    tag_index: DashMap<String, Vec<AppId>>,
    /// Download cache
    download_cache: DashMap<AppId, DownloadInfo>,
}

impl AppStore {
    /// Create a new app store
    pub fn new() -> Self {
        Self::with_config(AppStoreConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: AppStoreConfig) -> Self {
        AppStore {
            config,
            registry: Arc::new(AppRegistry::new()),
            tag_index: DashMap::new(),
            download_cache: DashMap::new(),
        }
    }

    // ==================== Publishing ====================

    /// Publish a chip app
    pub fn publish_chip_app(&self, mut app: ChipApp) -> AppStoreResult<AppId> {
        // Validate size
        if app.wasm.len() > self.config.max_chip_app_size {
            return Err(AppStoreError::SizeExceeded {
                size: app.wasm.len(),
                limit: self.config.max_chip_app_size,
            });
        }

        // Auto-approve chip apps if configured
        if self.config.auto_approve_chip_apps {
            app.metadata.publish()?;
        }

        // Index tags
        for tag in &app.metadata.tags {
            self.tag_index
                .entry(tag.to_lowercase())
                .or_insert_with(Vec::new)
                .push(app.id.clone());
        }

        let id = app.id.clone();
        self.registry.register_chip_app(app)?;

        Ok(id)
    }

    /// Publish a full app
    pub fn publish_full_app(&self, app: FullApp) -> AppStoreResult<AppId> {
        // Validate size
        let total_size = app.total_size();
        if total_size > self.config.max_full_app_size {
            return Err(AppStoreError::SizeExceeded {
                size: total_size,
                limit: self.config.max_full_app_size,
            });
        }

        // Index tags
        for tag in &app.metadata.tags {
            self.tag_index
                .entry(tag.to_lowercase())
                .or_insert_with(Vec::new)
                .push(app.id.clone());
        }

        let id = app.id.clone();
        self.registry.register_full_app(app)?;

        Ok(id)
    }

    /// Update a chip app
    pub fn update_chip_app(&self, app: ChipApp) -> AppStoreResult<()> {
        self.registry.update_chip_app(app)
    }

    /// Update a full app
    pub fn update_full_app(&self, app: FullApp) -> AppStoreResult<()> {
        self.registry.update_full_app(app)
    }

    /// Unpublish an app
    pub fn unpublish(&self, id: &AppId) -> AppStoreResult<()> {
        if let Some(mut app) = self.registry.get_chip_app(id) {
            app.metadata.archive();
            self.registry.update_chip_app(app)?;
        } else if let Some(mut app) = self.registry.get_full_app(id) {
            app.metadata.archive();
            self.registry.update_full_app(app)?;
        } else {
            return Err(AppStoreError::AppNotFound(id.clone()));
        }
        Ok(())
    }

    // ==================== Discovery ====================

    /// Search for apps
    pub fn search(&self, query: &str, limit: usize) -> Vec<SearchResult> {
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        let mut results: Vec<SearchResult> = Vec::new();
        let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Search by name and description
        for entry in self.registry.chip_apps.iter() {
            let app = entry.value();
            if app.metadata.status != AppStatus::Published {
                continue;
            }

            let score = Self::calculate_relevance(&app.metadata, &query_words);
            if score > 0.0 && !seen_ids.contains(&app.id) {
                seen_ids.insert(app.id.clone());
                results.push(SearchResult {
                    metadata: app.metadata.clone(),
                    relevance_score: score,
                    is_chip_app: true,
                });
            }
        }

        for entry in self.registry.full_apps.iter() {
            let app = entry.value();
            if app.metadata.status != AppStatus::Published {
                continue;
            }

            let score = Self::calculate_relevance(&app.metadata, &query_words);
            if score > 0.0 && !seen_ids.contains(&app.id) {
                seen_ids.insert(app.id.clone());
                results.push(SearchResult {
                    metadata: app.metadata.clone(),
                    relevance_score: score,
                    is_chip_app: false,
                });
            }
        }

        // Search by tags
        for word in &query_words {
            if let Some(ids) = self.tag_index.get(*word) {
                for id in ids.iter() {
                    if seen_ids.contains(id) {
                        continue;
                    }
                    if let Some(meta) = self.registry.get_metadata(id) {
                        if meta.status == AppStatus::Published {
                            seen_ids.insert(id.clone());
                            results.push(SearchResult {
                                metadata: meta.clone(),
                                relevance_score: 0.5,
                                is_chip_app: meta.is_chip_app(),
                            });
                        }
                    }
                }
            }
        }

        // Sort by relevance
        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
        results.truncate(limit);
        results
    }

    /// Calculate relevance score for search
    fn calculate_relevance(metadata: &AppMetadata, query_words: &[&str]) -> f64 {
        let name_lower = metadata.name.to_lowercase();
        let desc_lower = metadata.description.to_lowercase();

        let mut score = 0.0;

        for word in query_words {
            // Exact name match
            if name_lower == *word {
                score += 10.0;
            }
            // Name contains word
            else if name_lower.contains(word) {
                score += 5.0;
            }
            // Description contains word
            if desc_lower.contains(word) {
                score += 2.0;
            }
            // Tag match
            if metadata.tags.iter().any(|t| t.to_lowercase() == *word) {
                score += 3.0;
            }
        }

        // Boost for popularity
        score += (metadata.downloads as f64).log10().max(0.0) * 0.5;
        // Boost for rating
        score += metadata.rating as f64 * 0.2;
        // Boost for verified publisher
        if metadata.verified_publisher {
            score += 1.0;
        }
        // Boost for featured
        if metadata.featured {
            score += 2.0;
        }

        score
    }

    /// Get apps by category
    pub fn browse_category(&self, category: AppCategory, limit: usize, offset: usize) -> Vec<AppMetadata> {
        let mut apps = self.registry.get_category_apps(category);
        apps.sort_by(|a, b| b.downloads.cmp(&a.downloads)); // Sort by popularity
        apps.into_iter().skip(offset).take(limit).collect()
    }

    /// Get featured apps
    pub fn get_featured(&self) -> Vec<AppMetadata> {
        self.registry.get_featured_apps()
            .into_iter()
            .map(|(meta, _)| meta)
            .collect()
    }

    /// Get trending apps (most downloads in recent period)
    pub fn get_trending(&self, limit: usize) -> Vec<AppMetadata> {
        let mut apps: Vec<AppMetadata> = Vec::new();

        for entry in self.registry.chip_apps.iter() {
            if entry.metadata.status == AppStatus::Published {
                apps.push(entry.metadata.clone());
            }
        }
        for entry in self.registry.full_apps.iter() {
            if entry.metadata.status == AppStatus::Published {
                apps.push(entry.metadata.clone());
            }
        }

        // Sort by downloads (in production, would filter by time period)
        apps.sort_by(|a, b| b.downloads.cmp(&a.downloads));
        apps.truncate(limit);
        apps
    }

    /// Get new apps
    pub fn get_new(&self, limit: usize) -> Vec<AppMetadata> {
        let mut apps: Vec<AppMetadata> = Vec::new();

        for entry in self.registry.chip_apps.iter() {
            if entry.metadata.status == AppStatus::Published {
                apps.push(entry.metadata.clone());
            }
        }
        for entry in self.registry.full_apps.iter() {
            if entry.metadata.status == AppStatus::Published {
                apps.push(entry.metadata.clone());
            }
        }

        // Sort by creation date
        apps.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        apps.truncate(limit);
        apps
    }

    /// Get apps by publisher
    pub fn get_publisher_apps(&self, publisher_id: &PublisherId) -> Vec<AppMetadata> {
        self.registry.get_publisher_apps(publisher_id)
    }

    // ==================== Retrieval ====================

    /// Get app metadata
    pub fn get_metadata(&self, id: &AppId) -> Option<AppMetadata> {
        self.registry.get_metadata(id)
    }

    /// Get chip app
    pub fn get_chip_app(&self, id: &AppId) -> Option<ChipApp> {
        self.registry.get_chip_app(id)
    }

    /// Get full app
    pub fn get_full_app(&self, id: &AppId) -> Option<FullApp> {
        self.registry.get_full_app(id)
    }

    /// Download a chip app (increments download count)
    pub fn download_chip_app(&self, id: &AppId) -> AppStoreResult<ChipApp> {
        let mut app = self.registry.get_chip_app(id)
            .ok_or_else(|| AppStoreError::AppNotFound(id.clone()))?;

        if app.metadata.status != AppStatus::Published {
            return Err(AppStoreError::DownloadFailed("App is not published".to_string()));
        }

        app.metadata.record_download();
        self.registry.update_chip_app(app.clone())?;

        // Update download cache
        self.download_cache
            .entry(id.clone())
            .or_insert_with(|| DownloadInfo::new())
            .record_download();

        Ok(app)
    }

    /// Download a full app (increments download count)
    pub fn download_full_app(&self, id: &AppId) -> AppStoreResult<FullApp> {
        let mut app = self.registry.get_full_app(id)
            .ok_or_else(|| AppStoreError::AppNotFound(id.clone()))?;

        if app.metadata.status != AppStatus::Published {
            return Err(AppStoreError::DownloadFailed("App is not published".to_string()));
        }

        app.metadata.record_download();
        self.registry.update_full_app(app.clone())?;

        self.download_cache
            .entry(id.clone())
            .or_insert_with(|| DownloadInfo::new())
            .record_download();

        Ok(app)
    }

    // ==================== Ratings & Reviews ====================

    /// Rate an app
    pub fn rate_app(&self, id: &AppId, rating: f32) -> AppStoreResult<()> {
        if rating < 0.0 || rating > 5.0 {
            return Err(AppStoreError::InvalidApp("Rating must be 0-5".to_string()));
        }

        if let Some(mut app) = self.registry.get_chip_app(id) {
            app.metadata.update_rating(rating);
            self.registry.update_chip_app(app)?;
        } else if let Some(mut app) = self.registry.get_full_app(id) {
            app.metadata.update_rating(rating);
            self.registry.update_full_app(app)?;
        } else {
            return Err(AppStoreError::AppNotFound(id.clone()));
        }

        Ok(())
    }

    // ==================== Admin ====================

    /// Set app as featured
    pub fn feature_app(&self, id: &AppId, info: FeaturedInfo) -> AppStoreResult<()> {
        // Update featured flag in metadata
        if let Some(mut app) = self.registry.get_chip_app(id) {
            app.metadata.featured = true;
            self.registry.update_chip_app(app)?;
        } else if let Some(mut app) = self.registry.get_full_app(id) {
            app.metadata.featured = true;
            self.registry.update_full_app(app)?;
        } else {
            return Err(AppStoreError::AppNotFound(id.clone()));
        }

        self.registry.set_featured(id, info)
    }

    /// Remove featured status
    pub fn unfeature_app(&self, id: &AppId) -> AppStoreResult<()> {
        // Update featured flag in metadata
        if let Some(mut app) = self.registry.get_chip_app(id) {
            app.metadata.featured = false;
            self.registry.update_chip_app(app)?;
        } else if let Some(mut app) = self.registry.get_full_app(id) {
            app.metadata.featured = false;
            self.registry.update_full_app(app)?;
        }

        self.registry.remove_featured(id);
        Ok(())
    }

    /// Suspend an app
    pub fn suspend_app(&self, id: &AppId, reason: &str) -> AppStoreResult<()> {
        if let Some(mut app) = self.registry.get_chip_app(id) {
            app.metadata.suspend(reason);
            self.registry.update_chip_app(app)?;
        } else if let Some(mut app) = self.registry.get_full_app(id) {
            app.metadata.suspend(reason);
            self.registry.update_full_app(app)?;
        } else {
            return Err(AppStoreError::AppNotFound(id.clone()));
        }
        Ok(())
    }

    // ==================== Stats ====================

    /// Get store statistics
    pub fn stats(&self) -> AppStoreStats {
        let registry_stats = self.registry.stats();
        AppStoreStats {
            total_apps: registry_stats.total_chip_apps + registry_stats.total_full_apps,
            chip_apps: registry_stats.total_chip_apps,
            full_apps: registry_stats.total_full_apps,
            publishers: registry_stats.total_publishers,
            featured: registry_stats.featured_count,
            categories: AppCategory::all().len(),
        }
    }
}

impl Default for AppStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// App metadata
    pub metadata: AppMetadata,
    /// Relevance score
    pub relevance_score: f64,
    /// Is this a chip app
    pub is_chip_app: bool,
}

/// Download tracking info
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DownloadInfo {
    /// Total downloads
    pub total_downloads: u64,
    /// Downloads today
    pub today_downloads: u64,
    /// Downloads this week
    pub week_downloads: u64,
    /// Downloads this month
    pub month_downloads: u64,
    /// Last download timestamp
    pub last_download: Option<chrono::DateTime<chrono::Utc>>,
}

impl DownloadInfo {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_download(&mut self) {
        self.total_downloads += 1;
        self.today_downloads += 1;
        self.week_downloads += 1;
        self.month_downloads += 1;
        self.last_download = Some(chrono::Utc::now());
    }
}

/// App store statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppStoreStats {
    pub total_apps: usize,
    pub chip_apps: usize,
    pub full_apps: usize,
    pub publishers: usize,
    pub featured: usize,
    pub categories: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chip::{ChipApp, templates};

    fn create_test_store() -> AppStore {
        AppStore::new()
    }

    fn create_test_chip_app(name: &str) -> ChipApp {
        let mut app = ChipApp::new(
            name.to_string(),
            format!("A test {} app", name),
            "test_publisher".to_string(),
            templates::hello_world_wasm(),
        ).unwrap();
        app.metadata.tags = vec!["test".to_string(), "demo".to_string()];
        app
    }

    #[test]
    fn test_publish_chip_app() {
        let store = create_test_store();
        let app = create_test_chip_app("Calculator");

        let id = store.publish_chip_app(app).unwrap();
        assert!(!id.is_empty());

        let retrieved = store.get_chip_app(&id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().metadata.name, "Calculator");
    }

    #[test]
    fn test_search() {
        let store = create_test_store();

        let app1 = create_test_chip_app("Calculator");
        let app2 = create_test_chip_app("Note Taker");
        let app3 = create_test_chip_app("Math Helper");

        store.publish_chip_app(app1).unwrap();
        store.publish_chip_app(app2).unwrap();
        store.publish_chip_app(app3).unwrap();

        let results = store.search("calculator", 10);
        assert!(!results.is_empty());
        assert_eq!(results[0].metadata.name, "Calculator");

        let results = store.search("math", 10);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_download() {
        let store = create_test_store();
        let app = create_test_chip_app("Download Test");
        let id = store.publish_chip_app(app).unwrap();

        let downloaded = store.download_chip_app(&id).unwrap();
        assert_eq!(downloaded.metadata.downloads, 1);

        let downloaded2 = store.download_chip_app(&id).unwrap();
        assert_eq!(downloaded2.metadata.downloads, 2);
    }

    #[test]
    fn test_rating() {
        let store = create_test_store();
        let app = create_test_chip_app("Rated App");
        let id = store.publish_chip_app(app).unwrap();

        store.rate_app(&id, 5.0).unwrap();
        let meta = store.get_metadata(&id).unwrap();
        assert_eq!(meta.rating, 5.0);

        store.rate_app(&id, 3.0).unwrap();
        let meta = store.get_metadata(&id).unwrap();
        assert_eq!(meta.rating, 4.0); // Average of 5 and 3
    }

    #[test]
    fn test_stats() {
        let store = create_test_store();

        for i in 0..5 {
            let app = create_test_chip_app(&format!("App {}", i));
            store.publish_chip_app(app).unwrap();
        }

        let stats = store.stats();
        assert_eq!(stats.chip_apps, 5);
        assert_eq!(stats.total_apps, 5);
        assert!(stats.categories > 0);
    }

    #[test]
    fn test_feature_app() {
        let store = create_test_store();
        let app = create_test_chip_app("Featured App");
        let id = store.publish_chip_app(app).unwrap();

        store.feature_app(&id, FeaturedInfo {
            position: 1,
            featured_from: chrono::Utc::now(),
            featured_until: None,
            banner_url: None,
            reason: Some("Great app!".to_string()),
        }).unwrap();

        let featured = store.get_featured();
        assert_eq!(featured.len(), 1);
        assert!(featured[0].featured);
    }
}
