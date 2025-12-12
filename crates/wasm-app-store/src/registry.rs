//! App registry - storage and retrieval of apps

use std::collections::HashMap;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use crate::apps::{AppId, AppMetadata, AppStatus, PublisherId};
use crate::chip::ChipApp;
use crate::full::FullApp;
use crate::category::AppCategory;
use crate::version::Version;
use crate::error::{AppStoreError, AppStoreResult};

/// App registry - thread-safe storage for apps
pub struct AppRegistry {
    /// Chip apps by ID
    pub chip_apps: DashMap<AppId, ChipApp>,
    /// Full apps by ID
    pub full_apps: DashMap<AppId, FullApp>,
    /// App versions (app_id -> list of versions)
    versions: DashMap<AppId, Vec<AppVersion>>,
    /// Slug to ID mapping
    slug_to_id: DashMap<String, AppId>,
    /// Apps by publisher
    publisher_apps: DashMap<PublisherId, Vec<AppId>>,
    /// Apps by category
    category_apps: DashMap<AppCategory, Vec<AppId>>,
    /// Featured apps
    featured_apps: DashMap<AppId, FeaturedInfo>,
}

impl AppRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        AppRegistry {
            chip_apps: DashMap::new(),
            full_apps: DashMap::new(),
            versions: DashMap::new(),
            slug_to_id: DashMap::new(),
            publisher_apps: DashMap::new(),
            category_apps: DashMap::new(),
            featured_apps: DashMap::new(),
        }
    }

    // ==================== Chip Apps ====================

    /// Register a chip app
    pub fn register_chip_app(&self, app: ChipApp) -> AppStoreResult<()> {
        let id = app.id.clone();
        let slug = app.metadata.slug.clone();
        let publisher_id = app.metadata.publisher_id.clone();
        let category = app.metadata.category;
        let version = app.metadata.version.clone();

        // Check for duplicate slug
        if self.slug_to_id.contains_key(&slug) {
            return Err(AppStoreError::AppAlreadyExists(slug));
        }

        // Store app
        self.chip_apps.insert(id.clone(), app);

        // Update indexes
        self.slug_to_id.insert(slug, id.clone());
        self.publisher_apps
            .entry(publisher_id)
            .or_insert_with(Vec::new)
            .push(id.clone());
        self.category_apps
            .entry(category)
            .or_insert_with(Vec::new)
            .push(id.clone());
        self.versions
            .entry(id.clone())
            .or_insert_with(Vec::new)
            .push(AppVersion::new(version, true));

        Ok(())
    }

    /// Get a chip app by ID
    pub fn get_chip_app(&self, id: &AppId) -> Option<ChipApp> {
        self.chip_apps.get(id).map(|r| r.clone())
    }

    /// Get a chip app by slug
    pub fn get_chip_app_by_slug(&self, slug: &str) -> Option<ChipApp> {
        self.slug_to_id
            .get(slug)
            .and_then(|id| self.chip_apps.get(&*id).map(|r| r.clone()))
    }

    /// Update a chip app
    pub fn update_chip_app(&self, app: ChipApp) -> AppStoreResult<()> {
        if !self.chip_apps.contains_key(&app.id) {
            return Err(AppStoreError::AppNotFound(app.id.clone()));
        }
        self.chip_apps.insert(app.id.clone(), app);
        Ok(())
    }

    /// Remove a chip app
    pub fn remove_chip_app(&self, id: &AppId) -> AppStoreResult<ChipApp> {
        let (_, app) = self.chip_apps
            .remove(id)
            .ok_or_else(|| AppStoreError::AppNotFound(id.clone()))?;

        // Clean up indexes
        self.slug_to_id.remove(&app.metadata.slug);
        if let Some(mut apps) = self.publisher_apps.get_mut(&app.metadata.publisher_id) {
            apps.retain(|a| a != id);
        }
        if let Some(mut apps) = self.category_apps.get_mut(&app.metadata.category) {
            apps.retain(|a| a != id);
        }
        self.versions.remove(id);

        Ok(app)
    }

    // ==================== Full Apps ====================

    /// Register a full app
    pub fn register_full_app(&self, app: FullApp) -> AppStoreResult<()> {
        let id = app.id.clone();
        let slug = app.metadata.slug.clone();
        let publisher_id = app.metadata.publisher_id.clone();
        let category = app.metadata.category;
        let version = app.metadata.version.clone();

        // Check for duplicate slug
        if self.slug_to_id.contains_key(&slug) {
            return Err(AppStoreError::AppAlreadyExists(slug));
        }

        // Store app
        self.full_apps.insert(id.clone(), app);

        // Update indexes
        self.slug_to_id.insert(slug, id.clone());
        self.publisher_apps
            .entry(publisher_id)
            .or_insert_with(Vec::new)
            .push(id.clone());
        self.category_apps
            .entry(category)
            .or_insert_with(Vec::new)
            .push(id.clone());
        self.versions
            .entry(id.clone())
            .or_insert_with(Vec::new)
            .push(AppVersion::new(version, true));

        Ok(())
    }

    /// Get a full app by ID
    pub fn get_full_app(&self, id: &AppId) -> Option<FullApp> {
        self.full_apps.get(id).map(|r| r.clone())
    }

    /// Get a full app by slug
    pub fn get_full_app_by_slug(&self, slug: &str) -> Option<FullApp> {
        self.slug_to_id
            .get(slug)
            .and_then(|id| self.full_apps.get(&*id).map(|r| r.clone()))
    }

    /// Update a full app
    pub fn update_full_app(&self, app: FullApp) -> AppStoreResult<()> {
        if !self.full_apps.contains_key(&app.id) {
            return Err(AppStoreError::AppNotFound(app.id.clone()));
        }
        self.full_apps.insert(app.id.clone(), app);
        Ok(())
    }

    /// Remove a full app
    pub fn remove_full_app(&self, id: &AppId) -> AppStoreResult<FullApp> {
        let (_, app) = self.full_apps
            .remove(id)
            .ok_or_else(|| AppStoreError::AppNotFound(id.clone()))?;

        // Clean up indexes
        self.slug_to_id.remove(&app.metadata.slug);
        if let Some(mut apps) = self.publisher_apps.get_mut(&app.metadata.publisher_id) {
            apps.retain(|a| a != id);
        }
        if let Some(mut apps) = self.category_apps.get_mut(&app.metadata.category) {
            apps.retain(|a| a != id);
        }
        self.versions.remove(id);

        Ok(app)
    }

    // ==================== Queries ====================

    /// Get app metadata by ID (either chip or full)
    pub fn get_metadata(&self, id: &AppId) -> Option<AppMetadata> {
        self.chip_apps
            .get(id)
            .map(|r| r.metadata.clone())
            .or_else(|| self.full_apps.get(id).map(|r| r.metadata.clone()))
    }

    /// Get all apps by publisher
    pub fn get_publisher_apps(&self, publisher_id: &PublisherId) -> Vec<AppMetadata> {
        self.publisher_apps
            .get(publisher_id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.get_metadata(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all apps in a category
    pub fn get_category_apps(&self, category: AppCategory) -> Vec<AppMetadata> {
        self.category_apps
            .get(&category)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.get_metadata(id))
                    .filter(|m| m.status == AppStatus::Published)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get featured apps
    pub fn get_featured_apps(&self) -> Vec<(AppMetadata, FeaturedInfo)> {
        self.featured_apps
            .iter()
            .filter_map(|entry| {
                self.get_metadata(entry.key())
                    .map(|meta| (meta, entry.value().clone()))
            })
            .collect()
    }

    /// Set app as featured
    pub fn set_featured(&self, id: &AppId, info: FeaturedInfo) -> AppStoreResult<()> {
        if !self.chip_apps.contains_key(id) && !self.full_apps.contains_key(id) {
            return Err(AppStoreError::AppNotFound(id.clone()));
        }
        self.featured_apps.insert(id.clone(), info);
        Ok(())
    }

    /// Remove featured status
    pub fn remove_featured(&self, id: &AppId) {
        self.featured_apps.remove(id);
    }

    /// Get version history for an app
    pub fn get_versions(&self, id: &AppId) -> Vec<AppVersion> {
        self.versions
            .get(id)
            .map(|v| v.clone())
            .unwrap_or_default()
    }

    /// Add a new version
    pub fn add_version(&self, id: &AppId, version: Version) -> AppStoreResult<()> {
        let mut versions = self.versions
            .entry(id.clone())
            .or_insert_with(Vec::new);

        // Check for duplicate version
        if versions.iter().any(|v| v.version == version) {
            return Err(AppStoreError::VersionConflict(format!(
                "Version {} already exists", version
            )));
        }

        // Mark current latest as not latest
        for v in versions.iter_mut() {
            v.is_latest = false;
        }

        versions.push(AppVersion::new(version, true));
        Ok(())
    }

    // ==================== Stats ====================

    /// Get registry statistics
    pub fn stats(&self) -> RegistryStats {
        RegistryStats {
            total_chip_apps: self.chip_apps.len(),
            total_full_apps: self.full_apps.len(),
            total_publishers: self.publisher_apps.len(),
            featured_count: self.featured_apps.len(),
        }
    }

    /// Get total app count
    pub fn total_apps(&self) -> usize {
        self.chip_apps.len() + self.full_apps.len()
    }
}

impl Default for AppRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// App version info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppVersion {
    /// Version
    pub version: Version,
    /// Is this the latest version
    pub is_latest: bool,
    /// Release timestamp
    pub released_at: chrono::DateTime<chrono::Utc>,
    /// Download count for this version
    pub downloads: u64,
    /// Is this version deprecated
    pub deprecated: bool,
}

impl AppVersion {
    pub fn new(version: Version, is_latest: bool) -> Self {
        AppVersion {
            version,
            is_latest,
            released_at: chrono::Utc::now(),
            downloads: 0,
            deprecated: false,
        }
    }
}

/// Featured app information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeaturedInfo {
    /// Position in featured list
    pub position: u32,
    /// Featured from date
    pub featured_from: chrono::DateTime<chrono::Utc>,
    /// Featured until date
    pub featured_until: Option<chrono::DateTime<chrono::Utc>>,
    /// Featured banner URL
    pub banner_url: Option<String>,
    /// Featured reason/description
    pub reason: Option<String>,
}

/// Registry statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryStats {
    pub total_chip_apps: usize,
    pub total_full_apps: usize,
    pub total_publishers: usize,
    pub featured_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chip::{ChipApp, templates};

    fn create_test_chip_app(name: &str) -> ChipApp {
        ChipApp::new(
            name.to_string(),
            format!("Test app: {}", name),
            "test_publisher".to_string(),
            templates::hello_world_wasm(),
        ).unwrap()
    }

    #[test]
    fn test_register_chip_app() {
        let registry = AppRegistry::new();
        let app = create_test_chip_app("Test App");
        let id = app.id.clone();

        registry.register_chip_app(app).unwrap();

        assert!(registry.get_chip_app(&id).is_some());
        assert_eq!(registry.stats().total_chip_apps, 1);
    }

    #[test]
    fn test_get_by_slug() {
        let registry = AppRegistry::new();
        let app = create_test_chip_app("My Cool App");
        registry.register_chip_app(app).unwrap();

        let retrieved = registry.get_chip_app_by_slug("my-cool-app");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().metadata.name, "My Cool App");
    }

    #[test]
    fn test_duplicate_slug() {
        let registry = AppRegistry::new();
        let app1 = create_test_chip_app("Test App");
        let app2 = create_test_chip_app("Test App"); // Same name = same slug

        registry.register_chip_app(app1).unwrap();
        let result = registry.register_chip_app(app2);

        assert!(matches!(result, Err(AppStoreError::AppAlreadyExists(_))));
    }

    #[test]
    fn test_publisher_apps() {
        let registry = AppRegistry::new();

        let app1 = create_test_chip_app("App 1");
        let app2 = create_test_chip_app("App 2");

        registry.register_chip_app(app1).unwrap();
        registry.register_chip_app(app2).unwrap();

        let publisher_apps = registry.get_publisher_apps(&"test_publisher".to_string());
        assert_eq!(publisher_apps.len(), 2);
    }

    #[test]
    fn test_category_apps() {
        let registry = AppRegistry::new();

        let mut app = create_test_chip_app("Math App");
        app.metadata.category = AppCategory::Math;
        app.metadata.status = AppStatus::Published;
        registry.register_chip_app(app).unwrap();

        let math_apps = registry.get_category_apps(AppCategory::Math);
        assert_eq!(math_apps.len(), 1);
    }

    #[test]
    fn test_remove_app() {
        let registry = AppRegistry::new();
        let app = create_test_chip_app("To Remove");
        let id = app.id.clone();
        let slug = app.metadata.slug.clone();

        registry.register_chip_app(app).unwrap();
        assert!(registry.get_chip_app(&id).is_some());

        registry.remove_chip_app(&id).unwrap();
        assert!(registry.get_chip_app(&id).is_none());
        assert!(registry.get_chip_app_by_slug(&slug).is_none());
    }

    #[test]
    fn test_version_management() {
        let registry = AppRegistry::new();
        let app = create_test_chip_app("Versioned App");
        let id = app.id.clone();

        registry.register_chip_app(app).unwrap();

        // Add new versions
        registry.add_version(&id, Version::new(0, 2, 0)).unwrap();
        registry.add_version(&id, Version::new(0, 3, 0)).unwrap();

        let versions = registry.get_versions(&id);
        assert_eq!(versions.len(), 3);

        // Only latest should be marked as latest
        let latest_count = versions.iter().filter(|v| v.is_latest).count();
        assert_eq!(latest_count, 1);
    }
}
