//! WASM bindings for wasm-app-store
//!
//! This crate provides WebAssembly bindings for the app store, enabling
//! browser-based app discovery, download, and execution.

use wasm_bindgen::prelude::*;
use serde_wasm_bindgen::{to_value, from_value};

use wasm_app_store::{
    AppStore, AppStoreConfig,
    ChipApp, ChipAppBuilder,
    FullApp, FullAppBuilder,
    AppMetadata, AppPricing,
    AppCategory, Platform,
    Version,
    registry::FeaturedInfo,
};

/// Initialize panic hook for better error messages
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// WASM App Store wrapper
#[wasm_bindgen]
pub struct WasmAppStore {
    store: AppStore,
}

#[wasm_bindgen]
impl WasmAppStore {
    /// Create a new app store
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        WasmAppStore {
            store: AppStore::new(),
        }
    }

    /// Create with custom configuration
    #[wasm_bindgen(js_name = withConfig)]
    pub fn with_config(config: JsValue) -> Result<WasmAppStore, JsValue> {
        let config: AppStoreConfig = from_value(config)?;
        Ok(WasmAppStore {
            store: AppStore::with_config(config),
        })
    }

    // ==================== Publishing ====================

    /// Publish a chip app from WASM bytes
    #[wasm_bindgen(js_name = publishChipApp)]
    pub fn publish_chip_app(
        &self,
        name: String,
        description: String,
        publisher_id: String,
        wasm_bytes: Vec<u8>,
        tags: Option<Vec<String>>,
    ) -> Result<String, JsValue> {
        let mut app = ChipApp::new(name, description, publisher_id, wasm_bytes)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        if let Some(tags) = tags {
            app.metadata.tags = tags;
        }

        self.store.publish_chip_app(app)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Publish a chip app with full options
    #[wasm_bindgen(js_name = publishChipAppFull)]
    pub fn publish_chip_app_full(&self, options: JsValue) -> Result<String, JsValue> {
        let opts: ChipAppOptions = from_value(options)?;

        let mut builder = ChipAppBuilder::new()
            .name(opts.name)
            .description(opts.description)
            .publisher(opts.publisher_id)
            .wasm(opts.wasm_bytes);

        if let Some(category) = opts.category {
            builder = builder.category(parse_category(&category)?);
        }

        for tag in opts.tags.unwrap_or_default() {
            builder = builder.tag(tag);
        }

        if let Some(pricing) = opts.pricing {
            builder = builder.pricing(parse_pricing(pricing)?);
        }

        let app = builder.build()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        self.store.publish_chip_app(app)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Publish a full app
    #[wasm_bindgen(js_name = publishFullApp)]
    pub fn publish_full_app(
        &self,
        name: String,
        description: String,
        publisher_id: String,
        wasm_bytes: Vec<u8>,
    ) -> Result<String, JsValue> {
        let app = FullApp::new(name, description, publisher_id, wasm_bytes)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        self.store.publish_full_app(app)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    // ==================== Discovery ====================

    /// Search for apps
    #[wasm_bindgen]
    pub fn search(&self, query: String, limit: usize) -> Result<JsValue, JsValue> {
        let results = self.store.search(&query, limit);
        to_value(&results).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Browse category
    #[wasm_bindgen(js_name = browseCategory)]
    pub fn browse_category(
        &self,
        category: String,
        limit: usize,
        offset: usize,
    ) -> Result<JsValue, JsValue> {
        let cat = parse_category(&category)?;
        let apps = self.store.browse_category(cat, limit, offset);
        to_value(&apps).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get featured apps
    #[wasm_bindgen(js_name = getFeatured)]
    pub fn get_featured(&self) -> Result<JsValue, JsValue> {
        let apps = self.store.get_featured();
        to_value(&apps).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get trending apps
    #[wasm_bindgen(js_name = getTrending)]
    pub fn get_trending(&self, limit: usize) -> Result<JsValue, JsValue> {
        let apps = self.store.get_trending(limit);
        to_value(&apps).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get new apps
    #[wasm_bindgen(js_name = getNew)]
    pub fn get_new(&self, limit: usize) -> Result<JsValue, JsValue> {
        let apps = self.store.get_new(limit);
        to_value(&apps).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get publisher apps
    #[wasm_bindgen(js_name = getPublisherApps)]
    pub fn get_publisher_apps(&self, publisher_id: String) -> Result<JsValue, JsValue> {
        let apps = self.store.get_publisher_apps(&publisher_id);
        to_value(&apps).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    // ==================== Retrieval ====================

    /// Get app metadata
    #[wasm_bindgen(js_name = getMetadata)]
    pub fn get_metadata(&self, app_id: String) -> Result<JsValue, JsValue> {
        let metadata = self.store.get_metadata(&app_id)
            .ok_or_else(|| JsValue::from_str("App not found"))?;
        to_value(&metadata).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Download chip app (returns WASM bytes)
    #[wasm_bindgen(js_name = downloadChipApp)]
    pub fn download_chip_app(&self, app_id: String) -> Result<Vec<u8>, JsValue> {
        let app = self.store.download_chip_app(&app_id)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(app.wasm)
    }

    /// Download chip app compressed
    #[wasm_bindgen(js_name = downloadChipAppCompressed)]
    pub fn download_chip_app_compressed(&self, app_id: String) -> Result<Vec<u8>, JsValue> {
        let app = self.store.download_chip_app(&app_id)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(app.wasm_compressed)
    }

    /// Download chip app as distribution bytes
    #[wasm_bindgen(js_name = downloadChipAppDistribution)]
    pub fn download_chip_app_distribution(&self, app_id: String) -> Result<Vec<u8>, JsValue> {
        let app = self.store.download_chip_app(&app_id)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        app.to_distribution_bytes()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Download full app main module
    #[wasm_bindgen(js_name = downloadFullApp)]
    pub fn download_full_app(&self, app_id: String) -> Result<Vec<u8>, JsValue> {
        let app = self.store.download_full_app(&app_id)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(app.main_module.wasm)
    }

    // ==================== Ratings ====================

    /// Rate an app
    #[wasm_bindgen(js_name = rateApp)]
    pub fn rate_app(&self, app_id: String, rating: f32) -> Result<(), JsValue> {
        self.store.rate_app(&app_id, rating)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    // ==================== Admin ====================

    /// Feature an app
    #[wasm_bindgen(js_name = featureApp)]
    pub fn feature_app(&self, app_id: String, position: u32) -> Result<(), JsValue> {
        let info = FeaturedInfo {
            position,
            featured_from: chrono::Utc::now(),
            featured_until: None,
            banner_url: None,
            reason: None,
        };
        self.store.feature_app(&app_id, info)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Unfeature an app
    #[wasm_bindgen(js_name = unfeatureApp)]
    pub fn unfeature_app(&self, app_id: String) -> Result<(), JsValue> {
        self.store.unfeature_app(&app_id)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Suspend an app
    #[wasm_bindgen(js_name = suspendApp)]
    pub fn suspend_app(&self, app_id: String, reason: String) -> Result<(), JsValue> {
        self.store.suspend_app(&app_id, &reason)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Unpublish an app
    #[wasm_bindgen]
    pub fn unpublish(&self, app_id: String) -> Result<(), JsValue> {
        self.store.unpublish(&app_id)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    // ==================== Stats ====================

    /// Get store statistics
    #[wasm_bindgen(js_name = getStats)]
    pub fn get_stats(&self) -> Result<JsValue, JsValue> {
        let stats = self.store.stats();
        to_value(&stats).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl Default for WasmAppStore {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== Helper Types ====================

#[derive(serde::Deserialize)]
struct ChipAppOptions {
    name: String,
    description: String,
    publisher_id: String,
    wasm_bytes: Vec<u8>,
    category: Option<String>,
    tags: Option<Vec<String>>,
    pricing: Option<PricingOptions>,
}

#[derive(serde::Deserialize)]
#[serde(tag = "type")]
enum PricingOptions {
    #[serde(rename = "free")]
    Free,
    #[serde(rename = "oneTime")]
    OneTime { price: u64 },
    #[serde(rename = "payPerUse")]
    PayPerUse { price_per_use: u64 },
    #[serde(rename = "subscription")]
    Subscription { monthly_price: u64 },
    #[serde(rename = "freemium")]
    Freemium { free_uses_per_day: u32, price_per_use: u64 },
}

fn parse_category(s: &str) -> Result<AppCategory, JsValue> {
    match s.to_lowercase().as_str() {
        "utilities" => Ok(AppCategory::Utilities),
        "data_processing" | "dataprocessing" => Ok(AppCategory::DataProcessing),
        "ai_ml" | "aiml" | "ai" | "ml" => Ok(AppCategory::AiMl),
        "crypto" => Ok(AppCategory::Crypto),
        "media" => Ok(AppCategory::Media),
        "text" => Ok(AppCategory::Text),
        "math" => Ok(AppCategory::Math),
        "games" => Ok(AppCategory::Games),
        "finance" => Ok(AppCategory::Finance),
        "dev_tools" | "devtools" => Ok(AppCategory::DevTools),
        "web" => Ok(AppCategory::Web),
        "embedded" => Ok(AppCategory::Embedded),
        "education" => Ok(AppCategory::Education),
        "social" => Ok(AppCategory::Social),
        "health" => Ok(AppCategory::Health),
        "other" => Ok(AppCategory::Other),
        _ => Err(JsValue::from_str(&format!("Unknown category: {}", s))),
    }
}

fn parse_pricing(p: PricingOptions) -> Result<AppPricing, JsValue> {
    Ok(match p {
        PricingOptions::Free => AppPricing::Free,
        PricingOptions::OneTime { price } => AppPricing::OneTime { price },
        PricingOptions::PayPerUse { price_per_use } => AppPricing::PayPerUse { price_per_use },
        PricingOptions::Subscription { monthly_price } => AppPricing::Subscription {
            monthly_price,
            annual_price: monthly_price * 10,
        },
        PricingOptions::Freemium { free_uses_per_day, price_per_use } => AppPricing::Freemium {
            free_uses_per_day,
            price_per_use,
        },
    })
}

// ==================== Utility Functions ====================

/// Get all categories
#[wasm_bindgen(js_name = getAllCategories)]
pub fn get_all_categories() -> Result<JsValue, JsValue> {
    let categories: Vec<_> = AppCategory::all()
        .into_iter()
        .map(|c| CategoryInfo {
            id: format!("{:?}", c).to_lowercase(),
            name: c.display_name().to_string(),
            icon: c.icon().to_string(),
        })
        .collect();
    to_value(&categories).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[derive(serde::Serialize)]
struct CategoryInfo {
    id: String,
    name: String,
    icon: String,
}

/// Parse version string
#[wasm_bindgen(js_name = parseVersion)]
pub fn parse_version(version: String) -> Result<JsValue, JsValue> {
    let v = Version::parse(&version)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    to_value(&v).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Load chip app from distribution bytes
#[wasm_bindgen(js_name = loadChipAppFromBytes)]
pub fn load_chip_app_from_bytes(bytes: Vec<u8>) -> Result<JsValue, JsValue> {
    let app = ChipApp::from_distribution_bytes(&bytes)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    to_value(&app.metadata).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Get WASM from distribution bytes
#[wasm_bindgen(js_name = getWasmFromDistribution)]
pub fn get_wasm_from_distribution(bytes: Vec<u8>) -> Result<Vec<u8>, JsValue> {
    let app = ChipApp::from_distribution_bytes(&bytes)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(app.wasm)
}
