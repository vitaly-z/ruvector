//! ConceptNet Assertions CSV Loader
//!
//! Loads the full ConceptNet assertions dump efficiently using:
//! - Memory-mapped I/O for large files
//! - Streaming decompression for .gz files
//! - Parallel parsing with rayon
//! - Batch processing to control memory usage

use crate::api::{Edge, ConceptNode, Relation, RelationType, Source};
use crate::graph::ConceptNetGraphBuilder;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use flate2::read::GzDecoder;
use rayon::prelude::*;

/// Configuration for loading assertions
#[derive(Debug, Clone)]
pub struct AssertionsConfig {
    /// Path to assertions CSV file (supports .csv and .csv.gz)
    pub file_path: String,
    /// Languages to include (empty = all)
    pub languages: Vec<String>,
    /// Minimum edge weight to include
    pub min_weight: f64,
    /// Relations to include (empty = all)
    pub relations: Vec<String>,
    /// Maximum edges to load (0 = unlimited)
    pub max_edges: usize,
    /// Batch size for processing
    pub batch_size: usize,
    /// Number of parallel workers
    pub num_workers: usize,
    /// Skip edges with negative weight
    pub skip_negative_weights: bool,
    /// Deduplicate edges
    pub deduplicate: bool,
}

impl Default for AssertionsConfig {
    fn default() -> Self {
        Self {
            file_path: String::new(),
            languages: vec!["en".to_string()],
            min_weight: 1.0,
            relations: vec![], // All relations
            max_edges: 0,      // Unlimited
            batch_size: 100_000,
            num_workers: num_cpus::get(),
            skip_negative_weights: true,
            deduplicate: true,
        }
    }
}

/// Progress tracking for loading
#[derive(Debug, Clone)]
pub struct LoadProgress {
    pub lines_processed: usize,
    pub edges_loaded: usize,
    pub edges_skipped: usize,
    pub errors: usize,
    pub elapsed_ms: u64,
    pub estimated_total: Option<usize>,
}

impl LoadProgress {
    pub fn rate(&self) -> f64 {
        if self.elapsed_ms > 0 {
            self.lines_processed as f64 / (self.elapsed_ms as f64 / 1000.0)
        } else {
            0.0
        }
    }

    pub fn eta_seconds(&self) -> Option<f64> {
        if let Some(total) = self.estimated_total {
            let rate = self.rate();
            if rate > 0.0 {
                let remaining = total.saturating_sub(self.lines_processed);
                return Some(remaining as f64 / rate);
            }
        }
        None
    }
}

/// Loader for ConceptNet assertions CSV
pub struct AssertionsLoader {
    config: AssertionsConfig,
    progress: Arc<LoadProgress>,
    lines_counter: Arc<AtomicUsize>,
    edges_counter: Arc<AtomicUsize>,
    skip_counter: Arc<AtomicUsize>,
    error_counter: Arc<AtomicUsize>,
}

impl AssertionsLoader {
    /// Create a new loader with configuration
    pub fn new(config: AssertionsConfig) -> Self {
        Self {
            config,
            progress: Arc::new(LoadProgress {
                lines_processed: 0,
                edges_loaded: 0,
                edges_skipped: 0,
                errors: 0,
                elapsed_ms: 0,
                estimated_total: None,
            }),
            lines_counter: Arc::new(AtomicUsize::new(0)),
            edges_counter: Arc::new(AtomicUsize::new(0)),
            skip_counter: Arc::new(AtomicUsize::new(0)),
            error_counter: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Load assertions into a graph builder
    pub fn load(&self, builder: &mut ConceptNetGraphBuilder) -> Result<LoadProgress, LoadError> {
        let start = std::time::Instant::now();
        let path = Path::new(&self.config.file_path);

        if !path.exists() {
            return Err(LoadError::FileNotFound(self.config.file_path.clone()));
        }

        let file = File::open(path)?;
        let reader: Box<dyn BufRead> = if self.config.file_path.ends_with(".gz") {
            Box::new(BufReader::with_capacity(
                8 * 1024 * 1024, // 8MB buffer
                GzDecoder::new(file),
            ))
        } else {
            Box::new(BufReader::with_capacity(8 * 1024 * 1024, file))
        };

        // Process lines in batches
        let mut batch = Vec::with_capacity(self.config.batch_size);
        let mut seen_edges: HashSet<u64> = if self.config.deduplicate {
            HashSet::with_capacity(1_000_000)
        } else {
            HashSet::new()
        };

        let languages: HashSet<&str> = self.config.languages.iter().map(|s| s.as_str()).collect();
        let relations: HashSet<&str> = self.config.relations.iter().map(|s| s.as_str()).collect();

        for line_result in reader.lines() {
            let line = match line_result {
                Ok(l) => l,
                Err(_) => {
                    self.error_counter.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
            };

            self.lines_counter.fetch_add(1, Ordering::Relaxed);

            // Check max edges limit
            if self.config.max_edges > 0
                && self.edges_counter.load(Ordering::Relaxed) >= self.config.max_edges
            {
                break;
            }

            batch.push(line);

            if batch.len() >= self.config.batch_size {
                self.process_batch(&mut batch, builder, &languages, &relations, &mut seen_edges)?;
            }
        }

        // Process remaining
        if !batch.is_empty() {
            self.process_batch(&mut batch, builder, &languages, &relations, &mut seen_edges)?;
        }

        Ok(LoadProgress {
            lines_processed: self.lines_counter.load(Ordering::Relaxed),
            edges_loaded: self.edges_counter.load(Ordering::Relaxed),
            edges_skipped: self.skip_counter.load(Ordering::Relaxed),
            errors: self.error_counter.load(Ordering::Relaxed),
            elapsed_ms: start.elapsed().as_millis() as u64,
            estimated_total: None,
        })
    }

    fn process_batch(
        &self,
        batch: &mut Vec<String>,
        builder: &mut ConceptNetGraphBuilder,
        languages: &HashSet<&str>,
        relations: &HashSet<&str>,
        seen_edges: &mut HashSet<u64>,
    ) -> Result<(), LoadError> {
        // Parse lines in parallel
        let edges: Vec<Option<Edge>> = batch
            .par_iter()
            .map(|line| self.parse_line(line, languages, relations))
            .collect();

        // Add edges to builder (single-threaded for thread safety)
        for edge_opt in edges {
            if let Some(edge) = edge_opt {
                // Deduplicate
                if self.config.deduplicate {
                    let hash = self.edge_hash(&edge);
                    if seen_edges.contains(&hash) {
                        self.skip_counter.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    seen_edges.insert(hash);
                }

                match builder.add_edge(&edge) {
                    Ok(_) => {
                        self.edges_counter.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(_) => {
                        self.skip_counter.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }

        batch.clear();
        Ok(())
    }

    fn parse_line(
        &self,
        line: &str,
        languages: &HashSet<&str>,
        relations: &HashSet<&str>,
    ) -> Option<Edge> {
        // ConceptNet CSV format:
        // URI\tREL\tSTART\tEND\tJSON_DATA
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 5 {
            return None;
        }

        let uri = parts[0];
        let rel = parts[1];
        let start = parts[2];
        let end = parts[3];
        let json_str = parts[4];

        // Parse JSON for weight and other metadata
        let json: serde_json::Value = serde_json::from_str(json_str).ok()?;
        let weight = json.get("weight")?.as_f64()?;

        // Apply filters
        if self.config.skip_negative_weights && weight < 0.0 {
            return None;
        }
        if weight < self.config.min_weight {
            return None;
        }

        // Filter by relation
        if !relations.is_empty() && !relations.contains(rel) {
            return None;
        }

        // Filter by language
        let start_lang = Self::extract_language(start)?;
        let end_lang = Self::extract_language(end)?;
        if !languages.is_empty() && (!languages.contains(start_lang) || !languages.contains(end_lang))
        {
            return None;
        }

        // Extract surface text
        let surface_text = json.get("surfaceText").and_then(|v| v.as_str()).map(|s| s.to_string());

        // Extract sources
        let sources = json
            .get("sources")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|s| s.as_str().map(|s| Source {
                        id: Some(s.to_string()),
                        contributor: None,
                        activity: None,
                        process: None,
                    }))
                    .collect()
            })
            .unwrap_or_default();

        Some(Edge {
            id: uri.to_string(),
            rel: Relation {
                id: rel.to_string(),
                label: Some(Self::extract_label(rel).to_string()),
            },
            start: ConceptNode {
                id: start.to_string(),
                label: Some(Self::extract_term(start).to_string()),
                language: Some(start_lang.to_string()),
                term: Some(Self::extract_term(start).to_string()),
                sense_label: None,
            },
            end: ConceptNode {
                id: end.to_string(),
                label: Some(Self::extract_term(end).to_string()),
                language: Some(end_lang.to_string()),
                term: Some(Self::extract_term(end).to_string()),
                sense_label: None,
            },
            weight,
            surface_text,
            license: json.get("license").and_then(|v| v.as_str()).map(|s| s.to_string()),
            dataset: json.get("dataset").and_then(|v| v.as_str()).map(|s| s.to_string()),
            sources,
        })
    }

    fn extract_language(uri: &str) -> Option<&str> {
        // /c/en/dog -> en
        let parts: Vec<&str> = uri.split('/').collect();
        if parts.len() >= 3 {
            Some(parts[2])
        } else {
            None
        }
    }

    fn extract_term(uri: &str) -> &str {
        // /c/en/dog -> dog
        uri.rsplit('/').next().unwrap_or(uri)
    }

    fn extract_label(rel: &str) -> &str {
        // /r/IsA -> IsA
        rel.rsplit('/').next().unwrap_or(rel)
    }

    fn edge_hash(&self, edge: &Edge) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        edge.start.id.hash(&mut hasher);
        edge.rel.id.hash(&mut hasher);
        edge.end.id.hash(&mut hasher);
        hasher.finish()
    }

    /// Get current progress
    pub fn progress(&self) -> LoadProgress {
        LoadProgress {
            lines_processed: self.lines_counter.load(Ordering::Relaxed),
            edges_loaded: self.edges_counter.load(Ordering::Relaxed),
            edges_skipped: self.skip_counter.load(Ordering::Relaxed),
            errors: self.error_counter.load(Ordering::Relaxed),
            elapsed_ms: 0,
            estimated_total: None,
        }
    }
}

/// Errors during loading
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("File not found: {0}")]
    FileNotFound(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
}

/// Download ConceptNet assertions (helper function)
pub async fn download_assertions(output_path: &str) -> Result<(), LoadError> {
    const URL: &str = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz";

    println!("Downloading ConceptNet assertions from: {}", URL);
    println!("This is a ~1GB download and may take a while...");

    let response = reqwest::get(URL).await.map_err(|e| LoadError::Parse(e.to_string()))?;
    let bytes = response.bytes().await.map_err(|e| LoadError::Parse(e.to_string()))?;

    std::fs::write(output_path, bytes)?;
    println!("Downloaded to: {}", output_path);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_language() {
        assert_eq!(AssertionsLoader::extract_language("/c/en/dog"), Some("en"));
        assert_eq!(AssertionsLoader::extract_language("/c/es/perro"), Some("es"));
    }

    #[test]
    fn test_extract_term() {
        assert_eq!(AssertionsLoader::extract_term("/c/en/artificial_intelligence"), "artificial_intelligence");
    }
}
