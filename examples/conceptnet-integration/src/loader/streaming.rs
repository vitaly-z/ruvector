//! Streaming Data Loading
//!
//! For when you don't want to load everything into memory at once.

use crate::api::Edge;
use std::io::{BufRead, BufReader};
use std::fs::File;
use flate2::read::GzDecoder;

/// Streaming loader for processing large ConceptNet dumps
pub struct StreamingLoader {
    reader: Box<dyn BufRead>,
    buffer: String,
    line_count: usize,
}

impl StreamingLoader {
    /// Create from file path
    pub fn from_file(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader: Box<dyn BufRead> = if path.ends_with(".gz") {
            Box::new(BufReader::with_capacity(4 * 1024 * 1024, GzDecoder::new(file)))
        } else {
            Box::new(BufReader::with_capacity(4 * 1024 * 1024, file))
        };

        Ok(Self {
            reader,
            buffer: String::with_capacity(4096),
            line_count: 0,
        })
    }

    /// Get line count processed so far
    pub fn line_count(&self) -> usize {
        self.line_count
    }
}

impl Iterator for StreamingLoader {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        self.buffer.clear();
        match self.reader.read_line(&mut self.buffer) {
            Ok(0) => None,
            Ok(_) => {
                self.line_count += 1;
                Some(self.buffer.trim().to_string())
            }
            Err(_) => None,
        }
    }
}

/// Batch processor for controlled memory usage
pub struct BatchProcessor<T> {
    items: Vec<T>,
    batch_size: usize,
}

impl<T> BatchProcessor<T> {
    pub fn new(batch_size: usize) -> Self {
        Self {
            items: Vec::with_capacity(batch_size),
            batch_size,
        }
    }

    pub fn add(&mut self, item: T) -> Option<Vec<T>> {
        self.items.push(item);
        if self.items.len() >= self.batch_size {
            Some(std::mem::take(&mut self.items))
        } else {
            None
        }
    }

    pub fn flush(self) -> Vec<T> {
        self.items
    }
}
