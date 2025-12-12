//! Transaction processing for the payment system

use serde::{Deserialize, Serialize};
use crate::types::{AccountId, TransactionId, AppId, Credits, Timestamp, generate_id};
use crate::error::{PaymentError, PaymentResult};

/// Transaction status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransactionStatus {
    /// Transaction is pending
    Pending,
    /// Transaction is processing
    Processing,
    /// Transaction completed successfully
    Completed,
    /// Transaction failed
    Failed,
    /// Transaction was cancelled
    Cancelled,
    /// Transaction was refunded
    Refunded,
    /// Transaction is disputed
    Disputed,
}

/// Transaction type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransactionType {
    /// Credit purchase
    Purchase,
    /// App payment
    AppPayment,
    /// Subscription payment
    Subscription,
    /// Transfer between accounts
    Transfer,
    /// Refund
    Refund,
    /// Revenue payout to developer
    Payout,
    /// Bonus/promotional credit
    Bonus,
    /// Micropayment for chip app
    Micropayment,
}

/// Payment transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    /// Unique transaction ID
    pub id: TransactionId,
    /// Transaction type
    pub transaction_type: TransactionType,
    /// Source account ID
    pub from_account: AccountId,
    /// Destination account ID (for transfers/payouts)
    pub to_account: Option<AccountId>,
    /// Amount in credits
    pub amount: Credits,
    /// Platform fee (if applicable)
    pub platform_fee: Credits,
    /// Net amount after fees
    pub net_amount: Credits,
    /// Related app ID (for app payments)
    pub app_id: Option<AppId>,
    /// Transaction status
    pub status: TransactionStatus,
    /// Description
    pub description: String,
    /// Idempotency key (prevent duplicates)
    pub idempotency_key: Option<String>,
    /// Digital signature (for blockchain integration)
    pub signature: Option<String>,
    /// Metadata
    pub metadata: Option<serde_json::Value>,
    /// Created timestamp
    pub created_at: Timestamp,
    /// Updated timestamp
    pub updated_at: Timestamp,
    /// Completed timestamp
    pub completed_at: Option<Timestamp>,
}

impl Transaction {
    /// Create a new transaction
    pub fn new(
        transaction_type: TransactionType,
        from_account: AccountId,
        amount: Credits,
        description: String,
    ) -> Self {
        Transaction {
            id: generate_id(),
            transaction_type,
            from_account,
            to_account: None,
            amount,
            platform_fee: 0,
            net_amount: amount,
            app_id: None,
            status: TransactionStatus::Pending,
            description,
            idempotency_key: None,
            signature: None,
            metadata: None,
            created_at: Timestamp::now(),
            updated_at: Timestamp::now(),
            completed_at: None,
        }
    }

    /// Create a transfer transaction
    pub fn transfer(
        from_account: AccountId,
        to_account: AccountId,
        amount: Credits,
        description: String,
    ) -> Self {
        let mut tx = Self::new(TransactionType::Transfer, from_account, amount, description);
        tx.to_account = Some(to_account);
        tx
    }

    /// Create an app payment transaction
    pub fn app_payment(
        from_account: AccountId,
        to_account: AccountId,
        app_id: AppId,
        amount: Credits,
        platform_fee_percentage: u8,
    ) -> Self {
        let platform_fee = (amount * platform_fee_percentage as u64) / 100;
        let net_amount = amount.saturating_sub(platform_fee);

        let mut tx = Self::new(
            TransactionType::AppPayment,
            from_account,
            amount,
            format!("Payment for app: {}", app_id),
        );
        tx.to_account = Some(to_account);
        tx.app_id = Some(app_id);
        tx.platform_fee = platform_fee;
        tx.net_amount = net_amount;
        tx
    }

    /// Create a micropayment transaction for chip apps
    pub fn micropayment(
        from_account: AccountId,
        to_account: AccountId,
        app_id: AppId,
        amount: Credits,
    ) -> Self {
        // Micropayments have reduced platform fees (10% instead of 25%)
        let platform_fee = (amount * 10) / 100;
        let net_amount = amount.saturating_sub(platform_fee);

        let mut tx = Self::new(
            TransactionType::Micropayment,
            from_account,
            amount,
            format!("Micropayment for chip app: {}", app_id),
        );
        tx.to_account = Some(to_account);
        tx.app_id = Some(app_id);
        tx.platform_fee = platform_fee;
        tx.net_amount = net_amount;
        tx
    }

    /// Set idempotency key
    pub fn with_idempotency_key(mut self, key: String) -> Self {
        self.idempotency_key = Some(key);
        self
    }

    /// Set signature
    pub fn with_signature(mut self, signature: String) -> Self {
        self.signature = Some(signature);
        self
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Mark transaction as processing
    pub fn start_processing(&mut self) {
        self.status = TransactionStatus::Processing;
        self.updated_at = Timestamp::now();
    }

    /// Complete the transaction
    pub fn complete(&mut self) {
        self.status = TransactionStatus::Completed;
        self.updated_at = Timestamp::now();
        self.completed_at = Some(Timestamp::now());
    }

    /// Fail the transaction
    pub fn fail(&mut self, reason: &str) {
        self.status = TransactionStatus::Failed;
        self.updated_at = Timestamp::now();
        let mut meta = self.metadata.clone().unwrap_or(serde_json::json!({}));
        meta["failure_reason"] = serde_json::json!(reason);
        self.metadata = Some(meta);
    }

    /// Cancel the transaction
    pub fn cancel(&mut self) {
        self.status = TransactionStatus::Cancelled;
        self.updated_at = Timestamp::now();
    }

    /// Refund the transaction
    pub fn refund(&mut self) -> PaymentResult<Transaction> {
        if self.status != TransactionStatus::Completed {
            return Err(PaymentError::RefundNotAllowed(
                "Can only refund completed transactions".to_string(),
            ));
        }

        self.status = TransactionStatus::Refunded;
        self.updated_at = Timestamp::now();

        // Create refund transaction
        let mut refund_tx = Transaction::new(
            TransactionType::Refund,
            self.to_account.clone().unwrap_or(self.from_account.clone()),
            self.amount,
            format!("Refund for transaction: {}", self.id),
        );
        refund_tx.to_account = Some(self.from_account.clone());
        refund_tx.app_id = self.app_id.clone();

        let mut meta = serde_json::json!({});
        meta["original_transaction_id"] = serde_json::json!(self.id);
        refund_tx.metadata = Some(meta);

        Ok(refund_tx)
    }

    /// Check if transaction can be refunded
    pub fn is_refundable(&self) -> bool {
        self.status == TransactionStatus::Completed
            && matches!(
                self.transaction_type,
                TransactionType::AppPayment | TransactionType::Micropayment | TransactionType::Purchase
            )
    }
}

/// Transaction batch for atomic processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionBatch {
    /// Batch ID
    pub id: String,
    /// Transactions in the batch
    pub transactions: Vec<Transaction>,
    /// Total amount
    pub total_amount: Credits,
    /// Batch status
    pub status: TransactionStatus,
    /// Created timestamp
    pub created_at: Timestamp,
}

impl TransactionBatch {
    /// Create a new transaction batch
    pub fn new() -> Self {
        TransactionBatch {
            id: generate_id(),
            transactions: Vec::new(),
            total_amount: 0,
            status: TransactionStatus::Pending,
            created_at: Timestamp::now(),
        }
    }

    /// Add a transaction to the batch
    pub fn add(&mut self, transaction: Transaction) {
        self.total_amount = self.total_amount.saturating_add(transaction.amount);
        self.transactions.push(transaction);
    }

    /// Get transaction count
    pub fn len(&self) -> usize {
        self.transactions.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.transactions.is_empty()
    }
}

impl Default for TransactionBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Transaction receipt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionReceipt {
    /// Transaction ID
    pub transaction_id: TransactionId,
    /// Receipt number
    pub receipt_number: String,
    /// Amount
    pub amount: Credits,
    /// Status
    pub status: TransactionStatus,
    /// Description
    pub description: String,
    /// Timestamp
    pub timestamp: Timestamp,
    /// Signature for verification
    pub signature: Option<String>,
}

impl TransactionReceipt {
    /// Create a receipt from a transaction
    pub fn from_transaction(tx: &Transaction) -> Self {
        TransactionReceipt {
            transaction_id: tx.id.clone(),
            receipt_number: format!("RCP-{}", &tx.id[..8].to_uppercase()),
            amount: tx.amount,
            status: tx.status,
            description: tx.description.clone(),
            timestamp: Timestamp::now(),
            signature: tx.signature.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transaction_creation() {
        let tx = Transaction::new(
            TransactionType::Purchase,
            "user_123".to_string(),
            1000,
            "Credit purchase".to_string(),
        );

        assert_eq!(tx.amount, 1000);
        assert_eq!(tx.status, TransactionStatus::Pending);
        assert!(tx.to_account.is_none());
    }

    #[test]
    fn test_app_payment() {
        let tx = Transaction::app_payment(
            "user_123".to_string(),
            "dev_456".to_string(),
            "app_xyz".to_string(),
            100,
            25, // 25% platform fee
        );

        assert_eq!(tx.amount, 100);
        assert_eq!(tx.platform_fee, 25);
        assert_eq!(tx.net_amount, 75);
        assert_eq!(tx.transaction_type, TransactionType::AppPayment);
    }

    #[test]
    fn test_micropayment() {
        let tx = Transaction::micropayment(
            "user_123".to_string(),
            "dev_456".to_string(),
            "chip_app".to_string(),
            10,
        );

        assert_eq!(tx.amount, 10);
        assert_eq!(tx.platform_fee, 1); // 10% fee
        assert_eq!(tx.net_amount, 9);
        assert_eq!(tx.transaction_type, TransactionType::Micropayment);
    }

    #[test]
    fn test_transaction_lifecycle() {
        let mut tx = Transaction::new(
            TransactionType::Purchase,
            "user_123".to_string(),
            1000,
            "Test".to_string(),
        );

        tx.start_processing();
        assert_eq!(tx.status, TransactionStatus::Processing);

        tx.complete();
        assert_eq!(tx.status, TransactionStatus::Completed);
        assert!(tx.completed_at.is_some());

        assert!(tx.is_refundable());
    }

    #[test]
    fn test_transaction_batch() {
        let mut batch = TransactionBatch::new();

        batch.add(Transaction::new(
            TransactionType::Purchase,
            "user_1".to_string(),
            100,
            "Purchase 1".to_string(),
        ));
        batch.add(Transaction::new(
            TransactionType::Purchase,
            "user_2".to_string(),
            200,
            "Purchase 2".to_string(),
        ));

        assert_eq!(batch.len(), 2);
        assert_eq!(batch.total_amount, 300);
    }
}
