//! WASM bindings for agentic-payments
//!
//! This crate provides WebAssembly bindings for payment processing in browsers
//! and other WASM-capable environments.

use wasm_bindgen::prelude::*;
use serde_wasm_bindgen::{to_value, from_value};

use agentic_payments::{
    PaymentEngine, PaymentEngineConfig,
    Account, CreditBalance, CreditTransaction, CreditTransactionType,
    Transaction, TransactionReceipt,
    Subscription, SubscriptionTier, BillingPeriod,
    MicropaymentChannel, MicropaymentReceipt,
    Credits,
};

/// Initialize panic hook for better error messages
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// WASM Payment Engine wrapper
#[wasm_bindgen]
pub struct WasmPaymentEngine {
    engine: PaymentEngine,
}

#[wasm_bindgen]
impl WasmPaymentEngine {
    /// Create a new payment engine
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        WasmPaymentEngine {
            engine: PaymentEngine::new(),
        }
    }

    /// Create a payment engine with custom configuration
    #[wasm_bindgen(js_name = withConfig)]
    pub fn with_config(config: JsValue) -> Result<WasmPaymentEngine, JsValue> {
        let config: PaymentEngineConfig = from_value(config)?;
        Ok(WasmPaymentEngine {
            engine: PaymentEngine::with_config(config),
        })
    }

    // ==================== Account Management ====================

    /// Create a new account
    #[wasm_bindgen(js_name = createAccount)]
    pub fn create_account(&self, external_id: Option<String>) -> Result<JsValue, JsValue> {
        let account = self.engine.create_account(external_id)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        to_value(&account).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Create a developer account
    #[wasm_bindgen(js_name = createDeveloperAccount)]
    pub fn create_developer_account(
        &self,
        external_id: Option<String>,
        name: String,
    ) -> Result<JsValue, JsValue> {
        let (account, profile) = self.engine.create_developer_account(external_id, name)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        to_value(&(account, profile)).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get account by ID
    #[wasm_bindgen(js_name = getAccount)]
    pub fn get_account(&self, account_id: &str) -> Result<JsValue, JsValue> {
        let account = self.engine.get_account(&account_id.to_string())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        to_value(&account).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    // ==================== Credit Management ====================

    /// Get credit balance
    #[wasm_bindgen(js_name = getBalance)]
    pub fn get_balance(&self, account_id: &str) -> Result<JsValue, JsValue> {
        let balance = self.engine.get_balance(&account_id.to_string())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        to_value(&balance).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Add credits to account
    #[wasm_bindgen(js_name = addCredits)]
    pub fn add_credits(
        &self,
        account_id: &str,
        amount: u64,
        description: String,
    ) -> Result<JsValue, JsValue> {
        let tx = self.engine.add_credits(
            &account_id.to_string(),
            amount,
            CreditTransactionType::Purchase,
            description,
        ).map_err(|e| JsValue::from_str(&e.to_string()))?;
        to_value(&tx).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Purchase credits with money
    #[wasm_bindgen(js_name = purchaseCredits)]
    pub fn purchase_credits(
        &self,
        account_id: &str,
        amount_cents: u64,
    ) -> Result<JsValue, JsValue> {
        let (credits, tx) = self.engine.purchase_credits(&account_id.to_string(), amount_cents)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        to_value(&(credits, tx)).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get credit history
    #[wasm_bindgen(js_name = getCreditHistory)]
    pub fn get_credit_history(
        &self,
        account_id: &str,
        limit: usize,
    ) -> Result<JsValue, JsValue> {
        let history = self.engine.get_credit_history(&account_id.to_string(), limit)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        to_value(&history).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    // ==================== Transactions ====================

    /// Process an app payment
    #[wasm_bindgen(js_name = processAppPayment)]
    pub fn process_app_payment(
        &self,
        user_account_id: &str,
        developer_account_id: &str,
        app_id: &str,
        amount: u64,
    ) -> Result<JsValue, JsValue> {
        let receipt = self.engine.process_app_payment(
            &user_account_id.to_string(),
            &developer_account_id.to_string(),
            &app_id.to_string(),
            amount,
        ).map_err(|e| JsValue::from_str(&e.to_string()))?;
        to_value(&receipt).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get transaction by ID
    #[wasm_bindgen(js_name = getTransaction)]
    pub fn get_transaction(&self, transaction_id: &str) -> Result<JsValue, JsValue> {
        let tx = self.engine.get_transaction(&transaction_id.to_string())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        to_value(&tx).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    // ==================== Micropayments ====================

    /// Create a micropayment channel
    #[wasm_bindgen(js_name = createMicropaymentChannel)]
    pub fn create_micropayment_channel(
        &self,
        user_account_id: &str,
        developer_account_id: &str,
        app_id: &str,
        deposit: u64,
    ) -> Result<JsValue, JsValue> {
        let channel = self.engine.create_micropayment_channel(
            &user_account_id.to_string(),
            &developer_account_id.to_string(),
            &app_id.to_string(),
            deposit,
        ).map_err(|e| JsValue::from_str(&e.to_string()))?;
        to_value(&channel).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Process a micropayment
    #[wasm_bindgen(js_name = processMicropayment)]
    pub fn process_micropayment(
        &self,
        channel_id: &str,
        amount: u64,
    ) -> Result<JsValue, JsValue> {
        let receipt = self.engine.process_micropayment(channel_id, amount)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        to_value(&receipt).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Process a chip app use (with free tier)
    #[wasm_bindgen(js_name = processChipAppUse)]
    pub fn process_chip_app_use(
        &self,
        user_account_id: &str,
        developer_account_id: &str,
        app_id: &str,
    ) -> Result<JsValue, JsValue> {
        let receipt = self.engine.process_chip_app_use(
            &user_account_id.to_string(),
            &developer_account_id.to_string(),
            &app_id.to_string(),
        ).map_err(|e| JsValue::from_str(&e.to_string()))?;
        to_value(&receipt).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    // ==================== Subscriptions ====================

    /// Create a subscription
    #[wasm_bindgen(js_name = createSubscription)]
    pub fn create_subscription(
        &self,
        account_id: &str,
        tier: &str,
        billing_period: &str,
    ) -> Result<JsValue, JsValue> {
        let tier = match tier.to_lowercase().as_str() {
            "free" => SubscriptionTier::Free,
            "pro" => SubscriptionTier::Pro,
            "enterprise" => SubscriptionTier::Enterprise,
            _ => return Err(JsValue::from_str("Invalid tier")),
        };

        let period = match billing_period.to_lowercase().as_str() {
            "monthly" => BillingPeriod::Monthly,
            "annual" => BillingPeriod::Annual,
            _ => return Err(JsValue::from_str("Invalid billing period")),
        };

        let subscription = self.engine.create_subscription(
            &account_id.to_string(),
            tier,
            period,
        ).map_err(|e| JsValue::from_str(&e.to_string()))?;

        to_value(&subscription).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Create a trial subscription
    #[wasm_bindgen(js_name = createTrialSubscription)]
    pub fn create_trial_subscription(
        &self,
        account_id: &str,
        trial_days: u32,
    ) -> Result<JsValue, JsValue> {
        let subscription = self.engine.create_trial_subscription(
            &account_id.to_string(),
            trial_days,
        ).map_err(|e| JsValue::from_str(&e.to_string()))?;

        to_value(&subscription).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Upgrade subscription
    #[wasm_bindgen(js_name = upgradeSubscription)]
    pub fn upgrade_subscription(
        &self,
        account_id: &str,
        new_tier: &str,
    ) -> Result<u64, JsValue> {
        let tier = match new_tier.to_lowercase().as_str() {
            "pro" => SubscriptionTier::Pro,
            "enterprise" => SubscriptionTier::Enterprise,
            _ => return Err(JsValue::from_str("Invalid tier")),
        };

        self.engine.upgrade_subscription(&account_id.to_string(), tier)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Cancel subscription
    #[wasm_bindgen(js_name = cancelSubscription)]
    pub fn cancel_subscription(&self, account_id: &str) -> Result<(), JsValue> {
        self.engine.cancel_subscription(&account_id.to_string())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    // ==================== Stats ====================

    /// Get engine statistics
    #[wasm_bindgen(js_name = getStats)]
    pub fn get_stats(&self) -> Result<JsValue, JsValue> {
        let stats = self.engine.get_stats();
        to_value(&stats).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl Default for WasmPaymentEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Get subscription plans
#[wasm_bindgen(js_name = getSubscriptionPlans)]
pub fn get_subscription_plans() -> Result<JsValue, JsValue> {
    use agentic_payments::subscriptions::SubscriptionPlan;
    let plans = SubscriptionPlan::all_plans();
    to_value(&plans).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Get credit pricing info
#[wasm_bindgen(js_name = getCreditPricing)]
pub fn get_credit_pricing() -> Result<JsValue, JsValue> {
    use agentic_payments::credits::CreditPricing;
    let pricing = CreditPricing::default();
    to_value(&pricing).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Calculate credits for an amount
#[wasm_bindgen(js_name = calculateCredits)]
pub fn calculate_credits(amount_cents: u64) -> u64 {
    use agentic_payments::credits::CreditPricing;
    CreditPricing::default().calculate_credits(amount_cents)
}
