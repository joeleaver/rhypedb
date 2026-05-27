use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;
use serde::Serialize;

/// A change event emitted when an object is created, updated, or deleted.
#[derive(Debug, Clone, Serialize)]
pub struct ChangeEvent {
    pub version: u64,
    pub kind: ChangeKind,
    pub type_name: String,
    pub object_id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fields: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ChangeKind {
    Create,
    Update,
    Delete,
}

/// A subscription filter that determines which changes a subscriber cares about.
#[derive(Debug, Clone)]
pub struct SubscriptionFilter {
    /// Only match events for this type (None = all types).
    pub type_name: Option<String>,
    /// Only match events for this specific object ID (None = all objects of the type).
    pub object_id: Option<u64>,
    /// Only match these change kinds (empty = all kinds).
    pub kinds: Vec<ChangeKind>,
}

impl SubscriptionFilter {
    pub fn all() -> Self {
        Self {
            type_name: None,
            object_id: None,
            kinds: Vec::new(),
        }
    }

    pub fn for_type(type_name: impl Into<String>) -> Self {
        Self {
            type_name: Some(type_name.into()),
            object_id: None,
            kinds: Vec::new(),
        }
    }

    pub fn for_object(type_name: impl Into<String>, object_id: u64) -> Self {
        Self {
            type_name: Some(type_name.into()),
            object_id: Some(object_id),
            kinds: Vec::new(),
        }
    }

    fn matches(&self, event: &ChangeEvent) -> bool {
        if let Some(ref tn) = self.type_name
            && tn != &event.type_name {
                return false;
            }
        if let Some(oid) = self.object_id
            && oid != event.object_id {
                return false;
            }
        if !self.kinds.is_empty() && !self.kinds.contains(&event.kind) {
            return false;
        }
        true
    }
}

/// A registered subscription with its callback channel.
struct Subscription {
    id: u64,
    filter: SubscriptionFilter,
    sender: std::sync::mpsc::Sender<ChangeEvent>,
}

/// The subscription hub — manages subscriptions and dispatches change events.
///
/// The engine calls `publish()` on every committed mutation. The hub evaluates
/// each event against all active subscriptions and sends matching events to
/// their channels.
pub struct SubscriptionHub {
    subscriptions: RwLock<Vec<Subscription>>,
    next_id: AtomicU64,
}

impl Default for SubscriptionHub {
    fn default() -> Self {
        Self::new()
    }
}

impl SubscriptionHub {
    pub fn new() -> Self {
        Self {
            subscriptions: RwLock::new(Vec::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Register a new subscription. Returns a subscription ID and a receiver
    /// channel for change events.
    pub fn subscribe(
        &self,
        filter: SubscriptionFilter,
    ) -> (u64, std::sync::mpsc::Receiver<ChangeEvent>) {
        let (sender, receiver) = std::sync::mpsc::channel();
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);

        self.subscriptions.write().push(Subscription {
            id,
            filter,
            sender,
        });

        (id, receiver)
    }

    /// Unsubscribe by ID. Returns true if the subscription was found and removed.
    pub fn unsubscribe(&self, id: u64) -> bool {
        let mut subs = self.subscriptions.write();
        let len_before = subs.len();
        subs.retain(|s| s.id != id);
        subs.len() < len_before
    }

    /// Publish a change event to all matching subscriptions.
    /// Called by the engine on every committed mutation.
    pub fn publish(&self, event: ChangeEvent) {
        let subs = self.subscriptions.read();
        // Collect dead subscriptions (where the receiver was dropped).
        let mut dead_ids = Vec::new();

        for sub in subs.iter() {
            if sub.filter.matches(&event)
                && sub.sender.send(event.clone()).is_err() {
                    dead_ids.push(sub.id);
                }
        }

        drop(subs);

        // Clean up dead subscriptions.
        if !dead_ids.is_empty() {
            let mut subs = self.subscriptions.write();
            subs.retain(|s| !dead_ids.contains(&s.id));
        }
    }

    /// Number of active subscriptions.
    pub fn subscription_count(&self) -> usize {
        self.subscriptions.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_event(kind: ChangeKind, type_name: &str, object_id: u64) -> ChangeEvent {
        ChangeEvent {
            version: 1,
            kind,
            type_name: type_name.into(),
            object_id,
            fields: None,
        }
    }

    #[test]
    fn subscribe_and_receive() {
        let hub = SubscriptionHub::new();
        let (_id, rx) = hub.subscribe(SubscriptionFilter::for_type("User"));

        hub.publish(make_event(ChangeKind::Create, "User", 1));

        let event = rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(event.type_name, "User");
        assert_eq!(event.object_id, 1);
        assert_eq!(event.kind, ChangeKind::Create);
    }

    #[test]
    fn filter_by_type() {
        let hub = SubscriptionHub::new();
        let (_id, rx) = hub.subscribe(SubscriptionFilter::for_type("User"));

        hub.publish(make_event(ChangeKind::Create, "Post", 1));
        hub.publish(make_event(ChangeKind::Create, "User", 2));

        let event = rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(event.type_name, "User");
        assert_eq!(event.object_id, 2);

        // No more events (Post was filtered out).
        assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());
    }

    #[test]
    fn filter_by_object() {
        let hub = SubscriptionHub::new();
        let (_id, rx) = hub.subscribe(SubscriptionFilter::for_object("User", 5));

        hub.publish(make_event(ChangeKind::Update, "User", 3));
        hub.publish(make_event(ChangeKind::Update, "User", 5));

        let event = rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(event.object_id, 5);

        assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());
    }

    #[test]
    fn filter_by_kind() {
        let hub = SubscriptionHub::new();
        let mut filter = SubscriptionFilter::for_type("User");
        filter.kinds = vec![ChangeKind::Delete];
        let (_id, rx) = hub.subscribe(filter);

        hub.publish(make_event(ChangeKind::Create, "User", 1));
        hub.publish(make_event(ChangeKind::Update, "User", 1));
        hub.publish(make_event(ChangeKind::Delete, "User", 1));

        let event = rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(event.kind, ChangeKind::Delete);

        assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());
    }

    #[test]
    fn subscribe_all() {
        let hub = SubscriptionHub::new();
        let (_id, rx) = hub.subscribe(SubscriptionFilter::all());

        hub.publish(make_event(ChangeKind::Create, "User", 1));
        hub.publish(make_event(ChangeKind::Update, "Post", 2));
        hub.publish(make_event(ChangeKind::Delete, "Tag", 3));

        assert!(rx.recv_timeout(Duration::from_secs(1)).is_ok());
        assert!(rx.recv_timeout(Duration::from_secs(1)).is_ok());
        assert!(rx.recv_timeout(Duration::from_secs(1)).is_ok());
        assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());
    }

    #[test]
    fn multiple_subscribers() {
        let hub = SubscriptionHub::new();
        let (_id1, rx1) = hub.subscribe(SubscriptionFilter::for_type("User"));
        let (_id2, rx2) = hub.subscribe(SubscriptionFilter::for_type("User"));

        hub.publish(make_event(ChangeKind::Create, "User", 1));

        assert!(rx1.recv_timeout(Duration::from_secs(1)).is_ok());
        assert!(rx2.recv_timeout(Duration::from_secs(1)).is_ok());
    }

    #[test]
    fn unsubscribe() {
        let hub = SubscriptionHub::new();
        let (id, rx) = hub.subscribe(SubscriptionFilter::for_type("User"));

        assert_eq!(hub.subscription_count(), 1);
        assert!(hub.unsubscribe(id));
        assert_eq!(hub.subscription_count(), 0);

        hub.publish(make_event(ChangeKind::Create, "User", 1));
        assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());
    }

    #[test]
    fn dead_subscription_auto_cleaned() {
        let hub = SubscriptionHub::new();
        let (_id, rx) = hub.subscribe(SubscriptionFilter::all());

        // Drop the receiver — simulates client disconnect.
        drop(rx);

        hub.publish(make_event(ChangeKind::Create, "User", 1));

        // The dead subscription should have been cleaned up.
        assert_eq!(hub.subscription_count(), 0);
    }

    #[test]
    fn serialization() {
        let event = ChangeEvent {
            version: 42,
            kind: ChangeKind::Create,
            type_name: "User".into(),
            object_id: 1,
            fields: Some({
                let mut m = HashMap::new();
                m.insert("name".into(), serde_json::json!("Alice"));
                m
            }),
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"kind\":\"create\""));
        assert!(json.contains("\"version\":42"));
        assert!(json.contains("\"Alice\""));
    }
}
