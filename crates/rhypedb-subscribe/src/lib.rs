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
    /// An opaque caller-supplied write origin, threaded from the mutating verb
    /// (`create_with_origin` / `update_with_origin` / `delete_with_origin`) onto
    /// every event that write produces. `None` for an untagged write. Its meaning
    /// is the embedder's: it exists so a subscriber that also *writes in reaction*
    /// to the feed can deterministically skip its OWN changes (via
    /// [`SubscriptionFilter::exclude_origin`]) and not loop. A batch/cascade write
    /// stamps the SAME origin on all of its events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin: Option<u64>,
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
    /// Drop events carrying this write [`origin`](ChangeEvent::origin)
    /// (None = no origin filter). This is the loop-breaker for a subscriber that
    /// also writes in reaction to the feed: tag your writes with an origin via
    /// the engine's `*_with_origin` verbs, set this to the same value, and the
    /// hub filters out your OWN events before they reach you.
    ///
    /// The local [`SubscriptionHub`] applies this predicate directly. It is NOT
    /// carried on the binary subscribe protocol (the wire filter can't express
    /// it), so over the network the official clients apply it CLIENT-SIDE after
    /// decoding each event — the server still sends the event; the client drops
    /// it. Either way, `exclude_origin` breaks the write → react → write loop.
    pub exclude_origin: Option<u64>,
}

impl SubscriptionFilter {
    pub fn all() -> Self {
        Self {
            type_name: None,
            object_id: None,
            kinds: Vec::new(),
            exclude_origin: None,
        }
    }

    pub fn for_type(type_name: impl Into<String>) -> Self {
        Self {
            type_name: Some(type_name.into()),
            object_id: None,
            kinds: Vec::new(),
            exclude_origin: None,
        }
    }

    pub fn for_object(type_name: impl Into<String>, object_id: u64) -> Self {
        Self {
            type_name: Some(type_name.into()),
            object_id: Some(object_id),
            kinds: Vec::new(),
            exclude_origin: None,
        }
    }

    /// Chainable setter for [`exclude_origin`](Self::exclude_origin):
    /// `SubscriptionFilter::for_type("Post").excluding_origin(my_origin)`.
    pub fn excluding_origin(mut self, origin: u64) -> Self {
        self.exclude_origin = Some(origin);
        self
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
        // Loop-breaker: drop an event whose origin matches the excluded one. An
        // untagged event (origin = None) is never excluded.
        if let Some(ex) = self.exclude_origin
            && event.origin == Some(ex) {
                return false;
            }
        true
    }
}

/// The outcome of offering an event to a subscriber's [`EventSink`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delivery {
    /// The sink accepted the event, OR a bounded sink intentionally dropped it
    /// (e.g. after flagging a lag) but wants to stay registered. Keep the sub.
    Delivered,
    /// The consumer is permanently gone (channel closed / receiver dropped).
    /// The hub removes the subscription.
    Disconnected,
}

/// A transport-agnostic delivery target for change events. The hub invokes
/// [`EventSink::deliver`] for every matching subscription on every published
/// event — from the engine's SYNCHRONOUS commit context — so an implementation
/// MUST NOT block or await. A bounded network sink applies its own
/// slow-consumer policy here (drop + flag) and returns [`Delivery::Delivered`]
/// so the subscription survives; it returns [`Delivery::Disconnected`] only when
/// the consumer is truly gone.
pub trait EventSink: Send + Sync {
    fn deliver(&self, event: &ChangeEvent) -> Delivery;
}

/// The built-in unbounded sink backing [`SubscriptionHub::subscribe`]: a plain
/// `std::sync::mpsc::Sender`. Never reports `Full` (the channel is unbounded);
/// reports `Disconnected` once the receiver is dropped.
struct ChannelSink {
    sender: std::sync::mpsc::Sender<ChangeEvent>,
}

impl EventSink for ChannelSink {
    fn deliver(&self, event: &ChangeEvent) -> Delivery {
        match self.sender.send(event.clone()) {
            Ok(()) => Delivery::Delivered,
            Err(_) => Delivery::Disconnected,
        }
    }
}

/// A registered subscription with its delivery sink.
struct Subscription {
    id: u64,
    filter: SubscriptionFilter,
    sink: Box<dyn EventSink>,
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
    /// channel for change events. The channel is UNBOUNDED — suitable for
    /// in-process consumers that drain promptly; a network transport should use
    /// [`SubscriptionHub::subscribe_sink`] with a bounded sink instead.
    pub fn subscribe(
        &self,
        filter: SubscriptionFilter,
    ) -> (u64, std::sync::mpsc::Receiver<ChangeEvent>) {
        let (sender, receiver) = std::sync::mpsc::channel();
        let id = self.subscribe_sink(filter, Box::new(ChannelSink { sender }));
        (id, receiver)
    }

    /// Register a subscription backed by a caller-supplied [`EventSink`].
    /// Returns the subscription ID (pass it to [`SubscriptionHub::unsubscribe`]).
    /// This is the transport-agnostic entry point: the network layer supplies a
    /// BOUNDED sink that enforces its own slow-consumer policy, keeping the hub
    /// itself free of any buffering or transport concern.
    pub fn subscribe_sink(&self, filter: SubscriptionFilter, sink: Box<dyn EventSink>) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        self.subscriptions.write().push(Subscription {
            id,
            filter,
            sink,
        });
        id
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
        // Collect dead subscriptions (sink reported the consumer is gone).
        let mut dead_ids = Vec::new();

        for sub in subs.iter() {
            if sub.filter.matches(&event)
                && sub.sink.deliver(&event) == Delivery::Disconnected
            {
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
            origin: None,
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
    fn filter_exclude_origin() {
        let hub = SubscriptionHub::new();
        let (_id, rx) = hub.subscribe(SubscriptionFilter::all().excluding_origin(42));

        // Tagged with the excluded origin → dropped.
        hub.publish(ChangeEvent { origin: Some(42), ..make_event(ChangeKind::Create, "User", 1) });
        // A different origin → delivered.
        hub.publish(ChangeEvent { origin: Some(7), ..make_event(ChangeKind::Update, "User", 2) });
        // Untagged (origin = None) → delivered; None is never excluded.
        hub.publish(make_event(ChangeKind::Delete, "User", 3));

        let e1 = rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!((e1.object_id, e1.origin), (2, Some(7)));
        let e2 = rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!((e2.object_id, e2.origin), (3, None));
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
    fn custom_sink_delivered_keeps_disconnected_removes() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        // A sink that records deliveries and reports Disconnected once a shared
        // flag is set — exercising both Delivery outcomes through publish().
        struct CountingSink {
            count: Arc<AtomicUsize>,
            dead: Arc<std::sync::atomic::AtomicBool>,
        }
        impl EventSink for CountingSink {
            fn deliver(&self, _event: &ChangeEvent) -> Delivery {
                self.count.fetch_add(1, Ordering::SeqCst);
                if self.dead.load(Ordering::SeqCst) {
                    Delivery::Disconnected
                } else {
                    Delivery::Delivered
                }
            }
        }

        let hub = SubscriptionHub::new();
        let count = Arc::new(AtomicUsize::new(0));
        let dead = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let _id = hub.subscribe_sink(
            SubscriptionFilter::for_type("User"),
            Box::new(CountingSink {
                count: count.clone(),
                dead: dead.clone(),
            }),
        );

        // Non-matching event: sink not called.
        hub.publish(make_event(ChangeKind::Create, "Post", 1));
        assert_eq!(count.load(Ordering::SeqCst), 0);
        assert_eq!(hub.subscription_count(), 1);

        // Matching event while Delivered: kept.
        hub.publish(make_event(ChangeKind::Create, "User", 2));
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert_eq!(hub.subscription_count(), 1);

        // Now report Disconnected: the next matching publish removes the sub.
        dead.store(true, Ordering::SeqCst);
        hub.publish(make_event(ChangeKind::Update, "User", 2));
        assert_eq!(count.load(Ordering::SeqCst), 2);
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
            origin: Some(7),
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"kind\":\"create\""));
        assert!(json.contains("\"version\":42"));
        assert!(json.contains("\"Alice\""));
        // NB: this guards ChangeEvent's IN-CRATE serde contract, where origin is a
        // bare number (`"origin":7`). It is NOT proof of what crosses the network —
        // the wire form is `WireEvent` with a decimal-STRING origin (`"origin":"7"`),
        // covered by rhypedb-wire's `wire_event_roundtrip_and_lossless_ids`.
        assert!(json.contains("\"origin\":7"));

        // An untagged event omits `origin` entirely (skip_serializing_if).
        let untagged = ChangeEvent {
            origin: None,
            ..event
        };
        let json = serde_json::to_string(&untagged).unwrap();
        assert!(!json.contains("origin"), "untagged event must omit origin: {json}");
    }
}
