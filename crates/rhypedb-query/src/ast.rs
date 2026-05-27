use std::collections::HashMap;

/// A complete query — a chain of operations starting from a type source.
#[derive(Debug, Clone, PartialEq)]
pub struct Query {
    pub source: Source,
    pub steps: Vec<Step>,
}

/// The starting point of a query — always a type name.
#[derive(Debug, Clone, PartialEq)]
pub enum Source {
    /// `Type.get(id)` — fetch a single object by ID.
    Get { type_name: String, id: u64 },

    /// `Type.filter(...)` — scan a type with a predicate.
    Filter {
        type_name: String,
        predicate: Predicate,
    },

    /// `Type.create({...})` — create an object.
    Create {
        type_name: String,
        fields: HashMap<String, Literal>,
    },

    /// `Type` — reference to all objects of a type (for chaining).
    All { type_name: String },
}

/// A step in the query chain after the source.
#[derive(Debug, Clone, PartialEq)]
pub enum Step {
    /// `.field_name` — traverse a relationship.
    Traverse {
        field_name: String,
    },

    /// `.filter(.field op value)` — filter results.
    Filter {
        predicate: Predicate,
    },

    /// `.similar(.field, vec, k: N)` — vector similarity search.
    Similar {
        field_name: String,
        vector: Vec<f32>,
        k: usize,
    },

    /// `.update({...})` — update matched objects.
    Update {
        fields: HashMap<String, Literal>,
    },

    /// `.delete()` — delete matched objects.
    Delete,

    /// `.link(Target.get(id), {edge_props})` — create a relationship.
    Link {
        target_type: String,
        target_id: u64,
        edge_fields: HashMap<String, Literal>,
    },

    /// `.unlink(Target.get(id))` — remove a relationship.
    Unlink {
        target_type: String,
        target_id: u64,
    },

    /// `.limit(n)` — limit result count.
    Limit {
        count: usize,
    },

    /// `.offset(n)` — skip results.
    Offset {
        count: usize,
    },
}

/// A filter predicate.
#[derive(Debug, Clone, PartialEq)]
pub enum Predicate {
    /// `.field op value`
    Compare {
        field_path: String,
        op: CompareOp,
        value: Literal,
    },

    /// `pred AND pred`
    And(Box<Predicate>, Box<Predicate>),

    /// `pred OR pred`
    Or(Box<Predicate>, Box<Predicate>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CompareOp {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Eq => "==",
            Self::Ne => "!=",
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Gt => ">",
            Self::Ge => ">=",
        }
    }
}

/// A literal value in a query.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
}
