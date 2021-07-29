use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;

#[skip_serializing_none]
#[derive(Debug, Deserialize, Serialize)]
pub struct DataFrame {
    #[serialize_always]
    pub BodyMarkdown: Option<String>,
    #[serialize_always]
    pub Title: Option<String>,
}
