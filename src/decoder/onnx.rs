//! The small protobuf subset needed by homr's ONNX model. Unknown fields are skipped.
use std::collections::HashMap;

pub(super) struct Tensor {
    pub dims: Vec<usize>,
    pub kind: u64,
    pub data: Vec<u8>,
}
impl Tensor {
    pub fn floats(&self) -> Result<Vec<f32>, String> {
        if self.kind != 1 || !self.data.len().is_multiple_of(4) {
            return Err("expected float32 initializer".into());
        }
        Ok(self
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }
    pub fn scalar(&self) -> Result<f32, String> {
        let f = self.floats()?;
        if f.len() != 1 {
            return Err("expected scalar initializer".into());
        }
        Ok(f[0])
    }
}
#[derive(Default)]
pub(super) struct Node {
    pub name: String,
    pub op: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}
pub(super) struct Graph {
    pub tensors: HashMap<String, Tensor>,
    pub nodes: Vec<Node>,
}
struct Field<'a> {
    tag: u64,
    int: u64,
    bytes: &'a [u8],
}
fn varint(b: &mut &[u8]) -> Result<u64, String> {
    let mut n = 0;
    for shift in (0..70).step_by(7) {
        let (&v, rest) = b.split_first().ok_or("truncated protobuf varint")?;
        *b = rest;
        if shift == 63 && v > 1 {
            return Err("protobuf varint overflow".into());
        }
        n |= ((v & 127) as u64) << shift;
        if v < 128 {
            return Ok(n);
        }
    }
    Err("invalid varint".into())
}
fn fields(mut b: &[u8]) -> Result<Vec<Field<'_>>, String> {
    let mut out = Vec::new();
    while !b.is_empty() {
        let key = varint(&mut b)?;
        let mut f = Field {
            tag: key >> 3,
            int: 0,
            bytes: &[],
        };
        let len = match key & 7 {
            0 => {
                f.int = varint(&mut b)?;
                0
            }
            1 => 8,
            2 => usize::try_from(varint(&mut b)?).map_err(|_| "protobuf length overflow")?,
            5 => 4,
            _ => return Err("unsupported protobuf wire type".into()),
        };
        f.bytes = b.get(..len).ok_or("truncated protobuf field")?;
        b = &b[len..];
        if f.tag == 0 {
            return Err("invalid protobuf field zero".into());
        }
        out.push(f);
    }
    Ok(out)
}
fn string(b: &[u8]) -> Result<String, String> {
    String::from_utf8(b.to_vec()).map_err(|e| e.to_string())
}
impl Graph {
    pub fn read(bytes: &[u8]) -> Result<Self, String> {
        let model = fields(bytes)?;
        let graph = model
            .iter()
            .find(|f| f.tag == 7)
            .ok_or("ONNX graph missing")?;
        let mut g = Self {
            tensors: HashMap::new(),
            nodes: Vec::new(),
        };
        for f in fields(graph.bytes)? {
            if f.tag == 1 {
                let mut n = Node::default();
                for a in fields(f.bytes)? {
                    match a.tag {
                        1 => n.inputs.push(string(a.bytes)?),
                        2 => n.outputs.push(string(a.bytes)?),
                        3 => n.name = string(a.bytes)?,
                        4 => n.op = string(a.bytes)?,
                        _ => {}
                    }
                }
                g.nodes.push(n);
            } else if f.tag == 5 {
                let mut name = String::new();
                let mut t = Tensor {
                    dims: Vec::new(),
                    kind: 0,
                    data: Vec::new(),
                };
                let mut raw = None;
                for a in fields(f.bytes)? {
                    match a.tag {
                        1 => {
                            if a.bytes.is_empty() {
                                t.dims.push(a.int as usize);
                            } else {
                                let mut b = a.bytes;
                                while !b.is_empty() {
                                    t.dims.push(varint(&mut b)? as usize);
                                }
                            }
                        }
                        2 => t.kind = a.int,
                        4 => t.data.extend_from_slice(a.bytes),
                        5 | 7 => {
                            let mut vals = Vec::new();
                            if a.bytes.is_empty() {
                                vals.push(a.int);
                            } else {
                                let mut b = a.bytes;
                                while !b.is_empty() {
                                    vals.push(varint(&mut b)?);
                                }
                            }
                            for v in vals {
                                match t.kind {
                                    2 | 3 => t.data.push(v as u8),
                                    7 => t.data.extend_from_slice(&v.to_le_bytes()),
                                    _ => return Err("unsupported typed initializer".into()),
                                }
                            }
                        }
                        8 => name = string(a.bytes)?,
                        9 => raw = Some(a.bytes.to_vec()),
                        13 => {
                            return Err(
                                "external ONNX tensors are not supported; pass the complete model"
                                    .into(),
                            )
                        }
                        14 if a.int != 0 => {
                            return Err("external ONNX tensors are not supported".into())
                        }
                        _ => {}
                    }
                }
                if let Some(raw) = raw {
                    t.data = raw;
                }
                let width: usize = match t.kind {
                    1 => 4,
                    2 | 3 => 1,
                    7 => 8,
                    _ => return Err(format!("unsupported initializer type {}", t.kind)),
                };
                let size = t
                    .dims
                    .iter()
                    .try_fold(width, |n, &d| n.checked_mul(d))
                    .ok_or("initializer size overflow")?;
                if size != t.data.len() {
                    return Err(format!("initializer size mismatch: {name}"));
                }
                if g.tensors.insert(name, t).is_some() {
                    return Err("duplicate initializer".into());
                }
            }
        }
        Ok(g)
    }
    pub fn tensor(&self, name: &str) -> Result<&Tensor, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("initializer missing: {name}"))
    }
    pub fn node(&self, name: &str) -> Result<&Node, String> {
        self.nodes
            .iter()
            .find(|n| n.name == name)
            .ok_or_else(|| format!("node missing: {name}"))
    }
    pub fn producer(&self, output: &str) -> Result<&Node, String> {
        self.nodes
            .iter()
            .find(|n| n.outputs.iter().any(|s| s == output))
            .ok_or_else(|| format!("producer missing: {output}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_truncation_and_overflow() {
        for b in [&[0x3a, 12, 1][..], &[0x80][..], &[0xff; 12][..], &[0][..]] {
            assert!(Graph::read(b).is_err());
        }
    }
}
