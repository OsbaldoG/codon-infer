use std::fs::File;
use std::io::{self, BufRead, BufReader};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FastaRecord {
    pub id: String,
    pub header: String,
    pub sequence: String,
}

impl FastaRecord {
    pub fn new(id: &str, header: &str, sequence: &str) -> Self {
        Self {
            id: id.to_string(),
            header: header.to_string(),
            sequence: sequence.to_string(),
        }
    }

    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }
}

/// Normalizes and cleans a single line of amino acid characters.
/// Keeps standard IUPAC amino acid characters, uppercase, and mask token '*'.
fn clean_seq_line(line: &str, seq_buf: &mut String) {
    for c in line.chars() {
        if c.is_ascii_alphabetic() {
            seq_buf.push(c.to_ascii_uppercase());
        } else if c == '*' || c == '.' || c == '-' {
            seq_buf.push(c);
        }
    }
}

/// Parses FASTA format from any buffered reader stream.
pub fn parse_fasta_stream<R: BufRead>(reader: R) -> io::Result<Vec<FastaRecord>> {
    let mut records = Vec::new();
    let mut current_header = String::new();
    let mut current_seq = String::new();

    for line_result in reader.lines() {
        let line = line_result?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with(';') {
            continue; // Skip empty lines and legacy FASTA comment lines
        }

        if trimmed.starts_with('>') {
            if !current_seq.is_empty() {
                let id = current_header
                    .split_whitespace()
                    .next()
                    .unwrap_or("seq")
                    .to_string();
                records.push(FastaRecord {
                    id,
                    header: current_header.clone(),
                    sequence: current_seq.clone(),
                });
                current_seq.clear();
            }
            current_header = trimmed[1..].trim().to_string();
        } else {
            clean_seq_line(trimmed, &mut current_seq);
        }
    }

    if !current_seq.is_empty() {
        let id = current_header
            .split_whitespace()
            .next()
            .unwrap_or("seq")
            .to_string();
        records.push(FastaRecord {
            id,
            header: current_header,
            sequence: current_seq,
        });
    }

    Ok(records)
}

/// Parses a FASTA file from disk or standard input (if path is "-").
pub fn parse_fasta_file(path: &str) -> io::Result<Vec<FastaRecord>> {
    if path == "-" {
        let stdin = io::stdin();
        parse_fasta_stream(stdin.lock())
    } else {
        let file = File::open(path)?;
        parse_fasta_stream(BufReader::new(file))
    }
}

/// Parses an input string directly into a FastaRecord (handling either FASTA format or raw sequence).
pub fn parse_raw_sequence(input: &str, default_id: &str) -> FastaRecord {
    let trimmed = input.trim();
    if trimmed.starts_with('>') {
        let records = parse_fasta_stream(trimmed.as_bytes()).unwrap_or_default();
        if let Some(first) = records.into_iter().next() {
            return first;
        }
    }

    let mut clean_seq = String::with_capacity(trimmed.len());
    clean_seq_line(trimmed, &mut clean_seq);
    FastaRecord {
        id: default_id.to_string(),
        header: default_id.to_string(),
        sequence: clean_seq,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_fasta() {
        let fasta_data = ">sp|P00330|LDH_BACST L-lactate dehydrogenase\nMKVIGSGCNLDSARFR\nYLMGERLGVH\n";
        let records = parse_fasta_stream(fasta_data.as_bytes()).expect("should parse");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id, "sp|P00330|LDH_BACST");
        assert_eq!(records[0].sequence, "MKVIGSGCNLDSARFRYLMGERLGVH");
    }

    #[test]
    fn test_parse_multi_fasta_with_masks() {
        let fasta_data = ">prot1 Chlorotoxin fragment\nMCMPC*TTD\n>prot2 Masked active site\nMKVIGSG*NLDS\n";
        let records = parse_fasta_stream(fasta_data.as_bytes()).expect("should parse");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].id, "prot1");
        assert_eq!(records[0].sequence, "MCMPC*TTD");
        assert_eq!(records[1].id, "prot2");
        assert_eq!(records[1].sequence, "MKVIGSG*NLDS");
    }

    #[test]
    fn test_parse_raw_sequence_fallback() {
        let rec = parse_raw_sequence("  mkvigsGCNLds  ", "custom_id");
        assert_eq!(rec.id, "custom_id");
        assert_eq!(rec.sequence, "MKVIGSGCNLDS");
    }
}
