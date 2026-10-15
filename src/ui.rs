use std::io::{self, IsTerminal, Write};
use std::time::Instant;

/// Terminal styling and color theme configuration.
#[derive(Debug, Clone, Copy)]
pub struct TerminalTheme {
    pub enabled: bool,
    pub truecolor: bool,
}

impl TerminalTheme {
    /// Detects color capability using standard streams, environment variables, and CLI overrides.
    pub fn detect(color_arg: Option<&str>, no_color_flag: bool) -> Self {
        if no_color_flag || std::env::var_os("NO_COLOR").is_some() {
            return Self { enabled: false, truecolor: false };
        }

        let is_atty = io::stdout().is_terminal();

        match color_arg {
            Some("never") => Self { enabled: false, truecolor: false },
            Some("always") => Self { enabled: true, truecolor: true },
            Some("auto") | None => {
                if !is_atty {
                    Self { enabled: false, truecolor: false }
                } else {
                    let colorterm = std::env::var("COLORTERM").unwrap_or_default();
                    let term = std::env::var("TERM").unwrap_or_default();
                    let truecolor = colorterm == "truecolor" || colorterm == "24bit" || term.contains("256color") || term.contains("xterm");
                    Self { enabled: true, truecolor }
                }
            }
            _ => Self { enabled: is_atty, truecolor: is_atty },
        }
    }

    pub fn no_color() -> Self {
        Self { enabled: false, truecolor: false }
    }
}

/// Linear interpolation between two RGB points.
pub fn lerp_rgb(c1: (u8, u8, u8), c2: (u8, u8, u8), t: f32) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    (
        (c1.0 as f32 + (c2.0 as f32 - c1.0 as f32) * t).round() as u8,
        (c1.1 as f32 + (c2.1 as f32 - c1.1 as f32) * t).round() as u8,
        (c1.2 as f32 + (c2.2 as f32 - c1.2 as f32) * t).round() as u8,
    )
}

/// Standard bioinformatic physicochemical color palette (Zappo / Lesk standard).
pub fn aa_rgb(aa: char) -> (u8, u8, u8) {
    match aa {
        // Aliphatic / Hydrophobic (Blue)
        'A' | 'V' | 'L' | 'I' | 'M' => (59, 130, 246),
        // Aromatic (Purple)
        'F' | 'W' | 'Y' => (139, 92, 246),
        // Basic / Positively Charged (Cyan)
        'K' | 'R' | 'H' => (6, 182, 212),
        // Acidic / Negatively Charged (Red)
        'D' | 'E' => (239, 68, 68),
        // Polar Uncharged (Green)
        'S' | 'T' | 'N' | 'Q' => (16, 185, 129),
        // Cysteines / Disulfide (Amber)
        'C' => (245, 158, 11),
        // Conformationally flexible / Glycine & Proline (Pink)
        'G' | 'P' => (236, 72, 153),
        // Mask token '*' (Bright Yellow)
        '*' => (250, 204, 21),
        // Default / Unknown (Neutral Slate)
        _ => (156, 163, 175),
    }
}

/// Physicochemical category name for an amino acid.
pub fn aa_category(aa: char) -> &'static str {
    match aa {
        'A' | 'V' | 'L' | 'I' | 'M' => "Hydrophobic Aliphatic",
        'F' | 'W' | 'Y' => "Hydrophobic Aromatic",
        'K' | 'R' | 'H' => "Basic Positively Charged",
        'D' | 'E' => "Acidic Negatively Charged",
        'S' | 'T' | 'N' | 'Q' => "Polar Uncharged",
        'C' => "Cysteine (Disulfide / Redox)",
        'G' | 'P' => "Special Conformation (Loop/Kink)",
        '*' => "Masked Token (<mask_1>)",
        _ => "Non-standard Residue",
    }
}

/// Formats an amino acid character with ANSI TrueColor foreground.
pub fn format_aa(aa: char, theme: &TerminalTheme) -> String {
    if !theme.enabled {
        return aa.to_string();
    }
    let (r, g, b) = aa_rgb(aa);
    format!("\x1b[38;2;{};{};{}m{}\x1b[0m", r, g, b, aa)
}

/// Formats an entire protein sequence string with per-residue physicochemical colors.
pub fn format_sequence(seq: &str, theme: &TerminalTheme) -> String {
    if !theme.enabled {
        return seq.to_string();
    }
    let mut out = String::with_capacity(seq.len() * 20);
    for c in seq.chars() {
        let (r, g, b) = aa_rgb(c);
        out.push_str(&format!("\x1b[38;2;{};{};{}m{}", r, g, b, c));
    }
    out.push_str("\x1b[0m");
    out
}

/// Evaluates density character glyph for monochrome accessibility.
pub fn delta_glyph(delta: f32) -> &'static str {
    if delta >= 0.0 {
        "█"
    } else if delta >= -1.0 {
        "▓"
    } else if delta >= -2.5 {
        "▒"
    } else if delta >= -4.0 {
        "░"
    } else {
        "·"
    }
}

/// Formats a DMS heatmap cell with thermal background color and density glyph.
/// Gradient:
///   ΔLogP <= -4.0: Deep Crimson (#7F1D1D) [Strictly Lethal]
///   -4.0 < ΔLogP < -1.0: Dark Amber (#B45309) [Deleterious]
///   -1.0 <= ΔLogP < 0.0: Charcoal Slate (#1F2937) [Mildly Tolerated]
///   ΔLogP == 0.0: Neutral Dark (#111827) [Wild-Type Reference]
///   ΔLogP > 0.0: Electric Deep Blue (#1E3A8A) [Beneficial / Favorable]
pub fn format_dms_cell(delta: f32, is_wt: bool, theme: &TerminalTheme) -> String {
    let glyph = delta_glyph(delta);
    if !theme.enabled {
        return format!(" {} ", glyph);
    }

    let bg_rgb = if is_wt {
        (17, 24, 39) // Dark charcoal for WT reference
    } else if delta <= -4.0 {
        (127, 29, 29) // Deep crimson
    } else if delta < -1.0 {
        // Interpolate crimson -> amber
        let t = (delta + 4.0) / 3.0;
        lerp_rgb((127, 29, 29), (180, 83, 9), t)
    } else if delta < 0.0 {
        // Interpolate amber -> charcoal
        let t = delta + 1.0;
        lerp_rgb((180, 83, 9), (31, 41, 55), t)
    } else {
        // Interpolate charcoal -> deep electric blue
        let t = (delta / 1.5).min(1.0);
        lerp_rgb((31, 41, 55), (30, 58, 138), t)
    };

    let fg_rgb = if is_wt {
        (59, 130, 246) // Bright blue glyph for WT
    } else if delta <= -4.0 {
        (254, 202, 202) // Light pink/red text
    } else if delta >= 0.0 {
        (147, 197, 253) // Light cyan/blue text
    } else {
        (229, 231, 235) // Neutral light gray
    };

    format!(
        "\x1b[48;2;{};{};{}m\x1b[38;2;{};{};{}m {} \x1b[0m",
        bg_rgb.0, bg_rgb.1, bg_rgb.2,
        fg_rgb.0, fg_rgb.1, fg_rgb.2,
        glyph
    )
}

/// Dynamic console progress bar with fractional blocks and live metrics.
pub struct ProgressBar {
    total: usize,
    start_time: Instant,
    last_update: Instant,
    theme: TerminalTheme,
    description: String,
}

impl ProgressBar {
    pub fn new(total: usize, description: &str, theme: TerminalTheme) -> Self {
        let now = Instant::now();
        Self {
            total,
            start_time: now,
            last_update: now,
            theme,
            description: description.to_string(),
        }
    }

    pub fn update(&mut self, current: usize) {
        if !self.theme.enabled {
            return;
        }

        // Throttle updates to ~30 FPS (every 33ms) to minimize I/O overhead
        if current < self.total && self.last_update.elapsed().as_millis() < 33 {
            return;
        }
        self.last_update = Instant::now();

        let progress = (current as f64 / self.total as f64).clamp(0.0, 1.0);
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let speed = if elapsed > 0.0 { current as f64 / elapsed } else { 0.0 };
        let eta = if speed > 0.0 && current < self.total {
            (self.total - current) as f64 / speed
        } else {
            0.0
        };

        const BAR_WIDTH: usize = 28;
        let filled_fraction = progress * (BAR_WIDTH as f64);
        let full_blocks = filled_fraction.floor() as usize;
        let remainder = filled_fraction - full_blocks as f64;

        const FRACTIONAL_BLOCKS: [&str; 8] = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉"];
        let frac_index = (remainder * 8.0).floor() as usize;
        let frac_glyph = if frac_index > 0 && frac_index < 8 {
            FRACTIONAL_BLOCKS[frac_index]
        } else {
            " "
        };

        let mut bar = String::with_capacity(BAR_WIDTH * 4);
        for _ in 0..full_blocks.min(BAR_WIDTH) {
            bar.push('█');
        }
        if full_blocks < BAR_WIDTH {
            bar.push_str(frac_glyph);
            for _ in (full_blocks + 1)..BAR_WIDTH {
                bar.push('░');
            }
        }

        // Write to stderr so standard output data streams remain pristine
        eprint!(
            "\r\x1b[2K\x1b[38;2;139;92;246m{}\x1b[0m [\x1b[38;2;16;185;129m{}\x1b[0m] {:>5.1}% | {:>4}/{} | {:>6.1} ops/s | ETA: {:>4.1}s",
            self.description, bar, progress * 100.0, current, self.total, speed, eta
        );
        let _ = io::stderr().flush();
    }

    pub fn finish(&self) {
        if self.theme.enabled {
            eprintln!("\r\x1b[2K\x1b[38;2;16;185;129m✓\x1b[0m {} complete ({} items in {:.2}s).",
                self.description, self.total, self.start_time.elapsed().as_secs_f64());
            let _ = io::stderr().flush();
        }
    }
}
