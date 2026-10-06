use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput, measurement::Measurement, BenchmarkGroup};
use codon_infer::{EsmConfig, EsmWeights, RunState, forward, Tokenizer, SimpleRng, sample};
use std::fs::File;
use std::hint::black_box;
use std::time::Duration;

fn load_model() -> (EsmConfig, EsmWeights) {
    let config = EsmConfig {
        dim: 320,
        hidden_dim: 1280,
        n_layers: 6,
        n_heads: 20,
        vocab_size: 33,
        max_seq_len: 1024,
    };
    let mut file = File::open("esm2_8m_weights.bin").expect("Failed to open model weights");
    let weights = EsmWeights::load(&mut file, &config).expect("Failed to load weights");
    (config, weights)
}

fn configure_group<M: Measurement>(group: &mut BenchmarkGroup<M>) {
    group.sample_size(10).warm_up_time(Duration::from_secs(1)).measurement_time(Duration::from_secs(3));
}

pub fn bench_prefill_latency(c: &mut Criterion) {
    let (config, weights) = load_model();
    let mut group = c.benchmark_group("prefill_latency");
    configure_group(&mut group);
    
    let prompt_lengths = [16, 32, 64, 128];
    
    for &len in &prompt_lengths {
        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |b, &len| {
            let prompt = "M".repeat(len);
            let tokens = Tokenizer::encode(&prompt);
            let prompt_tokens = &tokens[..tokens.len().saturating_sub(1)];
            
            b.iter(|| {
                let mut run_state = RunState::new(&config);
                for (pos, &token) in prompt_tokens.iter().enumerate() {
                    forward(black_box(token), black_box(pos), black_box(&config), black_box(&weights), black_box(&mut run_state));
                }
                black_box(run_state.logits[0])
            });
        });
    }
    group.finish();
}

pub fn bench_decode_throughput(c: &mut Criterion) {
    let (config, weights) = load_model();
    let mut group = c.benchmark_group("decode_throughput");
    configure_group(&mut group);
    
    let prompt = "MCMPCFTTDHQMARKCDD";
    let prompt_tokens = Tokenizer::encode(prompt);
    let prompt_tokens = &prompt_tokens[..prompt_tokens.len().saturating_sub(1)];
    
    let num_decode_tokens = 20;
    group.throughput(Throughput::Elements(num_decode_tokens as u64));
    
    group.bench_function("decode_20_tokens", |b| {
        b.iter(|| {
            let mut run_state = RunState::new(&config);
            let mut pos = 0;
            
            for &token in prompt_tokens.iter() {
                forward(black_box(token), black_box(pos), black_box(&config), black_box(&weights), black_box(&mut run_state));
                pos += 1;
            }
            
            let mut next_token = prompt_tokens[prompt_tokens.len() - 1];
            let mut rng = SimpleRng::new(42);
            let temperature = 0.7;
            
            for _ in 0..num_decode_tokens {
                forward(black_box(next_token), black_box(pos), black_box(&config), black_box(&weights), black_box(&mut run_state));
                next_token = sample(black_box(&mut run_state.logits), black_box(temperature), black_box(&mut rng));
                pos += 1;
                if next_token == 2 {
                    break;
                }
            }
            black_box(pos)
        });
    });
    group.finish();
}

pub fn bench_kv_cache_memory(c: &mut Criterion) {
    let config = EsmConfig {
        dim: 320,
        hidden_dim: 1280,
        n_layers: 6,
        n_heads: 20,
        vocab_size: 33,
        max_seq_len: 1024,
    };
    let mut group = c.benchmark_group("kv_cache_memory");
    configure_group(&mut group);
    
    let seq_lengths = [64, 128, 256, 512, 1024];
    
    for &len in &seq_lengths {
        group.bench_function(format!("seq_len_{len}"), |b| {
            b.iter(|| {
                let run_state = RunState::new(black_box(&config));
                black_box(run_state.key_cache.len());
                black_box(run_state.value_cache.len());
            });
        });
    }
    group.finish();
}

pub fn bench_full_inference(c: &mut Criterion) {
    let (config, weights) = load_model();
    let mut group = c.benchmark_group("full_inference");
    configure_group(&mut group);
    
    let prompt = "MCMPCFTTDHQMARKCDD";
    let prompt_tokens = Tokenizer::encode(prompt);
    let prompt_tokens = &prompt_tokens[..prompt_tokens.len().saturating_sub(1)];
    let num_decode = 20;
    
    group.throughput(Throughput::Elements((prompt_tokens.len() + num_decode) as u64));
    
    group.bench_function("prompt_plus_20_decode", |b| {
        b.iter(|| {
            let mut run_state = RunState::new(&config);
            let mut pos = 0;
            
            for &token in prompt_tokens.iter() {
                forward(black_box(token), black_box(pos), black_box(&config), black_box(&weights), black_box(&mut run_state));
                pos += 1;
            }
            
            let mut next_token = prompt_tokens[prompt_tokens.len() - 1];
            let mut rng = SimpleRng::new(42);
            let temperature = 0.7;
            
            for _ in 0..num_decode {
                forward(black_box(next_token), black_box(pos), black_box(&config), black_box(&weights), black_box(&mut run_state));
                next_token = sample(black_box(&mut run_state.logits), black_box(temperature), black_box(&mut rng));
                pos += 1;
                if next_token == 2 {
                    break;
                }
            }
            black_box(pos)
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_prefill_latency,
    bench_decode_throughput,
    bench_kv_cache_memory,
    bench_full_inference
);
criterion_main!(benches);