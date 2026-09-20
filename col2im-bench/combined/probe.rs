use candle::{DType, Device, Module, Result, Shape, Tensor};
use candle_nn::{var_builder::SimpleBackend, Init, VarBuilder};
use std::{hint::black_box, time::Instant};

fn values(n: usize, mut seed: u64, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * scale
        })
        .collect()
}

struct Weights;
impl SimpleBackend for Weights {
    fn get(&self, s: Shape, name: &str, _: Init, dt: DType, dev: &Device) -> Result<Tensor> {
        let seed = name.bytes().fold(0xcbf29ce484222325u64, |a, b| {
            (a ^ b as u64).wrapping_mul(0x100000001b3)
        });
        let v = if name.ends_with("alpha") {
            vec![1.; s.elem_count()]
        } else if name.ends_with("weight_g") {
            vec![0.8; s.elem_count()]
        } else if name.ends_with("bias") {
            vec![0.; s.elem_count()]
        } else {
            values(s.elem_count(), seed, 1.)
        };
        Tensor::from_vec(v, s, dev)?.to_dtype(dt)
    }
    fn get_unchecked(&self, name: &str, _: DType, _: &Device) -> Result<Tensor> {
        candle::bail!("A shape is required for {name}")
    }
    fn contains_tensor(&self, _: &str) -> bool {
        true
    }
}

fn measure(name: &str, shape: &[usize], run: impl Fn() -> Result<Tensor>) -> Result<()> {
    let output = run()?;
    let out_shape = output.dims().to_vec();
    let output = output.flatten_all()?.to_vec1::<f32>()?;
    assert!(output.iter().all(|x| x.is_finite()));
    let hash = output.iter().fold(0xcbf29ce484222325u64, |a, b| {
        (a ^ b.to_bits() as u64).wrapping_mul(0x100000001b3)
    });
    if let Ok(dir) = std::env::var("OUTPUT_DIR") {
        let bytes: Vec<u8> = output.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(format!("{dir}/{name}.f32"), bytes)?;
    }
    for _ in 0..2 {
        black_box(run()?);
    }
    let start = Instant::now();
    black_box(run()?);
    let iters = (0.025 / start.elapsed().as_secs_f64())
        .ceil()
        .clamp(1., 10000.) as usize;
    let mut samples = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..iters {
            black_box(run()?);
        }
        samples.push(start.elapsed().as_secs_f64() * 1000. / iters as f64);
    }
    samples.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::json!({"case":name, "shape":shape, "out_shape":out_shape,
        "median_ms":samples[2], "samples_ms":samples, "iterations":iters,
        "hash":format!("{hash:016x}")})
    );
    Ok(())
}

fn main() -> Result<()> {
    let filter = std::env::var("CASE_FILTER").unwrap_or_default();
    for (name, ci, co, len, k, stride, pad, out_pad) in [
        ("dac0", 1536, 768, 16, 16, 8, 4, 0),
        ("dac1", 768, 384, 128, 16, 8, 4, 0),
        ("dac2", 384, 192, 1024, 8, 4, 2, 0),
        ("dac3", 192, 96, 4096, 4, 2, 1, 0),
        ("snac_odd", 256, 128, 256, 10, 5, 3, 1),
        ("padded_gap", 16, 64, 2048, 3, 5, 2, 3),
        ("outpad_only", 16, 64, 2048, 4, 4, 0, 3),
        ("tiny_pad", 2, 3, 7, 3, 2, 1, 1),
        ("old_overlap", 16, 64, 2048, 7, 1, 0, 0),
        ("old_no_overlap", 16, 64, 2048, 4, 4, 0, 0),
    ] {
        if !name.contains(&filter) {
            continue;
        }
        let x = Tensor::from_vec(values(ci * len, 42, 2.), (1, ci, len), &Device::Cpu)?;
        let w = Tensor::from_vec(values(ci * co * k, 73, 2.), (ci, co, k), &Device::Cpu)?;
        measure(name, &[1, ci, co, len, k, stride, pad, out_pad], || {
            x.conv_transpose1d(&w, pad, out_pad, stride, 1, 1)
        })?;
    }
    if "dac_decoder".contains(&filter) {
        let vb = VarBuilder::from_backend(Box::new(Weights), DType::F32, Device::Cpu);
        let model =
            candle_transformers::models::dac::Decoder::new(1024, 1536, &[8, 8, 4, 2], 1, vb)?;
        let x = Tensor::from_vec(values(1024 * 16, 29, 0.5), (1, 1024, 16), &Device::Cpu)?;
        measure("dac_decoder", &[1, 1024, 16], || model.forward(&x))?;
    }
    Ok(())
}
