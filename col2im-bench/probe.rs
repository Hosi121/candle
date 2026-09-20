use candle_core::{DType, Device, Result, Tensor};
use std::{hint::black_box, time::Instant};

fn main() -> Result<()> {
    let cases = [
        ("tiny", 1, 2, 3, 7, 3, 2),
        ("stream", 1, 64, 32, 8, 4, 2),
        ("encodec0", 1, 512, 256, 75, 16, 8),
        ("encodec1", 1, 256, 128, 600, 10, 5),
        ("encodec2", 1, 128, 64, 3000, 8, 4),
        ("encodec3", 1, 64, 32, 12000, 4, 2),
        ("long", 1, 8, 64, 8192, 16, 8),
        ("batch", 4, 16, 32, 1024, 8, 4),
        ("overlap", 1, 16, 64, 2048, 7, 1),
        ("no_overlap", 1, 16, 64, 2048, 4, 4),
        ("gaps", 1, 16, 64, 2048, 3, 5),
        ("pointwise", 1, 16, 64, 2048, 1, 1),
    ];
    let filter = std::env::var("CASE_FILTER").unwrap_or_default();
    let dtype = if std::env::var("DTYPE").as_deref() == Ok("f64") {
        DType::F64
    } else {
        DType::F32
    };
    for (name, b, ci, co, len, kernel, stride) in cases {
        if !name.contains(&filter) {
            continue;
        }
        let data = |n| {
            (0..n)
                .map(|i: usize| ((i.wrapping_mul(7919) % 127) as f32 - 63.) / 64.)
                .collect::<Vec<_>>()
        };
        let x =
            Tensor::from_vec(data(b * ci * len), (b, ci, len), &Device::Cpu)?.to_dtype(dtype)?;
        let x = if b > 1 {
            x.transpose(1, 2)?.contiguous()?.transpose(1, 2)?
        } else {
            x
        };
        let w = Tensor::from_vec(data(ci * co * kernel), (ci, co, kernel), &Device::Cpu)?
            .to_dtype(dtype)?;
        let run = || x.conv_transpose1d(&w, 0, 0, stride, 1, 1);
        let output = run()?
            .flatten_all()?
            .to_dtype(DType::F64)?
            .to_vec1::<f64>()?;
        let hash = output.iter().fold(0xcbf29ce484222325u64, |s, v| {
            (s ^ v.to_bits()).wrapping_mul(0x100000001b3)
        });
        for _ in 0..3 {
            black_box(run()?);
        }
        let start = Instant::now();
        for _ in 0..3 {
            black_box(run()?);
        }
        let iters = (0.025 / (start.elapsed().as_secs_f64() / 3.))
            .ceil()
            .clamp(3., 10000.) as usize;
        let mut samples = Vec::new();
        for _ in 0..7 {
            let start = Instant::now();
            for _ in 0..iters {
                black_box(run()?);
            }
            samples.push(start.elapsed().as_secs_f64() * 1000. / iters as f64);
        }
        samples.sort_by(f64::total_cmp);
        println!("{{\"case\":\"{name}\",\"dtype\":\"{dtype:?}\",\"shape\":[{b},{ci},{co},{len},{kernel},{stride}],\"iterations\":{iters},\"median_ms\":{},\"samples_ms\":{samples:?},\"hash\":\"{hash:016x}\"}}", samples[3]);
    }
    Ok(())
}
