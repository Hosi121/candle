# CPU ConvTranspose1d with and without padding

This record compares main `ddf1b879dc3a1760cbcb3f3c4a7c6467850cec4a` with `6db8617`, the two commits in PR #3988. The first commit changes output access. The second commit enables GEMM for padded calls. It keeps the existing path selection when both padding fields are zero.

| Case | Threads | main | PR | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| DAC decoder | 1 | 635.619 ms | 375.134 ms | 41.0% |
| DAC decoder | 4 | 285.905 ms | 206.007 ms | 27.9% |
| Layer without padding | 1 | 2.189 ms | 0.462 ms | 78.9% |
| DAC first transpose layer | 1 | 193.808 ms | 15.723 ms | 91.9% |

## Method

- Intel Core Ultra 7 255H, Linux/WSL2, Rust 1.97.1, release, `-C target-cpu=native`, F32, default CPU features.
- One thread uses CPU 0. Four threads use CPUs 0–3. Set `RAYON_NUM_THREADS` for each process.
- Five process pairs per thread count. Alternate the process order. Take five samples per case and report the median of process medians.
- Inputs and weights use fixed nonzero values. Each layer call includes GEMM, the fold, and allocation. Model setup is outside the timed loop.
- The full DAC decoder has 1024 input channels, width 1536, and rates `[8,8,4,2]`. Input `[1,1024,16]` produces `[1,1,8192]`. Parameter names set the seed for generated weights. The model uses its normal weight normalization code. No trained checkpoint is required. Audio quality was not tested.

`paired.json` has every sample and shape. `summary.json` has all 11 cases and output error measures. Each output value is finite. The script requires relative L2 error below 1e-5 and maximum absolute error below 1e-5 times the maximum reference magnitude.

## Memory and limits

The direct path has a `B * L * Cin` input copy. The new GEMM path has a `B * L * Cout * K` column buffer, plus GEMM packing space. The four DAC column buffers have sizes 0.75, 3, 6, and 6 MiB. The corresponding input copies have sizes 0.094, 0.375, 1.5, and 3 MiB. These are buffer sizes, not peak process memory.

The existing GEMM cases keep their addition order. The new cases can change rounding. Dilation other than 1 and noncontiguous kernels keep direct computation. The new cases also keep direct computation for unsupported types and native MatMul batch layouts affected by #3758. ARM, GPU, MKL, and Accelerate performance was not measured.

## Reproduce

Use the Cargo.lock in the parent directory. For each revision, copy `probe.rs` to `candle-transformers/examples/col2im_padding_probe.rs` and build:

```bash
CARGO_BUILD_JOBS=6 CARGO_INCREMENTAL=0 cargo build --locked --release -p candle-transformers --example col2im_padding_probe
```

Save `target/release/examples/col2im_padding_probe` beside `bench.py` as `baseline_probe` or `candidate_probe`. Run:

```bash
python3 bench.py
```

The script writes the full outputs for the first process pair. `OUTPUT_DIR` controls the output directory. `CASE_FILTER` selects one case when the probe runs on its own.

Checks on the final PR commit:

```bash
cargo test --locked --release -p candle-core --lib --tests
cargo bench --locked -p candle-core --bench bench_main -- cpu_conv_transpose1d --test
cargo clippy --locked --release -p candle-core --tests --examples --benches -- -D warnings
cargo fmt --all -- --check
```

All 187 core tests and nine Criterion cases pass. `metadata.json` records the commits, compiler, build options, and file hashes.
