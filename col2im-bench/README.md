# CPU ConvTranspose1d measurements

The [combined record](combined/README.md) covers the full PR at `6db8617`, including padded calls and the complete DAC decoder. This page records the first commit, `e15ffebc`.

The complete layer call includes GEMM, allocation, and col2im. Inputs and weights are fixed nonzero values. No model weights are required. The EnCodec cases use the decoder channel counts and upsampling ratios with synthetic input lengths. These are layer measurements.

The host is a Core Ultra 7 255H under WSL2. The build uses Rust 1.97.1, the release profile, and `-C target-cpu=native`. `RAYON_NUM_THREADS` is 1 or 4. CPU affinity is 0 or 0–3. Each process has warmup calls and seven timed samples. Five process pairs run in alternating order. Times in the summary are medians of process medians. Both the ratio of medians and each paired reduction are included.

`paired.json` contains every final sample. `summary.json` contains all 12 shapes and both thread counts. `metadata.json` gives the source commits and hashes. `probe.rs` also checks output hashes. The committed tests use independent references: 216 fold cases and 180 complete convolution cases. The tests cover sums that cancel, signed zero, infinities, NaNs, groups, offsets, and input views.

Contiguous batched input has a separate MatMul error on the base revision; see [#3758](https://github.com/huggingface/candle/pull/3758). The batched probe stores the input in NLC order and uses a view with NCL dimensions. Independent convolution tests also check this layout.

From this branch, copy the files in `col2im-bench` to a temporary directory. Build both commits with the same lock file and compiler:

```bash
bench_dir=$(mktemp -d)
cp col2im-bench/* "$bench_dir/"
for variant in baseline candidate; do
    if [ "$variant" = baseline ]; then
        ref=ddf1b879dc3a1760cbcb3f3c4a7c6467850cec4a
    else
        ref=e15ffebc421bd31a0ecb54ea255fa154e3e34716
    fi
    git switch --detach "$ref"
    cp "$bench_dir/probe.rs" candle-core/examples/col2im_probe.rs
    cp "$bench_dir/Cargo.lock" Cargo.lock
    CARGO_INCREMENTAL=0 cargo build --locked --release -p candle-core --example col2im_probe
    cp target/release/examples/col2im_probe "$bench_dir/${variant}_probe"
done
python3 "$bench_dir/bench.py"
```

Use a clean checkout for these commands. Results are written to the temporary directory. The same build directory is reused.

Checks on the candidate:

```bash
cargo test --release -p candle-core --lib --tests
cargo bench -p candle-core --bench bench_main -- cpu_conv_transpose1d --test
cargo clippy --release -p candle-core --tests --examples --benches -- -D warnings
cargo fmt --all -- --check
```
