# llama-inference-c

A from-scratch implementation of Llama-style transformer inference in C,
with a focus on **secure parsing** of the model file format.

> Work in progress. Currently: attention interaction, softmax, wo, and KV-cache

## Why this project

Most LLM inference is a black box behind `model()`. This project rebuilds it
in C, one piece at a time, to understand it from the inside: matmul, attention,
RoPE, quantization, while paying special attention to the **security of the
model-loading code**, an area with a real and active track record of
vulnerabilities.

## Current status

- [x] Parse the model file header (`Config`: dim, n_layers, n_heads, vocab_size…)
- [x] Compute the full expected weight layout, byte-for-byte
- [x] **Overflow-safe file validation** (see below)
- [x] Memory-map and load the weights
- [x] matmul
- [ ] attention + RoPE
- [ ] text generation
- [ ] int8/int4 quantization

## The security angle

The header of a `.bin` model file contains dimensions (`dim`, `n_layers`…)
that are read straight from the file, **unvalidated**. A malicious file can
declare huge dimensions, causing size calculations to overflow and the loader
to read/write out of bounds, a heap overflow.

This is not hypothetical. The same class of bug hit `llama.cpp` repeatedly:
- **CVE-2025-53630** : integer overflow in GGUF parsing → heap out-of-bounds
- **CVE-2026-27940** : bypass of the above fix (per-addition check, but not the total)
- **CVE-2026-33298** : integer overflow in `ggml_nbytes` → heap overflow → potential RCE

This loader defends against it: it computes the exact expected file size from
the header dimensions and compares it to the real file size, **rejecting any
mismatch**. Every size calculation is checked for overflow *before* the
operation (multiplication via `SIZE_MAX / n`, addition via `SIZE_MAX - n`),
so the validation itself can't be bypassed by the very overflow it guards against.

## Build & run

    gcc -o main main.c
    ./main

Expects `stories15M.bin` (Karpathy's tiny Llama model) in the same directory.

## Credits

Structure inspired by [llama2.c](https://github.com/karpathy/llama2.c) by
Andrej Karpathy. Reimplemented from scratch for learning, with added
security-focused input validation.