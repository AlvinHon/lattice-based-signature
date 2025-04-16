use criterion::{criterion_group, criterion_main, Criterion};

criterion_main!(signature_scheme);

criterion_group! {
    name = signature_scheme;
    config = Criterion::default().sample_size(10);
    targets = bench_set_1_sign, bench_set_1_verify,
              bench_set_2_sign, bench_set_2_verify
}

use lattice_based_signature::{
    basic::RandKeyGen,
    params::{set_1, set_2},
};
use ripemd::Ripemd160;

fn bench_set_1_sign(c: &mut Criterion) {
    let rng = &mut rand::thread_rng();
    let params = set_1();
    let sk = rng.gen_signing_key(&params);
    let vk = rng.gen_verifying_key(&params, &sk);
    let message = vec![
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    ];

    c.bench_function("set 1/sign", |b| {
        b.iter(|| {
            _ = sk.sign::<_, Ripemd160>(rng, &params, &vk, &message);
        })
    });
}

fn bench_set_1_verify(c: &mut Criterion) {
    let rng = &mut rand::thread_rng();
    let params = set_1();
    let sk = rng.gen_signing_key(&params);
    let vk = rng.gen_verifying_key(&params, &sk);
    let message = vec![
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    ];
    let sig = sk.sign::<_, Ripemd160>(rng, &params, &vk, &message);

    c.bench_function("set 1/verify", |b| {
        b.iter(|| {
            _ = vk.verify::<Ripemd160>(&params, &message, &sig);
        })
    });
}

fn bench_set_2_sign(c: &mut Criterion) {
    let rng = &mut rand::thread_rng();
    let params = set_2();
    let sk = rng.gen_signing_key(&params);
    let vk = rng.gen_verifying_key(&params, &sk);
    let message = vec![
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    ];

    c.bench_function("set 2/sign", |b| {
        b.iter(|| {
            _ = sk.sign::<_, Ripemd160>(rng, &params, &vk, &message);
        })
    });
}

fn bench_set_2_verify(c: &mut Criterion) {
    let rng = &mut rand::thread_rng();
    let params = set_2();
    let sk = rng.gen_signing_key(&params);
    let vk = rng.gen_verifying_key(&params, &sk);
    let message = vec![
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    ];
    let sig = sk.sign::<_, Ripemd160>(rng, &params, &vk, &message);

    c.bench_function("set 2/verify", |b| {
        b.iter(|| {
            _ = vk.verify::<Ripemd160>(&params, &message, &sig);
        })
    });
}
