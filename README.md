# Practical Lattice-Based Cryptography

A straightforward rust implementation of Lattice-based signature scheme defined in [Practical Lattice-Based Cryptography: A Signature Scheme for Embedded Systems](https://www.iacr.org/archive/ches2012/74280529/74280529.pdf).

> a signature scheme whose security is derived from the hardness of lattice problems. It is based on recent theoretical advances in lattice-based cryptography and is highly optimized for practicability and use in embedded systems.

Note: the current progress implements only the basic scheme which does not aim to run on embedded systems. In future, the optimized scheme should be implemented and compatible with FPGA (e.g. by using [RustHDL](https://rust-hdl.org/)).

## Example

There are two sets of parameters (`Set 1` and `Set 2`) provided as defined in the paper (`Table 1`).


In the following example, `set_1` is used:

```rust
use lattice_based_signature::{params::set_1, basic::RandKeyGen};
use rand::Rng;;
use ripemd::Ripemd160;

let rng = &mut rand::thread_rng();
let params = set_1();
let sk = rng.gen_signing_key(&params);
let vk = rng.gen_verifying_key(&params, &sk);

// random 20-byte message
let message = (0..20).map(|_| rng.gen()).collect::<Vec<u8>>();

let sig = sk.sign::<_, Ripemd160>(rng, &params, &vk, &message);
assert!(vk.verify::<Ripemd160>(&params, &message, &sig));
```