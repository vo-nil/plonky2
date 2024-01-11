use anyhow::Result;
use core::num::ParseIntError;
use core::ops::RangeInclusive;
use core::str::FromStr;

use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig, PoseidonGoldilocksConfig};

use plonky2::plonk::circuit_data::{CircuitConfig, CommonCircuitData, VerifierOnlyCircuitData};

use plonky2::plonk::proof::{CompressedProofWithPublicInputs, ProofWithPublicInputs};
use plonky2::util::timing::TimingTree;
use plonky2::util::serialization::DefaultGateSerializer;
use plonky2_field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::gates::noop::NoopGate;
use plonky2::plonk::prover::prove;
use structopt::StructOpt;

use rand::rngs::OsRng;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use log::{info, Level, LevelFilter};

type ProofTuple<F, C, const D: usize> = (
    ProofWithPublicInputs<F, C, D>,
    VerifierOnlyCircuitData<C, D>,
    CommonCircuitData<F, D>,
);

#[derive(Clone, StructOpt, Debug)]
#[structopt(name = "fibonacci_recursive")]
struct Options {
    /// Verbose mode (-v, -vv, -vvv, etc.)
    #[structopt(short, long, parse(from_occurrences))]
    verbose: usize,

    /// Apply an env_filter compatible log filter
    #[structopt(long, env, default_value)]
    log_filter: String,

    /// Random seed for deterministic runs.
    /// If not specified a new seed is generated from OS entropy.
    #[structopt(long, parse(try_from_str = parse_hex_u64))]
    seed: Option<u64>,

    /// Number of compute threads to use. Defaults to number of cores. Can be a single
    /// value or a rust style range.
    #[structopt(long, parse(try_from_str = parse_range_usize))]
    threads: Option<RangeInclusive<usize>>,

    /// Log2 gate count of the inner proof. Can be a single value or a rust style
    /// range.
    #[structopt(long, default_value="14", parse(try_from_str = parse_range_usize))]
    size: RangeInclusive<usize>,
}



fn recursive_proof<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    InnerC: GenericConfig<D, F = F>,
    const D: usize,
>(
    inner: &ProofTuple<F, InnerC, D>,
    config: &CircuitConfig,
    min_degree_bits: Option<usize>,
) -> Result<ProofTuple<F, C, D>>
where
    InnerC::Hasher: AlgebraicHasher<F>,
{
    let (inner_proof, inner_vd, inner_cd) = inner;
    let mut builder = CircuitBuilder::<F, D>::new(config.clone());
    let pt = builder.add_virtual_proof_with_pis(inner_cd);

    let inner_data = builder.add_virtual_verifier_data(inner_cd.config.fri_config.cap_height);

    builder.verify_proof::<InnerC>(&pt, &inner_data, inner_cd);
    builder.print_gate_counts(0);

    if let Some(min_degree_bits) = min_degree_bits {
        // We don't want to pad all the way up to 2^min_degree_bits, as the builder will
        // add a few special gates afterward. So just pad to 2^(min_degree_bits
        // - 1) + 1. Then the builder will pad to the next power of two,
        // 2^min_degree_bits.
        let min_gates = (1 << (min_degree_bits - 1)) + 1;
        for _ in builder.num_gates()..min_gates {
            builder.add_gate(NoopGate, vec![]);
        }
    }

    let data = builder.build::<C>();

    let mut pw = PartialWitness::new();
    pw.set_proof_with_pis_target(&pt, inner_proof);
    pw.set_verifier_data_target(&inner_data, inner_vd);

    let mut timing = TimingTree::new("prove", Level::Debug);
    let proof = prove::<F, C, D>(&data.prover_only, &data.common, pw, &mut timing)?;
    timing.print();

    data.verify(proof.clone())?;

    Ok((proof, data.verifier_only, data.common))
}

/// Creates a proof for 100th fibonacci number.
fn fibonacci_100th_proof<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    config: &CircuitConfig) -> Result<ProofTuple<F, C, D>> {

    info!("Constructing fibonacci proof");
    let mut builder = CircuitBuilder::<F, D>::new(config.clone());

    // The arithmetic circuit.
    let initial_a = builder.add_virtual_target();
    let initial_b = builder.add_virtual_target();
    let mut prev_target = initial_a;
    let mut cur_target = initial_b;

    for _ in 0..99 {
        let temp = builder.add(prev_target, cur_target);
        prev_target = cur_target;
        cur_target = temp;
    }

    // Public inputs are the two initial values (provided below) and the result (which is generated).
    builder.register_public_input(initial_a);
    builder.register_public_input(initial_b);
    builder.register_public_input(cur_target);

    // Provide initial values.
    let mut pw = PartialWitness::new();
    pw.set_target(initial_a, F::ZERO);
    pw.set_target(initial_b, F::ONE);

    builder.print_gate_counts(0);

    let data = builder.build::<C>();

    let mut timing = TimingTree::new("prove", Level::Debug);
    let proof = prove::<F, C, D>(&data.prover_only, &data.common, pw, &mut timing)?;
    timing.print();
    data.verify(proof.clone())?;

    Ok((proof, data.verifier_only, data.common ))
}

/// An example of using Plonky2 to prove a statement of the form
/// "I know the 100th element of the Fibonacci sequence, starting with constants a and b."
/// When a == 0 and b == 1, this is proving knowledge of the 100th (standard) Fibonacci number.
fn main() -> Result<()> {
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;


    // Parse command line arguments, see `--help` for details.
    let options = Options::from_args_safe()?;
    // Initialize logging
    let mut builder = env_logger::Builder::from_default_env();
    builder.parse_filters(&options.log_filter);
    builder.format_timestamp(None);
    match options.verbose {
        0 => &mut builder,
        1 => builder.filter_level(LevelFilter::Info),
        2 => builder.filter_level(LevelFilter::Debug),
        _ => builder.filter_level(LevelFilter::Trace),
    };
    builder.try_init()?;
    
    // Initialize randomness source
    let rng_seed = options.seed.unwrap_or_else(|| OsRng.next_u64());
    info!("Using random seed {rng_seed:16x}");
    let _rng = ChaCha8Rng::seed_from_u64(rng_seed);
    // TODO: Use `rng` to create deterministic runs

    let config = CircuitConfig::standard_recursion_config();

    let fib_proof = fibonacci_100th_proof(&config)?;
    let (_, _, common_data) = &fib_proof;

    info!("Fibonacci proof produced");
    info!(
        "Initial degree {} = 2^{}",
        common_data.degree(),
        common_data.degree_bits()
    );

    // Recursively verify the proof
    let middle = recursive_proof::<F, C, C, D>(&fib_proof, &config, None)?;
    let (_, _, common_data) = &middle;
    info!(
        "Single recursion degree {} = 2^{}",
        common_data.degree(),
        common_data.degree_bits()
    );

    // Recursively verify the proof
    let double = recursive_proof::<F, C, C, D>(&middle, &config, None)?;
    let (proof, vd, common_data) = &middle;
    info!(
        "Double recursion degree {} = 2^{}",
        common_data.degree(),
        common_data.degree_bits()
    );

    test_serialization(proof, vd, common_data)?;

    Ok(())

}

fn parse_hex_u64(src: &str) -> Result<u64, ParseIntError> {
    let src = src.strip_prefix("0x").unwrap_or(src);
    u64::from_str_radix(src, 16)
}

fn parse_range_usize(src: &str) -> Result<RangeInclusive<usize>, ParseIntError> {
    if let Some((left, right)) = src.split_once("..=") {
        Ok(RangeInclusive::new(
            usize::from_str(left)?,
            usize::from_str(right)?,
        ))
    } else if let Some((left, right)) = src.split_once("..") {
        Ok(RangeInclusive::new(
            usize::from_str(left)?,
            if right.is_empty() {
                usize::MAX
            } else {
                usize::from_str(right)?.saturating_sub(1)
            },
        ))
    } else {
        let value = usize::from_str(src)?;
        Ok(RangeInclusive::new(value, value))
    }
}

/// Test serialization and print some size info.
fn test_serialization<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    proof: &ProofWithPublicInputs<F, C, D>,
    vd: &VerifierOnlyCircuitData<C, D>,
    common_data: &CommonCircuitData<F, D>,
) -> Result<()> {
    let proof_bytes = proof.to_bytes();
    info!("Proof length: {} bytes", proof_bytes.len());
    let proof_from_bytes = ProofWithPublicInputs::from_bytes(proof_bytes, common_data)?;
    assert_eq!(proof, &proof_from_bytes);

    let now = std::time::Instant::now();
    let compressed_proof = proof.clone().compress(&vd.circuit_digest, common_data)?;
    let decompressed_compressed_proof = compressed_proof
        .clone()
        .decompress(&vd.circuit_digest, common_data)?;
    info!("{:.4}s to compress proof", now.elapsed().as_secs_f64());
    assert_eq!(proof, &decompressed_compressed_proof);

    let compressed_proof_bytes = compressed_proof.to_bytes();
    info!(
        "Compressed proof length: {} bytes",
        compressed_proof_bytes.len()
    );
    let compressed_proof_from_bytes =
        CompressedProofWithPublicInputs::from_bytes(compressed_proof_bytes, common_data)?;
    assert_eq!(compressed_proof, compressed_proof_from_bytes);

    let gate_serializer = DefaultGateSerializer;
    let common_data_bytes = common_data
        .to_bytes(&gate_serializer)
        .map_err(|_| anyhow::Error::msg("CommonCircuitData serialization failed."))?;
    info!(
        "Common circuit data length: {} bytes",
        common_data_bytes.len()
    );
    let common_data_from_bytes =
        CommonCircuitData::<F, D>::from_bytes(common_data_bytes, &gate_serializer)
            .map_err(|_| anyhow::Error::msg("CommonCircuitData deserialization failed."))?;
    assert_eq!(common_data, &common_data_from_bytes);

    Ok(())
}

