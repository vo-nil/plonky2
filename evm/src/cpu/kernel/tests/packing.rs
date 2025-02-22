use anyhow::Result;
use ethereum_types::U256;

use crate::cpu::kernel::aggregator::KERNEL;
use crate::cpu::kernel::interpreter::Interpreter;
use crate::memory::segments::Segment;

#[test]
fn test_mload_packing_1_byte() -> Result<()> {
    let mload_packing = KERNEL.global_labels["mload_packing"];

    let retdest = 0xDEADBEEFu32.into();
    let len = 1.into();
    let addr = (Segment::RlpRaw as u64 + 2).into();
    let initial_stack = vec![retdest, len, addr];

    let mut interpreter = Interpreter::new_with_kernel(mload_packing, initial_stack);
    interpreter.set_rlp_memory(vec![0, 0, 0xAB]);

    interpreter.run()?;
    assert_eq!(interpreter.stack(), vec![0xAB.into()]);

    Ok(())
}

#[test]
fn test_mload_packing_3_bytes() -> Result<()> {
    let mload_packing = KERNEL.global_labels["mload_packing"];

    let retdest = 0xDEADBEEFu32.into();
    let len = 3.into();
    let addr = (Segment::RlpRaw as u64 + 2).into();
    let initial_stack = vec![retdest, len, addr];

    let mut interpreter = Interpreter::new_with_kernel(mload_packing, initial_stack);
    interpreter.set_rlp_memory(vec![0, 0, 0xAB, 0xCD, 0xEF]);

    interpreter.run()?;
    assert_eq!(interpreter.stack(), vec![0xABCDEF.into()]);

    Ok(())
}

#[test]
fn test_mload_packing_32_bytes() -> Result<()> {
    let mload_packing = KERNEL.global_labels["mload_packing"];

    let retdest = 0xDEADBEEFu32.into();
    let len = 32.into();
    let addr = (Segment::RlpRaw as u64).into();
    let initial_stack = vec![retdest, len, addr];

    let mut interpreter = Interpreter::new_with_kernel(mload_packing, initial_stack);
    interpreter.set_rlp_memory(vec![0xFF; 32]);

    interpreter.run()?;
    assert_eq!(interpreter.stack(), vec![U256::MAX]);

    Ok(())
}

#[test]
fn test_mstore_unpacking() -> Result<()> {
    let mstore_unpacking = KERNEL.global_labels["mstore_unpacking"];

    let retdest = 0xDEADBEEFu32.into();
    let len = 4.into();
    let value = 0xABCD1234u32.into();
    let addr = (Segment::TxnData as u64).into();
    let initial_stack = vec![retdest, len, value, addr];

    let mut interpreter = Interpreter::new_with_kernel(mstore_unpacking, initial_stack);

    interpreter.run()?;
    assert_eq!(interpreter.stack(), vec![addr + U256::from(4)]);
    assert_eq!(
        &interpreter.get_txn_data(),
        &[0xAB.into(), 0xCD.into(), 0x12.into(), 0x34.into()]
    );

    Ok(())
}
