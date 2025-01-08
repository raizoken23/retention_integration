import torch
import pytest
from state_kernel.checks import (
    check_tensor_property_pairs,
    check_inputs_created_determinstically,
    check_fn_compiles,
    check_fn_compiles_with_backward,
    check_fake_fn_implementation_matches,
    check_forwards_match,
    check_backwards_match,
    save_to_csv,
)
from state_kernel.utils import partial_with_keywords

SEED = 41

## FORWARD TESTS ##
import pytest
from state_kernel._chunk_state.fwd import (
    create_inputs as create_inputs_fwd,
    chunk_state_fwd,
    chunk_state_fwd_fake,
    ExpandedDim
)

FWD_TEST_CASES = [
    (1, 4, 128, 1, 32, torch.float16, 'cuda'),    # Minimal case with float16
    (2, 8, 128, 4, 32, torch.float16, 'cuda'),    # Common case with float16 
    (4, 32, 128, 8, 64, torch.float16, 'cuda'),   # Large case with float16
    (1, 8, 128, 4, 32, torch.bfloat16, 'cuda'),   # Test bfloat16
    (2, 32, 128, 8, 64, torch.bfloat16, 'cuda'),  # Large case with bfloat16
]

@pytest.mark.parametrize("b,n,c,h,d,dtype,device", FWD_TEST_CASES)
def test_chunk_state_forward_create_inputs(b, n, c, h, d, dtype, device):
    torch.manual_seed(SEED)
    K, V, deg = create_inputs_fwd(b, n, c, h, d, dtype, device)    
    check_tensor_property_pairs(
        (K, ((b, n, c, h, d), dtype, device)), 
        (V, ((b, n, c, h, d), dtype, device))
    )
    assert deg == 2, "Degree must be 2"

@pytest.mark.parametrize("b,n,c,h,d,dtype,device", FWD_TEST_CASES)
def test_chunk_state_forward_output(b, n, c, h, d, dtype, device):
    torch.manual_seed(SEED)
    inputs = create_inputs_fwd(b, n, c, h, d, dtype, device)
    with torch.no_grad():
        S = chunk_state_fwd(*inputs)
    D = ExpandedDim(d, 2)
    check_tensor_property_pairs(
        (S, ((b, n, h, D, d), dtype, device))
    )

@pytest.mark.parametrize("args", FWD_TEST_CASES)
def test_chunk_state_forward_create_inputs_determinism(args):
    check_inputs_created_determinstically(create_inputs_fwd, (*args, SEED))

@pytest.mark.parametrize("args", FWD_TEST_CASES)
def test_chunk_state_forward_compiles(args):
    torch.manual_seed(SEED)
    check_fn_compiles(chunk_state_fwd, create_inputs_fwd(*args))

@pytest.mark.parametrize("args", FWD_TEST_CASES)
def test_chunk_state_forward_fake_implementation(args):
    torch.manual_seed(SEED)
    check_fake_fn_implementation_matches(chunk_state_fwd, chunk_state_fwd_fake, create_inputs_fwd(*args))

@pytest.mark.parametrize("args", FWD_TEST_CASES)
def test_chunk_state_forward_opcheck(args):
    torch.manual_seed(SEED)
    torch.library.opcheck(chunk_state_fwd, create_inputs_fwd(*args), 
        test_utils=('test_schema', 'test_faketensor', 'test_aot_dispatch_dynamic', 'test_aot_dispatch_static'))

## BACKWARD TESTS ##
from state_kernel._chunk_state.bwd import (
    create_inputs as create_inputs_bwd,
    chunk_state_bwd,
    chunk_state_bwd_fake,
)


BWD_TEST_CASES = [
    (1, 4, 128, 8, 32, torch.float16, 'cuda'),    # Minimal case
    (2, 8, 128, 8, 32, torch.float16, 'cuda'),    # Common case with float16
    (4, 32, 128, 8, 32, torch.float16, 'cuda'),   # Large case with float16
    (1, 8, 128, 8, 32, torch.bfloat16, 'cuda'),   # Test bfloat16
    (2, 32, 128, 8, 32, torch.bfloat16, 'cuda'),  # Large case with bfloat16
    (4, 4, 128, 8, 64, torch.bfloat16, 'cuda'),   # Small case with bfloat16
    (1, 32, 128, 8, 64, torch.float16, 'cuda'),   # Large case with float16
    (2, 8, 128, 8, 64, torch.bfloat16, 'cuda'),   # Common case with bfloat16
]

@pytest.mark.parametrize("b,n,c,h,d,dtype,device", BWD_TEST_CASES)
def test_chunk_state_backward_create_inputs(b, n, c, h, d, dtype, device):
    torch.manual_seed(SEED)
    K, V, dS, deg = create_inputs_bwd(b, n, c, h, d, dtype=dtype, device=device)
    D = ExpandedDim(d, deg)
    check_tensor_property_pairs(
        (K, ((b, n, c, h, d), dtype, device)),
        (V, ((b, n, c, h, d), dtype, device)),
        (dS, ((b, n, h, D, d), dtype, device))
    )

@pytest.mark.parametrize("b,n,c,h,d,dtype,device", BWD_TEST_CASES)
def test_chunk_state_backward_output(b, n, c, h, d, dtype, device):
    torch.manual_seed(SEED)
    inputs = create_inputs_bwd(b, n, c, h, d, dtype=dtype, device=device)
    with torch.no_grad():
        dK, dV = chunk_state_bwd(*inputs)
    check_tensor_property_pairs(
        (dK, ((b, n, c, h, d), dtype, device)),
        (dV, ((b, n, c, h, d), dtype, device))
    )

@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("n", [1, 8, 32])
@pytest.mark.parametrize("c", [128, 1024])
@pytest.mark.parametrize("h", [1, 8])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])
def test_chunk_state_backward_create_inputs_determinism(b, n, c, h, d, dtype, device):
    check_inputs_created_determinstically(create_inputs_bwd, (b, n, c, h, d, dtype, device, SEED))

@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("n", [1, 8, 32])
@pytest.mark.parametrize("c", [128, 1024])
@pytest.mark.parametrize("h", [1, 8])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])
def test_chunk_state_backward_compiles(b, n, c, h, d, dtype, device):
    torch.manual_seed(SEED)
    check_fn_compiles(chunk_state_bwd, create_inputs_bwd(b, n, c, h, d, dtype, device))

@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("n", [1, 8, 32])
@pytest.mark.parametrize("c", [128, 1024])
@pytest.mark.parametrize("h", [1, 8])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])
def test_chunk_state_backward_fake_implementation(b, n, c, h, d, dtype, device):
    torch.manual_seed(SEED)
    check_fake_fn_implementation_matches(chunk_state_bwd, chunk_state_bwd_fake, create_inputs_bwd(b, n, c, h, d, dtype, device))

@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("n", [1, 8, 32])
@pytest.mark.parametrize("c", [128, 1024])
@pytest.mark.parametrize("h", [1, 8])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])
def test_chunk_state_backward_opcheck(b, n, c, h, d, dtype, device):
    torch.manual_seed(SEED)
    torch.library.opcheck(chunk_state_bwd, create_inputs_bwd(b, n, c, h, d, dtype, device), 
        test_utils=('test_schema', 'test_faketensor', 'test_aot_dispatch_dynamic', 'test_aot_dispatch_static'))

## OP IMPL TESTS ##
from packages.state_kernel.state_kernel._chunk_state.impl import (
    chunk_state,
    chunk_state_fake,
    create_inputs as create_inputs_impl,
    compute_expanded_dim
)

IMPL_TEST_CASES = [
    (2, 8, 128, 4, 32, torch.float16, 'cuda'),    # Common case with float16
    (4, 32, 128, 4, 32, torch.float16, 'cuda'),   # Large case with float16
    (1, 8, 128, 4, 64, torch.float16, 'cuda'),    # Test float16 with large head
    (2, 32, 128, 4, 64, torch.float16, 'cuda'),   # Large case with float16
    (1, 8, 128, 4, 32, torch.bfloat16, 'cuda'),   # Test bfloat16
    (4, 32, 128, 4, 32, torch.bfloat16, 'cuda'),  # Large case with bfloat16
    (2, 8, 128, 4, 64, torch.bfloat16, 'cuda'),   # Test bfloat16 with large head
    (1, 32, 128, 4, 64, torch.bfloat16, 'cuda'),  # Large case with bfloat16
]

@pytest.mark.parametrize("b,n,c,h,d,dtype,device", IMPL_TEST_CASES)
def test_chunk_state_create_inputs(b, n, c, h, d, dtype, device):
    torch.manual_seed(SEED)
    K, V, deg = create_inputs_impl(b, n, c, h, d, dtype, device)
    check_tensor_property_pairs(
        (K, ((b, n, c, h, d), dtype, device)),
        (V, ((b, n, c, h, d), dtype, device))
    )

@pytest.mark.parametrize("b,n,c,h,d,dtype,device", IMPL_TEST_CASES)
def test_chunk_state_output(b, n, c, h, d, dtype, device):
    torch.manual_seed(SEED)
    K, V, deg = create_inputs_impl(b, n, c, h, d, dtype, device)
    D = compute_expanded_dim(d, deg)
    with torch.no_grad():
        S = chunk_state(K, V, deg)
    check_tensor_property_pairs(
        (S, ((b, n, h, D, d), dtype, device))
    )

@pytest.mark.parametrize("b,n,c,h,d,dtype,device", IMPL_TEST_CASES)
def test_chunk_state_backward(b, n, c, h, d, dtype, device):
    torch.manual_seed(SEED)
    K, V, deg = create_inputs_impl(b, n, c, h, d, dtype, device, requires_grad=True)    
    outputs = chunk_state(K, V, deg)
    torch.autograd.backward(outputs, outputs)
    check_tensor_property_pairs(
        (K.grad, ((b, n, c, h, d), dtype, device)),
        (V.grad, ((b, n, c, h, d), dtype, device))
    )

@pytest.mark.parametrize("args", IMPL_TEST_CASES)
def test_chunk_state_backward_create_inputs_determinism(args):
    check_inputs_created_determinstically(create_inputs_impl, (*args, SEED))

@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("n", [1, 8, 32])
@pytest.mark.parametrize("c", [128, 1024])
@pytest.mark.parametrize("h", [1, 8])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])
def test_chunk_state_compiles(b, n, c, h, d, dtype, device):
    torch.manual_seed(SEED)
    check_fn_compiles(chunk_state, create_inputs_impl(b, n, c, h, d, dtype, device))

@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("n", [1, 4, 32])
@pytest.mark.parametrize("c", [128, 1024])
@pytest.mark.parametrize("h", [4, 8])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])
def test_chunk_state_compiles_with_backward(b, n, c, h, d, dtype, device):
    torch.manual_seed(SEED)
    check_fn_compiles_with_backward(chunk_state, create_inputs_impl(b, n, c, h, d, dtype, device, requires_grad=True))

@pytest.mark.parametrize("args", IMPL_TEST_CASES)
def test_chunk_state_fake_fn_implementation_matches(args):
    torch.manual_seed(SEED)
    check_fake_fn_implementation_matches(chunk_state, chunk_state_fake, create_inputs_impl(*args))

@pytest.mark.parametrize("args", IMPL_TEST_CASES)
def test_chunk_state_opcheck(args):
    torch.manual_seed(SEED)
    torch.library.opcheck(chunk_state, create_inputs_impl(*args, requires_grad=True))

## REFERENCE TESTS ##
from state_kernel._chunk_state.reference import (
    SymmetricPowerChunkStateReference,
    chunk_state_reference,
    chunk_state_reference_fwd,
)


@pytest.mark.parametrize("b", [1, 2, 4])
@pytest.mark.parametrize("n", [1, 4])
@pytest.mark.parametrize("c", [16])
@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])    
def test_expansion(b, n, c, h, d, dtype, device):
    torch.manual_seed(SEED)
    K, V, deg = create_inputs_impl(b, n, c, h, d, dtype, device)
    phi = SymmetricPowerChunkStateReference.expand(K, deg) # [b, n, h, c, D]
    prod = phi @ phi.transpose(-1, -2) # [b, n, h, c, c]
    prod_ref = (K @ K.transpose(-1, -2))**2 # [b, n, h, c, c]


@pytest.mark.parametrize("b", [1, 2, 4])
@pytest.mark.parametrize("n", [1, 4])
@pytest.mark.parametrize("c", [128, 256])
@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])    
def test_chunk_state_reference_matches_autograd(b, n, c, h, d, dtype, device):
    torch.manual_seed(SEED)
    check_backwards_match(
        ref_fn=chunk_state_reference_fwd,
        gold_inputs=partial_with_keywords(create_inputs_impl, requires_grad=True)(b, n, c, h, d, torch.float32, device),
        test_fn=chunk_state_reference,
        test_inputs=partial_with_keywords(create_inputs_impl, requires_grad=True)(b, n, c, h, d, dtype, device),
        tol=3.,
        show_precision=True,
    )

@pytest.mark.parametrize("b", [1, 2, 4])
@pytest.mark.parametrize("n", [1, 8])
@pytest.mark.parametrize("c", [128, 256, 1024])
@pytest.mark.parametrize("h", [1, 8])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])    
def test_chunk_state_matches_reference(b, n, c, h, d, dtype, device):
    torch.manual_seed(SEED)
    check_forwards_match(
        ref_fn=chunk_state_reference,
        gold_inputs=create_inputs_impl(b, n, c, h, d, torch.float32, device),
        test_fn=chunk_state,
        test_inputs=create_inputs_impl(b, n, c, h, d, dtype, device),
        tol=.3,
        show_precision=True,
    )

@pytest.mark.parametrize("b", [1, 2, 4])
@pytest.mark.parametrize("n", [1, 8])
@pytest.mark.parametrize("c", [128, 256, 1024])
@pytest.mark.parametrize("h", [1, 3, 8])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])    
def test_chunk_state_grad_matches_reference(b, n, c, h, d, dtype, device):
    torch.manual_seed(SEED)
    check_backwards_match(
        ref_fn=chunk_state_reference,
        gold_inputs=create_inputs_impl(b, n, c, h, d, torch.float32, device, requires_grad=True),
        test_fn=chunk_state,
        test_inputs=create_inputs_impl(b, n, c, h, d, dtype, device, requires_grad=True),
        tol=.3,
        precision_scale=1e-4,
        show_precision=True,
    )
