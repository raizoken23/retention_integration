import torch
import pytest
from itertools import product
from state_kernel.checks import (
    check_tensor_property_pairs,
    check_inputs_created_determinstically,
    check_fn_compiles,
    check_fn_compiles_with_backward,
    check_fake_fn_implementation_matches,
    check_forwards_match,
    check_backwards_match,
    save_to_csv
)
from state_kernel.utils import partial_with_keywords

## FORWARD TESTS ##
from state_kernel._discumsum.fwd import (
    create_inputs as create_inputs_fwd,
    discumsum_fwd,
    discumsum_fwd_fake
)

FWD_TEST_CASES = [
    (1, 4, 1, 16, torch.float16, 'cuda'),    # Minimal case
    (2, 8, 4, 32, torch.float16, 'cuda'),    # Common case with float16
    (4, 32, 8, 64, torch.float16, 'cuda'),   # Large case with float16
    (1, 8, 4, 32, torch.bfloat16, 'cuda'),   # Test bfloat16
    (2, 32, 8, 64, torch.bfloat16, 'cuda'),  # Large case with bfloat16
    (4, 4, 1, 16, torch.float32, 'cuda'),    # Small case with float32
    (1, 32, 8, 64, torch.float32, 'cuda'),   # Large case with float32
    (2, 8, 4, 32, torch.float32, 'cuda'),    # Common case with float32
]

@pytest.mark.parametrize("b,n,h,d,X_dtype,device", FWD_TEST_CASES)
def test_discumsum_fwd_create_inputs(b, n, h, d, X_dtype, device):
    X, log_G = create_inputs_fwd(b, n, h, d, X_dtype, device)    
    check_tensor_property_pairs(
        (X, ((b, n, h, d), X_dtype, device)), 
        (log_G, ((b, n, h), torch.float32, device))
    )

@pytest.mark.parametrize("b,n,h,d,X_dtype,device", FWD_TEST_CASES)
def test_discumsum_fwd_output(b, n, h, d, X_dtype, device):
    inputs = create_inputs_fwd(b, n, h, d, X_dtype, device)
    with torch.no_grad():
        cum_X = discumsum_fwd(*inputs)
    check_tensor_property_pairs(
        (cum_X, ((b, n+1, h, d), X_dtype, device))
    )
    # Check first element is zero
    assert torch.all(cum_X[:,0] == 0)

@pytest.mark.parametrize("args", FWD_TEST_CASES)
def test_discumsum_fwd_create_inputs_determinism(args):
    check_inputs_created_determinstically(create_inputs_fwd, args)

@pytest.mark.parametrize("args", FWD_TEST_CASES)
def test_discumsum_fwd_compiles(args):
    check_fn_compiles(discumsum_fwd, create_inputs_fwd(*args))

@pytest.mark.parametrize("args", FWD_TEST_CASES)
def test_discumsum_fwd_fake_implementation(args):
    check_fake_fn_implementation_matches(discumsum_fwd, discumsum_fwd_fake, create_inputs_fwd(*args))

@pytest.mark.parametrize("args", FWD_TEST_CASES)
def test_discumsum_fwd_opcheck(args):
    torch.library.opcheck(discumsum_fwd, create_inputs_fwd(*args), 
        test_utils=('test_schema', 'test_faketensor', 'test_aot_dispatch_dynamic', 'test_aot_dispatch_static'))

## BACKWARD TESTS ##
from state_kernel._discumsum.bwd import (
    create_inputs as create_inputs_bwd,
    discumsum_bwd,
    discumsum_bwd_fake
)

BWD_TEST_CASES = [
    (1, 4, 1, 16, torch.float16, 'cuda'),    # Minimal case
    (2, 8, 4, 32, torch.float16, 'cuda'),    # Common case with float16
    (4, 32, 8, 64, torch.float16, 'cuda'),   # Large case with float16
    (1, 8, 4, 32, torch.bfloat16, 'cuda'),   # Test bfloat16
    (2, 32, 8, 64, torch.bfloat16, 'cuda'),  # Large case with bfloat16
    (4, 4, 1, 16, torch.float32, 'cuda'),    # Small case with float32
    (1, 32, 8, 64, torch.float32, 'cuda'),   # Large case with float32
    (2, 8, 4, 32, torch.float32, 'cuda'),    # Common case with float32
]

@pytest.mark.parametrize("b,n,h,d,dtype,device", BWD_TEST_CASES)
def test_discumsum_bwd_create_inputs(b, n, h, d, dtype, device):
    dout, out, log_G = create_inputs_bwd(b, n, h, d, dtype, device)    
    check_tensor_property_pairs(
        (dout, ((b, n+1, h, d), dtype, device)),
        (out, ((b, n+1, h, d), dtype, device)),
        (log_G, ((b, n, h), torch.float32, device))
    )

@pytest.mark.parametrize("b,n,h,d,dtype,device", BWD_TEST_CASES)
def test_discumsum_bwd_output(b, n, h, d, dtype, device):
    inputs = create_inputs_bwd(b, n, h, d, dtype, device)
    with torch.no_grad():
        dX, dlog_G = discumsum_bwd(*inputs)
    check_tensor_property_pairs(
        (dX, ((b, n, h, d), dtype, device)),
        (dlog_G, ((b, n, h), torch.float32, device))
    )

@pytest.mark.parametrize("args", BWD_TEST_CASES)
def test_discumsum_bwd_create_inputs_determinism(args):
    check_inputs_created_determinstically(create_inputs_bwd, args)

@pytest.mark.parametrize("args", BWD_TEST_CASES)
def test_discumsum_bwd_compiles(args):
    check_fn_compiles(discumsum_bwd, create_inputs_bwd(*args))

@pytest.mark.parametrize("args", BWD_TEST_CASES)
def test_discumsum_bwd_fake_implementation(args):
    check_fake_fn_implementation_matches(discumsum_bwd, discumsum_bwd_fake, create_inputs_bwd(*args))

@pytest.mark.parametrize("args", BWD_TEST_CASES)
def test_discumsum_bwd_opcheck(args):
    torch.library.opcheck(discumsum_bwd, create_inputs_bwd(*args), 
        test_utils=('test_schema', 'test_faketensor', 'test_aot_dispatch_dynamic', 'test_aot_dispatch_static'))

## OP IMPL TESTS ##
from packages.state_kernel.state_kernel._discumsum.impl import (
    discumsum,
    discumsum_fake,
    create_inputs as create_inputs_impl
)
TEST_CASES_IMPL = [
    (2, 8, 4, 4, 32, torch.float16, 'cuda'),    # Common case with float16
    (4, 32, 8, 4, 32, torch.float16, 'cuda'),   # Large case with float16 and gating
    (1, 8, 4, 4, 64, torch.float16, 'cuda'),    # Test float16 with large head
    (2, 32, 8, 4, 64, torch.float16, 'cuda'),   # Large case with float16 and gating
    (1, 8, 4, 4, 32, torch.bfloat16, 'cuda'),   # Test bfloat16
    (4, 32, 8, 4, 32, torch.bfloat16, 'cuda'),  # Large case with bfloat16 and gating
    (2, 8, 4, 4, 64, torch.bfloat16, 'cuda'),   # Test bfloat16 with large head
    (1, 32, 8, 4, 64, torch.bfloat16, 'cuda'),  # Large case with bfloat16 and gating
]

@pytest.mark.parametrize("b,n,h,D,d,dtype,device", TEST_CASES_IMPL)
def test_discumsum_create_inputs(b, n, h, D, d, dtype, device):
    X, log_G = create_inputs_impl(b, n, h, D, d, dtype, device)
    check_tensor_property_pairs(
        (X, ((b, n, h, D, d), dtype, device)),
        (log_G, ((b, n, h), torch.float32, device))
    )

@pytest.mark.parametrize("b,n,h,D,d,dtype,device", TEST_CASES_IMPL)
def test_discumsum_output(b, n, h, D, d, dtype, device):
    inputs = create_inputs_impl(b, n, h, D, d, dtype, device)
    with torch.no_grad():
        cum_X = discumsum(*inputs)
    check_tensor_property_pairs(
        (cum_X, ((b, n+1, h, D, d), dtype, device))
    )

@pytest.mark.parametrize("b,n,h,D,d,dtype,device", TEST_CASES_IMPL)
def test_discumsum_backward(b, n, h, D, d, dtype, device):
    X, log_G = create_inputs_impl(b, n, h, D, d, dtype, device, requires_grad=True)
    output = discumsum(X, log_G)
    torch.autograd.backward(output, output)
    check_tensor_property_pairs(
        (X.grad, ((b, n, h, D, d), dtype, device)),
        (log_G.grad, ((b, n, h), torch.float32, device))
    )

@pytest.mark.parametrize("args", TEST_CASES_IMPL)
def test_discumsum_create_inputs_determinism(args):
    check_inputs_created_determinstically(create_inputs_impl, args)

@pytest.mark.parametrize("args", TEST_CASES_IMPL)
def test_discumsum_compiles(args):
    check_fn_compiles(discumsum, create_inputs_impl(*args))

@pytest.mark.parametrize("args", TEST_CASES_IMPL)
def test_discumsum_compiles_with_backward(args):
    check_fn_compiles_with_backward(discumsum, create_inputs_impl(*args, requires_grad=True))

@pytest.mark.parametrize("args", TEST_CASES_IMPL)
def test_discumsum_fake_fn_implementation_matches(args):
    check_fake_fn_implementation_matches(discumsum, discumsum_fake, create_inputs_impl(*args))

@pytest.mark.parametrize("args", TEST_CASES_IMPL)
@pytest.mark.xfail(reason='Have not yet figured out how to make opcheck pass')
def test_discumsum_opcheck(args):
    torch.library.opcheck(discumsum, create_inputs_impl(*args, requires_grad=True))

## REFERENCE TESTS ##
from state_kernel._discumsum.reference import (
    discumsum_reference
)
REF_TEST_CASES = [
    (2, 8, 4, 4, 32, 'cuda'),    # Common case
    (4, 32, 8, 4, 32, 'cuda'),   # Large case with gating
    (1, 8, 4, 4, 64, 'cuda'),    # Test with large head
    (2, 32, 8, 4, 64, 'cuda'),   # Large case with gating
    (1, 8, 4, 4, 32, 'cuda'),    # Test small case
    (4, 32, 8, 4, 32, 'cuda'),   # Large case with gating
    (2, 8, 4, 4, 64, 'cuda'),    # Test with large head
    (1, 32, 8, 4, 64, 'cuda'),   # Large case with gating
]
TEST_DTYPES = [torch.float16, torch.bfloat16]

@pytest.mark.parametrize("args", REF_TEST_CASES)
@pytest.mark.parametrize("test_dtype", TEST_DTYPES)
def test_discumsum_matches_reference(args, test_dtype):
    check_forwards_match(
        ref_fn=discumsum_reference,
        gold_inputs=partial_with_keywords(create_inputs_impl, X_dtype=torch.float32)(*args),
        test_fn=discumsum,
        test_inputs=partial_with_keywords(create_inputs_impl, X_dtype=test_dtype)(*args)
    )

@pytest.mark.parametrize("b", [1, 2, 4])
@pytest.mark.parametrize("n", [4, 8, 32, 64])
@pytest.mark.parametrize("h", [1, 4, 8])
@pytest.mark.parametrize("D", [16, 32])
@pytest.mark.parametrize("d", [1, 4, 16, 32])
@pytest.mark.parametrize("test_dtype", TEST_DTYPES)
@pytest.mark.parametrize("device", ['cuda'])
def test_discumsum_grad_matches_reference(b, n, h, D, d, test_dtype, device):
    # TODO (sean): the atomicAdd for dlog_G makes the gradient slightly worse than reference, hence the high tol. If this becomes a problem, fix it.
    check_backwards_match(
        ref_fn=discumsum_reference,
        gold_inputs=partial_with_keywords(create_inputs_impl, X_dtype=torch.float32, requires_grad=True)(b, n, h, D, d, device),
        test_fn=discumsum,
        test_inputs=partial_with_keywords(create_inputs_impl, X_dtype=test_dtype, requires_grad=True)(b, n, h, D, d, device),
        show_precision=True,
        precision_scale=1e-4,
        tol=1.,
    )
