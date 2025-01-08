import torch
import pytest
from itertools import product
from functools import partial
from state_kernel.checks import (
    check_tensor_property_pairs,
    check_inputs_created_determinstically,
    check_fn_compiles,
    check_fn_compiles_with_backward,
    check_fake_fn_implementation_matches,
    check_forwards_match,
    check_backwards_match,
    check_output_match,
    save_to_csv,
    clone_inputs
)
from state_kernel.utils import partial_with_keywords, get_precision_bwd, plot_precision

SEED = 42

## FORWARD TESTS ##
from state_kernel._query_state.fwd import (
    query_state_fwd,
    query_state_fwd_fake,
    create_inputs as create_inputs_fwd,
    compute_expanded_dim_size
)
FWD_TEST_CASES = [
    (2, 8, 128, 4, 32, torch.float16, 'cuda'),    # Common case with float16
    (4, 32, 256, 8, 32, torch.float16, 'cuda'),   # Large case with float16
    (1, 8, 128, 4, 64, torch.float16, 'cuda'),    # Test float16 with large head
    (2, 32, 256, 8, 64, torch.float16, 'cuda'),   # Large case with float16
    (1, 8, 128, 4, 32, torch.bfloat16, 'cuda'),   # Test bfloat16
    (4, 32, 256, 8, 32, torch.bfloat16, 'cuda'),  # Large case with bfloat16
    (2, 8, 128, 4, 64, torch.bfloat16, 'cuda'),   # Test bfloat16 with large head
    (1, 32, 256, 8, 64, torch.bfloat16, 'cuda'),  # Large case with bfloat16
]

@pytest.mark.parametrize("b,n,c,h,d,dtype,device", FWD_TEST_CASES)
def test_query_state_fwd_create_inputs(b, n, c, h, d, dtype, device):
    Q, S, Y, rowmax, deg, stabilizer, zero_initial_state, eps = create_inputs_fwd(b, n, c, h, d, dtype, device, stabilizer=1.0)
    D = compute_expanded_dim_size(d, deg)
    check_tensor_property_pairs(
        (Q, ((b, n, c, h, d), dtype, device)),
        (S, ((b, n, h, D, d), dtype, device)),
        (Y, ((b, n, c, h, d), dtype, device))
    )

@pytest.mark.parametrize("b,n,c,h,d,dtype,device", FWD_TEST_CASES)
def test_query_state_fwd_output(b, n, c, h, d, dtype, device):
    inputs = create_inputs_fwd(b, n, c, h, d, dtype, device, stabilizer=1.0)
    with torch.no_grad():
        O = query_state_fwd(*inputs)
    check_tensor_property_pairs(
        (O, ((b, n, c, h, d), dtype, device))
    )

@pytest.mark.parametrize("args", FWD_TEST_CASES)
def test_query_state_fwd_determinism(args):
    check_inputs_created_determinstically(partial_with_keywords(create_inputs_fwd, stabilizer=1.0), args)

@pytest.mark.parametrize("args", FWD_TEST_CASES)
def test_query_state_fwd_compiles(args):
    check_fn_compiles(query_state_fwd, create_inputs_fwd(*args, stabilizer=1.0))

@pytest.mark.parametrize("args", FWD_TEST_CASES)
def test_query_state_fwd_fake_implementation(args):
    check_fake_fn_implementation_matches(query_state_fwd, query_state_fwd_fake, create_inputs_fwd(*args, stabilizer=1.0))

@pytest.mark.parametrize("args", FWD_TEST_CASES)
def test_query_state_fwd_opcheck(args):
    torch.library.opcheck(query_state_fwd, create_inputs_fwd(*args, stabilizer=1.0),
        test_utils=('test_schema', 'test_faketensor', 'test_aot_dispatch_dynamic', 'test_aot_dispatch_static'))

## BACKWARD TESTS ##
from state_kernel._query_state.bwd import (
    query_state_bwd,
    query_state_bwd_fake,
    create_inputs as create_inputs_bwd,
    ExpandedDim
)
BWD_TEST_CASES = [
    (1, 16, 128, 4, 32, torch.bfloat16, 'cuda'),   # Small batch with bfloat16
    (8, 8, 128, 8, 32, torch.bfloat16, 'cuda'),    # Large batch with bfloat16
    (2, 16, 128, 8, 64, torch.bfloat16, 'cuda'),   # Medium case with large head dim
    (4, 8, 128, 4, 32, torch.float16, 'cuda'),     # Medium batch with float16
    (1, 4, 128, 8, 32, torch.float16, 'cuda'),     # Long sequence with float16
    (8, 2, 128, 4, 64, torch.float16, 'cuda'),     # Large batch and large head dim
]

@pytest.mark.parametrize("b,n,c,h,d,dtype,device", BWD_TEST_CASES)
def test_query_state_backward_create_inputs(b, n, c, h, d, dtype, device):
    D = ExpandedDim(d, 2)
    Q, S, dO, rowmax, deg, stabilizer, zero_initial_state, deterministic = create_inputs_bwd(b, n, c, h, d, dtype, device, stabilizer=1.0)
    check_tensor_property_pairs(
        (Q, ((b, n, c, h, d), dtype, device)),
        (S, ((b, n, h, D, d), dtype, device)), 
        (dO, ((b, n, c, h, d), dtype, device))
    )

@pytest.mark.parametrize("b,n,c,h,d,dtype,device", BWD_TEST_CASES)
def test_query_state_backward_output(b, n, c, h, d, dtype, device):
    inputs = create_inputs_bwd(b, n, c, h, d, dtype, device, stabilizer=1.0)
    D = ExpandedDim(d, 2)
    with torch.no_grad():
        dQ, dS, ds = query_state_bwd(*inputs)
        check_tensor_property_pairs(
            (dQ, ((b, n, c, h, d), dtype, device)),
            (dS, ((b, n, h, D, d), dtype, device))
        )

@pytest.mark.parametrize("args", BWD_TEST_CASES)
def test_query_state_backward_determinism(args):
    check_inputs_created_determinstically(create_inputs_bwd, (*args, SEED))

@pytest.mark.parametrize("b", [1, 4])
@pytest.mark.parametrize("n", [1, 4])
@pytest.mark.parametrize("c", [128])
@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])
def test_query_state_backward_compiles(b, n, c, h, d, dtype, device):
    torch._dynamo.config.cache_size_limit = 32
    inputs = create_inputs_bwd(b, n, c, h, d, dtype, device, SEED, stabilizer=1.0)
    # TODO (sean): The backward pass of query_state doesn't support deterministic mode yet
    # hence the high rtol
    check_fn_compiles(query_state_bwd, inputs, rtol=1e-2)

@pytest.mark.parametrize("args", BWD_TEST_CASES)
def test_query_state_backward_fake_implementation(args):
    inputs = create_inputs_bwd(*args, stabilizer=1.0)
    check_fake_fn_implementation_matches(query_state_bwd, query_state_bwd_fake, inputs)

@pytest.mark.parametrize("args", BWD_TEST_CASES)
def test_query_state_backward_opcheck(args):
    inputs = create_inputs_bwd(*args, stabilizer=1.0)
    test_utils = ('test_schema', 'test_faketensor', 'test_aot_dispatch_dynamic', 'test_aot_dispatch_static')
    torch.library.opcheck(query_state_bwd, inputs, test_utils=test_utils)

## OP IMPL TESTS ##
from state_kernel._query_state.impl import (
    query_state,
    query_state_fake,
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

@pytest.mark.parametrize("b", [1, 2, 4])
@pytest.mark.parametrize("n", [1, 4, 8])
@pytest.mark.parametrize("c", [128, 1024])
@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("device", ['cuda'])
@pytest.mark.parametrize("deterministic", [True, False])
def test_query_state_create_inputs(b, n, c, h, d, dtype, fused, device, deterministic):
    Q, S, Y, rowmax, deg, stabilizer, zero_initial_state, eps, deterministic = create_inputs_impl(b, n, c, h, d, dtype, fused, device, deterministic=deterministic, stabilizer=1.0)
    D = compute_expanded_dim(d, deg)
    if fused:
        check_tensor_property_pairs(
            (Q, ((b, n, c, h, d), dtype, device)),
            (S, ((b, n, h, D, d), dtype, device)),
            (Y, ((b, n, c, h, d), dtype, device))
        )
        check_tensor_property_pairs(
            (rowmax, ((b, n, c, h), torch.float32, device)))
    else:
        check_tensor_property_pairs(
            (Q, ((b, n, c, h, d), dtype, device)),
            (S, ((b, n, h, D, d), dtype, device))
        )
        assert Y is None and y is None and rowmax is None

@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("n", [1, 4])
@pytest.mark.parametrize("c", [128, 1024])
@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("device", ['cuda'])
@pytest.mark.parametrize("deterministic", [True])
@pytest.mark.parametrize("stabilizer", [1.0, 100.0])
def test_query_state_output(b, n, c, h, d, dtype, fused, device, deterministic, stabilizer):
    inputs = create_inputs_impl(b, n, c, h, d, dtype, fused, device, deterministic=deterministic, stabilizer=stabilizer)
    with torch.no_grad():
        O = query_state(*inputs)
    check_tensor_property_pairs(
        (O, ((b, n, c, h, d), dtype, device))
    )

@pytest.mark.parametrize("b", [1, 2, 4])
@pytest.mark.parametrize("n", [1, 4, 8])
@pytest.mark.parametrize("c", [128, 1024])
@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("device", ['cuda'])
@pytest.mark.parametrize("deterministic", [True, False])
def test_query_state_backward(b, n, c, h, d, dtype, fused, device, deterministic):
    inputs = create_inputs_impl(b, n, c, h, d, dtype, fused, device, deterministic=deterministic, requires_grad=True, stabilizer=1.0)
    Q, S, Y, rowmax, deg, stabilizer, zero_initial_state, eps, deterministic = inputs
    with torch.autograd.enable_grad():
        outputs = query_state(Q, S, Y, rowmax, deg, stabilizer, zero_initial_state, eps, deterministic)
        torch.autograd.backward(outputs, outputs)
    D = compute_expanded_dim(d, deg)
    if fused:
        check_tensor_property_pairs(
            (Q.grad, ((b, n, c, h, d), dtype, device)),
            (S.grad, ((b, n, h, D, d), dtype, device)), 
            (Y.grad, ((b, n, c, h, d), dtype, device))
        )
    else:
        check_tensor_property_pairs(
            (Q.grad, ((b, n, c, h, d), dtype, device)),
            (S.grad, ((b, n, h, D, d), dtype, device))
        )

@pytest.mark.parametrize("args", IMPL_TEST_CASES)
def test_query_state_backward_determinism(args):
    check_inputs_created_determinstically(create_inputs_impl, args)

@pytest.mark.parametrize("b", [1, 2, 4])
@pytest.mark.parametrize("n", [1, 4, 8])
@pytest.mark.parametrize("c", [128, 1024])
@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("device", ['cuda'])
@pytest.mark.parametrize("deterministic", [True, False])
def test_query_state_compiles_fwd(b, n, c, h, d, dtype, fused, device, deterministic):
    check_fn_compiles(query_state, create_inputs_impl(b, n, c, h, d, dtype, fused, device, deterministic=deterministic, stabilizer=1.0))

@pytest.mark.parametrize("b", [1, 2, 4])
@pytest.mark.parametrize("n", [1, 4, 8])
@pytest.mark.parametrize("c", [128, 1024])
@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("device", ['cuda'])
@pytest.mark.parametrize("deterministic", [True, False])
def test_query_state_compiles_with_backward(b, n, c, h, d, dtype, fused, device, deterministic):
    check_fn_compiles_with_backward(query_state, create_inputs_impl(b, n, c, h, d, dtype, fused, device, deterministic=deterministic, requires_grad=True, stabilizer=1.0))

@pytest.mark.parametrize("args", IMPL_TEST_CASES)
def test_query_state_fake_fn_implementation_matches(args):
    check_fake_fn_implementation_matches(query_state, query_state_fake, create_inputs_impl(*args, stabilizer=1.0))

@pytest.mark.parametrize("args", IMPL_TEST_CASES)
@pytest.mark.skip(reason='Have not yet figured out how to make opcheck pass')
def test_query_state_opcheck(args):
    torch.library.opcheck(query_state, create_inputs_impl(*args, requires_grad=True, stabilizer=1.0))


## REFERENCE TESTS ##
from state_kernel._query_state.reference import (
    query_state_reference,
    query_state_reference_fwd,
)
REF_TEST_CASES = [
    (2, 8, 128, 4, 32, 'cuda'),    # Common case
    (4, 32, 128, 4, 32, 'cuda'),   # Large case
    (1, 8, 128, 4, 64, 'cuda'),    # Test large head
    (2, 32, 128, 4, 64, 'cuda'),   # Large case with large head
    (1, 8, 128, 4, 32, 'cuda'),    # Test small batch
    (4, 32, 128, 4, 32, 'cuda'),   # Large case
    (2, 8, 128, 4, 64, 'cuda'),    # Test large head
    (1, 32, 128, 4, 64, 'cuda'),   # Large case with large head
]
TEST_DTYPES = [torch.float16, torch.bfloat16]

@pytest.mark.parametrize("b", [1, 4])
@pytest.mark.parametrize("n", [1, 8])
@pytest.mark.parametrize("c", [128, 1024])
@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("device", ['cuda'])
@pytest.mark.parametrize("stabilizer", [1.0, 100.0])
@pytest.mark.parametrize("zero_initial_state", [True, False])
def test_query_state_reference_matches_autograd(b, n, c, h, d, dtype, fused, device, stabilizer, zero_initial_state):
    check_backwards_match(
        ref_fn=query_state_reference_fwd,
        gold_inputs=partial_with_keywords(create_inputs_impl, dtype=torch.float32, fused=fused, stabilizer=stabilizer, zero_initial_state=zero_initial_state, requires_grad=True)(b, n, c, h, d, device),
        test_fn=query_state_reference,
        test_inputs=partial_with_keywords(create_inputs_impl, dtype=dtype, fused=fused, stabilizer=stabilizer, zero_initial_state=zero_initial_state, requires_grad=True)(b, n, c, h, d, device),
        tol=3.,
        show_precision=True
    )
    torch.cuda.empty_cache()

@pytest.mark.parametrize("b", [1, 4])
@pytest.mark.parametrize("n", [1, 8])
@pytest.mark.parametrize("c", [512])
@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fused", [True])
@pytest.mark.parametrize("device", ['cuda'])
@pytest.mark.parametrize("stabilizer", [None, 100.0, 0.001])
@pytest.mark.parametrize("zero_initial_state", [True])
def test_query_state_matches_reference(b, n, c, h, d, dtype, fused, device, stabilizer, zero_initial_state):
    torch.manual_seed(SEED)

    gold_inputs = partial_with_keywords(create_inputs_impl, dtype=torch.float32, fused=fused, stabilizer=stabilizer, zero_initial_state=zero_initial_state)(b, n, c, h, d, device)
    test_inputs = partial_with_keywords(create_inputs_impl, dtype=dtype, fused=fused, stabilizer=stabilizer, zero_initial_state=zero_initial_state)(b, n, c, h, d, device)

    check_forwards_match(
        ref_fn=query_state_reference,
        gold_inputs=gold_inputs,
        test_fn=query_state,
        test_inputs=test_inputs,
        show_precision=True,
        tol=.1,
        precision_scale=1e-5,
        atol=0,
    )

@pytest.mark.parametrize("base_path", ['/home/sean/manifest3/packages/state_kernel'])
@pytest.mark.skip(reason='not needed')
def test_query_state_matches_reference_debug(base_path):
    Q = torch.load(f'{base_path}/Q.pt')
    S = torch.load(f'{base_path}/S.pt')
    attn_Y = torch.load(f'{base_path}/attn_Y.pt')
    rowmax = torch.load(f'{base_path}/rowmax.pt')
    deg = 2
    stabilizer = None
    zero_initial_state = True
    deterministic = False
    ε = 1e-7

    ref_inputs = (Q, S, attn_Y, rowmax, deg, stabilizer, zero_initial_state, ε, deterministic)
    test_inputs = (Q, S, attn_Y, rowmax, deg, stabilizer, zero_initial_state, ε, deterministic)

    assert Q.is_contiguous()
    assert S.is_contiguous()
    assert attn_Y.is_contiguous()
    assert rowmax.is_contiguous()

    check_output_match(
        ref_fn=query_state_reference,
        ref_inputs=ref_inputs,
        test_fn=query_state,
        test_inputs=test_inputs,
        show_precision=True,
        tol=.3,
        precision_scale=1e-5,
        atol=1e-4,
    )


@pytest.mark.parametrize("b", [1, 4])
@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("c", [128, 1024])
@pytest.mark.parametrize("h", [1, 4])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("device", ['cuda'])
@pytest.mark.parametrize("stabilizer", [None, 20.0])
@pytest.mark.parametrize("zero_initial_state", [True, False])
def test_query_state_grad_matches_reference(b, n, c, h, d, dtype, fused, device, stabilizer, zero_initial_state):
    gold_inputs = partial_with_keywords(create_inputs_impl, dtype=torch.float32, fused=fused, stabilizer=stabilizer, zero_initial_state=zero_initial_state, requires_grad=True, Y_std=1.0, q_std=1e-2)(b, n, c, h, d, device)
    test_inputs = partial_with_keywords(create_inputs_impl, dtype=dtype, fused=fused, stabilizer=stabilizer, zero_initial_state=zero_initial_state, requires_grad=True, Y_std=1.0, q_std=1e-2)(b, n, c, h, d, device)

    check_backwards_match(
        ref_fn=query_state_reference,
        gold_inputs=gold_inputs,
        test_fn=query_state,
        test_inputs=test_inputs,
        tol=.1,
        precision_scale=1e-5,
        show_precision=True,
        grad_scale=1.,
        atol=1e-4,
        use_random_grads=False,
    )


def query_state_precision(vary_std=True): 
    pass
    # bs = [1]
    # chunk_sizes = [1024]
    # hs = [12]
    # ds = [64]
    # num_chunks = [12]
    # dtypes = [torch.bfloat16]
    # devices = ['cuda']
    # deterministic = False
    # fused = True
    # q_stds = [1]
    # S_stds = [1]
    # s_stds = [100.]
    # Y_stds = [1]
    # y_stds = [100.]
    # rowmax_stds = [1.]
    # stabilizers = [100.]

    # dy_stds = [1e-3, 1e4]

    # if vary_std:
    #     for b, n, c, h, d, dtype, device, stabilizer, q_std, S_std, s_std, y_std, Y_std, rowmax_std in product(bs, num_chunks, chunk_sizes, hs, ds, dtypes, devices, stabilizers, q_stds, S_stds, s_stds, y_stds, Y_stds, rowmax_stds):
    #         print(f'{dtype=} {b=} {n=} {c=} {h=} {d=} {device=} {stabilizer=} {q_std=} {S_std=} {s_std=} {y_std=} {Y_std=}')
    #         ref_precisions = []
    #         test_precisions = []
    #         for dy_std in dy_stds:
    #             def fwd_input_factory(dtype, seed):
    #                 return create_inputs_impl(b, n, c, h, d, dtype, stabilizer=stabilizer, fused=fused, device=device, deterministic=deterministic, seed=seed, q_std=q_std, S_std=S_std, s_std=s_std, Y_std=Y_std, y_std=y_std, rowmax_std=rowmax_std, requires_grad=True)
                
    #             def load_debug_inputs(dtype, seed):
    #                 Q = torch.load('/home/sean/manifest2/projects/torch_nanogpt/Q.pt').to(dtype).requires_grad_(True)
    #                 S = torch.load('/home/sean/manifest2/projects/torch_nanogpt/S.pt').to(dtype).requires_grad_(True)
    #                 attn_Y = torch.load('/home/sean/manifest2/projects/torch_nanogpt/attn_Y.pt').to(dtype).requires_grad_(True)
    #                 rowmax = torch.load('/home/sean/manifest2/projects/torch_nanogpt/rowmax.pt')
    #                 deg = 2
    #                 stabilizer = 1.0 / h**0.5
    #                 initial_attention_chunk_n = 4
    #                 ε = 1e-7
    #                 deterministic = False
    #                 return Q, S, attn_Y, rowmax, deg, stabilizer, initial_attention_chunk_n == 0, ε, deterministic
 
    #             def create_grads_fn(output, seed):
    #                 torch.manual_seed(seed)
    #                 dY = torch.load('/home/sean/manifest2/projects/torch_nanogpt/Y_grad.pt')
    #                 return [dY, dy]

    #             ref_p, test_p = get_precision_bwd(
    #                 gold_fn=query_state_reference,
    #                 gold_input_factory=partial(load_debug_inputs, torch.float32),
    #                 ref_fn=query_state_reference,
    #                 ref_input_factory=partial(load_debug_inputs, dtype),
    #                 test_fn=query_state,
    #                 test_input_factory=partial(load_debug_inputs, dtype),
    #                 precision_threshold=1e-5,
    #                 create_grads_fn=create_grads_fn,
    #             )
    #             ref_precisions.append(ref_p)
    #             test_precisions.append(test_p)
    #         num_args = len(ref_precisions[0])
    #         ref_precisions = {i: [v[i] for v in ref_precisions] for i in range(num_args)}
    #         test_precisions = {i: [v[i] for v in test_precisions] for i in range(num_args)}
    #         plot_precision(ref_precisions, test_precisions, dy_stds, 'dy_std', f'Query State BWD error with {q_std=} {S_std=} {s_std=} {Y_std=} {y_std=}', x_log=True)


if __name__ == '__main__':
    query_state_precision(vary_std=True)
