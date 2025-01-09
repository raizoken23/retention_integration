from functools import partial
from itertools import product
import math

import pytest
import torch
from power_attention.utils import (
    partial_with_keywords, DummyCtx, make_fwd_bwd,
    get_precision, get_precision_bwd, plot_precision, plot_stats, get_stats_and_precision
)

from power_attention.checks import (
    check_backwards_match,
    check_backwards_match_,
    check_fake_fn_implementation_matches,
    check_fn_compiles,
    check_fn_compiles_with_backward,
    check_forwards_match,
    check_output_match,
    check_grads_match,
    check_inputs_created_determinstically,
    check_tensor_property_pairs,
    create_random_grads,
    save_to_csv,
)

SEED = 41

## FORWARD TESTS ##
from power_attention._attention.fwd import (
    attention_fwd,
    attention_fwd_fake,
)
from power_attention._attention.fwd import (
    create_inputs as create_inputs_fwd,
)

FWD_TEST_CASES = [
    (2, 8, 4, 32, torch.float16, 'cuda', False),    # Common case with float16
    (4, 32, 8, 32, torch.float16, 'cuda', True),    # Large case with float16 and gating
    (1, 8, 4, 64, torch.float16, 'cuda', False),    # Test float16 with large head
    (2, 32, 8, 64, torch.float16, 'cuda', True),    # Large case with float16 and gating
    (1, 8, 4, 32, torch.bfloat16, 'cuda', False),   # Test bfloat16
    (4, 32, 8, 32, torch.bfloat16, 'cuda', True),   # Large case with bfloat16 and gating
    (2, 8, 4, 64, torch.bfloat16, 'cuda', False),   # Test bfloat16 with large head
    (1, 32, 8, 64, torch.bfloat16, 'cuda', True),   # Large case with bfloat16 and gating
]

@pytest.mark.parametrize("b,t,h,d,dtype,device,gating", FWD_TEST_CASES)
def test_attention_fwd_create_inputs(b, t, h, d, dtype, device, gating):
    torch.manual_seed(SEED)
    Q, K, V, log_G_Q, log_G_K, deg, scale, eps = create_inputs_fwd(b, t, h, d, dtype, device, gating)
    check_tensor_property_pairs(
        (Q, ((b, t, h, d), dtype, device)),
        (K, ((b, t, h, d), dtype, device)),
        (V, ((b, t, h, d), dtype, device))
    )
    if gating:
        check_tensor_property_pairs(
            (log_G_Q, ((b, t, h), torch.float32, device)),
            (log_G_K, ((b, t, h), torch.float32, device))
        )

@pytest.mark.parametrize("b,t,h,d,dtype,device,gating", FWD_TEST_CASES)
def test_attention_fwd_output(b, t, h, d, dtype, device, gating):
    torch.manual_seed(SEED)
    inputs = create_inputs_fwd(b, t, h, d, dtype, device, gating)
    with torch.no_grad():
        Y, y, rowmax = attention_fwd(*inputs)
    check_tensor_property_pairs(
        (Y, ((b, t, h, d), dtype, device)),
        (y, ((b, t, h), torch.float32, device)),
        (rowmax, ((b, t, h), torch.float32, device))
    )

@pytest.mark.parametrize("args", FWD_TEST_CASES)
def test_attention_fwd_create_inputs_determinism(args):
    torch.manual_seed(SEED)
    check_inputs_created_determinstically(create_inputs_fwd, args)

@pytest.mark.parametrize("b", [1])
@pytest.mark.parametrize("t", [8192])
@pytest.mark.parametrize("h", [4]) 
@pytest.mark.parametrize("d", [64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])
@pytest.mark.parametrize("gating", [False, True])
def test_attention_fwd_compiles(b, t, h, d, dtype, device, gating):
    torch.manual_seed(SEED)
    check_fn_compiles(attention_fwd, create_inputs_fwd(b, t, h, d, dtype, device, gating))

@pytest.mark.parametrize("args", FWD_TEST_CASES)
def test_attention_fwd_fake_implementation(args):
    torch.manual_seed(SEED)
    check_fake_fn_implementation_matches(attention_fwd, attention_fwd_fake, create_inputs_fwd(*args))

@pytest.mark.parametrize("args", FWD_TEST_CASES)
def test_attention_fwd_opcheck(args):
    torch.manual_seed(SEED)
    torch.library.opcheck(attention_fwd, create_inputs_fwd(*args), 
        test_utils=('test_schema', 'test_faketensor', 'test_aot_dispatch_dynamic', 'test_aot_dispatch_static'))


## BACKWARD TESTS ##
from power_attention._attention.bwd import (
    attention_bwd_gating,
    attention_bwd_gating_fake,
    attention_bwd_gatingless,
    attention_bwd_gatingless_fake,
)
from power_attention._attention.bwd import (
    create_inputs as create_inputs_bwd,
)

BWD_TEST_CASES = [
    (2, 8, 4, 32, torch.float16, 'cuda'),    # Common case with float16
    (4, 32, 8, 32, torch.float16, 'cuda'),   # Large case with float16
    (1, 8, 4, 64, torch.float16, 'cuda'),    # Test float16 with large head
    (2, 32, 8, 64, torch.float16, 'cuda'),   # Large case with float16
    (1, 8, 4, 32, torch.bfloat16, 'cuda'),   # Test bfloat16
    (4, 32, 8, 32, torch.bfloat16, 'cuda'),  # Large case with bfloat16
    (2, 8, 4, 64, torch.bfloat16, 'cuda'),   # Test bfloat16 with large head
    (1, 32, 8, 64, torch.bfloat16, 'cuda'),  # Large case with bfloat16
]

GATING_VALUES = [True, False]

@pytest.mark.parametrize("b,t,h,d,dtype,device", BWD_TEST_CASES)
@pytest.mark.parametrize("gating", GATING_VALUES)
def test_attention_bwd_create_inputs(b, t, h, d, dtype, device, gating):
    torch.manual_seed(SEED)
    inputs = create_inputs_bwd(b, t, h, d, dtype, device, gating)
    if gating:
        Q, K, V, log_G_Q, log_G_K, dY, dy, rowmax, deg, scale, eps, deterministic = inputs
    else:
        Q, K, V, dY, dy, rowmax, deg, scale, eps, deterministic = inputs
        
    check_tensor_property_pairs(
        (Q, ((b, t, h, d), dtype, device)),
        (K, ((b, t, h, d), dtype, device)),
        (V, ((b, t, h, d), dtype, device)),
        (dY, ((b, t, h, d), dtype, device)),
        (dy, ((b, t, h), torch.float32, device)),
        (rowmax, ((b, t, h), torch.float32, device))
    )
    if gating:
        check_tensor_property_pairs(
            (log_G_Q, ((b, t, h), torch.float32, device)),
            (log_G_K, ((b, t, h), torch.float32, device))
        )

@pytest.mark.parametrize("b,t,h,d,dtype,device", BWD_TEST_CASES)
@pytest.mark.parametrize("gating", GATING_VALUES)
def test_attention_bwd_output(b, t, h, d, dtype, device, gating):
    torch.manual_seed(SEED)
    inputs = create_inputs_bwd(b, t, h, d, dtype, device, gating, scale=1e-4 if dtype == torch.float16 else 1e-2)
    with torch.no_grad():
        if gating:
            dQ, dK, dV, dlog_G = attention_bwd_gating(*inputs)
            check_tensor_property_pairs(
                (dQ, ((b, t, h, d), dtype, device)),
                (dK, ((b, t, h, d), dtype, device)), 
                (dV, ((b, t, h, d), dtype, device)),
                (dlog_G, ((b, t, h), torch.float32, device))
            )
        else:
            dQ, dK, dV = attention_bwd_gatingless(*inputs)
            check_tensor_property_pairs(
                (dQ, ((b, t, h, d), dtype, device)),
                (dK, ((b, t, h, d), dtype, device)),
                (dV, ((b, t, h, d), dtype, device))
            )


@pytest.mark.parametrize("args", BWD_TEST_CASES)
@pytest.mark.parametrize("gating", GATING_VALUES)
def test_attention_bwd_create_inputs_determinism(args, gating):
    check_inputs_created_determinstically(create_inputs_bwd, (*args, gating, 1.0, SEED))

@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("t", [8192])
@pytest.mark.parametrize("h", [16]) 
@pytest.mark.parametrize("d", [64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])
@pytest.mark.parametrize("gating", [False, True])
@pytest.mark.parametrize("deterministic", [True])
def test_attention_bwd_compiles(b, t, h, d, dtype, device, gating, deterministic):
    torch.manual_seed(SEED)
    inputs = create_inputs_bwd(b, t, h, d, dtype, device, gating, deterministic=deterministic)
    fn = attention_bwd_gating if gating else attention_bwd_gatingless
    check_fn_compiles(fn, inputs)

@pytest.mark.parametrize("args", BWD_TEST_CASES)
@pytest.mark.parametrize("gating", GATING_VALUES)
def test_attention_bwd_fake_implementation(args, gating):
    torch.manual_seed(SEED)
    inputs = create_inputs_bwd(*args, gating)
    if gating:
        fn = attention_bwd_gating
        fn_fake = attention_bwd_gating_fake
    else:
        fn = attention_bwd_gatingless
        fn_fake = attention_bwd_gatingless_fake
    check_fake_fn_implementation_matches(fn, fn_fake, inputs)

@pytest.mark.parametrize("args", BWD_TEST_CASES)
@pytest.mark.parametrize("gating", GATING_VALUES)
def test_attention_bwd_opcheck(args, gating):
    torch.manual_seed(SEED)
    inputs = create_inputs_bwd(*args, gating)
    fn = attention_bwd_gating if gating else attention_bwd_gatingless
    torch.library.opcheck(fn, inputs,
        test_utils=('test_schema', 'test_faketensor', 'test_aot_dispatch_dynamic', 'test_aot_dispatch_static'))

## OP IMPL TESTS ##
from power_attention._attention.impl import (
    attention,
    attention_fake,
)
from power_attention._attention.impl import (
    create_inputs as create_inputs_impl,
)


@pytest.mark.parametrize("b", [1])
@pytest.mark.parametrize("t", [8, 32])
@pytest.mark.parametrize("h", [4, 8]) 
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])
@pytest.mark.parametrize("gating", [False, True])
def test_attention_create_inputs(b, t, h, d, dtype, device, gating):
    torch.manual_seed(SEED)
    Q, K, V, log_G, deg, scale, eps, deterministic, normalize_output = create_inputs_impl(b, t, h, d, dtype, device, gating)
    check_tensor_property_pairs(
        (Q, ((b, t, h, d), dtype, device)),
        (K, ((b, t, h, d), dtype, device)),
        (V, ((b, t, h, d), dtype, device))
    )
    if gating:
        check_tensor_property_pairs(
            (log_G, ((b, t, h), torch.float32, device))
        )

@pytest.mark.parametrize("b", [1])
@pytest.mark.parametrize("t", [8, 32])
@pytest.mark.parametrize("h", [4, 8]) 
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])
@pytest.mark.parametrize("gating", [False, True])
def test_attention_output(b, t, h, d, dtype, device, gating):
    torch.manual_seed(SEED)
    inputs = create_inputs_impl(b, t, h, d, dtype, device, gating)
    with torch.no_grad():
        Y, y, rowmax = attention(*inputs)
    check_tensor_property_pairs(
        (Y, ((b, t, h, d), dtype, device)),
        (y, ((b, t, h), torch.float32, device)),
        (rowmax, ((b, t, h), torch.float32, device))
    )

@pytest.mark.parametrize("b", [1, 8])
@pytest.mark.parametrize("t", [128, 1025, 8192])
@pytest.mark.parametrize("h", [1, 16]) 
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])
@pytest.mark.parametrize("gating", [True])
def test_attention_backward(b, t, h, d, dtype, device, gating):
    torch.manual_seed(SEED)
    Q, K, V, log_G, *other_inputs = create_inputs_impl(b, t, h, d, dtype, device, gating, requires_grad=True)  
    outputs = attention(Q, K, V, log_G, *other_inputs)
    # too large of a scale will cause overflows in the backward kernel
    outputs_grad = create_random_grads(outputs, scale=1e-2)
    torch.autograd.backward(outputs, outputs_grad)
    check_tensor_property_pairs(
        (Q.grad, ((b, t, h, d), dtype, device)),
        (K.grad, ((b, t, h, d), dtype, device)), 
        (V.grad, ((b, t, h, d), dtype, device))
    )
    if gating:
        check_tensor_property_pairs(
            (log_G.grad, ((b, t, h), torch.float32, device))
        )

@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("t", [8, 32, 128])
@pytest.mark.parametrize("h", [1, 4, 8]) 
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])
@pytest.mark.parametrize("gating", [False, True])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_attention_create_inputs_determinism(b, t, h, d, dtype, device, gating, requires_grad):
    check_inputs_created_determinstically(create_inputs_impl, (b, t, h, d, dtype, device, gating, requires_grad))

@pytest.mark.parametrize("b", [1, 2, 4])
@pytest.mark.parametrize("t", [8, 32, 128, 512, 1024, 8192])
@pytest.mark.parametrize("h", [1, 3, 4, 8]) 
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])
@pytest.mark.parametrize("gating", [False, True])
def test_attention_compiles(b, t, h, d, dtype, device, gating):
    torch.manual_seed(SEED)
    check_fn_compiles(attention, create_inputs_impl(b, t, h, d, dtype, device, gating))

@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("t", [8, 32, 128, 8192])
@pytest.mark.parametrize("h", [1, 8]) 
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])
@pytest.mark.parametrize("gating", [False, True])
@pytest.mark.parametrize("deterministic", [True])
@pytest.mark.parametrize("seed", [42])
def test_attention_compiles_with_backward(b, t, h, d, dtype, device, gating, deterministic, seed):
    check_fn_compiles_with_backward(attention,
                                    create_inputs_impl(b, t, h, d, dtype, device, gating, requires_grad=True, deterministic=deterministic, seed=seed),
                                    grad_scale=1e-4 if dtype == torch.float16 else 1e-3)

@pytest.mark.parametrize("b", [1])
@pytest.mark.parametrize("t", [8, 32])
@pytest.mark.parametrize("h", [4, 8]) 
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])
@pytest.mark.parametrize("gating", [False, True])
def test_attention_fake_fn_implementation_matches(b, t, h, d, dtype, device, gating):
    torch.manual_seed(SEED)
    check_fake_fn_implementation_matches(attention, attention_fake, create_inputs_impl(b, t, h, d, dtype, device, gating))

@pytest.mark.parametrize("b", [1])
@pytest.mark.parametrize("t", [8, 32])
@pytest.mark.parametrize("h", [4, 8]) 
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])
@pytest.mark.parametrize("gating", [False, True])
@pytest.mark.xfail(reason='Have not yet figured out how to make opcheck pass')
def test_attention_opcheck(b, t, h, d, dtype, device, gating):
    torch.manual_seed(SEED)
    torch.library.opcheck(attention, create_inputs_impl(b, t, h, d, dtype, device, gating, requires_grad=True))

## REFERENCE TESTS ##
from power_attention._attention.reference import (
    attention_reference,
    attention_reference_fwd,
)


@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("t", [128, 1024])
@pytest.mark.parametrize("h", [1, 4]) 
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])
@pytest.mark.parametrize("gating", [False, True])
@pytest.mark.parametrize("deg", [1, 2, 3, 4])
@pytest.mark.parametrize("normal_space", [True, False])
@pytest.mark.parametrize("scale", [1.0, 1/10000])
def test_attention_reference_matches_autograd(b, t, h, d, dtype, device, gating, deg, normal_space, scale):
    torch.manual_seed(SEED)

    check_backwards_match(
        ref_fn=attention_reference_fwd,
        gold_inputs=partial_with_keywords(create_inputs_impl, dtype=torch.float32, requires_grad=True, p=deg, normal_space=normal_space, scale=scale)(b, t, h, d, device, gating),
        test_fn=attention_reference,
        test_inputs=partial_with_keywords(create_inputs_impl, dtype=dtype, requires_grad=True, p=deg, normal_space=normal_space, scale=scale)(b, t, h, d, device, gating),
        atol=0,
        grad_scale=.1,
        tol=1.0 # this precision tolerance is pretty high because the reference implementation does accumulation on reduced precision
    )

@pytest.mark.parametrize("b", [1])
@pytest.mark.parametrize("t", [128, 256])
@pytest.mark.parametrize("h", [1]) 
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])
@pytest.mark.parametrize("gating", [True, False])
@pytest.mark.parametrize("deg", [1, 2, 3, 4])
@pytest.mark.parametrize("normal_space", [True, False])
@pytest.mark.parametrize("scale", [1.0, 1/100])
def test_attention_matches_reference(b, t, h, d, dtype, device, gating, deg, normal_space, scale):
    torch.manual_seed(SEED)
    test_inputs=partial_with_keywords(create_inputs_impl, dtype=dtype, scale=scale, p=deg, normal_space=normal_space)(b, t, h, d, device, gating)
    scale = test_inputs[5]
    gold_inputs=partial_with_keywords(create_inputs_impl, dtype=torch.float32, scale=scale, p=deg, normal_space=normal_space)(b, t, h, d, device, gating)

    check_forwards_match(
        ref_fn=attention_reference,
        gold_inputs=gold_inputs,
        test_fn=attention,
        test_inputs=test_inputs,
        tol=.2,
        show_precision=True,
    )

@pytest.mark.parametrize("b", [1])
@pytest.mark.parametrize("t", [128, 256])
@pytest.mark.parametrize("h", [1]) 
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])
@pytest.mark.parametrize("gating", [True, False])
@pytest.mark.parametrize("deg", [4, 3, 2, 1])
@pytest.mark.parametrize("scale", [1.0, 1/100])
@pytest.mark.parametrize("fn", ["ref", "kernel"])
def test_normal_match_logspace(b, t, h, d, dtype, device, gating, deg, scale, fn):
    torch.manual_seed(SEED)
    normal_inputs=partial_with_keywords(create_inputs_impl, dtype=dtype, scale=scale, p=deg, normal_space=True)(b, t, h, d, device, gating)
    log_inputs=partial_with_keywords(create_inputs_impl, dtype=dtype, scale=scale, p=deg, normal_space=False)(b, t, h, d, device, gating)

    attention_fn = attention_reference if fn == "ref" else attention
    def return_k(fn, k):
        def _wrapper(*inputs):
            outputs = fn(*inputs)
            return outputs[:k]
        return _wrapper

    # check referencce match
    check_output_match(
        ref_fn=return_k(attention_fn, 1),
        ref_inputs=normal_inputs,
        test_fn=return_k(attention_fn, 1),
        test_inputs=log_inputs,
        tol=1e-1,
        pct_threshold=1e-2
    )


@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("t", [128, 1024])
@pytest.mark.parametrize("h", [1, 4]) 
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])
@pytest.mark.parametrize("scale", [1.0, 1.0/32**.5])
@pytest.mark.parametrize("gating", [True, False])
@pytest.mark.parametrize("deg", [1, 2, 3, 4])
@pytest.mark.parametrize("normal_space", [True, False]) 
def test_attention_grad_matches_reference(b, t, h, d, dtype, device, scale, gating, deg, normal_space):
    torch.manual_seed(SEED)
    check_backwards_match_(
        gold_fn=attention_reference,
        gold_inputs=partial_with_keywords(create_inputs_impl, dtype=torch.float32, requires_grad=True, scale=scale, p=deg, normal_space=normal_space)(b, t, h, d, device, gating),
        ref_fn=attention_reference,
        ref_inputs=partial_with_keywords(create_inputs_impl, dtype=dtype, requires_grad=True, scale=scale, p=deg, normal_space=normal_space)(b, t, h, d, device, gating),
        test_fn=attention,
        test_inputs=partial_with_keywords(create_inputs_impl, dtype=dtype, requires_grad=True, scale=scale, p=deg, normal_space=normal_space)(b, t, h, d, device, gating),
        tol=1.0,
        show_precision=True,
        seed=42,
        grad_scale=1.0,
        use_random_grads=True,
    )

# FLASH TESTS ##
from power_attention._attention.reference import (
    flash_attention_reference,
    flash_attention_reference_fwd,
    attention_reference_fwd_impl,
    attention_reference_bwd_impl,
    create_inputs_flash,
)
from flash_attn.flash_attn_interface import flash_attn_func


@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("t", [8, 32, 33, 128, 129, 1024, 1025])
@pytest.mark.parametrize("h", [1, 2, 4, 8]) 
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])
def test_flash_attention_reference_matches_flash_fwd(b, t, h, d, dtype, device):
    torch.manual_seed(SEED)

    def flash_attn_wrapper(Q, K, V, softmax_scale):
        return flash_attn_func(Q, K, V, dropout_p=0.0, causal=True, softmax_scale=softmax_scale, return_attn_probs=False)

    def flash_attention_reference(Q, K, V, softmax_scale):
        return flash_attention_reference_fwd(Q.to(torch.float32), K.to(torch.float32), V.to(torch.float32), softmax_scale=softmax_scale).to(dtype)
    
    inputs = partial_with_keywords(create_inputs_flash, dtype=dtype, requires_grad=False)(b, t, h, d, device, 1.0 / 32**.5)

    check_output_match(
        ref_fn=flash_attention_reference,
        ref_inputs=inputs,
        test_fn=flash_attn_wrapper,
        test_inputs=inputs,
        tol=1e-2,
    )

@pytest.mark.parametrize("b", [1])
@pytest.mark.parametrize("t", [8, 128, 1024])
@pytest.mark.parametrize("h", [1, 4]) 
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.parametrize("device", ['cuda'])
def test_nonlogspace_power_matches_flash(b, t, h, d, dtype, scale, device):
    torch.manual_seed(SEED)

    def flash_attn_wrapper(Q, K, V, softmax_scale):
        return flash_attn_func(Q, K, V, dropout_p=0.0, causal=True, softmax_scale=softmax_scale)

    def attention_wrapper(Q, K, V, softmax_scale):
        Y, y, _ = attention(Q, K, V, None, 2, softmax_scale, 0, False, False, True)
        return (Y / y.unsqueeze(-1)).to(dtype)

    def attention_ref_wrapper(Q, K, V, softmax_scale):
        Y, y, _ = attention_reference(Q, K, V, None, 2, softmax_scale, 0, False, False, True)
        return (Y / y.unsqueeze(-1)).to(dtype)

    def flash_attention_reference(Q, K, V, softmax_scale):
        return flash_attention_reference_fwd(Q, K, V, softmax_scale=softmax_scale)

    inputs = partial_with_keywords(create_inputs_flash, softmax_scale=scale, requires_grad=True)(b, t, h, d, dtype, device)
    gold_inputs = partial_with_keywords(create_inputs_flash, softmax_scale=scale, requires_grad=True)(b, t, h, d, torch.float32, device)

    # check forward with reference
    # check_forwards_match(
    #     ref_fn=attention_ref_wrapper,
    #     gold_inputs=gold_inputs,
    #     test_fn=attention_wrapper,
    #     test_inputs=inputs,
    #     tol=8e-3,
    #     precision_scale=1e-3,
    # )

    # # check forward with kernel
    check_backwards_match(
        ref_fn=attention_ref_wrapper,
        gold_inputs=gold_inputs,
        test_fn=attention_wrapper,
        test_inputs=inputs,
        tol=.3,
        precision_scale=1e-3,
    )

    # check backward

@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("t", [8, 32, 33, 128, 129, 1024, 1025, 8192])
@pytest.mark.parametrize("h", [1, 2, 4, 8]) 
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])
def test_flash_attention_reference_matches_flash_bwd(b, t, h, d, dtype, device):
    torch.manual_seed(SEED)

    def flash_attn_wrapper(Q, K, V, softmax_scale):
        return flash_attn_func(Q, K, V, dropout_p=0.0, causal=True, softmax_scale=softmax_scale)

    check_grads_match(
        ref_fn=flash_attn_wrapper,
        ref_inputs=partial_with_keywords(create_inputs_flash, dtype=dtype, requires_grad=True)(b, t, h, d, device, 1 / math.sqrt(d)),
        test_fn=flash_attention_reference_fwd,
        test_inputs=partial_with_keywords(create_inputs_flash, dtype=dtype, requires_grad=True)(b, t, h, d, device, 1 / math.sqrt(d)),
        tol=8e-3,
    )


@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("t", [8, 32, 33, 128, 129, 1024, 1025, 8192])
@pytest.mark.parametrize("h", [1, 2, 4, 8]) 
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ['cuda'])
def test_flash_attention_reference_matches_flash_bwd(b, t, h, d, dtype, device):
    torch.manual_seed(SEED)

    def flash_attn_wrapper(Q, K, V, softmax_scale):
        return flash_attn_func(Q, K, V, dropout_p=0.0, causal=True, softmax_scale=softmax_scale)

    check_grads_match(
        ref_fn=flash_attn_wrapper,
        ref_inputs=partial_with_keywords(create_inputs_flash, dtype=dtype, requires_grad=True)(b, t, h, d, device, 1 / math.sqrt(d)),
        test_fn=flash_attention_reference,
        test_inputs=partial_with_keywords(create_inputs_flash, dtype=dtype, requires_grad=True)(b, t, h, d, device, 1 / math.sqrt(d)),
        tol=8e-3,
    )

# Jacob tests ##
from power_attention._attention.reference import JacobAttentionLayer

@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("t", [8, 128, 1024])
@pytest.mark.parametrize("h", [1, 2, 4, 8]) 
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gating", [False, True])
@pytest.mark.parametrize("device", ['cuda'])
def test_jacob_attention_matches_reference(b, t, h, d, dtype, gating, device):
    torch.manual_seed(SEED)

    layer = JacobAttentionLayer(h, d, gating, dtype, device=device)

    p2_kernel = partial_with_keywords(layer.forward, attention_kernel='p2')
    p2_power_ref = partial_with_keywords(layer.forward, attention_kernel='powerref')
    p2_jacob_ref = partial_with_keywords(layer.forward, attention_kernel='p2ref')

    # check that the p2 ref matches the power reference
    check_output_match(
        ref_fn=p2_jacob_ref,
        ref_inputs=layer.create_inputs(b, t, dtype, device),
        test_fn=p2_power_ref,
        test_inputs=layer.create_inputs(b, t, dtype, device),
        tol=1e-2,
    )
    # # check that the p2 kernel matches the jacob reference
    check_output_match(
        ref_fn=p2_kernel,
        ref_inputs=layer.create_inputs(b, t, dtype, device),
        test_fn=p2_jacob_ref,
        test_inputs=layer.create_inputs(b, t, dtype, device),
        tol=1e-2,
    )


@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("t", [8, 128, 1024])
@pytest.mark.parametrize("h", [1, 2, 4, 8]) 
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gating", [True])
@pytest.mark.parametrize("device", ['cuda'])
def test_jacob_attention_matches_reference_bwd(b, t, h, d, dtype, gating, device):
    torch.manual_seed(SEED)

    layer = JacobAttentionLayer(h, d*h, gating, dtype, device=device)

    p2_kernel = partial_with_keywords(layer.forward, attention_kernel='p2')
    p2_power_ref = partial_with_keywords(layer.forward, attention_kernel='powerref')
    p2_jacob_ref = partial_with_keywords(layer.forward, attention_kernel='p2ref')

    # check that the p2 ref matches the power reference
    # check_backwards_match_(
    #     gold_fn=p2_jacob_ref,
    #     gold_inputs=layer.create_inputs(b, t, dtype, device),
    #     ref_fn=p2_jacob_ref,
    #     ref_inputs=layer.create_inputs(b, t, dtype, device),
    #     test_fn=p2_power_ref,
    #     test_inputs=layer.create_inputs(b, t, dtype, device),
    #     tol=1e-2,
    #     atol=5e-4,
    # )
    # check that the p2 kernel matches the jacob reference
    check_backwards_match_(
        gold_fn=p2_jacob_ref,
        gold_inputs=layer.create_inputs(b, t, dtype, device),
        ref_fn=p2_power_ref,
        ref_inputs=layer.create_inputs(b, t, dtype, device),
        test_fn=p2_kernel,
        test_inputs=layer.create_inputs(b, t, dtype, device),
        tol=1e-2,
        atol=1e-4,
    )



def attention_precision(vary_ctx_bwd=True,
                        vary_ctx_fwd=True,
                        vary_std_bwd=True,
                        vary_std_bwd_stats=True,
                        vary_ctx_bwd_jacob=True):
    bs = [1]
    ctxs = [8, 32, 128, 256, 512, 1024, 2048, 4096, 8192]
    hs = [1]
    ds = [32, 64]
    dtypes = [torch.float16, torch.bfloat16]
    devices = ['cuda']
    gatings = [False]
    deterministics = [False]
    stds = [1e-2, 1.]
    normalized_outputs = [True]

    if vary_ctx_fwd:
        print("fwd ctx...")
        for dtype, b, d, h, device, gating, deterministic, std, normalized_output in product(dtypes, bs, ds, hs, devices, gatings, deterministics, stds, normalized_outputs):
            print(f'{dtype=} {b=} {d=} {h=} {device=} {gating=} {deterministic=} {std=} {normalized_output=}')
            ref_precisions = []
            test_precisions = []
            for ctx in ctxs:
                def fwd_input_factory(dtype, seed):
                    return create_inputs_impl(b, ctx, h, d, dtype, device, gating, False, deterministic, seed, std=std, normalize_output=normalized_output)
                ref_p, test_p = get_precision(
                    gold_fn=attention_reference,
                    gold_input_factory=partial(fwd_input_factory, torch.float32),
                    ref_fn=attention_reference,
                    ref_input_factory=partial(fwd_input_factory, dtype),
                    test_fn=attention,
                    test_input_factory=partial(fwd_input_factory, dtype),
                    precision_threshold=1e-4
                )
                ref_precisions.append(ref_p)
                test_precisions.append(test_p)
            # flip dict inside out
            num_args = len(ref_precisions[0])
            ref_precisions = {i: [v[i] for v in ref_precisions] for i in range(num_args)}
            test_precisions = {i: [v[i] for v in test_precisions] for i in range(num_args)}
            plot_precision(ref_precisions, test_precisions, ctxs, 'Context Size', f'FWD error with {dtype=} {b=} {h=} {device=} {gating=} {deterministic=} {std=} {normalized_output=} all ref')
    
    if vary_ctx_bwd:
        print("bwd ctx...")
        for dtype, b, d, h, device, gating, deterministic, std, normalized_output in product(dtypes, bs, ds, hs, devices, gatings, deterministics, stds, normalized_outputs):
            print(f'{dtype=} {b=} {d=} {h=} {device=} {gating=} {deterministic=} {std=} {normalized_output=}')
            ref_precisions = []
            test_precisions = []
            for ctx in ctxs:
                def fwd_input_factory(dtype, seed):
                    return create_inputs_impl(b, ctx, h, d, dtype, device, gating, True, deterministic, seed, std=std, normalize_output=normalized_output)
                ref_p, test_p = get_precision_bwd(
                    gold_fn=attention_reference_fwd,
                    gold_input_factory=partial(fwd_input_factory, torch.float32),
                    ref_fn=attention_reference,
                    ref_input_factory=partial(fwd_input_factory, dtype),
                    test_fn=attention_reference,
                    test_input_factory=partial(fwd_input_factory, dtype),
                    precision_threshold=1e-5
                )
                ref_precisions.append(ref_p)
                test_precisions.append(test_p)
            # flip dict inside out
            num_args = len(ref_precisions[0])
            ref_precisions = {i: [v[i] for v in ref_precisions] for i in range(num_args)}
            test_precisions = {i: [v[i] for v in test_precisions] for i in range(num_args)}
            plot_precision(ref_precisions, test_precisions, ctxs, 'Context Size', f'BWD error with {dtype=} {b=} {h=} {d=} {device=} {gating=} {deterministic=} {std=} {normalized_output=} all ref')

    if vary_std_bwd:
        print("bwd std...")
        ctx = 2048
        stds = [1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100., 1000.]
        for dtype, b, d, h, device, gating, deterministic, normalized_output in product(dtypes, bs, ds, hs, devices, gatings, deterministics, [True]):
            print(f'{dtype=} {ctx=} {b=} {d=} {h=} {device=} {gating=} {deterministic=} {normalized_output=}')
            ref_precisions = []
            test_precisions = []
            for std in stds:
                def fwd_input_factory(dtype, seed):
                    return create_inputs_impl(b, ctx, h, d, dtype, device, gating, True, deterministic, seed, std=std, normalize_output=normalized_output)
                ref_p, test_p = get_precision_bwd(
                    gold_fn=attention_reference_fwd,
                    gold_input_factory=partial(fwd_input_factory, torch.float32),
                    ref_fn=attention_reference,
                    ref_input_factory=partial(fwd_input_factory, dtype),
                    test_fn=attention_reference,
                    test_input_factory=partial(fwd_input_factory, dtype),
                    precision_threshold=1e-5
                )
                ref_precisions.append(ref_p)
                test_precisions.append(test_p)
            # flip dict inside out
            num_args = len(ref_precisions[0])
            ref_precisions = {i: [v[i] for v in ref_precisions] for i in range(num_args)}
            test_precisions = {i: [v[i] for v in test_precisions] for i in range(num_args)}
            plot_precision(ref_precisions, test_precisions, stds, 'Std', f'BWD error varying std with {dtype=} {ctx=} {b=} {h=} {d=} {device=} {gating=} {deterministic=} {normalized_output=} all ref dP', x_log=True, y_log=True, ref_label='autograd', test_label='torch backward')

    if vary_std_bwd_stats:
        print("bwd std stats...")
        ctx = 128
        stds = [1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100., 1000.]
        dP_selector = lambda x: (x[9], x[10], x[11], x[12])

        # this wrapper makes it so that we unifying the output
        def attention_reference_fwd_impl_normalized(ctx, *args):
            O, y, Z_rowmax = attention_reference_fwd_impl(ctx, *args)
            normalize_output = args[-1]
            if not normalize_output:
                return (O / y).to(O.dtype), y.detach(), Z_rowmax
            else:
                return O.to(O.dtype), y.detach(), Z_rowmax
            
        def attention_reference_bwd_impl_normalized(ctx, *args):
            dO, dy, dZ_rowmax = args
            normalize_output = ctx.normalize_output
            _, _, _, _, Y, y = ctx.saved_tensors
            if not normalize_output: # normalize it manualy
                dY = (dO / y.unsqueeze(-1)).to(Y.dtype)
                dy_ = -(dO * Y).sum(dim=-1) / (y**2)
                return attention_reference_bwd_impl(ctx, dY, dy_, dZ_rowmax)
            else:
                return attention_reference_bwd_impl(ctx, dO, dy, dZ_rowmax)

        gold_fwd_bwd = make_fwd_bwd(attention_reference_fwd_impl, attention_reference_bwd_impl, grad_scale=1e-4, output_selector=dP_selector)
        gold_fwd_bwd_normalized = make_fwd_bwd(attention_reference_fwd_impl_normalized, attention_reference_bwd_impl_normalized, grad_scale=1e-4, output_selector=dP_selector)

        for dtype, b, d, h, device, gating, deterministic, normalized_output in product(dtypes, bs, ds, hs, devices, gatings, deterministics, [True]):
            print(f'{dtype=} {ctx=} {b=} {d=} {h=} {device=} {gating=} {deterministic=} {normalized_output=}')
            ref_stats = []
            test_stats = []
            for std in stds:
                def fwd_input_factory(dtype, seed):
                    return create_inputs_impl(b, ctx, h, d, dtype, device, gating, True, deterministic, seed, std=std, normalize_output=normalized_output)

                ref_p, test_p = get_stats_and_precision(
                    gold_fn=gold_fwd_bwd,
                    gold_input_factory=partial(fwd_input_factory, torch.float32),
                    ref_fn=gold_fwd_bwd,
                    ref_input_factory=partial(fwd_input_factory, dtype),
                    test_fn=gold_fwd_bwd,
                    test_input_factory=partial(fwd_input_factory, dtype),
                )

                ref_stats.append(ref_p)
                test_stats.append(test_p)
            # flip dict inside out
            num_args = len(ref_stats[0])
            ref_stats = {i: [v[i] for v in ref_stats] for i in range(num_args)}
            test_stats = {i: [v[i] for v in test_stats] for i in range(num_args)}
            plot_stats(ref_stats, test_stats, stds, 'Std', f'BWD error varying std with {dtype=} {ctx=} {b=} {h=} {d=} {device=} {gating=} {deterministic=} {normalized_output=} all ref', x_log=True, y_log=True, ref_label='reference manual backward', test_label='reference manual backward', only_ref=True, element_names=['dP_no_dy', 'dZ', 'dS', 'dy'])

    if vary_ctx_bwd_jacob:
        print("bwd ctx jacob...")
        
        for dtype, b, d, h, device, gating, deterministic, std, normalized_output in product(dtypes, bs, ds, hs, devices, [True], deterministics, stds, normalized_outputs):
            print(f'{dtype=} {b=} {d=} {h=} {device=} {gating=} {deterministic=} {std=} {normalized_output=}')

            layer = JacobAttentionLayer(h, d, gating, dtype, device=device)
            fp32_layer = JacobAttentionLayer(h, d, gating, torch.float32, device=device)

            p2_kernel = partial_with_keywords(layer.forward, attention_kernel='p2')
            p2_power_ref = partial_with_keywords(layer.forward, attention_kernel='powerref')
            p2_jacob_ref = partial_with_keywords(layer.forward, attention_kernel='p2ref')
            fp32_p2_jacob_ref = partial_with_keywords(fp32_layer.forward, attention_kernel='p2ref')

            ref_precisions = []
            test_precisions = []
            for ctx in ctxs:
                def fwd_input_factory(dtype, seed):
                    return layer.create_inputs(b, ctx, dtype, device, seed=seed)
                ref_p, test_p = get_precision_bwd(
                    gold_fn=fp32_p2_jacob_ref,
                    gold_input_factory=partial(fwd_input_factory, torch.float32),
                    ref_fn=p2_jacob_ref,
                    ref_input_factory=partial(fwd_input_factory, dtype),
                    test_fn=p2_kernel,
                    test_input_factory=partial(fwd_input_factory, dtype),
                    precision_threshold=1e-5
                )
                ref_precisions.append(ref_p)
                test_precisions.append(test_p)
            # flip dict inside out
            num_args = len(ref_precisions[0])
            ref_precisions = {i: [v[i] for v in ref_precisions] for i in range(num_args)}
            test_precisions = {i: [v[i] for v in test_precisions] for i in range(num_args)}
            plot_precision(ref_precisions, test_precisions, ctxs, 'Context Size', f'BWD error with {dtype=} {b=} {h=} {d=} {device=} {gating=} {deterministic=} {std=} {normalized_output=} jacob test')

if __name__ == '__main__':
    print("Generating precision data for attention...")
    attention_precision(vary_ctx_bwd=False, vary_ctx_fwd=False, vary_std_bwd=False, vary_std_bwd_stats=False, vary_ctx_bwd_jacob=True)
