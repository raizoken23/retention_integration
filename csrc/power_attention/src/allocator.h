// /**
//  * Allocator for power attention.
//  */

// #pragma once

// #include <type_traits>
// #include <concepts>
// #include <cute/tensor.hpp>

// namespace power {

//     using namespace cute;

//     /**
//      * Allocator for power attention. Usage pattern:
//      *     extern __shared__ char smem_[];
//      *     power::shared_allocator al(smem_);
//      *     auto sN = al.allocate_tensor<Layout>()
//      */

// }