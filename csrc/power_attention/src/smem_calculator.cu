#include <iostream>

#include "power.h"
#include "power_bwd_kernel.h"

template <typename Kernel_traits>
void print_kernel_traits() {
    std::cout << "================================================" << std::endl;
    std::cout << Kernel_traits() << std::endl;

    constexpr int smem_size_QdY = Kernel_traits::kSmemQdYSize;
    constexpr int smem_size_dy = Kernel_traits::kSememdySize;
    constexpr int smem_size_LogG = Kernel_traits::kSmemLogGSize;
    constexpr int smem_size_K = Kernel_traits::kSmemKSize;
    constexpr int smem_size_KV = Kernel_traits::kSmemKVSize;
    constexpr int smem_size_dS = Kernel_traits::kSmemdSSize;
    constexpr int smem_size_P = Kernel_traits::kSmemPSize;

    constexpr int smem_size_dq_dk_dv_gating = Kernel_traits::kSmemSize1colblock_dq_dk_dv_gating;
    constexpr int smem_size_dq_dk_dv_no_gating = Kernel_traits::kSmemSize1colblock_dq_dk_dv_no_gating;

    std::cout << "no gating smem: " << smem_size_dq_dk_dv_no_gating << " bytes" << std::endl;
    std::cout << "\t" << "QdY: " << smem_size_QdY << " bytes" << std::endl;
    std::cout << "\t" << "dy: " << smem_size_dy << " bytes" << std::endl;
    if (Kernel_traits::Is_V_in_regs) {
        std::cout << "\t" << "K: " << smem_size_K << " bytes" << std::endl;
    } else {
        std::cout << "\t" << "KV: " << smem_size_KV << " bytes" << std::endl;
    }
    std::cout << "\t" << "dS: " << smem_size_dS << " bytes" << std::endl;
    std::cout << "\t" << "P: " << smem_size_P << " bytes" << std::endl;
    std::cout << "gating smem: " << smem_size_dq_dk_dv_gating << " bytes" << std::endl;
    std::cout << "\t" << "LogG: " << smem_size_LogG << " bytes" << std::endl;
}

int main(int argc, char** argv) {
    print_kernel_traits<Power_bwd_kernel_traits<
        /*kHeadDim=*/32,
        /*kBlockM=*/128,
        /*kBlockN=*/128,
        /*kNWarps=*/8,
        /*AtomLayoutMSdP=*/4,
        /*AtomLayoutNdKV=*/4,
        /*AtomLayoutMdQ=*/4,
        /*Is_V_in_regs=*/false,
        /*No_double_buffer=*/true,
        cutlass::half_t>>();
    print_kernel_traits<Power_bwd_kernel_traits<
        /*kHeadDim=*/64,
        /*kBlockM=*/128,
        /*kBlockN=*/128,
        /*kNWarps=*/8,
        /*AtomLayoutMSdP=*/4,
        /*AtomLayoutNdKV=*/4,
        /*AtomLayoutMdQ=*/4,
        /*Is_V_in_regs=*/false,
        /*No_double_buffer=*/false,
        cutlass::half_t>>();
    print_kernel_traits<Power_bwd_kernel_traits<
        /*kHeadDim=*/64,
        /*kBlockM=*/128,
        /*kBlockN=*/128,
        /*kNWarps=*/8,
        /*AtomLayoutMSdP=*/4,
        /*AtomLayoutNdKV=*/4,
        /*AtomLayoutMdQ=*/4,
        /*Is_V_in_regs=*/false,
        /*No_double_buffer=*/true,
        cutlass::half_t>>();
    print_kernel_traits<Power_bwd_kernel_traits<
        /*kHeadDim=*/64,
        /*kBlockM=*/128,
        /*kBlockN=*/128,
        /*kNWarps=*/8,
        /*AtomLayoutMSdP=*/4,
        /*AtomLayoutNdKV=*/4,
        /*AtomLayoutMdQ=*/4,
        /*Is_V_in_regs=*/true,
        /*No_double_buffer=*/false,
        cutlass::half_t>>();
    print_kernel_traits<Power_bwd_kernel_traits<
        /*kHeadDim=*/64,
        /*kBlockM=*/128,
        /*kBlockN=*/128,
        /*kNWarps=*/8,
        /*AtomLayoutMSdP=*/4,
        /*AtomLayoutNdKV=*/4,
        /*AtomLayoutMdQ=*/4,
        /*Is_V_in_regs=*/true,
        /*No_double_buffer=*/true,
        cutlass::half_t>>();
    print_kernel_traits<Power_bwd_kernel_traits<
        /*kHeadDim=*/64,
        /*kBlockM=*/64,
        /*kBlockN=*/128,
        /*kNWarps=*/8,
        /*AtomLayoutMSdP=*/4,
        /*AtomLayoutNdKV=*/4,
        /*AtomLayoutMdQ=*/4,
        /*Is_V_in_regs=*/false,
        /*No_double_buffer=*/false,
        cutlass::half_t>>();

}
