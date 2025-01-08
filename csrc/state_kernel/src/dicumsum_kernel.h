#include <cute/tensor.hpp>
#include <iostream>
#include <random>

using namespace std;
using namespace cute;

// ----------- Fwd Kernel ------------
template<typename XYLayout, typename XYSmLayout, typename XCopy, typename YCopy,
         typename dLayout, typename dSmLayout, typename dCopy,
         typename NBlock, typename DBlock>
__global__ void discumsum_kernel(half_t*X, float*d, half_t *Y,
                        XYLayout XY_layout, dLayout d_layout, 
                        XYSmLayout XY_smlayout, dSmLayout d_smlayout,
                        XCopy X_copy, dCopy d_copy, YCopy Y_copy,
                        NBlock N_block, DBlock D_block) {
    /* X and Y have ranks [B, N, H, D] and d has rank [B, N, H].
    B, H are batch dimensions, handled by y, z gird axis respectively.
    The D dimension is the broadcast dimension that reuses the same discount factors.
    Each CTA in the x grid axis hadles a D_Block of the D dimension.
    Each CTA loads a tile of A of size N_Block x D_Block and performs a discounted
    cumsum along the N dimension. If the N dimension is larger than N_Block then
    the CTA will loop over tiles. */
    int B = size<0>(XY_layout); int N = size<1>(XY_layout); int H = size<2>(XY_layout); int D = size<3>(XY_layout);
    Tensor mX = make_tensor(make_gmem_ptr(X), XY_layout)(blockIdx.y, _, blockIdx.z, _);
    Tensor md = make_tensor(make_gmem_ptr(d), d_layout)(blockIdx.y, _, blockIdx.z);
    Tensor mY = make_tensor(make_gmem_ptr(Y), XY_layout)(blockIdx.y, _, blockIdx.z, _);
    auto XY_tiler = make_shape(N_block, D_block);
    auto XY_coord = make_coord(_, blockIdx.x);
    Tensor gX = local_tile(mX, XY_tiler, XY_coord);
    Tensor gd = local_tile(md, N_block, _);
    Tensor gY = local_tile(mY, XY_tiler, XY_coord);

    __shared__ half_t smemXY[cosize_v<XYSmLayout>];
    __shared__ float smemd[cosize_v<dSmLayout>];

    Tensor sXY = make_tensor(make_smem_ptr(smemXY), XY_smlayout);
    Tensor sd = make_tensor(make_smem_ptr(smemd), d_smlayout);
    const int tid = threadIdx.x;

    auto gmem_thr_copy_X = X_copy.get_thread_slice(tid);
    auto gmem_thr_copy_d = d_copy.get_thread_slice(tid);
    auto gmem_thr_copy_Y = Y_copy.get_thread_slice(tid);
    
    Tensor tXsX = gmem_thr_copy_X.partition_D(sXY);
    Tensor tdsd = gmem_thr_copy_d.partition_D(sd);
    Tensor tYsY = gmem_thr_copy_Y.partition_S(sXY);

    bool debug = false;
    Tensor tdgd_ = gmem_thr_copy_d.partition_S(gd(_, 0));
    int num_tiles = (N + NBlock::value - 1) / NBlock::value;
    float acc = 0.;
    for (int k = 0; k < num_tiles; k++) {
        Tensor tXgX = gmem_thr_copy_X.partition_S(gX(_, _, k));
        Tensor tdgd = gmem_thr_copy_d.partition_S(gd(_, k));
        Tensor tYgY = gmem_thr_copy_Y.partition_D(gY(_, _, k));
        copy(X_copy, tXgX, tXsX);
        if (tid < N_block) {
            sd(tid) = gd(tid, k);
            sd(tid) = exp(sd(tid));
        }
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();
        CUTE_UNROLL
        for (int n = 0; n < NBlock::value; n++) {
            acc = sd(n) * acc + sXY(n, tid);
            sXY(n, tid) = acc;
        }
        __syncthreads();
        for (int n = 0; n < N_block; n++) {
            if (threadIdx.x + blockIdx.x * D_block < D && k*N_block + n < N) {
                gY(n, tid, k) = sXY(n, tid);
            }
        }
    }
}

void discumsum_gpu(half_t *X, float*d, half_t *Y, int B, int N, int H, int D) {
    // Uses 8 warps. They handle a DBlock of 256 elements. Each thread will loop
    // over NBlock of 64 elements.
    // using NBlock = _64; using DBlock= _256;
    using NBlock = _64; using DBlock= _256;

    auto XY_layout = make_layout(make_shape(B, N, H, D), LayoutRight{});
    auto d_layout = make_layout(make_shape(B, N, H), LayoutRight{});
    auto XY_smlayout = Layout<Shape<NBlock, DBlock>, Stride<DBlock, _1>>{};
    auto d_smlayout = Layout<Shape<NBlock>>{};
    // Vectorized copies produce OOB reads when D is not a multiple of 8
    // TiledCopy X_copy = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, half_t>{},
    //                                     Layout<Shape<_8,_32>, Stride<_32,_1>>{},
    //                                     Layout<Shape<_1, _8>>{});
    // TiledCopy Y_copy = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, half_t>{},
    //                                     Layout<Shape<_8, _32>, Stride<_32, _1>>{},
    //                                     Layout<Shape<_1, _8>>{});
    TiledCopy X_copy = make_tiled_copy(Copy_Atom<UniversalCopy<half_t>, half_t>{},
                                        Layout<Shape<_1,_256>, Stride<_256,_1>>{},
                                        Layout<Shape<_1, _1>>{});
    TiledCopy Y_copy = X_copy;
    TiledCopy d_copy = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, float>{},
                                        Layout<Shape<_256>>{},
                                        Layout<Shape<_8>>{});
    int D_block_num = (D + DBlock::value - 1) / DBlock::value;
    dim3 grid(D_block_num, B, H);
    dim3 block(DBlock::value);
    discumsum_kernel<<<grid, block>>>(X, d, Y, XY_layout, d_layout, XY_smlayout, d_smlayout, X_copy, d_copy, Y_copy, NBlock{}, DBlock{});
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        auto msg = "Error in launching kernel: " + string(cudaGetErrorString(err));
        throw runtime_error(msg);
    }
}

// ----------- Bwd Kernel ------------

template<typename XYLayout, typename XYSmLayout, typename XCopy, typename YCopy,
         typename dLayout, typename dSmLayout, typename dCopy,
         typename NBlock, typename DBlock>
__global__ void discumsum_bwd_kernel(float*d, half_t *Y, half_t *Y_grad, half_t *X_grad, float *d_grad,
                        XYLayout XY_layout, dLayout d_layout, 
                        XYSmLayout XY_smlayout, dSmLayout d_smlayout,
                        XCopy X_copy, dCopy d_copy, YCopy Y_copy,
                        NBlock N_block, DBlock D_block) {
    int B = size<0>(XY_layout); int N = size<1>(XY_layout); int H = size<2>(XY_layout); int D = size<3>(XY_layout);
    Tensor md = make_tensor(make_gmem_ptr(d), d_layout)(blockIdx.y, _, blockIdx.z);
    Tensor mY = make_tensor(make_gmem_ptr(Y), XY_layout)(blockIdx.y, _, blockIdx.z, _);
    Tensor mY_grad = make_tensor(make_gmem_ptr(Y_grad), XY_layout)(blockIdx.y, _, blockIdx.z, _);
    Tensor mX_grad = make_tensor(make_gmem_ptr(X_grad), XY_layout)(blockIdx.y, _, blockIdx.z, _);
    Tensor md_grad = make_tensor(make_gmem_ptr(d_grad), d_layout)(blockIdx.y, _, blockIdx.z);
    auto XY_tiler = make_shape(N_block, D_block);
    auto XY_coord = make_coord(_, blockIdx.x);
    Tensor gd = local_tile(md, N_block, _);
    Tensor gY = local_tile(mY, XY_tiler, XY_coord);
    Tensor gY_grad = local_tile(mY_grad, XY_tiler, XY_coord);
    Tensor gX_grad = local_tile(mX_grad, XY_tiler, XY_coord);
    Tensor gd_grad = local_tile(md_grad, N_block, _);
    __shared__ half_t smemY[cosize_v<XYSmLayout>];
    __shared__ half_t smemY_grad[cosize_v<XYSmLayout>];
    __shared__ half_t smemX_grad[cosize_v<XYSmLayout>];
    __shared__ float smemd[cosize_v<dSmLayout>];
    __shared__ float smemd_grad[cosize_v<dSmLayout>];

    Tensor sd = make_tensor(make_smem_ptr(smemd), d_smlayout);
    Tensor sY = make_tensor(make_smem_ptr(smemY), XY_smlayout);
    Tensor sY_grad = make_tensor(make_smem_ptr(smemY_grad), XY_smlayout);
    Tensor sX_grad = make_tensor(make_smem_ptr(smemX_grad), XY_smlayout);
    Tensor sd_grad = make_tensor(make_smem_ptr(smemd_grad), d_smlayout);

    const int tid = threadIdx.x;
    int num_tiles = (N + NBlock::value - 1) / NBlock::value;
    float acc = 0.;
    float disc = 0.;
    for (int k = 0; k < num_tiles; k++) {
        // Copy d, Y, Y_grad to shared memory
        // Assumes that number of threads is greater than NBlock
        if (tid < N_block) {
            sd(tid) = gd(tid, k);
        }
        // Assumes that number of threads == NBlock
        for (int n=0; n < NBlock::value; n++) {
            if (k*N_block + tid < N) {
                sY(n, tid) = gY(n, tid, k);
                sY_grad(n, tid) = gY_grad(n, tid, k);
            }
       }

       // Compute the gradients for this tile
       for (int n = NBlock::value-1; n >= 1; n--) {
            acc = disc * acc + sY_grad(n, tid);
            sX_grad(n, tid) = acc;
            // Very inneficient
            atomicAdd(&sd_grad(n), sY_grad(n-1, tid));
            disc = exp(sd(n)); 
            if (thread0()) {printf("disc: %d\n", disc);}
            if (thread0()) {printf("acc: %d\n", acc);}
        }
        acc = disc * acc + sY_grad(0, tid);
        sX_grad(0, tid) = acc;
        if (thread0()) {
            sd_grad(0) = 0.;
        }

        // Copy the gradients to global memory
        if (tid < N_block) {
            //gd_grad(tid, k) = acc * sY_grad(0, tid);
            atomicAdd(&gd_grad(tid, k), acc * sY_grad(0, tid));
        }
        for (int n = 0; n < N_block; n++) {
            if (threadIdx.x + blockIdx.x * D_block < D && n + k*N_block < N) {
                gX_grad(n, tid, k) = sX_grad(n, tid);
                // gX_grad(n, tid, k) = 1.;
            }
        }
    }
}

void discumsum_bwd_gpu(float *d, half_t *Y, half_t *Y_grad, half_t *X_grad, float *d_grad, int B, int N, int H, int D) {
    // Uses 8 warps. They handle a DBlock of 256 elements. Each thread will loop
    // over NBlock of 64 elements.
    // using NBlock = _64; using DBlock= _256;
    // using NBlock = _64; using DBlock= _256;
    using NBlock = _16; using DBlock= _256;

    auto XY_layout = make_layout(make_shape(B, N, H, D), LayoutRight{});
    auto d_layout = make_layout(make_shape(B, N, H), LayoutRight{});
    auto XY_smlayout = Layout<Shape<NBlock, DBlock>, Stride<DBlock, _1>>{};
    auto d_smlayout = Layout<Shape<NBlock>>{};
    TiledCopy X_copy = make_tiled_copy(Copy_Atom<UniversalCopy<half_t>, half_t>{},
                                        Layout<Shape<_1,_256>, Stride<_256,_1>>{},
                                        Layout<Shape<_1, _1>>{});
    TiledCopy Y_copy = X_copy;
    TiledCopy d_copy = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, float>{},
                                        Layout<Shape<_256>>{},
                                        Layout<Shape<_8>>{});
    int D_block_num = (D + DBlock::value - 1) / DBlock::value;
    dim3 grid(D_block_num, B, H);
    dim3 block(DBlock::value);
    discumsum_bwd_kernel<<<grid, block>>>(d, Y, Y_grad, X_grad, d_grad, XY_layout, d_layout, XY_smlayout, d_smlayout, X_copy, d_copy, Y_copy, NBlock{}, DBlock{});
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        auto msg = "Error in launching kernel: " + string(cudaGetErrorString(err));
        throw runtime_error(msg);
    }
}

// ----------- Reference Implementations ------------

void discumsum_cpu(half_t *X, float*d, half_t *Y, int B, int N, int H, int D) {
    float acc = 0.;
    float disc = 0.;
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int l = 0; l < D; l++) {
                acc = 0.;
                for (int n = 0; n < N; n++) {
                    disc = exp(d[b*N*H + n*H + h]);
                    acc = disc * acc + X[b*N*H*D + n*H*D + h*D + l];
                    Y[b*N*H*D + n*H*D + h*D + l] = acc;
                }
            }
        }
    }
}

void discumsum_bwd_cpu(float *d, half_t *Y, half_t *Y_grad, half_t *X_grad, float *d_grad, int B, int N, int H, int D) {
    float acc = 0.;
    float disc = 0.;
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int l = 0; l < D; l++) {
                disc = 0.;
                for (int n = N-1; n >=1; n--) {
                    acc = disc * acc + Y_grad[b*N*H*D + n*H*D + h*D + l];
                    X_grad[b*N*H*D + n*H*D + h*D + l] = acc;
                    d_grad[b*N*H + n*H + h] += acc * Y_grad[b*N*H*D + (n-1)*H*D + h*D + l];
                    disc = exp(d[b*N*H + (n)*H + h]);
                }
                acc = disc * acc + Y_grad[b*N*H*D + h*D + l];
                X_grad[b*N*H*D + h*D + l] = acc;
                d_grad[b*N*H + h] += 0.;
                acc = 0.;
            }
        }
    }
}   



// --------------- Tests ----------------

template<typename T>
void compare_vectors(T *A, T *B, int size) {
    int fails = 0;
    for (int i = 0; i < size; i++) {
        float err = abs(A[i] - B[i]);
        if ( err > 1e-1 & fails < 10) {
            cout << "Mismatch at index " << i << ": " << A[i] << " " << B[i] << " error: " << err << endl;
            fails++;
        }
    }
    if (fails == 0) {
        cout << "Vectors match" << endl;
    }
    else {
        throw runtime_error("Mismatch in vectors");
    }
}


struct Dims {
    int B, N, H, D;
};
Dims test_dims(int c) {
    switch (c) {
        case 0:
            return {1, 64, 1, 512};
        case 1:
            return {1, 64, 1, 513};
        case 2:
            return {2, 64, 2, 2048};
        case 3:
            return {2, 128, 4, 2048};
        case 4:
            return {1, 64, 1, 1000};
        case 5:
            return {1, 4, 1, 512};
        case 6:
            return {2, 4, 2, 1000};
        case 7:
            return {2, 8, 2, 1000};
        case 8:
            return {2, 500, 2, 1000};
    }
    throw invalid_argument("Invalid test case");
}

struct DiscumsumData{
    half_t* h_X;
    float* h_d;
    half_t* h_Y;
    half_t* h_Y_ref;
    half_t* d_X;
    float* d_d;
    half_t* d_Y;
};
DiscumsumData create_data(int B, int N, int H, int D) {
    int XY_size = B*N*H*D;
    int d_size = B*N*H;
    half_t *h_X = new half_t[XY_size];
    float*h_d = new float[d_size];
    half_t *h_Y = new half_t[XY_size];
    half_t *h_Y_ref = new half_t[XY_size];

    // fill A, d with data
    int type = 1;
    std::random_device rd;  // Seed generator
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::normal_distribution<> dist(0.0, 1.0); // Standard normal distribution (mean=0, stddev=1)
    switch (type) {
        case 0:
            for (int i = 0; i < XY_size; i++) h_X[i] = 1.;
            for (int i = 0; i < d_size; i++) h_d[i] = 0.;
            break;
        case 1:
            for (int i = 0; i < XY_size; i++) h_X[i] = dist(gen);
            for (int i = 0; i < d_size; i++) h_d[i] = log(abs(dist(gen)));
            break;
    }
    half_t *d_X, *d_Y;
    float *d_d;
    cudaMalloc(&d_X, XY_size*sizeof(half_t));
    cudaMalloc(&d_d, d_size*sizeof(float));
    cudaMalloc(&d_Y, XY_size*sizeof(half_t));
    cudaMemcpy(d_X, h_X, XY_size*sizeof(half_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d, d_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, XY_size*sizeof(half_t), cudaMemcpyHostToDevice);
    return {h_X, h_d, h_Y, h_Y_ref, d_X, d_d, d_Y};
}

void test_fwd()
{
    cout << "Testing forward pass" << endl;
    for (int c = 0; c < 9; c++) {
        auto [B, N, H, D] = test_dims(c);
        // cout << "Test case " << c << " B, N, H, D: " << B << " " << N << " " << H << " " << D << endl;
        cout << "Test case " << c << " B=" << B << " N=" << N << " H=" << H << " D=" << D << endl;
        int XY_size = B*N*H*D;
        int d_size = B*N*H;
        auto data = create_data(B, N, H, D);
        auto& [h_X, h_d, h_Y, h_Y_ref, d_X, d_d, d_Y] = data;
        discumsum_gpu(d_X, d_d, d_Y, B, N, H, D);
        cudaMemcpy(h_Y, d_Y, XY_size*sizeof(half_t), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < N; i++) {
        //     cout << h_Y[i*D] << " ";
        // }
        discumsum_cpu(h_X, h_d, h_Y_ref, B, N, H, D);
        compare_vectors(h_Y, h_Y_ref, XY_size);
        cudaFree(d_X);
        cudaFree(d_d);
        cudaFree(d_Y);
    }
}

struct DiscumsumBwdData{
    // host inputs
    float* h_d;
    half_t* h_Y;
    half_t* h_Y_grad;
    // host outputs
    half_t* h_X_grad;
    float* h_d_grad;
    // host reference outputs
    half_t* h_X_grad_ref;
    float* h_d_grad_ref;
    // device inputs
    float* d_d;
    half_t* d_Y;
    half_t* d_Y_grad;
    // device outputs
    half_t* d_X_grad;
    float* d_d_grad;
};
DiscumsumBwdData create_bwd_data(int B, int N, int H, int D) {
    int XY_size = B*N*H*D;
    int d_size = B*N*H;
    half_t *h_Y = new half_t[XY_size];
    half_t *h_Y_grad = new half_t[XY_size];
    half_t *h_X_grad = new half_t[XY_size];
    float *h_d = new float[d_size];
    float *h_d_grad = new float[d_size];
    half_t *h_X_grad_ref = new half_t[XY_size];
    float *h_d_grad_ref = new float[d_size];
    // fill A, d with data
    std::random_device rd;  // Seed generator
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::normal_distribution<> dist(0.0, 1.0); // Standard normal distribution (mean=0, stddev=1)
    for (int i = 0; i < XY_size; i++) h_Y[i] = dist(gen);
    for (int i = 0; i < XY_size; i++) h_Y_grad[i] = dist(gen);
    for (int i = 0; i < d_size; i++) h_d[i] = log(abs(dist(gen)));
    for (int i = 0; i < d_size; i++) h_d_grad[i] = 0.;
    half_t *d_Y, *d_Y_grad, *d_X_grad;
    float *d_d, *d_d_grad;
    cudaMalloc(&d_Y, XY_size*sizeof(half_t));
    cudaMalloc(&d_Y_grad, XY_size*sizeof(half_t));
    cudaMalloc(&d_X_grad, XY_size*sizeof(half_t));
    cudaMalloc(&d_d, d_size*sizeof(float));
    cudaMalloc(&d_d_grad, d_size*sizeof(float));
    cudaMemcpy(d_Y, h_Y, XY_size*sizeof(half_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y_grad, h_Y_grad, XY_size*sizeof(half_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X_grad, h_X_grad, XY_size*sizeof(half_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d, d_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d_grad, h_d_grad, d_size*sizeof(float), cudaMemcpyHostToDevice);
    return {h_d, h_Y, h_Y_grad, h_X_grad, h_d_grad, h_X_grad_ref, h_d_grad_ref, d_d, d_Y, d_Y_grad, d_X_grad, d_d_grad};
}

void test_bwd() {
    cout << "\nTesting backward pass" << endl;
    auto [B, N, H, D] = test_dims(0);
    cout << "Test case " << " B=" << B << " N=" << N << " H=" << H << " D=" << D << endl;
    auto data_bwd = create_bwd_data(B, N, H, D);
    auto& [h_d, h_Y, h_Y_grad, h_X_grad, h_d_grad, h_X_grad_ref, h_d_grad_ref, d_d, d_Y, d_Y_grad, d_X_grad, d_d_grad] = data_bwd;

    discumsum_bwd_cpu(h_d, h_Y, h_Y_grad, h_X_grad_ref, h_d_grad_ref, B, N, H, D);
    discumsum_bwd_gpu(d_d, d_Y, d_Y_grad, d_X_grad, d_d_grad, B, N, H, D);
    cudaMemcpy(h_X_grad, d_X_grad, B*N*H*D*sizeof(half_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d_grad, d_d_grad, B*N*H*sizeof(float), cudaMemcpyDeviceToHost);

    compare_vectors(h_X_grad, h_X_grad_ref, B*N*H*D);
    cout << "X_grad: ";
    for (int i = 0; i < N; i++) {
        cout << h_X_grad[i*D] << " ";
    }
    cout << endl;
    cout << "X_grad_ref: ";
    for (int i = 0; i < N; i++) {
        cout << h_X_grad_ref[i*D] << " ";
    }
    // compare_vectors(h_d_grad, h_d_grad_ref, B*N*H);

    cudaFree(d_Y);
    cudaFree(d_Y_grad);
    cudaFree(d_X_grad);
    cudaFree(d_d);
    cudaFree(d_d_grad);
}

int main()
{
    // test_fwd();
    test_bwd();
    return 0;
}
 