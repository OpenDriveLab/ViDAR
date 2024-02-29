#include <torch/extension.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>

// #define MAX_D 2048
#define MAX_D 1026 // 704 + 400 + 27 + 1

template <typename scalar_t>
__global__ void init_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> points,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> tindex,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> occupancy) {

    // batch index
    const auto n = blockIdx.y;

    // ray index
    const auto c = blockIdx.x * blockDim.x + threadIdx.x;

    // num of rays
    const auto M = points.size(1);
    const auto T = occupancy.size(1);

    // we allocated more threads than num_rays
    if (c < M) {
        // ray end point
        const auto t = tindex[n][c];

        // invalid points
        assert(T == 1 || t < T);

        // if t < 0, it is a padded point
        if (t < 0) return;

        // time index for sigma
        // when T = 1, we have a static sigma
        const auto ts = (T == 1) ? 0 : t;

        // grid shape
        const int vzsize = occupancy.size(2);
        const int vysize = occupancy.size(3);
        const int vxsize = occupancy.size(4);
        // assert(vzsize + vysize + vxsize <= MAX_D);

        // end point
        const int vx = int(points[n][c][0]);
        const int vy = int(points[n][c][1]);
        const int vz = int(points[n][c][2]);

        //
        if (0 <= vx && vx < vxsize &&
            0 <= vy && vy < vysize &&
            0 <= vz && vz < vzsize) {
            occupancy[n][ts][vz][vy][vx] = 1;
        }
    }
}

template <typename scalar_t>
__global__ void get_grad_sigma_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> elementwise_mult,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> indices,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> tindex,
    // const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> occupancy,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_sigma
    // torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_sigma_count,
    ) {

    // batch index
    const auto n = blockIdx.y;

    // ray index
    const auto c = blockIdx.x * blockDim.x + threadIdx.x;

    // num of rays
    const auto M = elementwise_mult.size(1);
    const auto T = grad_sigma.size(1);
    const auto maxlen = elementwise_mult.size(2);

    // we allocated more threads than num_rays
    if (c < M) {
        // ray end point
        const auto t = tindex[n][c];

        // invalid points
        // assert(t < T);
        assert(T == 1 || t < T);

        // time index for sigma
        // when T = 1, we have a static sigma
        const auto ts = (T == 1) ? 0 : t;

        // if t < 0, it is a padded point
        if (t < 0) return;

        // WHEN THERE IS AN INTERSECTION BETWEEN THE RAY AND THE VOXEL GRID
        for (int i = 0; i < maxlen; i ++) {
            const int z = indices[n][c][i][0];
            const int y = indices[n][c][i][1];
            const int x = indices[n][c][i][2];
            // NOTE: potential race conditions when writing gradients
            // grad_sigma[n][ts][z][y][x] = elementwise_mult[n][c][i];
            // grad_sigma_count[n][ts][v.z][v.y][v.x] += 1;

            atomicAdd(&grad_sigma[n][ts][z][y][x], elementwise_mult[n][c][i]);
        }
    }
}

/*
 * input shape
 *   elementwisemult    : N x M x MAX_D
 *   indices  : N x M x MAX_D x 3
 *   tindex   : N x M
 *   sigma_shape : 5
 * output shape
 *   grad_sigma : N x T x H x L x W
 */
std::vector<torch::Tensor> get_grad_sigma_cuda(
    torch::Tensor elementwise_mult,
    torch::Tensor indices,
    torch::Tensor tindex,
    torch::Tensor sigma_shape) {

    const auto N = elementwise_mult.size(0); // batch size
    const auto M = elementwise_mult.size(1); // num of rays
    const auto device = elementwise_mult.device();

    const int threads = 1024;
    const dim3 blocks((M + threads - 1) / threads, N);

    // perform rendering
    /* auto grad_sigma = torch::zeros({sigma_shape[0], sigma_shape[1], sigma_shape[2], sigma_shape[3], sigma_shape[4]}, device); */
    auto grad_sigma = torch::zeros_like(sigma_shape);
    // auto grad_sigma_count = torch::zeros_like(sigma);

    AT_DISPATCH_FLOATING_TYPES(sigma_shape.type(), "get_grad_sigma_cuda", ([&] {
                get_grad_sigma_cuda_kernel<scalar_t><<<blocks, threads>>>(
                    elementwise_mult.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    indices.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                    tindex.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    grad_sigma.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>()
                    );
            }));

    cudaDeviceSynchronize();

    // grad_sigma_count += (grad_sigma_count == 0);
    // grad_sigma /= grad_sigma_count;

    return {grad_sigma};
}



template <typename scalar_t>
__global__ void render_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> sigma,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> origin,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> points,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> tindex,
    // const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> occupancy,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> pred_dist,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> gt_dist,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> dd_dsigma,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> indices
    // torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_sigma_count,
    ) {

    // batch index
    const auto n = blockIdx.y;

    // ray index
    const auto c = blockIdx.x * blockDim.x + threadIdx.x;

    // num of rays
    const auto M = points.size(1);
    const auto T = sigma.size(1);

    // we allocated more threads than num_rays
    if (c < M) {
        // ray end point
        const auto t = tindex[n][c];

        // invalid points
        // assert(t < T);
        assert(T == 1 || t < T);

        // time index for sigma
        // when T = 1, we have a static sigma
        const auto ts = (T == 1) ? 0 : t;

        // if t < 0, it is a padded point
        if (t < 0) return;

        // grid shape
        const int vzsize = sigma.size(2);
        const int vysize = sigma.size(3);
        const int vxsize = sigma.size(4);
        /* constexpr int MAX_D = int(sqrt(vzsize * vzsize + vysize * vysize + vxsize * vxsize) / 2) + 1; */
        // assert(vzsize + vysize + vxsize <= MAX_D);

        // origin
        const double xo = origin[n][t][0];
        const double yo = origin[n][t][1];
        const double zo = origin[n][t][2];

        // end point
        const double xe = points[n][c][0];
        const double ye = points[n][c][1];
        const double ze = points[n][c][2];

        // locate the voxel where the origin resides
        const int vxo = int(xo);
        const int vyo = int(yo);
        const int vzo = int(zo);

        //
        const int vxe = int(xe);
        const int vye = int(ye);
        const int vze = int(ze);

        // NOTE: for computing ray-casting path.
        int vx = vxo;
        int vy = vyo;
        int vz = vzo;

        // NOTE: stored as the path.
        double path_vx = (double) vx;
        double path_vy = (double) vy;
        double path_vz = (double) vz;

        // origin to end
        const double rx = xe - xo;
        const double ry = ye - yo;
        const double rz = ze - zo;
        double gt_d = sqrt(rx * rx + ry * ry + rz * rz);

        // directional vector
        const double dx = rx / gt_d;
        const double dy = ry / gt_d;
        const double dz = rz / gt_d;

        // In which direction the voxel ids are incremented.
        const int stepX = (dx >= 0) ? 1 : -1;
        const int stepY = (dy >= 0) ? 1 : -1;
        const int stepZ = (dz >= 0) ? 1 : -1;

        // Distance along the ray to the next voxel border from the current position (tMaxX, tMaxY, tMaxZ).
        // NOTE (tom modified here):
        //  Since origin is (0, 0, 0) in our case,
        //  next_voxel_boundary will always be (vx, vy, vz)
        //  which means tMax will always be 0.
        const double next_voxel_boundary_x = vx + (stepX < 0 ? -1 : 1);
        const double next_voxel_boundary_y = vy + (stepY < 0 ? -1 : 1);
        const double next_voxel_boundary_z = vz + (stepZ < 0 ? -1 : 1);

        // tMaxX, tMaxY, tMaxZ -- distance until next intersection with voxel-border
        // the value of t at which the ray crosses the first vertical voxel boundary
        double tMaxX = (dx!=0) ? (next_voxel_boundary_x - xo)/dx : DBL_MAX; //
        double tMaxY = (dy!=0) ? (next_voxel_boundary_y - yo)/dy : DBL_MAX; //
        double tMaxZ = (dz!=0) ? (next_voxel_boundary_z - zo)/dz : DBL_MAX; //

        // tDeltaX, tDeltaY, tDeltaZ --
        // how far along the ray we must move for the horizontal component to equal the width of a voxel
        // the direction in which we traverse the grid
        // can only be FLT_MAX if we never go in that direction
        const double tDeltaX = (dx!=0) ? stepX/dx : DBL_MAX;
        const double tDeltaY = (dy!=0) ? stepY/dy : DBL_MAX;
        const double tDeltaZ = (dz!=0) ? stepZ/dz : DBL_MAX;

        int3 path[MAX_D];
        double csd[MAX_D];  // cumulative sum of sigma times delta
        double p[MAX_D];  // alpha
        double d[MAX_D];
        double dt[MAX_D];

        // forward raymarching with voxel traversal
        int step = 0;  // total number of voxels traversed
        int count = 0;  // number of voxels traversed inside the voxel grid
        double last_d = 0.0;  // correct initialization

        // voxel traversal raycasting
        bool was_inside = false;
        while (true) {
            bool inside = (0 <= vx && vx < vxsize) &&
                (0 <= vy && vy < vysize) &&
                (0 <= vz && vz < vzsize);
            if (inside) { // now inside
                was_inside = true;

                // Origin version, which has great quantization issue.
                // path[count] = make_int3(vx, vy, vz);

                // NOTE: modified version by tom.
                //  consider the nearest voxel grid as the voxel center.
                int path_vx_tmp = (int) round(path_vx);
                path_vx_tmp = path_vx_tmp < vxsize ? path_vx_tmp : vxsize-1;
                path_vx_tmp = path_vx_tmp >= 0 ? path_vx_tmp : 0;

                int path_vy_tmp = (int) round(path_vy);
                path_vy_tmp = path_vy_tmp < vysize ? path_vy_tmp : vysize-1;
                path_vy_tmp = path_vy_tmp >= 0 ? path_vy_tmp : 0;

                int path_vz_tmp = (int) round(path_vz);
                path_vz_tmp = path_vz_tmp < vzsize ? path_vz_tmp : vzsize-1;
                path_vz_tmp = path_vz_tmp >= 0 ? path_vz_tmp : 0;

                path[count] = make_int3(path_vx_tmp, path_vy_tmp, path_vz_tmp);
            }

            else if (was_inside) { // was inside but no longer
                // we know we are not coming back so terminate
                break;
            } else if (last_d > gt_d) {
                break;
            } /* else { // has not gone inside yet
                // assert(count == 0);
                // (1) when we have hit the destination but haven't gone inside the voxel grid
                // (2) when we have traveled MAX_D voxels but haven't found one valid voxel
                //     handle intersection corner cases in case of infinite loop
                // bool hit = (vx == vxe && vy == vye && vz == vze);
                // if (hit || step >= MAX_D)
                //     break;
                if (last_d >= gt_d || step >= MAX_D) break;
            } */
            // _d represents the ray distance has traveled before escaping the current voxel cell
            double _d = 0.0;
            // voxel traversal
            if (tMaxX < tMaxY) {
                if (tMaxX < tMaxZ) {
                    _d = tMaxX;
                    vx += stepX;
                    tMaxX += tDeltaX;
                } else {
                    _d = tMaxZ;
                    vz += stepZ;
                    tMaxZ += tDeltaZ;
                }
            } else {
                if (tMaxY < tMaxZ) {
                    _d = tMaxY;
                    vy += stepY;
                    tMaxY += tDeltaY;
                } else {
                    _d = tMaxZ;
                    vz += stepZ;
                    tMaxZ += tDeltaZ;
                }
            }

            // _d - last_d: motion of this iteration.
            path_vx += max(0.0, _d - last_d) * dx;
            path_vy += max(0.0, _d - last_d) * dy;
            path_vz += max(0.0, _d - last_d) * dz;

            if (inside) {
                // get sigma at the current voxel
                const int3 &v = path[count];  // use the recorded index
                const double _sigma = sigma[n][ts][v.z][v.y][v.x];

                if (count >= 1){
                    const int3 &v_last = path[count-1];
                    bool passed = (v_last.x == v.x && v_last.y == v.y && v_last.z == v.z);
                    if (passed) {
                        count --;
                        last_d -= dt[count];
                    }
                }

                const double _delta = max(0.0, _d - last_d);  // THIS TURNS OUT IMPORTANT
                const double sd = _sigma * _delta;
                if (count == 0) { // the first voxel inside
                    csd[count] = sd;
                    p[count] = 1 - exp(-sd);
                } else {
                    csd[count] = csd[count-1] + sd;
                    // exp(-csd[count-1]) * (1 - exp(-_sigma * _delta))
                    p[count] = exp(-csd[count-1]) - exp(-csd[count]);
                }
                // record the traveled distance
                d[count] = _d;
                dt[count] = _delta;
                // count the number of voxels we have escaped
                count ++;
            }
            last_d = _d;
            step ++;
        }

        // the total number of voxels visited should not exceed this number
        if (count > MAX_D) {
            printf("%d\n", count);
        }
        assert(count <= MAX_D);

        // WHEN THERE IS AN INTERSECTION BETWEEN THE RAY AND THE VOXEL GRID
        if (count > 0) {
            // compute the expected ray distance
            double exp_d = 0.0;
            for (int i = 0; i < count; i ++)
                exp_d += p[i] * d[i];

            // add an imaginary sample at the end point should gt_d exceeds max_d
            double p_out = exp(-csd[count-1]);
            double max_d = d[count-1];

            // if we have reached the end within the grid, this should be false
            // this will not affect gradient, but it will make loss more informative
            // if (gt_d > max_d)
            //     exp_d += (p_out * gt_d);

            exp_d += (p_out * max_d);
            gt_d = min(gt_d, max_d);

            // write the rendered ray distance (max_d)
            pred_dist[n][c] = exp_d;
            gt_dist[n][c] = gt_d;

            /* backward raymarching */
            double dd_dsigma_[MAX_D];
            for (int i = count - 1; i >= 0; i --) {
                // NOTE: probably need to double check again
                if (i == count - 1)
                    dd_dsigma_[i] = p_out * max_d;
                else
                    dd_dsigma_[i] = dd_dsigma_[i+1] - exp(-csd[i]) * (d[i+1] - d[i]);
            }

            for (int i = count - 1; i >= 0; i --)
                dd_dsigma_[i] *= dt[i];

            // option 1: no cap on a stack
            // if (gt_d > max_d)
            //     for (int i = count - 1; i >= 0; i --)
            //         dd_dsigma_[i] -= dt[i] * p_out * gt_d;

            // option 2: cap at the boundary
            for (int i = count - 1; i >= 0; i --)
                dd_dsigma_[i] -= dt[i] * p_out * max_d;

            for (int i = 0; i < count; i ++) {
                const int3 &v = path[i];
                // NOTE: potential race conditions when writing gradients
                dd_dsigma[n][c][i] = dd_dsigma_[i];
                indices[n][c][i][0] = v.z;
                indices[n][c][i][1] = v.y;
                indices[n][c][i][2] = v.x;
                // grad_sigma_count[n][ts][v.z][v.y][v.x] += 1;
            }
        }
    }
}

/*
 * input shape
 *   sigma      : N x T x H x L x W
 *   origin   : N x T x 3
 *   points   : N x M x 4
 * output shape
 *   dist     : N x M
 *   loss     : N x M
 *   grad_sigma : N x T x H x L x W
 */
std::vector<torch::Tensor> render_cuda(
    torch::Tensor sigma,
    torch::Tensor origin,
    torch::Tensor points,
    torch::Tensor tindex) {

    const auto N = points.size(0); // batch size
    const auto M = points.size(1); // num of rays
    const auto T = sigma.size(1); // num of timesteps
    const auto H = sigma.size(2); // height of voxel grid
    const auto W = sigma.size(3); // width of voxel grid
    const auto L = sigma.size(4); // length of voxel grid

    //const int MAXLEN = int(sqrt(H * H + W * W + L * L) / 2) + 1;

    const auto device = sigma.device();

    const int threads = 512;
    const dim3 blocks((M + threads - 1) / threads, N);

    // perform rendering
    auto gt_dist = -torch::ones({N, M}, device);
    auto pred_dist = -torch::ones({N, M}, device);
    auto dd_dsigma = torch::zeros({N, M, MAX_D}, device);
    auto indices = torch::zeros({N, M, MAX_D, 3}, device);
    // auto grad_sigma_count = torch::zeros_like(sigma);

    AT_DISPATCH_FLOATING_TYPES(sigma.type(), "render_cuda", ([&] {
                render_cuda_kernel<scalar_t><<<blocks, threads>>>(
                    sigma.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                    origin.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    points.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    tindex.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    // occupancy.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                    pred_dist.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    gt_dist.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    dd_dsigma.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    indices.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>()
                    // grad_sigma_count.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                    );
            }));

    cudaDeviceSynchronize();

    // grad_sigma_count += (grad_sigma_count == 0);
    // grad_sigma /= grad_sigma_count;

    return {pred_dist, gt_dist, dd_dsigma, indices};
}


/*
 * input shape
 *   origin   : N x T x 3
 *   points   : N x M x 3
 *   tindex   : N x M
 * output shape
 *   occupancy: N x T x H x L x W
 */
torch::Tensor init_cuda(
    torch::Tensor points,
    torch::Tensor tindex,
    const std::vector<int> grid) {

    const auto N = points.size(0); // batch size
    const auto M = points.size(1); // num of rays

    const auto T = grid[0];
    const auto H = grid[1];
    const auto L = grid[2];
    const auto W = grid[3];

    const auto dtype = points.dtype();
    const auto device = points.device();
    const auto options = torch::TensorOptions().dtype(dtype).device(device).requires_grad(false);
    auto occupancy = torch::zeros({N, T, H, L, W}, options);

    const int threads = 1024;
    const dim3 blocks((M + threads - 1) / threads, N);

    // initialize occupancy such that every voxel with one or more points is occupied
    AT_DISPATCH_FLOATING_TYPES(points.type(), "init_cuda", ([&] {
                init_cuda_kernel<scalar_t><<<blocks, threads>>>(
                    points.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    tindex.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    occupancy.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
            }));

    // synchronize
    cudaDeviceSynchronize();

    return occupancy;
}
