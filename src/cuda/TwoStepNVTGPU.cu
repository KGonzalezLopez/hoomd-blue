/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

#include "TwoStepNVTGPU.cuh"
#include "gpu_settings.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

//! The texture for reading the pdata pos array
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;
//! The texture for reading the pdata vel array
texture<float4, 1, cudaReadModeElementType> pdata_vel_tex;
//! The texture for reading the pdata accel array
texture<float4, 1, cudaReadModeElementType> pdata_accel_tex;
//! The texture for reading in the pdata image array
texture<int4, 1, cudaReadModeElementType> pdata_image_tex;
//! The texture for reading particle mass
texture<float, 1, cudaReadModeElementType> pdata_mass_tex;

//! Shared memory used in reducing the 2K sum
extern __shared__ float nvt_sdata[];

/*! \file TwoStepNVTGPU.cu
    \brief Defines GPU kernel code for NVT integration on the GPU. Used by TwoStepNVTGPU.
*/

//! Takes the first 1/2 step forward in the NVT integration step
/*! \param pdata Particle Data to step forward in time
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param box Box dimensions for periodic boundary condition handling
    \param d_partial_sum2K Stores one partial 2K sum per block run
    \param denominv Intermediate variable computed on the host and used in the NVT integration step
    \param deltaT Amount of real time to step forward in one time step
    
    This kernel must be executed with a 1D grid of a power of 2 block size such that the number of threads is greater
    than or equal to the number of members in the group. The kernel's implementation simply reads one particle in each
    thread and updates that particle. It then calculates the contribution of each particle to sum2K and performs the
    first pass of the sum reduction into \a d_partial_sum2K.
    
    Due to the shared memory usage, this kernel must be executed with block_size * sizeof(float) bytes of dynamic
    shared memory.
    
    See gpu_nve_step_one_kernel() for some performance notes on how to handle the group data reads efficiently.
*/
extern "C" __global__ 
void gpu_nvt_step_one_kernel(gpu_pdata_arrays pdata,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             gpu_boxsize box,
                             float *d_partial_sum2K,
                             float denominv,
                             float deltaT)
    {
    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float psq2; //p^2 * 2 for use later
    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
   
        // update positions to the next timestep and update velocities to the next half step
        float4 pos = tex1Dfetch(pdata_pos_tex, idx);
        
        float px = pos.x;
        float py = pos.y;
        float pz = pos.z;
        float pw = pos.w;
        
        float4 vel = tex1Dfetch(pdata_vel_tex, idx);
        float4 accel = tex1Dfetch(pdata_accel_tex, idx);
        
        vel.x = (vel.x + (1.0f/2.0f) * accel.x * deltaT) * denominv;
        px += vel.x * deltaT;
        
        vel.y = (vel.y + (1.0f/2.0f) * accel.y * deltaT) * denominv;
        py += vel.y * deltaT;
        
        vel.z = (vel.z + (1.0f/2.0f) * accel.z * deltaT) * denominv;
        pz += vel.z * deltaT;
        
        // read in the image flags
        int4 image = tex1Dfetch(pdata_image_tex, idx);
        
        // time to fix the periodic boundary conditions
        float x_shift = rintf(px * box.Lxinv);
        px -= box.Lx * x_shift;
        image.x += (int)x_shift;
        
        float y_shift = rintf(py * box.Lyinv);
        py -= box.Ly * y_shift;
        image.y += (int)y_shift;
        
        float z_shift = rintf(pz * box.Lzinv);
        pz -= box.Lz * z_shift;
        image.z += (int)z_shift;
        
        float4 pos2;
        pos2.x = px;
        pos2.y = py;
        pos2.z = pz;
        pos2.w = pw;
        
        // write out the results
        pdata.pos[idx] = pos2;
        pdata.vel[idx] = vel;
        pdata.image[idx] = image;
        
        // now we need to do the partial 2K sums
        
        // compute our contribution to the sum
        float mass = tex1Dfetch(pdata_mass_tex, idx);
        psq2 = mass * (vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
        }
    else
        {
        psq2 = 0.0f;
        }
        
    nvt_sdata[threadIdx.x] = psq2;
    __syncthreads();
    
    // reduce the sum in parallel
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (threadIdx.x < offs)
            nvt_sdata[threadIdx.x] += nvt_sdata[threadIdx.x + offs];
        offs >>= 1;
        __syncthreads();
        }
        
    // write out our partial sum
    if (threadIdx.x == 0)
        {
        d_partial_sum2K[blockIdx.x] = nvt_sdata[0];
        }
    }

/*! \param pdata Particle Data to step forward in time
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param box Box dimensions for periodic boundary condition handling
    \param d_partial_sum2K Stores one partial 2K sum per block run
    \param block_size Size of the block to run
    \param num_blocks Number of blocks to execute
    \param Xi Current value of the NVT degree of freedom Xi
    \param deltaT Amount of real time to step forward in one time step
*/
cudaError_t gpu_nvt_step_one(const gpu_pdata_arrays &pdata,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             const gpu_boxsize &box,
                             float *d_partial_sum2K,
                             unsigned int block_size,
                             unsigned int num_blocks,
                             float Xi,
                             float deltaT)
    {
    // setup the grid to run the kernel
    dim3 grid( num_blocks, 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // bind the textures
    cudaError_t error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, pdata_vel_tex, pdata.vel, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, pdata_accel_tex, pdata.accel, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    error = cudaBindTexture(0, pdata_image_tex, pdata.image, sizeof(int4) * pdata.N);
    if (error != cudaSuccess)
        return error;

    error = cudaBindTexture(0, pdata_mass_tex, pdata.mass, sizeof(float) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    // run the kernel
    gpu_nvt_step_one_kernel<<< grid, threads, block_size * sizeof(float) >>>(pdata,
                                                                             d_group_members,
                                                                             group_size,
                                                                             box,
                                                                             d_partial_sum2K,
                                                                             1.0f / (1.0f + deltaT/2.0f * Xi),
                                                                             deltaT);
    if (!g_gpu_error_checking)
        {
        return cudaSuccess;
        }
    else
        {
        cudaThreadSynchronize();
        return cudaGetLastError();
        }
    }

//! The texture for reading the net force
texture<float4, 1, cudaReadModeElementType> net_force_tex;

//! Takes the second 1/2 step forward in the NVT integration step
/*! \param pdata Particle Data to step forward in time
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param Xi current value of the NVT degree of freedom Xi
    \param deltaT Amount of real time to step forward in one time step
*/
extern "C" __global__ 
void gpu_nvt_step_two_kernel(gpu_pdata_arrays pdata,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             float Xi,
                             float deltaT)
    {
    // determine which particle this thread works on
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
   
        // read in the net force and calculate the acceleration
        float4 accel = tex1Dfetch(net_force_tex, idx);
        float mass = tex1Dfetch(pdata_mass_tex, idx);
        accel.x /= mass;
        accel.y /= mass;
        accel.z /= mass;
        
        float4 vel = tex1Dfetch(pdata_vel_tex, idx);
        
        vel.x += (1.0f/2.0f) * deltaT * (accel.x - Xi * vel.x);
        vel.y += (1.0f/2.0f) * deltaT * (accel.y - Xi * vel.y);
        vel.z += (1.0f/2.0f) * deltaT * (accel.z - Xi * vel.z);
        
        // write out data
        pdata.vel[idx] = vel;
        // since we calculate the acceleration, we need to write it for the next step
        pdata.accel[idx] = accel;
        }
    }

/*! \param pdata Particle Data to step forward in time
    \param d_group_members Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Net force on each particle
    \param block_size Size of the block to execute on the device
    \param num_blocks Number of blocks to execute
    \param Xi current value of the NVT degree of freedom Xi
    \param deltaT Amount of real time to step forward in one time step
*/
cudaError_t gpu_nvt_step_two(const gpu_pdata_arrays &pdata,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             float4 *d_net_force,
                             unsigned int block_size,
                             unsigned int num_blocks,
                             float Xi,
                             float deltaT)
    {
    // setup the grid to run the kernel
    dim3 grid( num_blocks, 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // bind the textures
    cudaError_t error = cudaBindTexture(0, pdata_vel_tex, pdata.vel, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
    
    error = cudaBindTexture(0, pdata_mass_tex, pdata.mass, sizeof(float) * pdata.N);
    if (error != cudaSuccess)
        return error;

    error = cudaBindTexture(0, net_force_tex, d_net_force, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;
        
    // run the kernel
    gpu_nvt_step_two_kernel<<< grid, threads >>>(pdata, d_group_members, group_size, Xi, deltaT);
    
    if (!g_gpu_error_checking)
        {
        return cudaSuccess;
        }
    else
        {
        cudaThreadSynchronize();
        return cudaGetLastError();
        }
    }


//! Makes the final 2K sum on the GPU
/*! \param d_sum2K Pointer to write the final sum to
    \param d_partial_sum2K Already computed partial sums
    \param num_blocks Number of partial sums in \a d_partial_sum2k

    nvt_step_one_kernel reduces the 2K sum per block. This kernel completes the task
    and makes the final 2K sum on the GPU. It is up to the host to read this value to do something with the final total

    This kernel is designed to be a really simple 1-block kernel for summing the total Ksum.
    
    \note \a num_blocks is the number of partial sums! Not the number of blocks executed in this kernel.
*/
extern "C" __global__ void gpu_nvt_reduce_sum2K_kernel(float *d_sum2K, float *d_partial_sum2K, unsigned int num_blocks)
    {
    float sum2K = 0.0f;
    
    // sum up the values in the partial sum via a sliding window
    for (int start = 0; start < num_blocks; start += blockDim.x)
        {
        __syncthreads();
        if (start + threadIdx.x < num_blocks)
            nvt_sdata[threadIdx.x] = d_partial_sum2K[start + threadIdx.x];
        else
            nvt_sdata[threadIdx.x] = 0.0f;
        __syncthreads();
        
        // reduce the sum in parallel
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                nvt_sdata[threadIdx.x] += nvt_sdata[threadIdx.x + offs];
            offs >>= 1;
            __syncthreads();
            }
            
        // everybody sums up sum2K
        sum2K += nvt_sdata[0];
        }
        
    if (threadIdx.x == 0)
        *d_sum2K = sum2K;
    }

/*! \param d_sum2K Pointer to write the final sum to
    \param d_partial_sum2K Already computed partial sums
    \param num_blocks Number of partial sums in \a d_partial_sum2k

    this is just a driver for gpu_nvt_reduce_sum2K_kernel() see it for details
*/
cudaError_t gpu_nvt_reduce_sum2K(float *d_sum2K, float *d_partial_sum2K, unsigned int num_blocks)
    {
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid( 1, 1, 1);
    dim3 threads(block_size, 1, 1);
    
    // run the kernel
    gpu_nvt_reduce_sum2K_kernel<<< grid, threads, block_size*sizeof(float) >>>(d_sum2K, d_partial_sum2K, num_blocks);
    
    if (!g_gpu_error_checking)
        {
        return cudaSuccess;
        }
    else
        {
        cudaThreadSynchronize();
        return cudaGetLastError();
        }
    }


// vim:syntax=cpp

