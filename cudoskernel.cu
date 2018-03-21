#include <stdio.h>
#include <chrono>

#define KPOINTS_PER_BLOCK 1
#define FREQUENCIES_PER_BLOCK 64
#define MAX_NUM_ORBITALS 16
#define CUBES_PER_BLOCK 64

__constant__ int dev_tetradefs[24];

__device__ float Sqr(float x){
    //Calculates the square of x.
    return x * x;
}

__device__ float Cub(float x){
    //Calculates x to the power of 3.
    return x * x * x;
}

__global__ void DensityOfStatesPrecalculateCubeMomentumIndices(uint nk, uint nx, uint ny, uint nz, uint* cubes){
    //Calculates the indices of k-points at the corners of a cube on the rectangular momentum grid.
    const uint ik = blockIdx.x * blockDim.x + threadIdx.x;
    if(ik < nk){
        const uint ix = ik / ny / nz;
        const uint iy = (ik - ix * ny * nz) / nz;
        const uint iz = ik - ix * ny * nz - iy * nz;
        //the elemental cube has eight indices in the linear grid that we now determine
        uint xindices[8] = {ix, ix + 1, ix, ix + 1, ix, ix + 1, ix, ix + 1};
        uint yindices[8] = {iy, iy, iy + 1, iy + 1, iy, iy, iy + 1, iy + 1};
        uint zindices[8] = {iz, iz, iz, iz, iz + 1, iz + 1, iz + 1, iz + 1};
        //take residual with respect to nx, ny, nz because cubes at the boundary go past borders of grid and are folded back
        for(uint i=0;i<8;i++){
            xindices[i] %= nx;
            yindices[i] %= ny;
            zindices[i] %= nz;
        }
        //get linear indices of relevant k-points within linear grid defined by input projectors
        for(uint i=0;i<8;i++){
            cubes[ik*8+i] = xindices[i] * ny * nz + yindices[i] * nz + zindices[i];
        }
    }
}

__global__ void DensityOfStatesPresortEnergiesInTetrahedron(uint nk, uint nx, uint ny, uint nz, uint nband, float* bandenergies, uint* cubes, float* bandenergiesSorted, uint* linearizedIndicesForSortedEnergies){
    //Calculates a map of indices for corners of a tetrahedron sorted by energies.
    const uint ic = blockIdx.x * blockDim.x + threadIdx.x; //cube index
    const uint ib = blockIdx.y;

    if(ic < nk){    
        __shared__ float senergies[CUBES_PER_BLOCK*8]; //store all energies, eight per cube
        const uint im = cubes[ic*8 + threadIdx.y];
        senergies[threadIdx.x*8 + threadIdx.y] = bandenergies[im * nband + ib];
        __syncthreads();

        const uint it = threadIdx.y;
        if(it < 6){
            uint map[4];
            float be[4]; //band energies
            for(uint i=0;i<4;i++){
                const uint ia = dev_tetradefs[it*4+i];
                map[i] = cubes[ic*8 + ia];
                be[i] = senergies[threadIdx.x*8 + ia];
            }
            //sort energies within tetrahedron
            for(uint n=4;n>1;n--){
                for(uint i=0; i<n-1; i++){
                    if (be[i] > be[i+1]){
                        const float temp = be[i+1];
                        const int tempm = map[i+1];
                        be[i+1] = be[i];
                        be[i] = temp;
                        map[i+1] = map[i];
                        map[i] = tempm;
                    }
                }
            }
            //store energies and linearized indices on the original grid
            for(uint i=0;i<4;i++){
                const uint il = ic*nband*24 + ib*24 + it*4 + i;
                bandenergiesSorted[il] = be[i];
                linearizedIndicesForSortedEnergies[il] = map[i];
            }
        }
    }
}

__global__ void DensityOfStates(uint nk, uint nx, uint ny, uint nz, uint norb, uint nband, uint ne, float* energies, float* realpart, float* imagpart, float* bandenergiesSorted, uint* indicesSorted, float* totaldos, float* orbitaldos){
    //Calculates the total and orbital-resolved density of states.
    float totaldosThisThread = 0.0;
    float orbitaldosThisThread[MAX_NUM_ORBITALS]; //beware, limits maximum number of orbitals
    for(uint iorb=0;iorb<norb;iorb++){
        orbitaldosThisThread[iorb] = 0.0;
    }

    __shared__ float rparts[4*MAX_NUM_ORBITALS]; //holds real part of matrix element for specific momentum and band index but for all corners of tetrahedron
    __shared__ float iparts[4*MAX_NUM_ORBITALS]; //holds imag part of matrix element for specific momentum and band index but for all corners of tetrahedron

    //get all indices
    const uint ifreq = blockIdx.x * FREQUENCIES_PER_BLOCK + threadIdx.x;
    const float e = (ifreq < ne) ? energies[ifreq] : 0;
    const uint ikstart = blockIdx.y * KPOINTS_PER_BLOCK;
    const uint ikend = min(nk,(blockIdx.y+1)*KPOINTS_PER_BLOCK);

    for(uint ik=ikstart;ik<ikend;ik++){
        for(uint it=0;it<6;it++){
            //get tetrahedron energies
            const float tetvol = 1.0 / 6.0 / nx / ny / nz;
            for(uint ib=0;ib<nband;ib++){
                const uint il = ik*nband*24 + ib * 24 + it*4;
                uint *IndicesThisTetrahedron = &indicesSorted[il];
                float* be = &bandenergiesSorted[il];
                //calculate DOS
                float weights[4] = {0.0,0.0,0.0,0.0};
                if(be[0] < e && e < be[1]){
                    totaldosThisThread += tetvol * 3.0 * Sqr(e-be[0])/((be[1] - be[0])*(be[2] - be[0])*(be[3] - be[0]));
                    const float EminusE1 = e - be[0];
                    const float E21 = be[1] - be[0];
                    const float E31 = be[2] - be[0];
                    const float E41 = be[3] - be[0];
                    const float C = tetvol / 4.0 / E21 / E31 / E41 * Cub(EminusE1);
                    const float dC = 3.0 * tetvol / 4.0 / E21 / E31 / E41 * Sqr(EminusE1);
                    weights[0] = dC * (4 - EminusE1 * (1.0 / E21 + 1.0 / E31 + 1.0 / E41)) - C * (1.0 / E21 + 1.0 / E31 + 1.0 / E41);
                    weights[1] = dC * EminusE1 / E21 + C / E21;
                    weights[2] = dC * EminusE1 / E31 + C / E31;
                    weights[3] = dC * EminusE1 / E41 + C / E41;
                }
                else if(be[1] < e && e < be[2]){
                    totaldosThisThread += tetvol / ((be[2] - be[0])*(be[3] - be[0])) * (3.0*(be[1] - be[0]) + 6.0*(e - be[1]) - 3.0*((be[2] - be[0] + be[3] - be[1]) * Sqr(e - be[1]))/((be[2] - be[1])*(be[3] - be[1])));
                    const float EminusE1 = e - be[0];
                    const float EminusE2 = e - be[1];
                    const float EminusE3 = e - be[2];
                    const float EminusE4 = e - be[3];
                    const float E31 = be[2] - be[0];
                    const float E32 = be[2] - be[1];
                    const float E41 = be[3] - be[0];
                    const float E42 = be[3] - be[1];
                    const float C1 = tetvol / 4.0 / E41 / E31 * Sqr(EminusE1);
                    const float C2 = -tetvol / 4.0 / E41 / E32 / E31 * EminusE1 * EminusE2 * EminusE3;
                    const float C3 = -tetvol / 4.0 /  E42 / E32 / E41 * Sqr(EminusE2) * EminusE4;
                    const float dC1 = tetvol / 2.0 /  E41 / E31 * EminusE1;
                    const float dC2 = -tetvol / 4.0 /  E41 / E32 / E31 * (EminusE1 * EminusE2 + EminusE1 * EminusE3 + EminusE2 * EminusE3);
                    const float dC3 = -tetvol / 4.0 /  E42 / E32 / E41 * EminusE2 * (2 * EminusE4 + EminusE2);
                    weights[0] = dC1 - (dC1 + dC2) * EminusE3 / E31 - (C1 + C2) / E31 - (dC1 + dC2 + dC3) * EminusE4 / E41 - (C1 + C2 + C3) / E41;
                    weights[1] = dC1 + dC2 + dC3 - (dC2 + dC3) * EminusE3 / E32 - (C2 + C3) / E32 - dC3 * EminusE4 / E42 - C3 / E42;
                    weights[2] = (dC1 + dC2) * EminusE1 / E31 + (C1 + C2) / E31 + (dC2 + dC3) * EminusE2 / E32 + (C2 + C3) / E32;
                    weights[3] = (dC1 + dC2 + dC3) * EminusE1 / E41 + (C1 + C2 + C3) / E41 + dC3 * EminusE2 / E42 + C3 / E42;
                }
                else if(be[2] < e && e < be[3]){
                    totaldosThisThread += tetvol * 3.0*Sqr(be[3]-e)/((be[3] - be[0])*(be[3] - be[1])*(be[3] - be[2]));
                    const float EminusE4 = e - be[3];
                    const float E41 = be[3] - be[0];
                    const float E42 = be[3] - be[1];
                    const float E43 = be[3] - be[2];
                    const float C = -tetvol / 4.0 * Cub(EminusE4) / E41 / E42 / E43;
                    const float dC = -3.0 * tetvol / 4.0 /  E41 / E42 / E43 * Sqr(EminusE4);
                    weights[0] = dC * EminusE4 / E41 + C / E41;
                    weights[1] = dC * EminusE4 / E42 + C / E42;
                    weights[2] = dC * EminusE4 / E43 + C / E43;
                    weights[3] = -dC * (4.0 + (1.0 / E41 + 1.0 / E42 + 1.0 / E43) * EminusE4) - C * (1.0 / E41 + 1.0 / E42 + 1.0 / E43);
                }
                //calculate orbital-resolved density of states from weights, mind that energies were sorted and that weights refer to sorted energies
                for(uint j=threadIdx.x;j<8*norb;j+=blockDim.x){
                    uint ipart = j / 4 / norb;
                    uint icorner = (j - ipart * 4 * norb) / norb;
                    uint iorb = j % norb;
                    uint il1 = IndicesThisTetrahedron[icorner] * norb * nband + iorb * nband + ib;
                    uint il2 = icorner * norb + iorb;
                    if(0 == ipart){
                        rparts[il2] = realpart[il1];
                    }
                    else{
                        iparts[il2] = imagpart[il1];
                    }
                }
                __syncthreads();
                for(uint i=0;i<4;i++){
                    for(uint io=0;io<norb;io++){
                        uint il2 = i * norb + io;
                        orbitaldosThisThread[io] += weights[i] * (rparts[il2] * rparts[il2] + iparts[il2] * iparts[il2]);
                    }//end loop over orbitals
                }//end loop over corners of tetrahedron
            }//end loop over bands
        }
    }//end loop over ik

    if(ifreq < ne){
        if(1 == gridDim.y){ //if all k-points are calculated in one block, no atomics are required
        //write results to global array after looping is finished
            totaldos[ifreq] = totaldosThisThread;
            for(uint iorb=0;iorb<norb;iorb++){
                uint il2 = iorb * ne + ifreq;
                orbitaldos[il2] = orbitaldosThisThread[iorb];
            }
        }
        else{
            //write results to global array after looping is finished
            atomicAdd(&(totaldos[ifreq]), totaldosThisThread);
            for(uint iorb=0;iorb<norb;iorb++){
                uint il2 = iorb * ne + ifreq;
                atomicAdd(&(orbitaldos[il2]), orbitaldosThisThread[iorb]);
            }
        }
    }
}

void DosWrapper(double* elapsedTimeInMilliseconds, int nk, int nx, int ny, int nz, int norb, int nband, int ne, float* energies, float* realpart, float* imagpart, float* bandenergies, float* tdosptr, float* dosorbptr){
    /*
    Wrapper function that can be plugged into host code.

    nk is the total number of k-points on the rectangular grid. nk = nx * ny * nz.
    nx, ny, nz are the grid dimensions.
    norb is the number of orbitals in the model.
    nband is the number of bands in the model. Usually norb = nband.
    ne is the number of energies at which DOS is calculated.
    energies contains the energy values at which DOS is calculated.
    realpart contains the real part of the eigenvector elements of the Hamiltonian in a flattened array. Original array had structure [nk, norb, nband].
    imagpart contains the imaginary part of the eigenvector elements of the Hamiltonian in a flattened array. Original array had structure [nk, norb, nband].
    bandenergies contains band energies of the Hamiltonian in a flattened array. Original array had structure [nk, nband].
    tdosptr is a pointer to the flattened output array that will contain total DOS for each of the energies in energies array.
    dosorbptr is a pointer to the flattened output array that will contain orbital-resolved DOS for each of the energies in energies array. Expanded array structure is [ne, norb].

    Conversion from expanded array [i,j,k] to flattened array [l] is done in the following way:
    for i in ni:
        for j in nj:
            for k in nk:
                l = i * nj * nk + j * nk + k
                v_l <- v_ijk
    */

    //allocate device memory
    uint *dev_cubes, *dev_si;
    float *dev_e, *dev_r, *dev_i, *dev_b, *dev_tdos, *dev_odos, *dev_se;
    cudaMalloc((void**)&dev_e, ne*sizeof(float));
    cudaMalloc((void**)&dev_r, nk*norb*nband*sizeof(float));
    cudaMalloc((void**)&dev_i, nk*norb*nband*sizeof(float));
    cudaMalloc((void**)&dev_b, nk*nband*sizeof(float));
    cudaMalloc((void**)&dev_tdos, ne*sizeof(float));
    cudaMalloc((void**)&dev_odos, ne*norb*sizeof(float));
    cudaMalloc((void**)&dev_se, nk*nband*24*sizeof(float));
    cudaMalloc((void**)&dev_cubes, nk*8*sizeof(uint));
    cudaMalloc((void**)&dev_si, nk*nband*24*sizeof(uint));
    int tetradefs[24] = {0,1,2,5,1,2,3,5,2,3,5,7,2,5,6,7,2,4,5,6,0,2,4,5}; //map from tetrahedron corners to corners of an elemental cube
    //copy data to GPU
    cudaMemcpy(dev_e, energies, ne*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_r, realpart, nk*norb*nband*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_i, imagpart, nk*norb*nband*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, bandenergies, nk*nband*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(dev_tetradefs, tetradefs, 24*sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemset(dev_tdos, 0, ne*sizeof(float));
    cudaMemset(dev_odos, 0, ne*norb*sizeof(float));
    cudaMemset(dev_se, 0, nk*nband*24*sizeof(float));
    cudaMemset(dev_cubes, 0, nk*8*sizeof(uint));
    cudaMemset(dev_si, 0, nk*nband*24*sizeof(uint));

    auto start = std::chrono::steady_clock::now(); //start recording time

    //precalculate tetrahedron indices
    const uint num_cubes_per_block = 1024;
    const uint num_cube_blocks = (nk - 1) / num_cubes_per_block + 1;
    dim3 numBlocksIndices(num_cube_blocks, 1, 1);
    dim3 threadsPerBlockIndices(num_cubes_per_block, 1, 1);
    DensityOfStatesPrecalculateCubeMomentumIndices<<<numBlocksIndices, threadsPerBlockIndices>>>(nk, nx, ny, nz, dev_cubes);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));

    //precalculate energy sorting within tetrahedrons
    const uint num_cube_blocks_cubes = (nk - 1) / CUBES_PER_BLOCK + 1;
    dim3 numBlocksCubes(num_cube_blocks_cubes, nband, 1);
    dim3 threadsPerBlockCubes(CUBES_PER_BLOCK, 8, 1);
    DensityOfStatesPresortEnergiesInTetrahedron<<<numBlocksCubes,threadsPerBlockCubes>>>(nk, nx, ny, nz, nband, dev_b, dev_cubes, dev_se, dev_si);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
    
    //calculate density of states
    const uint num_frequency_blocks = (ne - 1) / FREQUENCIES_PER_BLOCK + 1;
    const uint num_kpoint_blocks = (nk - 1) / KPOINTS_PER_BLOCK + 1;
    dim3 numBlocks(num_frequency_blocks, num_kpoint_blocks, 1);
    dim3 threadsPerBlock(FREQUENCIES_PER_BLOCK, 1, 1);
    
    DensityOfStates<<<numBlocks,threadsPerBlock>>>(nk, nx, ny, nz, norb, nband, ne, dev_e, dev_r, dev_i, dev_se, dev_si, dev_tdos, dev_odos);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));

    auto end = std::chrono::steady_clock::now(); //stop recording time
    *elapsedTimeInMilliseconds = ((std::chrono::duration<double>)(end - start)).count() * 1000;

    //free device memory
    cudaFree(dev_r);
    cudaFree(dev_i);
    cudaFree(dev_b);

    //copy result to CPU
    cudaMemcpy(tdosptr, dev_tdos, ne*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dosorbptr, dev_odos, ne*norb*sizeof(float), cudaMemcpyDeviceToHost);
    
    //free device memory
    cudaFree(dev_tdos);
    cudaFree(dev_odos);
}