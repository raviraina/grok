#include "grok_includes.h"

#include <Synthesis.h>

namespace grk {

#ifdef _WIN32
#define ATTR_ALIGNED_64
#else
#define ATTR_ALIGNED_64 __attribute__ ((aligned (64)))
#endif


template <typename T, size_t W, size_t H> void Synthesis<T,W,H>::test(size_t size){
    const size_t num_threads = ThreadPool::get()->num_threads();
    const size_t band_width = size/2;
    const size_t band_height = band_width;
    const size_t band_stride = band_width;
    const size_t band_area = band_stride * band_height;
    const auto ATTR_ALIGNED_64 src =
    		(T*)grk_aligned_malloc(band_area * 4 * sizeof(T));
    const auto ATTR_ALIGNED_64 dest =
    		(T*)grk_aligned_malloc(band_area * 4 * sizeof(T));
    const auto ATTR_ALIGNED_64 windows =
    		(T*)grk_aligned_malloc(W*H*num_threads * sizeof(T));
    const T* band[4] =
    		{src, src + band_area, src + 2*band_area, src + 3*band_area};

	//1. in raster pattern, read 4 W/2xH/2 windows, one in each band,
	// line by line.
	//2.Each line is transformed with dwt53 transform,
	// transposed, and stored in window buffer.
	//3. horizontal lines are transformed, then written out
	//   to dest
	const size_t grid_width = band_width /(W/2);
	const size_t grid_height = band_height /(H/2);
	const size_t num_windows = grid_width * grid_height;

	auto synth = [size,
					band,
					band_width,band_stride,
					grid_width,grid_height,
					src,dest,
					windows, num_windows,
					num_threads](){
		auto threadnum =  num_threads == 1 ? 0 :
				ThreadPool::get()->thread_number(std::this_thread::get_id());
	    const VREG two = LOAD_CST(2);
		const T*  ATTR_ALIGNED_64 window = windows + threadnum*W*H;
		for (size_t k = threadnum; k < num_windows; k +=num_threads){
			// read from src
			const size_t windowX = k % grid_width;
			const size_t windowY = k / grid_width;
			const size_t offset =  windowX * (W/2) + windowY * (H/2) * band_stride;
			const T*  bandPtr[4] = {band[0] + offset,
										 band[1] + offset,
										 band[2] + offset,
										 band[3] + offset};
			const T*  windowPtr[2] =	{window,
											 window + (W*H)/2};

			const size_t num_loads_src = (W/2)/VREG_INT_COUNT;


			// low pass band (y axis)
		    VREG s1n_L = LOADU(bandPtr[0]);
		    VREG d1n_L = LOADU(bandPtr[1]);

		    /* s0n = s1n - ((d1n + 1) >> 1); <==> */
		    /* s0n = s1n - ((d1n + d1n + 2) >> 2); */
		    VREG s0n_L = SUB(s1n_L, SAR(ADD3(d1n_L, d1n_L, two), 2));

		    // high pass band (y axis)
		    VREG s1n_U = LOADU(bandPtr[2]);
		    VREG d1n_U = LOADU(bandPtr[3]);

		    /* s0n = s1n - ((d1n + 1) >> 1); <==> */
		    /* s0n = s1n - ((d1n + d1n + 2) >> 2); */
		    VREG s0n_U = SUB(s1n_U, SAR(ADD3(d1n_U, d1n_U, two), 2));

			for (size_t p = 0; p < H/2; ++p){
				for (size_t r = 1; r < num_loads_src; r++){
					// low pass band (y axis)
					VREG d1c_L = d1n_L;
					VREG s0c_L = s0n_L;

				    s1n_L = LOAD(bandPtr[0] + p * W + r * VREG_INT_COUNT);
				    VREG s0n_L = SUB(s1n_L, SAR(ADD3(d1n_L, d1n_L, two), 2));

				    s1n_L = LOAD(bandPtr[1] + p * W + r * VREG_INT_COUNT);
				    s0n_L = SUB(s1n_L, SAR(ADD3(d1n_L, d1n_L, two), 2));

				    STORE(windowPtr[0] +  r * VREG_INT_COUNT, s0c_L);
					/* d1c + ((s0c + s0n) >> 1) */
					STORE(windowPtr[0] + W/2 +  r * VREG_INT_COUNT,
							ADD(d1c_L, SAR(ADD(s0c_L, s0n_L), 1)));

					// high pass band (y axis)
					VREG d1c_U = d1n_U;
					VREG s0c_U = s0n_U;

				    s1n_U = LOAD(bandPtr[2] + p * W + r * VREG_INT_COUNT);
				    VREG s0n_U = SUB(s1n_U, SAR(ADD3(d1n_U, d1n_U, two), 2));

				    s1n_U = LOAD(bandPtr[3] + p * W + r * VREG_INT_COUNT);
				    s0n_U = SUB(s1n_U, SAR(ADD3(d1n_U, d1n_U, two), 2));


				    STORE(windowPtr[1] +  r * VREG_INT_COUNT, s0c_U);
					/* d1c + ((s0c + s0n) >> 1) */
					STORE(windowPtr[1] + W/2 +  r * VREG_INT_COUNT,
							ADD(d1c_U, SAR(ADD(s0c_U, s0n_U), 1)));
				}
				windowPtr[0] += W;
				windowPtr[1] += W;
			}
			//2. write to dest
			windowPtr[0] = window;
			const T*  ATTR_ALIGNED_64 destPtr = 	dest + k * W * H;
			const size_t num_loads_dest = W/VREG_INT_COUNT;
			for (size_t p = 0; p < H; ++p){
				for (size_t r = 0; r < num_loads_dest; ++r){
					VREG ld0 = LOAD(windowPtr[0] + r * VREG_INT_COUNT);
					STORE(destPtr + r * VREG_INT_COUNT, ld0);
				}
				destPtr += W;
				windowPtr[0] += W;
			}
		}
		return 0;

	};
	std::vector< std::future<int> > results;

	if (num_threads == 1)
		synth();
	else
		for (size_t j = 0; j < num_threads; ++j)
			results.emplace_back(ThreadPool::get()->enqueue(synth));


	for(auto && result: results)
		result.get();

	grk_aligned_free(src);
    grk_aligned_free(dest);
    grk_aligned_free(windows);
}

} /* namespace grk */
