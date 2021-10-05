/*
 *    Copyright (C) 2016-2021 Grok Image Compression Inc.
 *
 *    This source code is free software: you can redistribute it and/or  modify
 *    it under the terms of the GNU Affero General Public License, version 3,
 *    as published by the Free Software Foundation.
 *
 *    This source code is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU Affero General Public License for more details.
 *
 *    You should have received a copy of the GNU Affero General Public License
 *    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 *    This source code incorporates work covered by the following copyright and
 *    permission notice:
 *
 * The copyright in this software is being made available under the 2-clauses
 * BSD License, included below. This software may be subject to other third
 * party and contributor rights, including patent rights, and no such rights
 * are granted under this license.
 *
 * Copyright (c) 2017, IntoPix SA <contact@intopix.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS `AS IS'
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include "grk_includes.h"
#include "spdlog/spdlog.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/times.h>
#endif /* _WIN32 */

#include <chrono> // for high_resolution_clock
#define TCLAP_NAMESTARTSTRING "-"
#include "tclap/CmdLine.h"
using namespace TCLAP;

using namespace grk;

#define GRK_TEST_SYNTHESIS
#ifdef GRK_TEST_SYNTHESIS
#include "Synthesis.h"
#include "Synthesis.cpp"
#endif

namespace grk {

int32_t getValue(uint32_t i){
    return ((int32_t)i % 511) - 256;
}

bool init_tilec(TileComponent* tilec, uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1,
				uint32_t numresolutions, GrkImage* outputImage)
{
	tilec->numresolutions = numresolutions;
	tilec->resolutions_to_decompress = numresolutions;
	tilec->tileCompResolution = new Resolution[tilec->numresolutions];
	for(auto i = 0; i < tilec->numresolutions; ++i)
		memset(tilec->tileCompResolution + i, 0, sizeof(Resolution));
	uint32_t leveno = tilec->numresolutions - 1;
	auto res = tilec->tileCompResolution;

	/* Adapted from grk_init_tile() */
	for(uint32_t resno = 0; resno < tilec->numresolutions; ++resno)
	{
		/* border for each resolution level (global) */
		res->x0 = ceildivpow2<uint32_t>(x0, leveno);
		res->y0 = ceildivpow2<uint32_t>(y0, leveno);
		res->x1 = ceildivpow2<uint32_t>(x1, leveno);
		res->y1 = ceildivpow2<uint32_t>(y1, leveno);
		++res;
		--leveno;
	}
	bool isCompressor = (outputImage == nullptr);
	auto highestNumberOfResolutions =
		(!isCompressor) ? tilec->resolutions_to_decompress : numresolutions;
	auto hightestResolution = tilec->tileCompResolution + highestNumberOfResolutions - 1;
	tilec->set(hightestResolution);
	grkRectU32 unreduced_tile_comp_window_dims;
	if(!isCompressor)
	{
		unreduced_tile_comp_window_dims = grkRectU32(
			ceildiv<uint32_t>(outputImage->x0, 1), ceildiv<uint32_t>(outputImage->y0, 1),
			ceildiv<uint32_t>(outputImage->x1, 1), ceildiv<uint32_t>(outputImage->y1, 1));
	}
	tilec->allocWindowBuffer(tilec, unreduced_tile_comp_window_dims);
	tilec->getBuffer()->alloc();
	auto data = tilec->getBuffer()->ptr();
	for(size_t i = 0; i < tilec->getBuffer()->stridedArea(); i++)
		data[i] = getValue((uint32_t)i);
	return true;
}

void usage(void)
{
	printf("bench_dwt [-size value] [-check] [-display] [-num_resolutions val] [-lossy]\n");
	printf("[-offset x y] [-num_threads val]\n");
}

class GrokOutput : public StdOutput
{
  public:
	virtual void usage(CmdLineInterface& c)
	{
		(void)c;
		::usage();
	}
};
} // namespace grk

int main(int argc, char** argv)
{
	uint32_t num_threads = 0;
	GrkImage image;
	GrkImage outputImage;
	Tile tile;
	TileComponent tilec;
	grk_image_comp imageComp;
	grk_image_comp output_image_comp;
	int32_t i, j;
	bool display = false;
	bool check = false;
	bool lossy = false;
	bool forward = false;
	uint32_t size = 16385 - 1;
	uint32_t offset_x = (uint32_t)((size + 1) / 2 - 1);
	uint32_t offset_y = (uint32_t)((size + 1) / 2 - 1);
	uint32_t num_resolutions = 6;

	CmdLine cmd("bench_dwt command line", ' ', grk_version());

	// set the output
	GrokOutput output;
	cmd.setOutput(&output);

	SwitchArg displayArg("d", "display", "display", cmd);
	SwitchArg checkArg("c", "check", "check", cmd);
	ValueArg<uint32_t> sizeArg("s", "size", "Size of image", false, 0, "unsigned integer", cmd);
	ValueArg<uint32_t> numThreadsArg("H", "num_threads", "Number of threads", false, 0,
									 "unsigned integer", cmd);
	ValueArg<uint32_t> numResolutionsArg("n", "Resolutions", "Number of resolutions", false, 0,
										 "unsigned integer", cmd);
	SwitchArg lossyArg("I", "irreversible", "irreversible dwt", cmd);
	SwitchArg forwardArg("F", "forward", "forward dwt", cmd);

	SwitchArg threadScalingArg("S", "ThreadScaling", "Thread scaling", cmd);

	cmd.parse(argc, argv);

	if(displayArg.isSet())
	{
		display = true;
		check = true;
	}
	if(checkArg.isSet())
		check = true;
	if(lossyArg.isSet())
		lossy = true;
	if(sizeArg.isSet())
	{
		size = sizeArg.getValue();
		offset_x = (uint32_t)((size + 1) / 2 - 1);
		offset_y = (uint32_t)((size + 1) / 2 - 1);
	}
	if(numThreadsArg.isSet())
		num_threads = numThreadsArg.getValue();
	if(num_threads == 0)
		num_threads = ThreadPool::hardware_concurrency();
	if(numResolutionsArg.isSet())
	{
		num_resolutions = numResolutionsArg.getValue();
		if(num_resolutions == 0 || num_resolutions > 32)
		{
			spdlog::error("Invalid value for num_resolutions. "
						  "Should be >= 1 and <= 32");
			exit(1);
		}
	}
	if(forwardArg.isSet())
		forward = forwardArg.getValue();

	size_t begin = num_threads;
	size_t end = num_threads;
	if(threadScalingArg.isSet())
		begin = 1;

	CodeStream codeStream(!forwardArg.isSet(), nullptr);
	codeStream.m_headerImage = &image;

	for(size_t k = begin; k <= end; ++k)
	{
		memset(&image, 0, sizeof(image));
		image.numcomps = 1;
		image.x0 = offset_x;
		image.y0 = offset_y;
		image.x1 = offset_x + size;
		image.y1 = offset_y + size;
		image.comps = &imageComp;
		memset(&imageComp, 0, sizeof(imageComp));
		imageComp.dx = 1;
		imageComp.dy = 1;
		imageComp.w = size;
		imageComp.stride = size;
		imageComp.h = size;

		memset(&outputImage, 0, sizeof(outputImage));
		outputImage.numcomps = 1;
		outputImage.x0 = offset_x;
		outputImage.y0 = offset_y;
		outputImage.x1 = offset_x + size;
		outputImage.y1 = offset_y + size;
		outputImage.comps = &output_image_comp;
		memset(&output_image_comp, 0, sizeof(output_image_comp));
		output_image_comp.dx = 1;
		output_image_comp.dy = 1;
		output_image_comp.w = size;
		output_image_comp.stride = size;
		output_image_comp.h = size;

		std::unique_ptr<TileProcessor> tileProcessor(new TileProcessor(&codeStream, nullptr));
		grk_initialize(nullptr, k);
		init_tilec(&tilec, offset_x, offset_y, offset_x + size, offset_y + size, num_resolutions,
				   &image);
		auto data = tilec.getBuffer()->ptr();

		if(display)
		{
			spdlog::info("Before");
			k = 0;
			for(j = 0; j < (int32_t)(tilec.height()); j++)
			{
				for(i = 0; i < (int32_t)(tilec.width()); i++)
				{
					printf("%u ", data[k]);
					k++;
				}
				printf("\n");
			}
		}
		memset(&tile, 0, sizeof(tile));
		tile.x0 = tilec.x0;
		tile.y0 = tilec.y0;
		tile.x1 = tilec.x1;
		tile.y1 = tilec.y1;
		tile.numcomps = 1;
		tile.comps = &tilec;

		std::chrono::time_point<std::chrono::high_resolution_clock> start, finish;
		std::chrono::duration<double> elapsed;

		start = std::chrono::high_resolution_clock::now();
#ifdef GRK_TEST_SYNTHESIS
	    Synthesis<int32_t, 64,64> synth;
	    synth.test(size);
#else
		bool rc = false;
		if(forward)
		{
			WaveletFwdImpl w;
			rc = w.compress(&tilec, lossy ? 0 : 1);
		}
		else
		{
			WaveletReverse w;
			rc = w.decompress(tileProcessor.get(), &tilec, tilec.getBuffer()->unreducedBounds(),
							  tilec.numresolutions, lossy ? 0 : 1);
		}
		assert(rc);
#endif
		finish = std::chrono::high_resolution_clock::now();
		elapsed = finish - start;
		spdlog::info("{} dwt {} with {:02d} threads: {} ms", lossy ? "lossy" : "lossless",
					 forward ? "compress" : "decompress", k, (uint32_t)(elapsed.count() * 1000));

		if(display || check)
		{
			if(display)
			{
				spdlog::info("After IDWT\n");
				k = 0;
				for(j = 0; j < (int32_t)(tilec.height()); j++)
				{
					for(i = 0; i < (int32_t)(tilec.width()); i++)
					{
						printf("%u ", data[k]);
						k++;
					}
					printf("\n");
				}
			}
			WaveletFwdImpl w;
			w.compress(&tilec, 1);
			if(display)
			{
				spdlog::info("After FDWT\n");
				k = 0;
				for(j = 0; j < (int32_t)(tilec.height()); j++)
				{
					for(i = 0; i < (int32_t)(tilec.width()); i++)
					{
						printf("%u ", data[k]);
						k++;
					}
					printf("\n");
				}
			}
			if(check)
			{
				size_t idx;
				size_t nValues = (size_t)tilec.area();
				for(idx = 0; idx < nValues; idx++)
				{
					if(data[idx] != getValue((uint32_t)idx))
					{
						printf("Difference found at idx = %u\n", (uint32_t)idx);
						return 1;
					}
				}
			}
		}
		grk_deinitialize();
	}

	codeStream.m_headerImage = nullptr;
	return 0;
}
