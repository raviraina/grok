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
 *    This source code incorporates work covered by the BSD 2-clause license.
 *    Please see the LICENSE file in the root directory for details.
 *
 */

#pragma once

#include <cstdint>

namespace grk
{
/**
 * Quantization stepsize
 */
struct grk_stepsize
{
	grk_stepsize() : expn(0), mant(0) {}
	/** exponent - 5 bits */
	uint8_t expn;
	/** mantissa  -11 bits */
	uint16_t mant;
};

class CodeStream;
class CodeStreamDecompress;
struct TileComponentCodingParams;
struct BufferedStream;
struct Subband;
struct TileCodingParams;
struct TileProcessor;

class Quantizer
{
  public:
	bool setBandStepSizeAndBps(TileCodingParams* tcp, Subband* band, uint32_t resno,
							   uint8_t bandIndex, TileComponentCodingParams* tccp,
							   uint8_t image_precision, bool compress);

	uint32_t get_SQcd_SQcc_size(CodeStream* codeStream, uint32_t comp_no);
	bool compare_SQcd_SQcc(CodeStream* codeStream, uint32_t first_comp_no, uint32_t second_comp_no);
	bool read_SQcd_SQcc(CodeStreamDecompress* codeStream, bool fromQCC, uint32_t comp_no,
						uint8_t* headerData, uint16_t* header_size);
	bool write_SQcd_SQcc(CodeStream* codeStream, uint32_t comp_no, IBufferedStream* stream);
	void apply_quant(TileComponentCodingParams* src, TileComponentCodingParams* dest);
};

} // namespace grk
