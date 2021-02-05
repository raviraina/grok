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
 *    This source code incorporates work covered by the BSD 2-clause license.
 *    Please see the LICENSE file in the root directory for details.
 *
 */

#include "grk_includes.h"

namespace grk {

struct j2k_prog_order {
	GRK_PROG_ORDER enum_prog;
	char str_prog[5];
};

static j2k_prog_order j2k_prog_order_list[] = { { GRK_CPRL, "CPRL" }, {
		GRK_LRCP, "LRCP" }, { GRK_PCRL, "PCRL" }, { GRK_RLCP, "RLCP" }, {
		GRK_RPCL, "RPCL" }, { (GRK_PROG_ORDER) -1, "" } };


static const j2k_mct_function j2k_mct_write_functions_from_float[] = {
		j2k_write_float_to_int16, j2k_write_float_to_int32,
		j2k_write_float_to_float, j2k_write_float_to_float64 };

/**
 * The read header procedure.
 *
 * @param       codeStream          JPEG 2000 code stream
 *
 */
static bool j2k_read_header_procedure(CodeStream *codeStream);

/**
 * The default compressing validation procedure without any extension.
 *
 * @param       codeStream          JPEG 2000 code stream
 *
 * @return true if the parameters are correct.
 */
static bool j2k_compress_validation(CodeStream *codeStream);

/**
 * The default decoding validation procedure without any extension.
 *
 * @param       codeStream          JPEG 2000 code stream
 *
 * @return true if the parameters are correct.
 */
static bool j2k_decompress_validation(CodeStream *codeStream);

/**
 * The mct compressing validation procedure.
 *
 * @param       codeStream          JPEG 2000 code stream
 *
 * @return true if the parameters are correct.
 */
static bool j2k_mct_validation(CodeStream *codeStream);


/**
 * Updates the rates of the tcp.
 *
 * @param       codeStream          JPEG 2000 code stream
 */
static bool j2k_update_rates(CodeStream *codeStream);

/**
 * Copies the decoding tile parameters onto all the tile parameters.
 * Creates also the tile decompressor.
 *
 * @param       codeStream          JPEG 2000 code stream
 */
static bool j2k_copy_default_tcp(CodeStream *codeStream);

/**
 * Read the tiles.
 *
 * @param       codeStream          JPEG 2000 code stream
 */
static bool j2k_decompress_tiles(CodeStream *codeStream);

/**
 * Gets the offset of the header.
 *
 * @param       codeStream          JPEG 2000 code stream
 */
static bool j2k_get_end_header(CodeStream *codeStream);

/**
 * Ends the compressing, i.e. frees memory.
 *
 * @param       codeStream          JPEG 2000 code stream
 */
static bool j2k_end_encoding(CodeStream *codeStream);

/**
 * Inits the Info
 *
 * @param       codeStream          JPEG 2000 code stream
 */
static bool j2k_init_info(CodeStream *codeStream);


/**
 * Checks the progression order changes values. Tells of the poc given as input are valid.
 * A nice message is outputted at errors.
 *
 * @param       p_pocs                the progression order changes.
 * @param       nb_pocs               the number of progression order changes.
 * @param       nb_resolutions        the number of resolutions.
 * @param       numcomps              the number of components
 * @param       numlayers             the number of layers.

 *
 * @return      true if the pocs are valid.
 */
static bool j2k_check_poc_val(const  grk_progression  *p_pocs, uint32_t nb_pocs,
		uint32_t nb_resolutions, uint32_t numcomps, uint32_t numlayers);

/**
 * Gets the number of tile parts used for the given change of progression (if any) and the given tile.
 *
 * @param               cp                      the coding parameters.
 * @param               pino            the offset of the given poc (i.e. its position in the coding parameter).
 * @param               tileno          the given tile.
 *
 * @return              the number of tile parts.
 */
static uint64_t j2k_get_num_tp(CodingParams *cp, uint32_t pino, uint16_t tileno);

/**
 * Calculates the total number of tile parts needed by the compressor to
 * compress such an image. If not enough memory is available, then the function return false.
 *
 * @param       cp              coding parameters for the image.
 * @param       p_nb_tile_parts total number of tile parts in whole image.
 * @param       image           image to compress.

 *
 * @return true if the function was successful, false else.
 */
static bool j2k_calculate_tp(CodingParams *cp, uint16_t *p_nb_tile_parts, GrkImage *image);

/**
 * LUP decomposition
 */
static bool lupDecompose(float *matrix, uint32_t *permutations,
		float *p_swap_area, uint32_t nb_compo);
/**
 * LUP solving
 */
static void lupSolve(float *pResult, float *pMatrix, float *pVector,
		uint32_t *pPermutations, uint32_t nb_compo, float *p_intermediate_data);

/**
 *LUP inversion (call with the result of lupDecompose)
 */
static void lupInvert(float *pSrcMatrix, float *pDestMatrix, uint32_t nb_compo,
		uint32_t *pPermutations, float *p_src_temp, float *p_dest_temp,
		float *p_swap_area);

static bool j2k_decompress_validation(CodeStream *codeStream) {

	return codeStream->decompress_validation();
}
static bool j2k_read_header_procedure(CodeStream *codeStream) {
	bool rc = false;
	try {
		rc = codeStream->read_header_procedure();
	} catch (InvalidMarkerException &ime){
		GRK_ERROR("Found invalid marker : 0x%x", ime.m_marker);
		rc = false;
	}
	return rc;
}
static bool j2k_decompress_tiles(CodeStream *codeStream) {
	return codeStream->decompress_tiles();
}
static bool j2k_decompress_tile(CodeStream *codeStream) {
	return codeStream->decompress_tile();
}
bool j2k_check_poc_val(const grk_progression *p_pocs, uint32_t nb_pocs,
		uint32_t nb_resolutions, uint32_t num_comps, uint32_t num_layers) {
	uint32_t index, resno, compno, layno;
	uint32_t i;
	uint32_t step_c = 1;
	uint32_t step_r = num_comps * step_c;
	uint32_t step_l = nb_resolutions * step_r;
	if (nb_pocs == 0)
		return true;

	auto packet_array = new uint32_t[step_l * num_layers];
	memset(packet_array, 0, step_l * num_layers * sizeof(uint32_t));

	/* iterate through all the pocs */
	for (i = 0; i < nb_pocs; ++i) {
		index = step_r * p_pocs->resS;
		/* take each resolution for each poc */
		for (resno = p_pocs->resS;	resno < std::min<uint32_t>(p_pocs->resE, nb_resolutions);++resno) {
			uint32_t res_index = index + p_pocs->compS * step_c;

			/* take each comp of each resolution for each poc */
			for (compno = p_pocs->compS;compno < std::min<uint32_t>(p_pocs->compE, num_comps);	++compno) {
				uint32_t comp_index = res_index + 0 * step_l;

				/* and finally take each layer of each res of ... */
				for (layno = 0;
						layno < std::min<uint32_t>(p_pocs->layE, num_layers);
						++layno) {
					/*index = step_r * resno + step_c * compno + step_l * layno;*/
					packet_array[comp_index] = 1;
					comp_index += step_l;
				}
				res_index += step_c;
			}
			index += step_r;
		}
		++p_pocs;
	}
	bool loss = false;
	index = 0;
	for (layno = 0; layno < num_layers; ++layno) {
		for (resno = 0; resno < nb_resolutions; ++resno) {
			for (compno = 0; compno < num_comps; ++compno) {
				loss |= (packet_array[index] != 1);
				index += step_c;
			}
		}
	}
	if (loss)
		GRK_ERROR("Missing packets possible loss of data");
	delete[] packet_array;

	return !loss;
}
static bool j2k_mct_validation(CodeStream *codeStream) {
	return codeStream->mct_validation();
}
static bool j2k_compress_validation(CodeStream *codeStream) {
	return codeStream->compress_validation();
}
bool j2k_init_mct_encoding(TileCodingParams *p_tcp, GrkImage *p_image) {
	uint32_t i;
	uint32_t indix = 1;
	grk_mct_data *mct_deco_data = nullptr, *mct_offset_data = nullptr;
	grk_simple_mcc_decorrelation_data *mcc_data;
	uint32_t mct_size, nb_elem;
	float *data, *current_data;
	TileComponentCodingParams *tccp;

	assert(p_tcp != nullptr);

	if (p_tcp->mct != 2)
		return true;

	if (p_tcp->m_mct_decoding_matrix) {
		if (p_tcp->m_nb_mct_records == p_tcp->m_nb_max_mct_records) {
			p_tcp->m_nb_max_mct_records += default_number_mct_records;

			auto new_mct_records = (grk_mct_data*) grk_realloc(p_tcp->m_mct_records,
					p_tcp->m_nb_max_mct_records * sizeof(grk_mct_data));
			if (!new_mct_records) {
				grk_free(p_tcp->m_mct_records);
				p_tcp->m_mct_records = nullptr;
				p_tcp->m_nb_max_mct_records = 0;
				p_tcp->m_nb_mct_records = 0;
				/* GRK_ERROR( "Not enough memory to set up mct compressing"); */
				return false;
			}
			p_tcp->m_mct_records = new_mct_records;
			mct_deco_data = p_tcp->m_mct_records + p_tcp->m_nb_mct_records;

			memset(mct_deco_data, 0,
					(p_tcp->m_nb_max_mct_records - p_tcp->m_nb_mct_records)
							* sizeof(grk_mct_data));
		}
		mct_deco_data = p_tcp->m_mct_records + p_tcp->m_nb_mct_records;
		grk_free(mct_deco_data->m_data);
		mct_deco_data->m_data = nullptr;

		mct_deco_data->m_index = indix++;
		mct_deco_data->m_array_type = MCT_TYPE_DECORRELATION;
		mct_deco_data->m_element_type = MCT_TYPE_FLOAT;
		nb_elem = p_image->numcomps * p_image->numcomps;
		mct_size = nb_elem * MCT_ELEMENT_SIZE[mct_deco_data->m_element_type];
		mct_deco_data->m_data = (uint8_t*) grk_malloc(mct_size);

		if (!mct_deco_data->m_data)
			return false;

		j2k_mct_write_functions_from_float[mct_deco_data->m_element_type](
				p_tcp->m_mct_decoding_matrix, mct_deco_data->m_data, nb_elem);

		mct_deco_data->m_data_size = mct_size;
		++p_tcp->m_nb_mct_records;
	}

	if (p_tcp->m_nb_mct_records == p_tcp->m_nb_max_mct_records) {
		grk_mct_data *new_mct_records;
		p_tcp->m_nb_max_mct_records += default_number_mct_records;
		new_mct_records = (grk_mct_data*) grk_realloc(p_tcp->m_mct_records,
				p_tcp->m_nb_max_mct_records * sizeof(grk_mct_data));
		if (!new_mct_records) {
			grk_free(p_tcp->m_mct_records);
			p_tcp->m_mct_records = nullptr;
			p_tcp->m_nb_max_mct_records = 0;
			p_tcp->m_nb_mct_records = 0;
			/* GRK_ERROR( "Not enough memory to set up mct compressing"); */
			return false;
		}
		p_tcp->m_mct_records = new_mct_records;
		mct_offset_data = p_tcp->m_mct_records + p_tcp->m_nb_mct_records;

		memset(mct_offset_data, 0,
				(p_tcp->m_nb_max_mct_records - p_tcp->m_nb_mct_records)
						* sizeof(grk_mct_data));
		if (mct_deco_data)
			mct_deco_data = mct_offset_data - 1;
	}
	mct_offset_data = p_tcp->m_mct_records + p_tcp->m_nb_mct_records;
	if (mct_offset_data->m_data) {
		grk_free(mct_offset_data->m_data);
		mct_offset_data->m_data = nullptr;
	}
	mct_offset_data->m_index = indix++;
	mct_offset_data->m_array_type = MCT_TYPE_OFFSET;
	mct_offset_data->m_element_type = MCT_TYPE_FLOAT;
	nb_elem = p_image->numcomps;
	mct_size = nb_elem * MCT_ELEMENT_SIZE[mct_offset_data->m_element_type];
	mct_offset_data->m_data = (uint8_t*) grk_malloc(mct_size);
	if (!mct_offset_data->m_data)
		return false;

	data = (float*) grk_malloc(nb_elem * sizeof(float));
	if (!data) {
		grk_free(mct_offset_data->m_data);
		mct_offset_data->m_data = nullptr;
		return false;
	}
	tccp = p_tcp->tccps;
	current_data = data;

	for (i = 0; i < nb_elem; ++i) {
		*(current_data++) = (float) (tccp->m_dc_level_shift);
		++tccp;
	}
	j2k_mct_write_functions_from_float[mct_offset_data->m_element_type](data,
			mct_offset_data->m_data, nb_elem);
	grk_free(data);
	mct_offset_data->m_data_size = mct_size;
	++p_tcp->m_nb_mct_records;

	if (p_tcp->m_nb_mcc_records == p_tcp->m_nb_max_mcc_records) {
		grk_simple_mcc_decorrelation_data *new_mcc_records;
		p_tcp->m_nb_max_mcc_records += default_number_mct_records;
		new_mcc_records = (grk_simple_mcc_decorrelation_data*) grk_realloc(
				p_tcp->m_mcc_records,
				p_tcp->m_nb_max_mcc_records
						* sizeof(grk_simple_mcc_decorrelation_data));
		if (!new_mcc_records) {
			grk_free(p_tcp->m_mcc_records);
			p_tcp->m_mcc_records = nullptr;
			p_tcp->m_nb_max_mcc_records = 0;
			p_tcp->m_nb_mcc_records = 0;
			/* GRK_ERROR( "Not enough memory to set up mct compressing"); */
			return false;
		}
		p_tcp->m_mcc_records = new_mcc_records;
		mcc_data = p_tcp->m_mcc_records + p_tcp->m_nb_mcc_records;
		memset(mcc_data, 0,
				(p_tcp->m_nb_max_mcc_records - p_tcp->m_nb_mcc_records)
						* sizeof(grk_simple_mcc_decorrelation_data));

	}
	mcc_data = p_tcp->m_mcc_records + p_tcp->m_nb_mcc_records;
	mcc_data->m_decorrelation_array = mct_deco_data;
	mcc_data->m_is_irreversible = 1;
	mcc_data->m_nb_comps = p_image->numcomps;
	mcc_data->m_index = indix++;
	mcc_data->m_offset_array = mct_offset_data;
	++p_tcp->m_nb_mcc_records;

	return true;
}

static bool j2k_end_encoding(CodeStream *codeStream) {
	(void) codeStream;
	return true;
}
static bool j2k_init_info(CodeStream *codeStream) {
	assert(codeStream);

	return j2k_calculate_tp(&codeStream->m_cp,
							&codeStream->m_encoder.m_total_tile_parts,
							codeStream->getHeaderImage());
}

static bool j2k_get_end_header(CodeStream *codeStream) {
	return codeStream->get_end_header();
}
static bool j2k_copy_default_tcp(CodeStream *codeStream) {
	return codeStream->copy_default_tcp();
}
static bool j2k_update_rates(CodeStream *codeStream) {
	return codeStream->update_rates();
}
char* j2k_convert_progression_order(GRK_PROG_ORDER prg_order) {
	j2k_prog_order *po;
	for (po = j2k_prog_order_list; po->enum_prog != -1; po++) {
		if (po->enum_prog == prg_order)
			return (char*) po->str_prog;
	}

	return po->str_prog;
}
static uint64_t j2k_get_num_tp(CodingParams *cp, uint32_t pino,	uint16_t tileno) {
	uint64_t num_tp = 1;

	/*  preconditions */
	assert(tileno < (cp->t_grid_width * cp->t_grid_height));
	assert(pino < (cp->tcps[tileno].numpocs + 1));

	/* get the given tile coding parameter */
	auto tcp = &cp->tcps[tileno];
	assert(tcp != nullptr);

	auto current_poc = &(tcp->progression[pino]);
	assert(current_poc != 0);

	/* get the progression order as a character string */
	auto prog = j2k_convert_progression_order(tcp->prg);
	assert(strlen(prog) > 0);

	if (cp->m_coding_params.m_enc.m_tp_on) {
		for (uint32_t i = 0; i < 4; ++i) {
			switch (prog[i]) {
			/* component wise */
			case 'C':
				num_tp *= current_poc->tpCompE;
				break;
				/* resolution wise */
			case 'R':
				num_tp *= current_poc->tpResE;
				break;
				/* precinct wise */
			case 'P':
				num_tp *= current_poc->tpPrecE;
				break;
				/* layer wise */
			case 'L':
				num_tp *= current_poc->tpLayE;
				break;
			}
			//we start a new tile part with every progression change
			if (cp->m_coding_params.m_enc.m_tp_flag == prog[i]) {
				cp->m_coding_params.m_enc.m_tp_pos = i;
				break;
			}
		}
	} else {
		num_tp = 1;
	}
	return num_tp;
}

static bool j2k_calculate_tp(CodingParams *cp, uint16_t *p_nb_tile_parts,
		GrkImage *image) {
	assert(p_nb_tile_parts != nullptr);
	assert(cp != nullptr);
	assert(image != nullptr);

	uint32_t nb_tiles = (uint16_t) (cp->t_grid_width * cp->t_grid_height);
	*p_nb_tile_parts = 0;
	auto tcp = cp->tcps;
	for (uint16_t tileno = 0; tileno < nb_tiles; ++tileno) {
		uint8_t totnum_tp = 0;
		pi_update_encoding_parameters(image, cp, tileno);
		for (uint32_t pino = 0; pino <= tcp->numpocs; ++pino) {
			uint64_t num_tp = j2k_get_num_tp(cp, pino, tileno);
			if (num_tp > max_num_tile_parts_per_tile){
				GRK_ERROR("Number of tile parts %d exceeds maximum number of "
						"tile parts %d", num_tp,max_num_tile_parts_per_tile );
				return false;
			}

			uint64_t total = num_tp + *p_nb_tile_parts;
			if (total > max_num_tile_parts){
				GRK_ERROR("Total number of tile parts %d exceeds maximum total number of "
						"tile parts %d", total,max_num_tile_parts );
				return false;
			}

			*p_nb_tile_parts = (uint16_t)(*p_nb_tile_parts + num_tp);
			totnum_tp = (uint8_t) (totnum_tp + num_tp);
		}
		tcp->m_nb_tile_parts = totnum_tp;
		++tcp;
	}

	return true;
}

/*
 ==========================================================
 Matric inversion interface
 ==========================================================
 */
/**
 * Matrix inversion.
 */
static bool matrix_inversion_f(float *pSrcMatrix, float *pDestMatrix,
		uint32_t nb_compo) {
	uint8_t *data = nullptr;
	uint32_t permutation_size = nb_compo * (uint32_t) sizeof(uint32_t);
	uint32_t swap_size = nb_compo * (uint32_t) sizeof(float);
	uint32_t total_size = permutation_size + 3 * swap_size;
	uint32_t *lPermutations = nullptr;
	float *double_data = nullptr;

 data = (uint8_t*) grk_malloc(total_size);
	if (data == 0) {
		return false;
	}
	lPermutations = (uint32_t*) data;
 double_data = (float*) (data + permutation_size);
	memset(lPermutations, 0, permutation_size);

	if (!lupDecompose(pSrcMatrix, lPermutations, double_data, nb_compo)) {
		grk_free(data);
		return false;
	}

	lupInvert(pSrcMatrix, pDestMatrix, nb_compo, lPermutations, double_data,
		 double_data + nb_compo, double_data + 2 * nb_compo);
	grk_free(data);

	return true;
}

static bool lupDecompose(float *matrix, uint32_t *permutations,
		float *p_swap_area, uint32_t nb_compo) {
	uint32_t *tmpPermutations = permutations;
	uint32_t *dstPermutations;
	uint32_t k2 = 0, t;
	float temp;
	uint32_t i, j, k;
	uint32_t lLastColum = nb_compo - 1;
	uint32_t lSwapSize = nb_compo * (uint32_t) sizeof(float);
	auto lTmpMatrix = matrix;
	uint32_t offset = 1;
	uint32_t lStride = nb_compo - 1;

	/*initialize permutations */
	for (i = 0; i < nb_compo; ++i) {
		*tmpPermutations++ = i;
	}
	/* now make a pivot with column switch */
	tmpPermutations = permutations;
	for (k = 0; k < lLastColum; ++k) {
		float p = 0.0;

		/* take the middle element */
		auto lColumnMatrix = lTmpMatrix + k;

		/* make permutation with the biggest value in the column */
		for (i = k; i < nb_compo; ++i) {
			temp = ((*lColumnMatrix > 0) ? *lColumnMatrix : -(*lColumnMatrix));
			if (temp > p) {
				p = temp;
				k2 = i;
			}
			/* next line */
			lColumnMatrix += nb_compo;
		}

		/* a whole rest of 0 -> non singular */
		if (p == 0.0) {
			return false;
		}

		/* should we permute ? */
		if (k2 != k) {
			/*exchange of line */
			/* k2 > k */
			dstPermutations = tmpPermutations + k2 - k;
			/* swap indices */
			t = *tmpPermutations;
			*tmpPermutations = *dstPermutations;
			*dstPermutations = t;

			/* and swap entire line. */
			lColumnMatrix = lTmpMatrix + (k2 - k) * nb_compo;
			memcpy(p_swap_area, lColumnMatrix, lSwapSize);
			memcpy(lColumnMatrix, lTmpMatrix, lSwapSize);
			memcpy(lTmpMatrix, p_swap_area, lSwapSize);
		}

		/* now update data in the rest of the line and line after */
		auto lDestMatrix = lTmpMatrix + k;
		lColumnMatrix = lDestMatrix + nb_compo;
		/* take the middle element */
		temp = *(lDestMatrix++);

		/* now compute up data (i.e. coeff up of the diagonal). */
		for (i = offset; i < nb_compo; ++i) {
			/*lColumnMatrix; */
			/* divide the lower column elements by the diagonal value */

			/* matrix[i][k] /= matrix[k][k]; */
			/* p = matrix[i][k] */
			p = *lColumnMatrix / temp;
			*(lColumnMatrix++) = p;

			for (j = /* k + 1 */offset; j < nb_compo; ++j) {
				/* matrix[i][j] -= matrix[i][k] * matrix[k][j]; */
				*(lColumnMatrix++) -= p * (*(lDestMatrix++));
			}
			/* come back to the k+1th element */
			lDestMatrix -= lStride;
			/* go to kth element of the next line */
			lColumnMatrix += k;
		}

		/* offset is now k+2 */
		++offset;
		/* 1 element less for stride */
		--lStride;
		/* next line */
		lTmpMatrix += nb_compo;
		/* next permutation element */
		++tmpPermutations;
	}
	return true;
}

static void lupSolve(float *pResult, float *pMatrix, float *pVector,
		uint32_t *pPermutations, uint32_t nb_compo,
		float *p_intermediate_data) {
	int32_t k;
	uint32_t i, j;
	float sum;
	uint32_t lStride = nb_compo + 1;
	float *lCurrentPtr;
	float *lIntermediatePtr;
	float *lDestPtr;
	float *lTmpMatrix;
	float *lLineMatrix = pMatrix;
	float *lBeginPtr = pResult + nb_compo - 1;
	float *lGeneratedData;
	uint32_t *lCurrentPermutationPtr = pPermutations;

	lIntermediatePtr = p_intermediate_data;
	lGeneratedData = p_intermediate_data + nb_compo - 1;

	for (i = 0; i < nb_compo; ++i) {
		sum = 0.0;
		lCurrentPtr = p_intermediate_data;
		lTmpMatrix = lLineMatrix;
		for (j = 1; j <= i; ++j) {
			/* sum += matrix[i][j-1] * y[j-1]; */
			sum += (*(lTmpMatrix++)) * (*(lCurrentPtr++));
		}
		/*y[i] = pVector[pPermutations[i]] - sum; */
		*(lIntermediatePtr++) = pVector[*(lCurrentPermutationPtr++)] - sum;
		lLineMatrix += nb_compo;
	}

	/* we take the last point of the matrix */
	lLineMatrix = pMatrix + nb_compo * nb_compo - 1;

	/* and we take after the last point of the destination vector */
	lDestPtr = pResult + nb_compo;

	assert(nb_compo != 0);
	for (k = (int32_t) nb_compo - 1; k != -1; --k) {
		sum = 0.0;
		lTmpMatrix = lLineMatrix;
		float u = *(lTmpMatrix++);
		lCurrentPtr = lDestPtr--;
		for (j = (uint32_t) (k + 1); j < nb_compo; ++j) {
			/* sum += matrix[k][j] * x[j] */
			sum += (*(lTmpMatrix++)) * (*(lCurrentPtr++));
		}
		/*x[k] = (y[k] - sum) / u; */
		*(lBeginPtr--) = (*(lGeneratedData--) - sum) / u;
		lLineMatrix -= lStride;
	}
}

static void lupInvert(float *pSrcMatrix, float *pDestMatrix, uint32_t nb_compo,
		uint32_t *pPermutations, float *p_src_temp, float *p_dest_temp,
		float *p_swap_area) {
	uint32_t j, i;
	auto lLineMatrix = pDestMatrix;
	uint32_t lSwapSize = nb_compo * (uint32_t) sizeof(float);

	for (j = 0; j < nb_compo; ++j) {
		auto lCurrentPtr = lLineMatrix++;
		memset(p_src_temp, 0, lSwapSize);
		p_src_temp[j] = 1.0;
		lupSolve(p_dest_temp, pSrcMatrix, p_src_temp, pPermutations, nb_compo,
				p_swap_area);

		for (i = 0; i < nb_compo; ++i) {
			*(lCurrentPtr) = p_dest_temp[i];
			lCurrentPtr += nb_compo;
		}
	}
}

CodeStream::CodeStream(bool decompress, BufferedStream *stream) : m_output_image(nullptr),
																cstr_index(nullptr),
																m_headerImage(nullptr),
																m_tileProcessor(nullptr),
																m_tileCache(new TileCache()),
																m_stream(stream),
																m_tile_ind_to_dec(-1),
																m_marker_scratch(nullptr),
																m_marker_scratch_size(0),
																m_multiTile(false),
																m_curr_marker(0),
																wholeTileDecompress(true),
																current_plugin_tile(nullptr),
																 m_nb_tile_parts_correction_checked(false),
																 m_nb_tile_parts_correction(0),
																 m_headerError(false){
    memset(&m_cp, 0 , sizeof(CodingParams));
    if (decompress){
		m_decompressor.m_default_tcp = new TileCodingParams();
		m_decompressor.m_last_sot_read_pos = 0;

		/* code stream index creation */
		cstr_index = j2k_create_cstr_index();
		if (!cstr_index) {
			delete m_decompressor.m_default_tcp;
			throw std::runtime_error("Out of memory");
		}
    }

    marker_map = {
    {J2K_MS_SOT, new marker_handler(J2K_MS_SOT,J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH_SOT, j2k_read_sot )},
    {J2K_MS_COD, new marker_handler(J2K_MS_COD,J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_cod )},
    {J2K_MS_COC, new marker_handler(J2K_MS_COC,J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_coc )},
    {J2K_MS_RGN, new marker_handler(J2K_MS_RGN,J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_rgn )},
    {J2K_MS_QCD, new marker_handler(J2K_MS_QCD,J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_qcd )},
    {J2K_MS_QCC, new marker_handler(J2K_MS_QCC,J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_qcc )},
    {J2K_MS_POC, new marker_handler(J2K_MS_POC,J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_poc )},
    {J2K_MS_SIZ, new marker_handler(J2K_MS_SIZ,J2K_DEC_STATE_MH_SIZ, j2k_read_siz )},
    {J2K_MS_CAP, new marker_handler(J2K_MS_CAP,J2K_DEC_STATE_MH,j2k_read_cap )},
    {J2K_MS_TLM, new marker_handler(J2K_MS_TLM,J2K_DEC_STATE_MH, j2k_read_tlm )},
    {J2K_MS_PLM, new marker_handler(J2K_MS_PLM,J2K_DEC_STATE_MH, j2k_read_plm )},
    {J2K_MS_PLT, new marker_handler(J2K_MS_PLT,J2K_DEC_STATE_TPH, j2k_read_plt )},
    {J2K_MS_PPM, new marker_handler(J2K_MS_PPM,J2K_DEC_STATE_MH,j2k_read_ppm )},
    {J2K_MS_PPT, new marker_handler(J2K_MS_PPT,J2K_DEC_STATE_TPH, j2k_read_ppt )},
    {J2K_MS_SOP, new marker_handler(J2K_MS_SOP,0, nullptr )},
    {J2K_MS_CRG, new marker_handler(J2K_MS_CRG,J2K_DEC_STATE_MH, j2k_read_crg )},
    {J2K_MS_COM, new marker_handler(J2K_MS_COM,J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_com )},
    {J2K_MS_MCT, new marker_handler(J2K_MS_MCT,J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_mct )},
    {J2K_MS_CBD, new marker_handler(J2K_MS_CBD,J2K_DEC_STATE_MH, j2k_read_cbd )},
    {J2K_MS_MCC, new marker_handler(J2K_MS_MCC,J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_mcc )},
    {J2K_MS_MCO, new marker_handler(J2K_MS_MCO,J2K_DEC_STATE_MH | J2K_DEC_STATE_TPH, j2k_read_mco )}
    };
}
CodeStream::~CodeStream(){
	delete m_decompressor.m_default_tcp;
	m_cp.destroy();
	j2k_destroy_cstr_index(cstr_index);
	delete m_headerImage;
	delete m_output_image;
	delete[] m_marker_scratch;
	delete m_tileCache;
	for (auto &val : marker_map)
		delete val.second;
}

bool CodeStream::exec(std::vector<j2k_procedure> &procs) {
    bool result = std::all_of(procs.begin(), procs.end(),[&](const j2k_procedure &proc){
    	return proc(this);
    });
	procs.clear();

	return result;
}

BufferedStream* CodeStream::getStream(){
	return m_stream;
}

GrkImage* CodeStream::getHeaderImage(void){
	return m_headerImage;
}

int32_t CodeStream::tileIndexToDecode(){
	return m_tile_ind_to_dec;
}

GrkImage* CodeStream::getCompositeImage(){
	return m_tileCache->getComposite();
}

TileProcessor* CodeStream::allocateProcessor(uint16_t tile_index){
	auto tileCache = m_tileCache->get(tile_index);
	auto tileProcessor = tileCache ? tileCache->processor : nullptr;
	if (!tileProcessor){
		tileProcessor = new TileProcessor(this,m_stream);
		tileProcessor->m_tile_index = tile_index;
		if (!m_output_image) {
			m_output_image = new GrkImage();
			getCompositeImage()->copyHeader(m_output_image);
		}
		m_tileCache->put(tile_index, tileProcessor);
	}
	m_tileProcessor = tileProcessor;


	return m_tileProcessor;
}
TileProcessor* CodeStream::currentProcessor(void){
	return m_tileProcessor;
}

GrkImage* CodeStream::get_image(uint16_t tileIndex){
	auto entry = m_tileCache->get(tileIndex);
	return entry ? entry->image : nullptr;
}

GrkImage* CodeStream::get_image(){
	return getCompositeImage();
}

std::vector<GrkImage*> CodeStream::getAllImages(void){
	return m_tileCache->getAllImages();
}

/** Main header reading function handler */
bool CodeStream::read_header(grk_header_info  *header_info){
	if (m_headerError)
		return false;

	if (!m_headerImage) {
		m_headerImage = new GrkImage();

		/* customization of the validation */
		m_validation_list.push_back((j2k_procedure) j2k_decompress_validation);

		/* validation of the parameters codec */
		if (!exec(m_validation_list)){
			m_headerError = true;
			return false;
		}

		m_procedure_list.push_back(	(j2k_procedure) j2k_read_header_procedure);
		// custom procedures here
		m_procedure_list.push_back(	(j2k_procedure) j2k_copy_default_tcp);

		/* read header */
		if (!exec(m_procedure_list)){
			m_headerError = true;
			return false;
		}

		/* Copy code stream image information to composite image */
		m_headerImage->copyHeader(getCompositeImage());
	}

	if (header_info) {
		CodingParams *cp = nullptr;
		TileCodingParams *tcp = nullptr;
		TileComponentCodingParams *tccp = nullptr;

		cp = &(m_cp);
		tcp = m_decompressor.m_default_tcp;
		tccp = &tcp->tccps[0];

		header_info->cblockw_init = 1 << tccp->cblkw;
		header_info->cblockh_init = 1 << tccp->cblkh;
		header_info->irreversible = tccp->qmfbid == 0;
		header_info->mct = tcp->mct;
		header_info->rsiz = cp->rsiz;
		header_info->numresolutions = tccp->numresolutions;
		// !!! assume that coding style is constant across all tile components
		header_info->csty = tccp->csty;
		// !!! assume that mode switch is constant across all tiles
		header_info->cblk_sty = tccp->cblk_sty;
		for (uint32_t i = 0; i < header_info->numresolutions; ++i) {
			header_info->prcw_init[i] = 1 << tccp->prcw_exp[i];
			header_info->prch_init[i] = 1 << tccp->prch_exp[i];
		}
		header_info->tx0 = m_cp.tx0;
		header_info->ty0 = m_cp.ty0;

		header_info->t_width = m_cp.t_width;
		header_info->t_height = m_cp.t_height;

		header_info->t_grid_width = m_cp.t_grid_width;
		header_info->t_grid_height = m_cp.t_grid_height;

		header_info->tcp_numlayers = tcp->numlayers;

		header_info->num_comments = m_cp.num_comments;
		for (size_t i = 0; i < header_info->num_comments; ++i) {
			header_info->comment[i] = m_cp.comment[i];
			header_info->comment_len[i] = m_cp.comment_len[i];
			header_info->isBinaryComment[i] = m_cp.isBinaryComment[i];
		}
	}
	return true;
}


bool CodeStream::set_decompress_window(grk_rect_u32 window) {
	auto cp = &(m_cp);
	auto image = m_headerImage;
	auto compositeImage = getCompositeImage();
	auto decompressor = &m_decompressor;

	/* Check if we have read the main header */
	if (decompressor->getState() != J2K_DEC_STATE_TPH_SOT) {
		GRK_ERROR("Need to decompress the main header before setting decompress window");
		return false;
	}

	if (window == grk_rect_u32(0,0,0,0)) {
		decompressor->m_start_tile_x_index = 0;
		decompressor->m_start_tile_y_index = 0;
		decompressor->m_end_tile_x_index = cp->t_grid_width;
		decompressor->m_end_tile_y_index = cp->t_grid_height;
		return true;
	}

	/* Check if the window provided by the user are correct */

	uint32_t start_x = window.x0 + image->x0;
	uint32_t start_y = window.y0 + image->y0;
	uint32_t end_x 	 = window.x1 + image->x0;
	uint32_t end_y   = window.y1 + image->y0;
	/* Left */
	if (start_x > image->x1) {
		GRK_ERROR("Left position of the decompress window (%u)"
				" is outside of the image area (Xsiz=%u).", start_x, image->x1);
		return false;
	} else {
		decompressor->m_start_tile_x_index = (start_x - cp->tx0) / cp->t_width;
		compositeImage->x0 = start_x;
	}

	/* Up */
	if (start_y > image->y1) {
		GRK_ERROR("Top position of the decompress window (%u)"
				" is outside of the image area (Ysiz=%u).", start_y, image->y1);
		return false;
	} else {
		decompressor->m_start_tile_y_index = (start_y - cp->ty0) / cp->t_height;
		compositeImage->y0 = start_y;
	}

	/* Right */
	assert(end_x > 0);
	assert(end_y > 0);
	if (end_x > image->x1) {
		GRK_WARN("Right position of the decompress window (%u)"
				" is outside the image area (Xsiz=%u).", end_x, image->x1);
		decompressor->m_end_tile_x_index = cp->t_grid_width;
		compositeImage->x1 = image->x1;
	} else {
		// avoid divide by zero
		if (cp->t_width == 0)
			return false;
		decompressor->m_end_tile_x_index = ceildiv<uint32_t>(end_x - cp->tx0,
				cp->t_width);
		compositeImage->x1 = end_x;
	}

	/* Bottom */
	if (end_y > image->y1) {
		GRK_WARN("Bottom position of the decompress window (%u)"
				" is outside of the image area (Ysiz=%u).", end_y, image->y1);
		decompressor->m_end_tile_y_index = cp->t_grid_height;
		compositeImage->y1 = image->y1;
	} else {
		// avoid divide by zero
		if (cp->t_height == 0)
			return false;
		decompressor->m_end_tile_y_index = ceildiv<uint32_t>(end_y - cp->ty0,
				cp->t_height);
		compositeImage->y1 = end_y;
	}
	wholeTileDecompress = false;
	if (!compositeImage->subsampleAndReduce(cp->m_coding_params.m_dec.m_reduce))
		return false;

	GRK_INFO("Decompress window set to (%d,%d,%d,%d)", window.x0,window.y0,window.x1,window.y1);

	return true;
}

bool CodeStream::read_header_procedure(void) {
	bool has_siz = false;
	bool has_cod = false;
	bool has_qcd = false;

	/*  We enter in the main header */
	m_decompressor.setState(J2K_DEC_STATE_MH_SOC);

	/* Try to read the SOC marker, the code stream must begin with SOC marker */
	if (!j2k_read_soc(this)) {
		GRK_ERROR("Code stream must begin with SOC marker ");
		return false;
	}
	// read next marker
	if (!read_marker())
		return false;

	if (m_curr_marker != J2K_MS_SIZ){
		GRK_ERROR("Code-stream must contain a valid SIZ marker segment, immediately after the SOC marker ");
		return false;
	}

	/* Try to read until the SOT is detected */
	while (m_curr_marker != J2K_MS_SOT) {

		/* Get the marker handler from the marker ID */
		auto marker_handler = get_marker_handler(m_curr_marker);

		/* Manage case where marker is unknown */
		if (!marker_handler) {
			if (!read_unk(&m_curr_marker)) {
				GRK_ERROR("Unable to read unknown marker 0x%02x.",
						m_curr_marker);
				return false;
			}
			continue;
		}

		if (marker_handler->id == J2K_MS_SIZ)
			has_siz = true;
		else if (marker_handler->id == J2K_MS_COD)
			has_cod = true;
		else if (marker_handler->id == J2K_MS_QCD)
			has_qcd = true;

		/* Check if the marker is known and if it is in the correct location (main, tile, end of code stream)*/
		if (!(m_decompressor.getState() & marker_handler->states)) {
			GRK_ERROR("Marker %d is not compliant with its position",m_curr_marker);
			return false;
		}

		uint16_t marker_size;
		if (!read_short(&marker_size))
			return false;
		/* Check marker size (does not include marker ID but includes marker size) */
		else if (marker_size < 2) {
			GRK_ERROR("Inconsistent marker size");
			return false;
		}
		else if (marker_size == 2) {
			GRK_ERROR("Zero-size marker in header.");
			return false;
		}
		marker_size = (uint16_t)(marker_size - 2); /* Subtract the size of the marker ID already read */

		if (!process_marker(marker_handler, marker_size))
			return false;

		if (cstr_index) {
			/* Add the marker to the code stream index*/
			if (!j2k_add_mhmarker(cstr_index, marker_handler->id,
					m_stream->tell() - marker_size - 4, marker_size + 4)) {
				GRK_ERROR("Not enough memory to add mh marker");
				return false;
			}
		}
		// read next marker
		if (!read_marker())
			return false;
	}
	if (!has_siz) {
		GRK_ERROR("required SIZ marker not found in main header");
		return false;
	} else if (!has_cod) {
		GRK_ERROR("required COD marker not found in main header");
		return false;
	} else if (!has_qcd) {
		GRK_ERROR("required QCD marker not found in main header");
		return false;
	}
	if (!j2k_merge_ppm(&(m_cp))) {
		GRK_ERROR("Failed to merge PPM data");
		return false;
	}
	/* Position of the last element if the main header */
	if (cstr_index)
		cstr_index->main_head_end = (uint32_t) m_stream->tell() - 2;
	/* Next step: read a tile-part header */
	m_decompressor.setState(J2K_DEC_STATE_TPH_SOT);

	return true;
}

/** Set up decompressor function handler */
void CodeStream::init_decompress(grk_dparameters  *parameters){
	if (parameters) {
		m_cp.m_coding_params.m_dec.m_layer = parameters->cp_layer;
		m_cp.m_coding_params.m_dec.m_reduce = parameters->cp_reduce;
		m_tileCache->setStrategy(parameters->tileCacheStrategy);
	}
}
bool CodeStream::decompress( grk_plugin_tile *tile){
	/* customization of the decoding */
	m_procedure_list.push_back((j2k_procedure) j2k_decompress_tiles);
	current_plugin_tile = tile;

	return exec_decompress();
}
bool CodeStream::exec_decompress(void){

	/* Decompress the code stream */
	if (!exec(m_procedure_list))
		return false;

	/* Move data from output image to composite image*/
	m_output_image->transferDataTo(getCompositeImage());

	return true;
}


bool CodeStream::decompress_tiles(void) {
	bool go_on = true;
	uint16_t num_tiles_to_decompress = (uint16_t)(m_cp.t_grid_height* m_cp.t_grid_width);
	m_multiTile = num_tiles_to_decompress > 1;
	std::atomic<bool> success(true);
	std::atomic<uint32_t> num_tiles_decoded(0);
	ThreadPool pool(std::min<uint32_t>((uint32_t)ThreadPool::get()->num_threads(), num_tiles_to_decompress));
	std::vector< std::future<int> > results;

	if (cstr_index) {
		/*Allocate and initialize some elements of codestream index*/
		if (!j2k_allocate_tile_element_cstr_index(this)) {
			m_headerError = true;
			return false;
		}
	}

	// parse header and perform T2 followed by asynch T1
	for (uint16_t tileno = 0; tileno < num_tiles_to_decompress; tileno++) {
		//1. read header
		try {
			if (!parse_tile_header_markers(&go_on)){
				success = false;
				goto cleanup;
			}
		} catch (InvalidMarkerException &ime){
			GRK_ERROR("Found invalid marker : 0x%x", ime.m_marker);
			success = false;
			goto cleanup;
		}
		if (!go_on)
			break;
		if (!m_tileProcessor){
			GRK_ERROR("Missing SOT marker in tile %d", tileno);
			success = false;
			goto cleanup;
		}
		//2. T2 decompress
		auto processor = m_tileProcessor;
		m_tileProcessor = nullptr;
		bool breakAfterT1 = false;
		try {
			if (!decompress_tile_t2(processor)){
					GRK_ERROR("Failed to decompress tile %u/%u",
							processor->m_tile_index + 1,
							num_tiles_to_decompress);
					success = false;
					goto cleanup;
			}
		}  catch (DecodeUnknownMarkerAtEndOfTileException &e){
			breakAfterT1 = true;
		}
		// once we schedule a processor for T1 compression, we will destroy it
		// regardless of success or not
		if (pool.num_threads() > 1) {
			results.emplace_back(
				pool.enqueue([this,processor,
							  num_tiles_to_decompress,
							  &num_tiles_decoded, &success] {
					if (success) {
						if (!decompress_tile_t2t1(processor)){
							GRK_ERROR("Failed to decompress tile %u/%u",
									processor->m_tile_index + 1,num_tiles_to_decompress);
							success = false;
						} else {
							num_tiles_decoded++;
						}
					}
					return 0;
				})
			);
		} else {
			if (!decompress_tile_t2t1(processor)){
					GRK_ERROR("Failed to decompress tile %u/%u",
							processor->m_tile_index + 1,num_tiles_to_decompress);
					success = false;
			} else {
				num_tiles_decoded++;
			}
			if (!success)
				goto cleanup;
		}
		if (breakAfterT1)
			break;
		if (m_stream->get_number_byte_left() == 0|| m_decompressor.getState() == J2K_DEC_STATE_NO_EOC)
			break;
	}
	for(auto &result: results){
		result.get();
	}
	results.clear();
	if (m_multiTile && m_output_image) {
		if (!m_output_image->allocData()){
			success = false;
			goto cleanup;
		}
	}
	for (uint16_t tileno = 0; tileno < num_tiles_to_decompress; tileno++) {
		auto entry = m_tileCache->get(tileno);
		if (!entry)
			continue;
		auto img = entry->image;
		if (!img)
			continue;
		if (!m_output_image->compositeFrom(img))
			return false;
	}

	// check if there is another tile that has not been processed
	// we will reject if it has the TPSot problem (https://github.com/uclouvain/openjpeg/issues/254)
	if (m_curr_marker == J2K_MS_SOT && m_stream->get_number_byte_left()){
		uint16_t marker_size;
		if (!read_short(&marker_size)){
			success = false;
			goto cleanup;
		}
		marker_size = (uint16_t)(marker_size - 2); /* Subtract the size of the marker ID already read */
		auto marker_handler = get_marker_handler(m_curr_marker);
		if (!(m_decompressor.getState() & marker_handler->states)) {
			GRK_ERROR("Marker %d is not compliant with its position",m_curr_marker);
			success = false;
			goto cleanup;
		}
		if (!process_marker(marker_handler, marker_size)){
			success = false;
			goto cleanup;
		}
	}
	// sanity checks
	if (num_tiles_decoded == 0) {
		GRK_ERROR("No tiles were decompressed.");
		success = false;
		goto cleanup;
	} else if (num_tiles_decoded < num_tiles_to_decompress) {
		uint32_t decompressed = num_tiles_decoded;
		GRK_WARN("Only %u out of %u tiles were decompressed", decompressed,
				num_tiles_to_decompress);
	}
cleanup:
	for(auto &result: results){
		result.get();
	}
	return success;
}


bool CodeStream::decompress_tile(uint16_t tile_index){
	auto entry = m_tileCache->get(tile_index);
	if (entry)
		return true;

	// another tile has already been decoded
	if (m_output_image){
		/* Copy code stream image information to composite image */
		m_headerImage->copyHeader(getCompositeImage());
	}

	if (cstr_index) {
		/*Allocate and initialize some elements of codestream index*/
		if (!j2k_allocate_tile_element_cstr_index(this)) {
			m_headerError = true;
			return false;
		}
	}

	auto compositeImage = getCompositeImage();
	if (tile_index >= m_cp.t_grid_width * m_cp.t_grid_height) {
		GRK_ERROR(	"Tile index %u is greater than maximum tile index %u",	tile_index,
				      m_cp.t_grid_width * m_cp.t_grid_height - 1);
		return false;
	}

	/* Compute the dimension of the desired tile*/
	uint32_t tile_x = tile_index % m_cp.t_grid_width;
	uint32_t tile_y = tile_index / m_cp.t_grid_width;

	auto imageBounds = grk_rect_u32(compositeImage->x0,
									compositeImage->y0,
									compositeImage->x1,
									compositeImage->y1);
	auto tileBounds = m_cp.getTileBounds(compositeImage, tile_x, tile_y);
	// crop tile bounds with image bounds
	auto croppedImageBounds = imageBounds.intersection(tileBounds);
	if (imageBounds.non_empty() && tileBounds.non_empty() && croppedImageBounds.non_empty()) {
		compositeImage->x0 = (uint32_t) croppedImageBounds.x0;
		compositeImage->y0 = (uint32_t) croppedImageBounds.y0;
		compositeImage->x1 = (uint32_t) croppedImageBounds.x1;
		compositeImage->y1 = (uint32_t) croppedImageBounds.y1;
	} else {
		GRK_WARN("Decompress bounds <%u,%u,%u,%u> do not overlap with requested tile %u. Decompressing full image",
				imageBounds.x0,
				imageBounds.y0,
				imageBounds.x1,
				imageBounds.y1,
				tile_index);
		croppedImageBounds = imageBounds;
	}

	auto reduce = m_cp.m_coding_params.m_dec.m_reduce;
	for (uint32_t compno = 0; compno < compositeImage->numcomps; ++compno) {
		auto comp = compositeImage->comps + compno;
		auto compBounds = croppedImageBounds.rectceildiv(comp->dx,comp->dy);
		auto reducedCompBounds = compBounds.rectceildivpow2(reduce);
		comp->x0 = reducedCompBounds.x0;
		comp->y0 = reducedCompBounds.y0;
		comp->w = reducedCompBounds.width();
		comp->h = reducedCompBounds.height();
	}
	m_tile_ind_to_dec = (int32_t) tile_index;

	// reset tile part numbers, in case we are re-using the same codec object
	// from previous decompress
	uint32_t nb_tiles = m_cp.t_grid_width * m_cp.t_grid_height;
	for (uint32_t i = 0; i < nb_tiles; ++i)
		m_cp.tcps[i].m_tile_part_index = -1;

	/* customization of the decoding */
	m_procedure_list.push_back((j2k_procedure) j2k_decompress_tile);

	return exec_decompress();
}

/*
 * Read and decompress one tile.
 */
bool CodeStream::decompress_tile() {
	m_multiTile = false;
	if (tileIndexToDecode() == -1) {
		GRK_ERROR("j2k_decompress_tile: Unable to decompress tile "
				"since first tile SOT has not been detected");
		return false;
	}
	auto tileCache = m_tileCache->get((uint16_t)tileIndexToDecode());
	auto tileProcessor = tileCache ? tileCache->processor : nullptr;
	bool rc = false;
	if (!tileProcessor) {
		/*Allocate and initialize some elements of code stream index if not already done*/
		if (!cstr_index->tile_index) {
			if (!j2k_allocate_tile_element_cstr_index(this))
				return false;
		}

		/* Move into the code stream to the first SOT used to decompress the desired tile */
		uint16_t tile_index_to_decompress =	(uint16_t) (tileIndexToDecode());
		if (cstr_index->tile_index && cstr_index->tile_index->tp_index) {
			if (!cstr_index->tile_index[tile_index_to_decompress].nb_tps) {
				/* the index for this tile has not been built,
				 *  so move to the last SOT read */
				if (!(m_stream->seek(m_decompressor.m_last_sot_read_pos	+ 2))) {
					GRK_ERROR("Problem with seek function");
					return false;
				}
			} else {
				if (!(m_stream->seek(cstr_index->tile_index[tile_index_to_decompress].tp_index[0].start_pos	+ 2))) {
					GRK_ERROR("Problem with seek function");
					return false;
				}
			}
			/* Special case if we have previously read the EOC marker
			 * (if the previous tile decompressed is the last ) */
			if (m_decompressor.getState() == J2K_DEC_STATE_EOC)
				m_decompressor.setState(J2K_DEC_STATE_TPH_SOT);
		}

		// if we have a TLM marker, then we can skip tiles until
		// we get to desired tile
		if (m_cp.tlm_markers){
			m_cp.tlm_markers->getInit();
			auto tl = m_cp.tlm_markers->getNext();
			//GRK_INFO("TLM : index: %u, length : %u", tl.tile_number, tl.length);
			uint16_t tileNumber = 0;
			while (m_stream->get_number_byte_left() != 0 &&	tileNumber != tileIndexToDecode()){
				if (tl.length == 0){
					GRK_ERROR("j2k_decompress_tile: corrupt TLM marker");
					return false;
				}
				m_stream->skip(tl.length);
				tl = m_cp.tlm_markers->getNext();
				tileNumber = tl.has_tile_number ? tl.tile_number : tileNumber+1;
			}
		}
		bool go_on = true;
		try {
			if (!parse_tile_header_markers(&go_on))
				goto cleanup;
		} catch (InvalidMarkerException &ime){
			GRK_ERROR("Found invalid marker : 0x%x", ime.m_marker);
			goto cleanup;
		}

		tileProcessor = m_tileProcessor;
		if (!decompress_tile_t2t1(tileProcessor))
			goto cleanup;
	} else {

	}

	rc = true;
cleanup:
	return rc;
}

bool CodeStream::decompress_tile_t2t1(TileProcessor *tileProcessor) {
	auto decompressor = &m_decompressor;
	uint16_t tile_index = tileProcessor->m_tile_index;
	auto tcp = m_cp.tcps + tile_index;
	if (!tcp->m_tile_data) {
		tcp->destroy();
		return false;
	}
	if (!tileProcessor->decompress_tile_t2(tcp->m_tile_data)) {
		tcp->destroy();
		decompressor->orState(J2K_DEC_STATE_ERR);
		return false;
	}
	if (tileProcessor->m_corrupt_packet){
		GRK_WARN("Tile %d was not decompressed", tileProcessor->m_tile_index+1);
		return true;
	}
	bool rc = true;
	bool doPost = !tileProcessor->current_plugin_tile ||
			(tileProcessor->current_plugin_tile->decompress_flags & GRK_DECODE_POST_T1);
	if (!tileProcessor->decompress_tile_t1()) {
		tcp->destroy();
		decompressor->orState(J2K_DEC_STATE_ERR);
		return false;
	}
	if (doPost) {
		auto tile = tileProcessor->tile;
		/* copy/transfer data from tile component to output image */
		m_tileCache->put(tile_index, m_output_image, tile);
		if (m_multiTile) {
			for (uint16_t compno = 0; compno < tile->numcomps; ++compno)
				(tile->comps + compno)->release_mem(true);
		} else {
			m_output_image->transferDataFrom(tile);
		}
	}

	return rc;
}
bool CodeStream::decompress_tile_t2(TileProcessor *tileProcessor) {
	auto decompressor = &m_decompressor;

	if (!(decompressor->getState() & J2K_DEC_STATE_DATA)){
	   GRK_ERROR("j2k_decompress_tile: no data.");
	   return false;
	}
	auto tcp = m_cp.tcps + tileProcessor->m_tile_index;
	if (!tcp->m_tile_data) {
		GRK_ERROR("Missing SOD marker");
		tcp->destroy();
		return false;
	}
	// find next tile
	bool rc = true;
	bool doPost = !tileProcessor->current_plugin_tile
			|| (tileProcessor->current_plugin_tile->decompress_flags
					& GRK_DECODE_POST_T1);
	if (doPost){
		rc =  decompressor->findNextTile(this);
	}
	return rc;
}
bool CodeStream::decompress_validation(void) {
	bool is_valid = true;
	is_valid &=	(m_decompressor.getState() == J2K_DEC_STATE_NONE);

	return is_valid;
}

void CodeStream::dump(uint32_t flag, FILE *out_stream){
	j2k_dump(this, flag, out_stream);
}

grk_codestream_info_v2* CodeStream::get_cstr_info(void){
	return j2k_get_cstr_info(this);
}

grk_codestream_index* CodeStream::get_cstr_index(void){
	return j2k_get_cstr_index(this);
}

bool CodeStream::process_marker(const marker_handler* marker_handler, uint16_t marker_size){
	if (!m_marker_scratch) {
		m_marker_scratch = new uint8_t[default_header_size];
		m_marker_scratch_size = default_header_size;
	}
	if (marker_size > m_marker_scratch_size) {
		if (marker_size > m_stream->get_number_byte_left()) {
			GRK_ERROR("Marker size inconsistent with stream length");
			return false;
		}
		delete[] m_marker_scratch;
		m_marker_scratch = new uint8_t[2 * marker_size];
		m_marker_scratch_size = 2 * marker_size;
	}
	if (m_stream->read(m_marker_scratch, marker_size) != marker_size) {
		GRK_ERROR("Stream too short");
		return false;
	}

	return (*(marker_handler->callback))(this,	m_marker_scratch, marker_size);
}

bool CodeStream::isDecodingTilePartHeader() {
	return (m_decompressor.getState() & J2K_DEC_STATE_TPH);
}
TileCodingParams* CodeStream::get_current_decode_tcp() {
    auto tileProcessor = m_tileProcessor;

	return (isDecodingTilePartHeader()) ?
			m_cp.tcps + tileProcessor->m_tile_index :
			m_decompressor.m_default_tcp;
}

bool CodeStream::read_short(uint16_t *val){
	uint8_t temp[2];
	if (m_stream->read(temp, 2) != 2)
		return false;

	grk_read<uint16_t>(temp, val);
	return true;
}

const marker_handler* CodeStream::get_marker_handler(	uint16_t id) {
	auto iter = marker_map.find(id);
	if (iter != marker_map.end())
		return iter->second;
	else {
		GRK_WARN("Unknown marker 0x%02x detected.", id);
		return nullptr;
	}
}

bool CodeStream::read_marker(){
	if (!read_short(&m_curr_marker))
		return false;

	/* Check if the current marker ID is valid */
	if (m_curr_marker < 0xff00) {
		GRK_WARN("marker ID 0x%.4x does not match JPEG 2000 marker format 0xffxx",
				m_curr_marker);
		throw InvalidMarkerException(m_curr_marker);
	}

	return true;
}
bool CodeStream::get_end_header(void) {
	cstr_index->main_head_end = m_stream->tell();

	return true;
}
bool CodeStream::mct_validation(void) {
	bool is_valid = true;
	if ((m_cp.rsiz & 0x8200) == 0x8200) {
		uint32_t nb_tiles = m_cp.t_grid_height
				* m_cp.t_grid_width;
		for (uint32_t i = 0; i < nb_tiles; ++i) {
			auto tcp = m_cp.tcps + i;
			if (tcp->mct == 2) {
				is_valid &= (tcp->m_mct_coding_matrix != nullptr);
				for (uint32_t j = 0; j < m_headerImage->numcomps; ++j) {
					auto tccp = tcp->tccps + j;
					is_valid &= !(tccp->qmfbid & 1);
				}
			}
		}
	}

	return is_valid;
}
bool CodeStream::read_unk(uint16_t *output_marker) {
	const marker_handler *marker_handler = nullptr;
	uint32_t size_unk = 2;
	while (true) {
		if (!read_marker())
			return false;
		/* Get the marker handler from the marker ID*/
		marker_handler = get_marker_handler(m_curr_marker);
		if (marker_handler == nullptr)	{
			size_unk += 2;
		} else {
			if (!(m_decompressor.getState() & marker_handler->states)) {
				GRK_ERROR("Marker %d is not compliant with its position",m_curr_marker);
				return false;
			} else {
				/* Add the marker to the code stream index*/
				if (cstr_index && marker_handler->id != J2K_MS_SOT) {
					bool res = j2k_add_mhmarker(cstr_index,
					J2K_MS_UNK, m_stream->tell() - size_unk, size_unk);
					if (res == false) {
						GRK_ERROR("Not enough memory to add mh marker");
						return false;
					}
				}
				break; /* next marker is known and located correctly  */
			}
		}
	}
	if (!marker_handler){
		GRK_ERROR("Unable to read unknown marker");
		return false;
	}
	*output_marker = marker_handler->id;

	return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////






/** Reading function used after code stream if necessary */
bool CodeStream::end_decompress(void){
	return true;
}

bool CodeStream::start_compress(void){
	/* customization of the validation */
	m_validation_list.push_back(
			(j2k_procedure) j2k_compress_validation);
	//custom validation here
	m_validation_list.push_back((j2k_procedure) j2k_mct_validation);

	/* validation of the parameters codec */
	if (!exec(m_validation_list))
		return false;

	/* customization of the compressing */
	if (!init_header_writing())
		return false;

	/* write header */
	return exec(m_procedure_list);
}

bool CodeStream::init_compress(grk_cparameters  *parameters,GrkImage *image){
	if (!parameters || !image)
		return false;

	//sanity check on image
	if (image->numcomps < 1 || image->numcomps > max_num_components) {
		GRK_ERROR(
				"Invalid number of components specified while setting up JP2 compressor");
		return false;
	}
	if ((image->x1 < image->x0) || (image->y1 < image->y0)) {
		GRK_ERROR(
				"Invalid input image dimensions found while setting up JP2 compressor");
		return false;
	}
	for (uint32_t i = 0; i < image->numcomps; ++i) {
		auto comp = image->comps + i;
		if (comp->w == 0 || comp->h == 0) {
			GRK_ERROR(
					"Invalid input image component dimensions found while setting up JP2 compressor");
			return false;
		}
		if (comp->prec == 0) {
			GRK_ERROR(
					"Invalid component precision of 0 found while setting up JP2 compressor");
			return false;
		}
	}

	// create private sanitized copy of image
	m_headerImage = new GrkImage();
	if (!image->copyHeader(m_headerImage)){
		GRK_ERROR("Failed to copy image header.");
		return false;
	}
	if (image->comps) {
		for (uint32_t compno = 0; compno < image->numcomps; compno++) {
			if (image->comps[compno].data) {
				m_headerImage->comps[compno].data =	image->comps[compno].data;
				image->comps[compno].data = nullptr;

			}
		}
	}

	if ((parameters->numresolution == 0)
			|| (parameters->numresolution > GRK_J2K_MAXRLVLS)) {
		GRK_ERROR("Invalid number of resolutions : %u not in range [1,%u]",
				parameters->numresolution, GRK_J2K_MAXRLVLS);
		return false;
	}

	if (GRK_IS_IMF(parameters->rsiz) && parameters->max_cs_size > 0
			&& parameters->tcp_numlayers == 1
			&& parameters->tcp_rates[0] == 0) {
		parameters->tcp_rates[0] = (float) (image->numcomps * image->comps[0].w
				* image->comps[0].h * image->comps[0].prec)
				/ (float) (((uint32_t) parameters->max_cs_size) * 8
						* image->comps[0].dx * image->comps[0].dy);
	}

	/* if no rate entered, lossless by default */
	if (parameters->tcp_numlayers == 0) {
		parameters->tcp_rates[0] = 0;
		parameters->tcp_numlayers = 1;
		parameters->cp_disto_alloc = true;
	}

	/* see if max_codestream_size does limit input rate */
	double image_bytes = ((double) image->numcomps * image->comps[0].w
			* image->comps[0].h * image->comps[0].prec)
			/ (8 * image->comps[0].dx * image->comps[0].dy);
	if (parameters->max_cs_size == 0) {
		if (parameters->tcp_numlayers > 0
				&& parameters->tcp_rates[parameters->tcp_numlayers - 1] > 0) {
			parameters->max_cs_size = (uint64_t) floor(
					image_bytes
							/ parameters->tcp_rates[parameters->tcp_numlayers
									- 1]);
		}
	} else {
		bool cap = false;
		auto min_rate = image_bytes / (double) parameters->max_cs_size;
		for (uint32_t i = 0; i < parameters->tcp_numlayers; i++) {
			if (parameters->tcp_rates[i] < min_rate) {
				parameters->tcp_rates[i] = min_rate;
				cap = true;
			}
		}
		if (cap) {
			GRK_WARN("The desired maximum code stream size has limited\n"
					"at least one of the desired quality layers");
		}
	}

	/* Manage profiles and applications and set RSIZ */
	/* set cinema parameters if required */
	if (parameters->isHT) {
		parameters->rsiz |= GRK_JPH_RSIZ_FLAG;
	}
	if (GRK_IS_CINEMA(parameters->rsiz)) {
		if ((parameters->rsiz == GRK_PROFILE_CINEMA_S2K)
				|| (parameters->rsiz == GRK_PROFILE_CINEMA_S4K)) {
			GRK_WARN(
					"JPEG 2000 Scalable Digital Cinema profiles not supported");
			parameters->rsiz = GRK_PROFILE_NONE;
		} else {
			if (Profile::is_cinema_compliant(image, parameters->rsiz))
				Profile::set_cinema_parameters(parameters, image);
			else
				parameters->rsiz = GRK_PROFILE_NONE;
		}
	} else if (GRK_IS_STORAGE(parameters->rsiz)) {
		GRK_WARN("JPEG 2000 Long Term Storage profile not supported");
		parameters->rsiz = GRK_PROFILE_NONE;
	} else if (GRK_IS_BROADCAST(parameters->rsiz)) {
		Profile::set_broadcast_parameters(parameters);
		if (!Profile::is_broadcast_compliant(parameters, image))
			parameters->rsiz = GRK_PROFILE_NONE;
	} else if (GRK_IS_IMF(parameters->rsiz)) {
		Profile::set_imf_parameters(parameters, image);
		if (!Profile::is_imf_compliant(parameters, image))
			parameters->rsiz = GRK_PROFILE_NONE;
	} else if (GRK_IS_PART2(parameters->rsiz)) {
		if (parameters->rsiz == ((GRK_PROFILE_PART2) | (GRK_EXTENSION_NONE))) {
			GRK_WARN("JPEG 2000 Part-2 profile defined\n"
					"but no Part-2 extension enabled.\n"
					"Profile set to NONE.");
			parameters->rsiz = GRK_PROFILE_NONE;
		} else if (parameters->rsiz
				!= ((GRK_PROFILE_PART2) | (GRK_EXTENSION_MCT))) {
			GRK_WARN("Unsupported Part-2 extension enabled\n"
					"Profile set to NONE.");
			parameters->rsiz = GRK_PROFILE_NONE;
		}
	}

	if (parameters->numpocs) {
		/* initialisation of POC */
		if (!j2k_check_poc_val(parameters->progression, parameters->numpocs,
				parameters->numresolution, image->numcomps,
				parameters->tcp_numlayers)) {
			GRK_ERROR("Failed to initialize POC");
			return false;
		}
	}

	/*
	 copy user compressing parameters
	 */

	/* keep a link to cp so that we can destroy it later in j2k_destroy_compress */
	auto cp = &(m_cp);

	/* set default values for cp */
	cp->t_grid_width = 1;
	cp->t_grid_height = 1;

	cp->m_coding_params.m_enc.m_max_comp_size = parameters->max_comp_size;
	cp->rsiz = parameters->rsiz;
	cp->m_coding_params.m_enc.m_disto_alloc = parameters->cp_disto_alloc;
	cp->m_coding_params.m_enc.m_fixed_quality = parameters->cp_fixed_quality;
	cp->m_coding_params.m_enc.writePLT = parameters->writePLT;
	cp->m_coding_params.m_enc.writeTLM = parameters->writeTLM;
	cp->m_coding_params.m_enc.rateControlAlgorithm =
			parameters->rateControlAlgorithm;

	/* tiles */
	cp->t_width = parameters->t_width;
	cp->t_height = parameters->t_height;

	/* tile offset */
	cp->tx0 = parameters->tx0;
	cp->ty0 = parameters->ty0;

	/* comment string */
	if (parameters->cp_num_comments) {
		for (size_t i = 0; i < parameters->cp_num_comments; ++i) {
			cp->comment_len[i] = parameters->cp_comment_len[i];
			if (!cp->comment_len[i]) {
				GRK_WARN("Empty comment. Ignoring");
				continue;
			}
			if (cp->comment_len[i] > GRK_MAX_COMMENT_LENGTH) {
				GRK_WARN(
						"Comment length %s is greater than maximum comment length %u. Ignoring",
						cp->comment_len[i], GRK_MAX_COMMENT_LENGTH);
				continue;
			}
			cp->comment[i] = (char*) new uint8_t[cp->comment_len[i]];
			if (!cp->comment[i]) {
				GRK_ERROR(
						"Not enough memory to allocate copy of comment string");
				return false;
			}
			memcpy(cp->comment[i], parameters->cp_comment[i],
					cp->comment_len[i]);
			cp->isBinaryComment[i] = parameters->cp_is_binary_comment[i];
			cp->num_comments++;
		}
	} else {
		/* Create default comment for code stream */
		const char comment[] = "Created by Grok     version ";
		const size_t clen = strlen(comment);
		const char *version = grk_version();

		cp->comment[0] = (char*) new uint8_t[clen + strlen(version) + 1];
		if (!cp->comment[0]) {
			GRK_ERROR("Not enough memory to allocate comment string");
			return false;
		}
		sprintf(cp->comment[0], "%s%s", comment, version);
		cp->comment_len[0] = (uint16_t) strlen(cp->comment[0]);
		cp->num_comments = 1;
		cp->isBinaryComment[0] = false;
	}

	/*
	 calculate other compressing parameters
	 */

	if (parameters->tile_size_on) {
		// avoid divide by zero
		if (cp->t_width == 0 || cp->t_height == 0) {
			GRK_ERROR("Invalid tile dimensions (%u,%u)",cp->t_width, cp->t_height);
			return false;
		}
		cp->t_grid_width  = ceildiv<uint32_t>((image->x1 - cp->tx0), cp->t_width);
		cp->t_grid_height = ceildiv<uint32_t>((image->y1 - cp->ty0), cp->t_height);
	} else {
		cp->t_width  = image->x1 - cp->tx0;
		cp->t_height = image->y1 - cp->ty0;
	}

	if (parameters->tp_on) {
		cp->m_coding_params.m_enc.m_tp_flag = parameters->tp_flag;
		cp->m_coding_params.m_enc.m_tp_on = true;
	}

	uint8_t numgbits = parameters->isHT ? 1 : 2;
	cp->tcps = new TileCodingParams[cp->t_grid_width * cp->t_grid_height];
	for (uint32_t tileno = 0; tileno < cp->t_grid_width * cp->t_grid_height; tileno++) {
		TileCodingParams *tcp = cp->tcps + tileno;
		tcp->setIsHT(parameters->isHT);
		tcp->qcd.generate(numgbits, (uint32_t) (parameters->numresolution - 1),
				!parameters->irreversible, image->comps[0].prec, tcp->mct > 0,
				image->comps[0].sgnd);
		tcp->numlayers = parameters->tcp_numlayers;

		for (uint16_t j = 0; j < tcp->numlayers; j++) {
			if (cp->m_coding_params.m_enc.m_fixed_quality)
				tcp->distoratio[j] = parameters->tcp_distoratio[j];
			else
				tcp->rates[j] = parameters->tcp_rates[j];
		}

		tcp->csty = parameters->csty;
		tcp->prg = parameters->prog_order;
		tcp->mct = parameters->tcp_mct;
		tcp->POC = false;

		if (parameters->numpocs) {
			/* initialisation of POC */
			tcp->POC = true;
			uint32_t numpocs_tile = 0;
			for (uint32_t i = 0; i < parameters->numpocs; i++) {
				if (tileno + 1 == parameters->progression[i].tileno) {
					auto tcp_poc = &tcp->progression[numpocs_tile];

					tcp_poc->resS = parameters->progression[numpocs_tile].resS;
					tcp_poc->compS = parameters->progression[numpocs_tile].compS;
					tcp_poc->layE = parameters->progression[numpocs_tile].layE;
					tcp_poc->resE = parameters->progression[numpocs_tile].resE;
					tcp_poc->compE = parameters->progression[numpocs_tile].compE;
					tcp_poc->prg1 = parameters->progression[numpocs_tile].prg1;
					tcp_poc->tileno = parameters->progression[numpocs_tile].tileno;
					numpocs_tile++;
				}
			}
			if (numpocs_tile == 0) {
				GRK_ERROR("Problem with specified progression order changes");
				return false;
			}
			tcp->numpocs = numpocs_tile - 1;
		} else {
			tcp->numpocs = 0;
		}

		tcp->tccps = new TileComponentCodingParams[image->numcomps];
		if (parameters->mct_data) {

			uint32_t lMctSize = image->numcomps * image->numcomps
					* (uint32_t) sizeof(float);
			auto lTmpBuf = (float*) grk_malloc(lMctSize);
			auto dc_shift = (int32_t*) ((uint8_t*) parameters->mct_data
					+ lMctSize);

			if (!lTmpBuf) {
				GRK_ERROR("Not enough memory to allocate temp buffer");
				return false;
			}

			tcp->mct = 2;
			tcp->m_mct_coding_matrix = (float*) grk_malloc(lMctSize);
			if (!tcp->m_mct_coding_matrix) {
				grk_free(lTmpBuf);
				lTmpBuf = nullptr;
				GRK_ERROR(
						"Not enough memory to allocate compressor MCT coding matrix ");
				return false;
			}
			memcpy(tcp->m_mct_coding_matrix, parameters->mct_data, lMctSize);
			memcpy(lTmpBuf, parameters->mct_data, lMctSize);

			tcp->m_mct_decoding_matrix = (float*) grk_malloc(lMctSize);
			if (!tcp->m_mct_decoding_matrix) {
				grk_free(lTmpBuf);
				lTmpBuf = nullptr;
				GRK_ERROR(
						"Not enough memory to allocate compressor MCT decoding matrix ");
				return false;
			}
			if (matrix_inversion_f(lTmpBuf, (tcp->m_mct_decoding_matrix),
					image->numcomps) == false) {
				grk_free(lTmpBuf);
				lTmpBuf = nullptr;
				GRK_ERROR("Failed to inverse compressor MCT decoding matrix ");
				return false;
			}

			tcp->mct_norms = (double*) grk_malloc(
					image->numcomps * sizeof(double));
			if (!tcp->mct_norms) {
				grk_free(lTmpBuf);
				lTmpBuf = nullptr;
				GRK_ERROR("Not enough memory to allocate compressor MCT norms ");
				return false;
			}
			mct::calculate_norms(tcp->mct_norms, image->numcomps,
					tcp->m_mct_decoding_matrix);
			grk_free(lTmpBuf);

			for (uint32_t i = 0; i < image->numcomps; i++) {
				auto tccp = &tcp->tccps[i];
				tccp->m_dc_level_shift = dc_shift[i];
			}

			if (j2k_init_mct_encoding(tcp, image) == false) {
				/* free will be handled by j2k_destroy */
				GRK_ERROR("Failed to set up j2k mct compressing");
				return false;
			}
		} else {
			if (tcp->mct == 1) {
				if (image->color_space == GRK_CLRSPC_EYCC
						|| image->color_space == GRK_CLRSPC_SYCC) {
					GRK_WARN("Disabling MCT for sYCC/eYCC colour space");
					tcp->mct = 0;
				} else if (image->numcomps >= 3) {
					if ((image->comps[0].dx != image->comps[1].dx)
							|| (image->comps[0].dx != image->comps[2].dx)
							|| (image->comps[0].dy != image->comps[1].dy)
							|| (image->comps[0].dy != image->comps[2].dy)) {
						GRK_WARN(
								"Cannot perform MCT on components with different dimensions. Disabling MCT.");
						tcp->mct = 0;
					}
				}
			}
			for (uint32_t i = 0; i < image->numcomps; i++) {
				auto tccp = tcp->tccps + i;
				auto comp = image->comps + i;
				if (!comp->sgnd) {
					tccp->m_dc_level_shift = 1 << (comp->prec - 1);
				}
			}
		}

		for (uint32_t i = 0; i < image->numcomps; i++) {
			auto tccp = &tcp->tccps[i];

			/* 0 => one precinct || 1 => custom precinct  */
			tccp->csty = parameters->csty & J2K_CP_CSTY_PRT;
			tccp->numresolutions = parameters->numresolution;
			tccp->cblkw = (uint8_t)floorlog2<uint32_t>(parameters->cblockw_init);
			tccp->cblkh = (uint8_t)floorlog2<uint32_t>(parameters->cblockh_init);
			tccp->cblk_sty = parameters->cblk_sty;
			tccp->qmfbid = parameters->irreversible ? 0 : 1;
			tccp->qntsty = parameters->irreversible ? J2K_CCP_QNTSTY_SEQNT : J2K_CCP_QNTSTY_NOQNT;
			tccp->numgbits = numgbits;
			if ((int32_t) i == parameters->roi_compno)
				tccp->roishift = (uint8_t)parameters->roi_shift;
			else
				tccp->roishift = 0;
			if ((parameters->csty & J2K_CCP_CSTY_PRT) && parameters->res_spec) {
				uint32_t p = 0;
				int32_t it_res;
				assert(tccp->numresolutions > 0);
				for (it_res = (int32_t) tccp->numresolutions - 1; it_res >= 0;
						it_res--) {
					if (p < parameters->res_spec) {
						if (parameters->prcw_init[p] < 1) {
							tccp->prcw_exp[it_res] = 1;
						} else {
							tccp->prcw_exp[it_res] = floorlog2<uint32_t>(
									parameters->prcw_init[p]);
						}
						if (parameters->prch_init[p] < 1) {
							tccp->prch_exp[it_res] = 1;
						} else {
							tccp->prch_exp[it_res] = floorlog2<uint32_t>(
									parameters->prch_init[p]);
						}
					} else {
						uint32_t res_spec = parameters->res_spec;
						uint32_t size_prcw = 0;
						uint32_t size_prch = 0;
						size_prcw = parameters->prcw_init[res_spec - 1]	>> (p - (res_spec - 1));
						size_prch = parameters->prch_init[res_spec - 1]	>> (p - (res_spec - 1));
						if (size_prcw < 1) {
							tccp->prcw_exp[it_res] = 1;
						} else {
							tccp->prcw_exp[it_res] = floorlog2<uint32_t>(size_prcw);
						}
						if (size_prch < 1) {
							tccp->prch_exp[it_res] = 1;
						} else {
							tccp->prch_exp[it_res] = floorlog2<uint32_t>(size_prch);
						}
					}
					p++;
					/*printf("\nsize precinct for level %u : %u,%u\n", it_res,tccp->prcw_exp[it_res], tccp->prch_exp[it_res]); */
				} /*end for*/
			} else {
				for (uint32_t j = 0; j < tccp->numresolutions; j++) {
					tccp->prcw_exp[j] = 15;
					tccp->prch_exp[j] = 15;
				}
			}
			tcp->qcd.pull(tccp->stepsizes, !parameters->irreversible);
		}
	}
	grk_free(parameters->mct_data);
	parameters->mct_data = nullptr;

	return true;
}

bool CodeStream::compress(grk_plugin_tile* tile){
	uint32_t nb_tiles = (uint32_t) m_cp.t_grid_height* m_cp.t_grid_width;
	if (nb_tiles > max_num_tiles) {
		GRK_ERROR("Number of tiles %u is greater than %u max tiles "
				"allowed by the standard.", nb_tiles, max_num_tiles);
		return false;
	}
	auto pool_size = std::min<uint32_t>((uint32_t)ThreadPool::get()->num_threads(), nb_tiles);
	ThreadPool pool(pool_size);
	std::vector< std::future<int> > results;
	std::unique_ptr<TileProcessor*[]> procs = std::make_unique<TileProcessor*[]>(nb_tiles);
	std::atomic<bool> success(true);
	bool rc = false;

	for (uint16_t i = 0; i < nb_tiles; ++i)
		procs[i] = nullptr;

	if (pool_size > 1){
		for (uint16_t i = 0; i < nb_tiles; ++i) {
			uint16_t tile_ind = i;
			results.emplace_back(
					pool.enqueue([this,
								  &procs,
								  tile,
								  tile_ind,
								  &success] {
						if (success) {
							auto tileProcessor = new TileProcessor(this,m_stream);
							procs[tile_ind] = tileProcessor;
							tileProcessor->m_tile_index = tile_ind;
							tileProcessor->current_plugin_tile = tile;
							if (!tileProcessor->pre_write_tile())
								success = false;
							else {
								if (!tileProcessor->do_compress())
									success = false;
							}
						}
						return 0;
					})
				);
		}
	} else {
		for (uint16_t i = 0; i < nb_tiles; ++i) {
			auto tileProcessor = new TileProcessor(this,m_stream);
			tileProcessor->m_tile_index = i;
			tileProcessor->current_plugin_tile = tile;
			if (!tileProcessor->pre_write_tile()){
				delete tileProcessor;
				goto cleanup;
			}
			if (!tileProcessor->do_compress()){
				delete tileProcessor;
				goto cleanup;
			}
			if (!post_write_tile(tileProcessor)){
				delete tileProcessor;
				goto cleanup;
			}
			delete tileProcessor;
		}
	}
	if (pool_size > 1) {
		for(auto &result: results){
			result.get();
		}
		if (!success)
			goto cleanup;
		for (uint16_t i = 0; i < nb_tiles; ++i) {
			bool write_success = post_write_tile(procs[i]);
			delete procs[i];
			procs[i] = nullptr;
			if (!write_success)
				goto cleanup;
		}
	}
	rc = true;
cleanup:
	for (uint16_t i = 0; i < nb_tiles; ++i)
		delete procs[i];

	return rc;
}

bool CodeStream::compress_tile(uint16_t tile_index,	uint8_t *p_data, uint64_t uncompressed_data_size){
	if (!p_data)
		return false;
	bool rc = false;

	m_tileProcessor = new TileProcessor(this,m_stream);
	m_tileProcessor->m_tile_index = tile_index;

	if (!m_tileProcessor->pre_write_tile()) {
		GRK_ERROR("Error while pre_write_tile with tile index = %u",
				tile_index);
		goto cleanup;
	}
	/* now copy data into the tile component */
	if (!m_tileProcessor->copy_uncompressed_data_to_tile(p_data,	uncompressed_data_size)) {
		GRK_ERROR("Size mismatch between tile data and sent data.");
		goto cleanup;
	}
	if (!m_tileProcessor->do_compress())
		goto cleanup;
	if (!post_write_tile(m_tileProcessor)) {
		GRK_ERROR("Error while j2k_post_write_tile with tile index = %u",
				tile_index);
		goto cleanup;
	}
	rc = true;
cleanup:
	delete m_tileProcessor;

	return rc;
}

bool CodeStream::end_compress(void){
	/* customization of the compressing */
	m_procedure_list.push_back((j2k_procedure) j2k_write_eoc);
	if (m_cp.m_coding_params.m_enc.writeTLM)
		m_procedure_list.push_back((j2k_procedure) j2k_write_tlm_end);
	m_procedure_list.push_back((j2k_procedure) j2k_write_epc);
	m_procedure_list.push_back((j2k_procedure) j2k_end_encoding);

	return  exec(m_procedure_list);
}

bool CodeStream::parse_tile_header_markers(bool *can_decode_tile_data) {
	if (m_decompressor.getState() == J2K_DEC_STATE_EOC) {
		m_curr_marker = J2K_MS_EOC;
		return true;
	}
	/* We need to encounter a SOT marker (a new tile-part header) */
	if (m_decompressor.getState() != J2K_DEC_STATE_TPH_SOT){
		GRK_ERROR("parse_markers: no SOT marker found");
		return false;
	}

	/* Seek in code stream for SOT marker specifying desired tile index.
	 * If we don't find it, we stop when we read the EOC or run out of data */
	while (!m_decompressor.last_tile_part_was_read && (m_curr_marker != J2K_MS_EOC)) {

		/* read markers until SOD is detected */
		while (m_curr_marker != J2K_MS_SOD) {
			// end of stream with no EOC
			if (m_stream->get_number_byte_left() == 0) {
				m_decompressor.setState(J2K_DEC_STATE_NO_EOC);
				break;
			}
			uint16_t marker_size;
			if (!read_short(&marker_size))
				return false;
			else if (marker_size < 2) {
				GRK_ERROR("Inconsistent marker size");
				return false;
			}
			else if (marker_size == 2) {
				GRK_ERROR("Zero-size marker in header.");
				return false;
			}

			// subtract tile part header and header marker size
			if (m_decompressor.getState() & J2K_DEC_STATE_TPH)
				m_tileProcessor->tile_part_data_length -= (marker_size + 2);

			marker_size = (uint16_t)(marker_size - 2); /* Subtract the size of the marker ID already read */

			auto marker_handler = get_marker_handler(m_curr_marker);
			if (!marker_handler) {
				GRK_ERROR("Unknown marker encountered while seeking SOT marker");
				return false;
			}
			if (!(m_decompressor.getState() & marker_handler->states)) {
				GRK_ERROR("Marker 0x%x is not compliant with its expected position", m_curr_marker);
				return false;
			}
			if (!process_marker(marker_handler, marker_size))
				return false;


			/* Add the marker to the code stream index*/
			if (cstr_index) {
				if (!TileLengthMarkers::add_to_index(m_tileProcessor->m_tile_index, cstr_index,
													marker_handler->id,
													(uint32_t) m_stream->tell() - marker_size - grk_marker_length,
													marker_size + grk_marker_length)) {
					GRK_ERROR("Not enough memory to add tl marker");
					return false;
				}
			}
			if (marker_handler->id == J2K_MS_SOT) {
				uint64_t sot_pos = m_stream->tell() - marker_size - grk_marker_length;
				if (sot_pos > m_decompressor.m_last_sot_read_pos)
					m_decompressor.m_last_sot_read_pos = sot_pos;
				if (m_decompressor.m_skip_tile_data) {
					if (!m_stream->skip(m_tileProcessor->tile_part_data_length)) {
						GRK_ERROR("Stream too short");
						return false;
					}
					break;
				}
			}
			if (!read_marker())
				return false;
		}

		// no bytes left and no EOC marker : we're done!
		if (!m_stream->get_number_byte_left() && m_decompressor.getState() == J2K_DEC_STATE_NO_EOC)
			break;

		/* If we didn't skip data before, we need to read the SOD marker*/
		if (!m_decompressor.m_skip_tile_data) {
			if (!m_tileProcessor->prepare_sod_decoding(this))
				return false;
			if (!m_decompressor.last_tile_part_was_read) {
				if (!read_marker()){
					m_decompressor.setState(J2K_DEC_STATE_NO_EOC);
					break;
				}
			}
		} else {
			if (!read_marker()){
				m_decompressor.setState(J2K_DEC_STATE_NO_EOC);
				break;
			}
			/* Indicate we will try to read a new tile-part header*/
			m_decompressor.m_skip_tile_data = false;
			m_decompressor.last_tile_part_was_read = false;
			m_decompressor.setState(J2K_DEC_STATE_TPH_SOT);
		}
	}

	if (!m_tileProcessor) {
		GRK_ERROR("Missing SOT marker");
		return false;
	}

	// ensure lossy wavelet has quantization set
	auto tcp = get_current_decode_tcp();
	auto numComps = m_headerImage->numcomps;
	for (uint32_t k = 0; k < numComps; ++k) {
		auto tccp = tcp->tccps + k;
		if (tccp->qmfbid == 0 && tccp->qntsty == J2K_CCP_QNTSTY_NOQNT) {
			GRK_ERROR("Tile-components compressed using the irreversible processing path\n"
					"must have quantization parameters specified in the QCD/QCC marker segments,\n"
					"either explicitly, or through implicit derivation from the quantization\n"
					"parameters for the LL subband, as explained in the JPEG2000 standard, ISO/IEC\n"
					"15444-1.  The present set of code-stream parameters is not legal.");
			return false;
		}
	}
	// do QCD marker quantization step size sanity check
	// see page 553 of Taubman and Marcellin for more details on this check
	if (tcp->main_qcd_qntsty != J2K_CCP_QNTSTY_SIQNT) {
		//1. Check main QCD
		uint8_t maxDecompositions = 0;
		for (uint32_t k = 0; k < numComps; ++k) {
			auto tccp = tcp->tccps + k;
			if (tccp->numresolutions == 0)
				continue;
			// only consider number of resolutions from a component
			// whose scope is covered by main QCD;
			// ignore components that are out of scope
			// i.e. under main QCC scope, or tile QCD/QCC scope
			if (tccp->fromQCC || tccp->fromTileHeader)
				continue;
			auto decomps = (uint8_t)(tccp->numresolutions - 1);
			if (maxDecompositions < decomps)
				maxDecompositions = decomps;
		}
		if ((tcp->main_qcd_numStepSizes < 3 * (uint32_t)maxDecompositions + 1)) {
			GRK_ERROR("From Main QCD marker, "
					"number of step sizes (%u) is less than "
					"3* (maximum decompositions) + 1, "
					"where maximum decompositions = %u ",
					tcp->main_qcd_numStepSizes, maxDecompositions);
			return false;
		}
		//2. Check Tile QCD
		TileComponentCodingParams *qcd_comp = nullptr;
		for (uint32_t k = 0; k < numComps; ++k) {
			auto tccp = tcp->tccps + k;
			if (tccp->fromTileHeader && !tccp->fromQCC) {
				qcd_comp = tccp;
				break;
			}
		}
		if (qcd_comp && (qcd_comp->qntsty != J2K_CCP_QNTSTY_SIQNT)) {
			uint32_t maxTileDecompositions = 0;
			for (uint32_t k = 0; k < numComps; ++k) {
				auto tccp = tcp->tccps + k;
				if (tccp->numresolutions == 0)
					continue;
				// only consider number of resolutions from a component
				// whose scope is covered by Tile QCD;
				// ignore components that are out of scope
				// i.e. under Tile QCC scope
				if (tccp->fromQCC && tccp->fromTileHeader)
					continue;
				auto decomps = (uint8_t)(tccp->numresolutions - 1);
				if (maxTileDecompositions < decomps)
					maxTileDecompositions = decomps;
			}
			if ((qcd_comp->numStepSizes < 3 * maxTileDecompositions + 1)) {
				GRK_ERROR("From Tile QCD marker, "
						"number of step sizes (%u) is less than"
						" 3* (maximum tile decompositions) + 1, "
						"where maximum tile decompositions = %u ",
						qcd_comp->numStepSizes, maxTileDecompositions);

				return false;
			}
		}
	}
	/* Current marker is the EOC marker ?*/
	if (m_curr_marker == J2K_MS_EOC && m_decompressor.getState() != J2K_DEC_STATE_EOC)
		m_decompressor.setState( J2K_DEC_STATE_EOC);

	//if we are not ready to decompress tile part data,
    // then skip tiles with no tile data i.e. no SOD marker
	if (!m_decompressor.last_tile_part_was_read) {
		tcp = m_cp.tcps + m_tileProcessor->m_tile_index;
		if (!tcp->m_tile_data){
			*can_decode_tile_data = false;
			return true;
		}
	}
	if (!j2k_merge_ppt(
			m_cp.tcps + m_tileProcessor->m_tile_index)) {
		GRK_ERROR("Failed to merge PPT data");
		return false;
	}
	if (!m_tileProcessor->init(m_output_image, false)) {
		GRK_ERROR("Cannot decompress tile %u",
				m_tileProcessor->m_tile_index);
		return false;
	}
	*can_decode_tile_data = true;
	m_decompressor.orState(J2K_DEC_STATE_DATA);

	return true;
}

bool CodeStream::init_header_writing(void) {
	m_procedure_list.push_back((j2k_procedure) j2k_init_info);
	m_procedure_list.push_back((j2k_procedure) j2k_write_soc);
	m_procedure_list.push_back((j2k_procedure) j2k_write_siz);
	if (m_cp.tcps[0].getIsHT())
		m_procedure_list.push_back((j2k_procedure) j2k_write_cap);
	m_procedure_list.push_back((j2k_procedure) j2k_write_cod);
	m_procedure_list.push_back((j2k_procedure) j2k_write_qcd);
	m_procedure_list.push_back((j2k_procedure) j2k_write_all_coc);
	m_procedure_list.push_back((j2k_procedure) j2k_write_all_qcc);

	if (m_cp.m_coding_params.m_enc.writeTLM)
		m_procedure_list.push_back((j2k_procedure) j2k_write_tlm_begin);
	if (m_cp.rsiz == GRK_PROFILE_CINEMA_4K)
		m_procedure_list.push_back((j2k_procedure) j2k_write_poc);

	m_procedure_list.push_back((j2k_procedure) j2k_write_regions);
	m_procedure_list.push_back((j2k_procedure) j2k_write_com);
	//begin custom procedures
	if ((m_cp.rsiz & (GRK_PROFILE_PART2 | GRK_EXTENSION_MCT))
			== (GRK_PROFILE_PART2 | GRK_EXTENSION_MCT)) {
		m_procedure_list.push_back(
				(j2k_procedure) j2k_write_mct_data_group);
	}
	//end custom procedures

	if (cstr_index)
		m_procedure_list.push_back((j2k_procedure) j2k_get_end_header);
	m_procedure_list.push_back((j2k_procedure) j2k_update_rates);

	return true;
}
bool CodeStream::write_tile_part(TileProcessor *tileProcessor) {
	uint16_t currentTileNumber = tileProcessor->m_tile_index;
	auto cp = &m_cp;
	bool firstTilePart = tileProcessor->m_tile_part_index == 0;
	//1. write SOT
	SOTMarker sot(this);
	if (!sot.write())
		return false;
	uint32_t tile_part_bytes_written = sot_marker_segment_len;
	//2. write POC (only in first tile part)
	if (firstTilePart) {
		if (!GRK_IS_CINEMA(cp->rsiz)) {
			if (cp->tcps[currentTileNumber].numpocs) {
				auto tcp = m_cp.tcps + currentTileNumber;
				auto image = m_headerImage;
				uint32_t nb_comp = image->numcomps;
				if (!j2k_write_poc(this))
					return false;
				tile_part_bytes_written += getPocSize(nb_comp,
						1 + tcp->numpocs);
			}
		}
		/* set packno to zero when writing the first tile part
		 * (packno is used for SOP markers)
		 */
		tileProcessor->tile->packno = 0;
	}
	// 3. compress tile part
	if (!tileProcessor->compress_tile_part(&tile_part_bytes_written)) {
		GRK_ERROR("Cannot compress tile");
		return false;
	}
	/* 4. write Psot in SOT marker */
	if (!sot.write_psot(tile_part_bytes_written))
		return false;
	// 5. update TLM
	if (m_cp.tlm_markers)
		j2k_update_tlm(this, currentTileNumber, tile_part_bytes_written);
	++tileProcessor->m_tile_part_index;

	return true;
}
bool CodeStream::post_write_tile(TileProcessor *tileProcessor) {
	m_tileProcessor = tileProcessor;
	assert(tileProcessor->m_tile_part_index == 0);

	//1. write first tile part
	tileProcessor->pino = 0;
	tileProcessor->m_first_poc_tile_part = true;
	if (!write_tile_part(tileProcessor))
		return false;
	//2. write the other tile parts
	uint32_t pino;
	auto cp = &(m_cp);
	auto tcp = cp->tcps + tileProcessor->m_tile_index;
	// write tile parts for first progression order
	uint64_t num_tp = j2k_get_num_tp(cp, 0, tileProcessor->m_tile_index);
	if (num_tp > max_num_tile_parts_per_tile){
		GRK_ERROR("Number of tile parts %d for first POC exceeds maximum number of tile parts %d", num_tp,max_num_tile_parts_per_tile );
		return false;
	}
	tileProcessor->m_first_poc_tile_part = false;
	for (uint8_t tilepartno = 1; tilepartno < (uint8_t)num_tp; ++tilepartno) {
		if (!write_tile_part(tileProcessor))
			return false;
	}
	// write tile parts for remaining progression orders
	for (pino = 1; pino <= tcp->numpocs; ++pino) {
		tileProcessor->pino = pino;
		num_tp = j2k_get_num_tp(cp, pino,
				tileProcessor->m_tile_index);
		if (num_tp > max_num_tile_parts_per_tile){
			GRK_ERROR("Number of tile parts %d exceeds maximum number of "
					"tile parts %d", num_tp,max_num_tile_parts_per_tile );
			return false;
		}
		for (uint8_t tilepartno = 0; tilepartno < num_tp; ++tilepartno) {
			tileProcessor->m_first_poc_tile_part = (tilepartno == 0);
			if (!write_tile_part(tileProcessor))
				return false;
		}
	}
	++tileProcessor->m_tile_index;

	return true;
}

bool CodeStream::copy_default_tcp(void) {
	auto image = m_headerImage;
	uint32_t nb_tiles = m_cp.t_grid_height * m_cp.t_grid_width;
	uint32_t tccp_size = image->numcomps * (uint32_t) sizeof(TileComponentCodingParams);
	auto default_tcp = m_decompressor.m_default_tcp;
	uint32_t mct_size = image->numcomps * image->numcomps * (uint32_t) sizeof(float);

	/* For each tile */
	for (uint32_t i = 0; i < nb_tiles; ++i) {
		auto tcp = m_cp.tcps + i;
		/* keep the tile-compo coding parameters pointer of the current tile coding parameters*/
		auto current_tccp = tcp->tccps;
		/*Copy default coding parameters into the current tile coding parameters*/
		*tcp = *default_tcp;
		/* Initialize some values of the current tile coding parameters*/
		tcp->cod = false;
		tcp->ppt = false;
		tcp->ppt_data = nullptr;
		/* Remove memory not owned by this tile in case of early error return. */
		tcp->m_mct_decoding_matrix = nullptr;
		tcp->m_nb_max_mct_records = 0;
		tcp->m_mct_records = nullptr;
		tcp->m_nb_max_mcc_records = 0;
		tcp->m_mcc_records = nullptr;
		/* Reconnect the tile-compo coding parameters pointer to the current tile coding parameters*/
		tcp->tccps = current_tccp;

		/* Get the mct_decoding_matrix of the dflt_tile_cp and copy them into the current tile cp*/
		if (default_tcp->m_mct_decoding_matrix) {
			tcp->m_mct_decoding_matrix = (float*) grk_malloc(mct_size);
			if (!tcp->m_mct_decoding_matrix)
				return false;
			memcpy(tcp->m_mct_decoding_matrix,
					default_tcp->m_mct_decoding_matrix, mct_size);
		}

		/* Get the mct_record of the dflt_tile_cp and copy them into the current tile cp*/
		uint32_t mct_records_size = default_tcp->m_nb_max_mct_records
				* (uint32_t) sizeof(grk_mct_data);
		tcp->m_mct_records = (grk_mct_data*) grk_malloc(mct_records_size);
		if (!tcp->m_mct_records)
			return false;
		memcpy(tcp->m_mct_records, default_tcp->m_mct_records,
				mct_records_size);

		/* Copy the mct record data from dflt_tile_cp to the current tile*/
		auto src_mct_rec = default_tcp->m_mct_records;
		auto dest_mct_rec = tcp->m_mct_records;

		for (uint32_t j = 0; j < default_tcp->m_nb_mct_records; ++j) {
			if (src_mct_rec->m_data) {
				dest_mct_rec->m_data = (uint8_t*) grk_malloc(
						src_mct_rec->m_data_size);
				if (!dest_mct_rec->m_data)
					return false;
				memcpy(dest_mct_rec->m_data, src_mct_rec->m_data,
						src_mct_rec->m_data_size);
			}
			++src_mct_rec;
			++dest_mct_rec;
			/* Update with each pass to free exactly what has been allocated on early return. */
			tcp->m_nb_max_mct_records += 1;
		}

		/* Get the mcc_record of the dflt_tile_cp and copy them into the current tile cp*/
		uint32_t mcc_records_size = default_tcp->m_nb_max_mcc_records
				* (uint32_t) sizeof(grk_simple_mcc_decorrelation_data);
		tcp->m_mcc_records = (grk_simple_mcc_decorrelation_data*) grk_malloc(
				mcc_records_size);
		if (!tcp->m_mcc_records)
			return false;
		memcpy(tcp->m_mcc_records, default_tcp->m_mcc_records,
				mcc_records_size);
		tcp->m_nb_max_mcc_records = default_tcp->m_nb_max_mcc_records;

		/* Copy the mcc record data from dflt_tile_cp to the current tile*/
		for (uint32_t j = 0; j < default_tcp->m_nb_max_mcc_records; ++j) {
			auto src_mcc_rec = default_tcp->m_mcc_records + j;
			auto dest_mcc_rec = tcp->m_mcc_records + j;
			if (src_mcc_rec->m_decorrelation_array) {
				uint32_t offset = (uint32_t) (src_mcc_rec->m_decorrelation_array
						- default_tcp->m_mct_records);
				dest_mcc_rec->m_decorrelation_array = tcp->m_mct_records
						+ offset;
			}
			if (src_mcc_rec->m_offset_array) {
				uint32_t offset = (uint32_t) (src_mcc_rec->m_offset_array
						- default_tcp->m_mct_records);
				dest_mcc_rec->m_offset_array = tcp->m_mct_records + offset;
			}
		}
		/* Copy all the dflt_tile_compo_cp to the current tile cp */
		memcpy(current_tccp, default_tcp->tccps, tccp_size);
	}

	return true;
}
bool CodeStream::update_rates(void) {
	auto cp = &(m_cp);
	auto image = m_headerImage;
	auto tcp = cp->tcps;
	auto width = image->x1 - image->x0;
	auto height = image->y1 - image->y0;
	if (width <= 0 || height <= 0)
		return false;

	uint32_t bits_empty = 8 * image->comps->dx * image->comps->dy;
	uint32_t size_pixel = image->numcomps * image->comps->prec;
	auto header_size = (double) m_stream->tell();

	for (uint32_t tile_y = 0; tile_y < cp->t_grid_height; ++tile_y) {
		for (uint32_t tile_x = 0; tile_x < cp->t_grid_width; ++tile_x) {
			double stride = 0;
			if (cp->m_coding_params.m_enc.m_tp_on)
				stride = (tcp->m_nb_tile_parts - 1) * 14;
			double offset = stride / tcp->numlayers;
			auto tileBounds = cp->getTileBounds(image,  tile_x, tile_y);
			uint64_t numTilePixels = tileBounds.area();
			for (uint16_t k = 0; k < tcp->numlayers; ++k) {
				double *rates = tcp->rates + k;
				if (*rates > 0.0f)
					*rates = ((((double) size_pixel * (double) numTilePixels))
							/ ((double) *rates * (double) bits_empty)) - offset;
			}
			++tcp;
		}
	}
	tcp = cp->tcps;

	for (uint32_t tile_y = 0; tile_y < cp->t_grid_height; ++tile_y) {
		for (uint32_t tile_x = 0; tile_x < cp->t_grid_width; ++tile_x) {
			double *rates = tcp->rates;

			auto tileBounds = cp->getTileBounds(image,  tile_x, tile_y);
			uint64_t numTilePixels = tileBounds.area();

			double sot_adjust = ((double) numTilePixels * (double) header_size)
					/ ((double) width * height);
			if (*rates > 0.0) {
				*rates -= sot_adjust;
				if (*rates < 30.0f)
					*rates = 30.0f;
			}
			++rates;
			for (uint16_t k = 1; k < (uint16_t)(tcp->numlayers - 1); ++k) {
				if (*rates > 0.0) {
					*rates -= sot_adjust;
					if (*rates < *(rates - 1) + 10.0)
						*rates = (*(rates - 1)) + 20.0;
				}
				++rates;
			}
			if (*rates > 0.0) {
				*rates -= (sot_adjust + 2.0);
				if (*rates < *(rates - 1) + 10.0)
					*rates = (*(rates - 1)) + 20.0;
			}
			++tcp;
		}
	}

	return true;
}

bool CodeStream::compress_validation() {
	bool is_valid = true;
	is_valid &=	(m_decompressor.getState() == J2K_DEC_STATE_NONE);

	/* ISO 15444-1:2004 states between 1 & 33
	 * ergo (number of decomposition levels between 0 -> 32) */
	if ((m_cp.tcps->tccps->numresolutions == 0)
			|| (m_cp.tcps->tccps->numresolutions > GRK_J2K_MAXRLVLS)) {
		GRK_ERROR("Invalid number of resolutions : %u not in range [1,%u]",
				m_cp.tcps->tccps->numresolutions, GRK_J2K_MAXRLVLS);
		return false;
	}
	if (m_cp.t_width == 0) {
		GRK_ERROR("Tile x dimension must be greater than zero ");
		return false;
	}
	if (m_cp.t_height == 0) {
		GRK_ERROR("Tile y dimension must be greater than zero ");
		return false;
	}

	return is_valid;
}

}
