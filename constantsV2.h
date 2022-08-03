/*! @file constantsV2.h
  *
 *  
 *  The code is provided under the GPL license, see LICENSE for
 *  more details.
 */

#include <stdint.h>
#include <stddef.h>

#define ALIGN __attribute__ ((aligned (32)))


/*bitslice of 8*/
const uint8_t* KMatrixBitslice(uint32_t round, int stateSizeBits);
const uint8_t* LMatrixBitslice(uint32_t round, int stateSizeBits);
const uint8_t* ConstantBitslice(uint32_t round, int stateSizeBits);






