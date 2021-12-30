#include <float.h>
#include <stdint.h>

#include "numpy/halffloat.h"
#include "numpy/npy_common.h"


// This is temporary: it needs to go in "halffloat.h"
#define NPY_HALF_MANT_DIG    (11)


// Constants (optimized out by compiler)
// t: target type
// s: number of bits in significand, +1 implicit (i.e, FLT/DBL/LDBL_MANT_DIG)
#define II_ZERO_MASK(t)                (((t)(-1)) >> 1)
#define II_EXPONENT_MASK(t, s)         ((t)(((t)(-1)) << (s)) >> 1)
#define II_EXPONENT_BIAS_MASK(t, s)    ((t)(((t)(-1)) << ((s) + 1)) >> 2)
#define II_EXPONENT_BIAS(t, s)         ((t)(((t)(-1)) << ((s) + 1)) >> ((s) + 1))
#define II_EXPONENT_SIG_MASK(t, s)     (II_EXPONENT_BIAS_MASK(t, s) + (((t)((s) - 1)) << ((s) - 1)))
#define II_SIGNIFICAND_MASK(t, s)      (((t)1 << ((s) - 1)) - 1)
// Compile-time check for endianness using https://stackoverflow.com/a/4240257/2988730 (requires C99)
#define II_IS_LITTLE_ENDIAN            (((union { uint16_t s; uint8_t b; }){1}).b)

// Code generation:

/**begin repeat
 * #itype = uint16_t, uint32_t, uint64_t#
 * #type = half, float, double#
 * #TYPE = NPY_HALF, FLT, DBL#
 * #NTYPE = HALF, FLOAT, DOUBLE#
 */
/*
static inline npy_bool isint_@type@(npy_@type@ n)
{
    // Zero when everything except sign bit is zero
    if((*((@itype@ *)&n) & ZERO_MASK(@itype@)) == 0) return 1;

    uint16_t exponent = *((@itype@ *)&n) & II_EXPONENT_MASK(@itype@, @TYPE@_MANT_DIG);

    // NaN or Inf when the exponent bits are all ones
    if(exponent == II_EXPONENT_MASK(@itype@, @TYPE@_MANT_DIG)) return 0;
    // Guaranteed fraction when exponent < 0
    if(exponent < II_EXPONENT_BIAS_MASK(@itype@, @TYPE@_MANT_DIG)) return 0;
    // Guaranteed integer when exponent >= @TYPE@_MANT_DIG - 1
    if(exponent >= II_EXPONENT_SIG_MASK(@itype@, @TYPE@_MANT_DIG)) return 1;
    // Otherwise, check that the significand bits past the exponent power are zeros
    return (*((@itype@ *)&n) & (II_SIGNIFICAND_MASK(@itype@, @TYPE@_MANT_DIG) >> ((exponent >> (@TYPE@_MANT_DIG - 1)) - II_EXPONENT_BIAS(@itype@, @TYPE@_MANT_DIG)))) == 0;
}
*/
/**end repeat */

static inline npy_bool isint_half(npy_half n)
{
    // Zero when everything except sign bit is zero
    if((*((uint16_t *)&n) & II_ZERO_MASK(uint16_t)) == 0) return 1;

    uint16_t exponent = *((uint16_t *)&n) & II_EXPONENT_MASK(uint16_t, NPY_HALF_MANT_DIG);

    // NaN or Inf when the exponent bits are all ones
    if(exponent == II_EXPONENT_MASK(uint16_t, NPY_HALF_MANT_DIG)) return 0;
    // Guaranteed fraction when exponent < 0
    if(exponent < II_EXPONENT_BIAS_MASK(uint16_t, NPY_HALF_MANT_DIG)) return 0;
    // Guaranteed integer when exponent >= NPY_HALF_MANT_DIG - 1
    if(exponent >= II_EXPONENT_SIG_MASK(uint16_t, NPY_HALF_MANT_DIG)) return 1;
    // Otherwise, check that the significand bits past the exponent are zeros
    return (*((uint16_t *)&n) & (II_SIGNIFICAND_MASK(uint16_t, NPY_HALF_MANT_DIG) >> ((exponent >> (NPY_HALF_MANT_DIG - 1)) - II_EXPONENT_BIAS(uint16_t, NPY_HALF_MANT_DIG)))) == 0;
}

static inline npy_bool isint_float(npy_float n)
{
    // Zero when everything except sign bit is zero
    if((*((uint32_t *)&n) & II_ZERO_MASK(uint32_t)) == 0) return 1;

    uint32_t exponent = *((uint32_t *)&n) & II_EXPONENT_MASK(uint32_t, FLT_MANT_DIG);

    // NaN or Inf when the exponent bits are all ones
    if(exponent == II_EXPONENT_MASK(uint32_t, FLT_MANT_DIG)) return 0;
    // Guaranteed fraction when exponent < 0
    if(exponent < II_EXPONENT_BIAS_MASK(uint32_t, FLT_MANT_DIG)) return 0;
    // Guaranteed integer when exponent >= FLT_MANT_DIG - 1
    if(exponent >= II_EXPONENT_SIG_MASK(uint32_t, FLT_MANT_DIG)) return 1;
    // Otherwise, check that the significand bits past the exponent are zeros
    return (*((uint32_t *)&n) & (II_SIGNIFICAND_MASK(uint32_t, FLT_MANT_DIG) >> ((exponent >> (FLT_MANT_DIG - 1)) - II_EXPONENT_BIAS(uint32_t, FLT_MANT_DIG)))) == 0;
}

static inline npy_bool isint_double(npy_double n)
{
    // Zero when everything except sign bit is zero
    if((*((uint64_t *)&n) & II_ZERO_MASK(uint64_t)) == 0) return 1;

    uint64_t exponent = *((uint64_t *)&n) & II_EXPONENT_MASK(uint64_t, DBL_MANT_DIG);

    // NaN or Inf when the exponent bits are all ones
    if(exponent == II_EXPONENT_MASK(uint64_t, DBL_MANT_DIG)) return 0;
    // Guaranteed fraction when exponent < 0
    if(exponent < II_EXPONENT_BIAS_MASK(uint64_t, DBL_MANT_DIG)) return 0;
    // Guaranteed integer when exponent >= DBL_MANT_DIG - 1
    if(exponent >= II_EXPONENT_SIG_MASK(uint64_t, DBL_MANT_DIG)) return 1;
    // Otherwise, check that the significand bits past the exponent are zeros
    return (*((uint64_t *)&n) & (II_SIGNIFICAND_MASK(uint64_t, DBL_MANT_DIG) >> ((exponent >> (DBL_MANT_DIG - 1)) - II_EXPONENT_BIAS(uint64_t, DBL_MANT_DIG)))) == 0;
}

// long double can be a weird type. In its 80-bit incartation, it has an
// explicit integer bit that is nominally not part of the significand. It can
// also be either double precision, extended precision (80-bit) or quadruple
// precision (128-bit)

npy_bool isint_longdouble(npy_longdouble n)
{
    // These ifs evaluate constants, so should effectively be compiled out
    if(sizeof(npy_longdouble) == sizeof(npy_float)) {
        return isint_float((npy_float)n);
    } else if(sizeof(npy_longdouble) == sizeof(npy_double)) {
        return isint_double((npy_double)n);
    } else if(LDBL_MANT_DIG == 64) {
        // 80-bit integer represented in 10, 12 or 16 bytes (only 10 used)
        uint64_t significand;
        uint16_t exponent;
        if(sizeof(npy_longdouble) == 16) {
            // Treat it as two uint64_ts: high for exponent and low for significand
            // Exponent is only in the lowest two bytes, significand is in all the bytes
            uint64_t *i64 = ((union { npy_longdouble ld; uint64_t i64[2]; }){n}).i64;
            if(II_IS_LITTLE_ENDIAN) {
                // Little endian
                significand = i64[0];
                exponent = i64[1] & 0xFFFF;
            } else {
                // Big endian
                significand = (uint64_t)(i64[0] << 16) | (i64[1] >> 48);
                exponent = (i64[0] >> 48) & 0xFFFF;
            }
        } else if(sizeof(npy_longdouble) == 12) {
            // Treat it as a uint64_t significand and a uint32_t
            struct container { uint64_t i64; uint32_t i32; };
            struct container i = ((union { npy_longdouble ld; struct container i; }){n}).i;
            if(II_IS_LITTLE_ENDIAN) {
                // Little endian
                significand = i.i64;
                exponent = i.i32 & 0xFFFF;
            } else {
                // Big endian
                significand = (uint64_t)(i.i64 << 16) | (i.i32 >> 16);
                exponent = (i.i64 >> 48) & 0xFFFF;
            }
        } else if(sizeof(npy_longdouble) == 10) {
            // Assume 10-byte number: uint64_t significand, uint16_t
            struct container { uint64_t i64; uint16_t i16; };
            struct container i = ((union { npy_longdouble ld; struct container i; }){n}).i;
            if(II_IS_LITTLE_ENDIAN) {
                // Little endian
                significand = i.i64;
                exponent = i.i16;
            } else {
                // Big endian
                significand = (uint64_t)(i.i64 << 16) | i.i16;
                exponent = (i.i64 >> 48) & 0xFFFF;
            }
        } else {
            // error: unsupported format
            return (npy_bool)-1;
        }

        // Zero when exponent and significand are zero.
        // Denormal or pseudo denormal (fractions) if significand is nonzero
        exponent &= II_ZERO_MASK(uint16_t);
        if(exponent == 0) return (npy_bool)(significand == 0);
        // NaN or Inf (or invalid) when the exponent bits are all ones
        // Unnormal (invalid) if integer bit is cleared (technically could pass as an integer)
        // Guaranteed fraction when exponent < 0
        if(exponent == II_EXPONENT_MASK(uint16_t, 1) || (significand & ~II_ZERO_MASK(uint64_t)) == 0 || exponent < II_EXPONENT_BIAS(uint16_t, 1)) return 0;
        // Guaranteed integer when exponent >= LDBL_MANT_DIG - 1
        if(exponent >= II_EXPONENT_BIAS(uint16_t, 1) + (LDBL_MANT_DIG - 1)) return 1;
        // Otherwise, check that the significand bits past the exponent are zeros
        return (significand & ((uint64_t)(-1) >> (exponent - II_EXPONENT_BIAS(uint16_t, 1)))) == 0;
    } else if(LDBL_MANT_DIG == 113) {
        // Quadruple precision
        uint64_t hi, lo;
        uint64_t *i64 = ((union { npy_longdouble ld; uint64_t i64[2]; }){n}).i64;
        if(II_IS_LITTLE_ENDIAN) {
            // Little endian
            lo = i64[0];
            hi = i64[1];
        } else {
            // Big endian
            lo = i64[1];
            hi = i64[0];
        }
        // Zero when everything except sign bit is zero
        if((hi & II_ZERO_MASK(uint64_t)) == 0 && lo == 0) return 1;

// Number of significand digits in upper half
#define QDBL_MANT_DIG  (LDBL_MANT_DIG - 64)

        uint64_t exponent = hi & II_EXPONENT_MASK(uint64_t, QDBL_MANT_DIG);

        // NaN or Inf when the exponent bits are all ones
        if(exponent == II_EXPONENT_MASK(uint64_t, QDBL_MANT_DIG)) return 0;
        // Guaranteed fraction when exponent < 0
        if(exponent < II_EXPONENT_BIAS_MASK(uint64_t, QDBL_MANT_DIG)) return 0;
        // Guaranteed integer when exponent >= LBL_MANT_DIG - 1
        if(exponent >= II_EXPONENT_BIAS_MASK(uint64_t, QDBL_MANT_DIG) + (uint64_t)((LDBL_MANT_DIG - 1) << (QDBL_MANT_DIG - 1))) return 1;
        // Otherwise, check that the significand bits past the exponent are zeros
        if(exponent >= II_EXPONENT_SIG_MASK(uint64_t, QDBL_MANT_DIG)) {
            // Only the low half matters
            return (lo & ((uint64_t)(-1) >> ((exponent >> (QDBL_MANT_DIG - 1)) - II_EXPONENT_BIAS(uint64_t, QDBL_MANT_DIG)))) == 0;
        } else {
            // Both parts of significand matter
            return (hi & (II_SIGNIFICAND_MASK(uint32_t, QDBL_MANT_DIG) >> ((exponent >> (QDBL_MANT_DIG - 1)) - II_EXPONENT_BIAS(uint32_t, FLT_MANT_DIG)))) == 0 && lo == 0;
        }
    } else {
        // Return error code
        return (npy_bool)(-1);
    }
}

