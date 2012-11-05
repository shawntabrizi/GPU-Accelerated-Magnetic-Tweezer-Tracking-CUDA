/*
* Copyright (c) 2008-2009 Christian Buchner <Christian.Buchner@gmail.com>
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*        * Redistributions of source code must retain the above copyright
*          notice, this list of conditions and the following disclaimer.
*        * Redistributions in binary form must reproduce the above copyright
*          notice, this list of conditions and the following disclaimer in the
*          documentation and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY Christian Buchner ''AS IS'' AND ANY 
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL Christian Buchner BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef _CUDACOMPLEX_H
#define _CUDACOMPLEX_H

#include <vector_types.h>  // required for float2

// Depending on whether we're running inside the CUDA compiler, define the __host_
// and __device__ intrinsics, otherwise just make the functions static to prevent
// linkage issues (duplicate symbols and such)
//#define __CUDACC__
#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#define M_HOST __host__
#define M_HOSTDEVICE __host__ __device__
#else
#define HOST static inline
#define DEVICE static inline
#define HOSTDEVICE static inline
#define M_HOST inline      // note there is no static here
#define M_HOSTDEVICE inline // (static has a different meaning for class member functions)
#endif

// Struct alignment is handled differently between the CUDA compiler and other
// compilers (e.g. GCC, MS Visual C++ .NET)
#ifdef __CUDACC__
#define ALIGN(x)  __align__(x)
#else
#if defined(_MSC_VER) && (_MSC_VER >= 1300)
// Visual C++ .NET and later
#define ALIGN(x) __declspec(align(x)) 
#else
#if defined(__GNUC__)
// GCC
#define ALIGN(x)  __attribute__ ((aligned (x)))
#else
// all other compilers
#define ALIGN(x) 
#endif
#endif
#endif

// Somehow in emulation mode the code won't compile Mac OS X 1.1 CUDA SDK when the
// operators below make use of references (compiler bug?). So instead we compile
// the code to pass everything through the stack. Slower, but works.
// I am not sure how the Linux CUDA SDK will behave, so currently when I detect
// Microsoft's Visual C++.NET I always allow it to use references.
#if !defined(__DEVICE_EMULATION__) || (defined(_MSC_VER) && (_MSC_VER >= 1300))
#define REF(x) &x
#define ARRAYREF(x,y) (&x)[y]
#else
#define REF(x) x
#define ARRAYREF(x,y) x[y]
#endif

/**
 * A complex number type for use with CUDA, single precision accuracy.
 * This is deliberately designed to use few C++ features in order to work with most
 * CUDA SDK versions. It is friendlier to use than the cuComplex type because it
 * provides more operator overloads.
 * The class should work in host code and in device code and also in emulation mode.
 * Also this has been tested on any OS that the CUDA SDK is available for.
 */
typedef struct ALIGN(8) _cudacomplex {

  // float2 is a native CUDA type and allows for coalesced 128 bit access
  // when accessed according to CUDA's memory coalescing rules.
  // x member is real component
  // y member is imaginary component
  float2 value;

  // assignment of a scalar to complex
  _cudacomplex& operator=(const float REF(a)) {
         value.x = a; value.y = 0;
         return *this;
  };

  // assignment of a pair of floats to complex
  _cudacomplex& operator=(const float ARRAYREF(a,2)) {
         value.x = a[0]; value.y = a[1];
         return *this;
  };

  // return references to the real and imaginary components
  M_HOSTDEVICE float& real() {return value.x;};
  M_HOSTDEVICE float& imag() {return value.y;};

} cudacomplex;

// add complex numbers
HOSTDEVICE cudacomplex operator+(const cudacomplex REF(a), const cudacomplex REF(b)) {
   cudacomplex result = {{ a.value.x + b.value.x, a.value.y  + b.value.y }};
   return result;
}

// add scalar to complex
HOSTDEVICE cudacomplex operator+(const cudacomplex REF(a), const float REF(b)) {
   cudacomplex result = {{ a.value.x + b, a.value.y }};
   return result;
}

// add complex to scalar
HOSTDEVICE cudacomplex operator+(const float REF(a), const cudacomplex REF(b)) {
   cudacomplex result = {{ a + b.value.x, b.value.y }};
   return result;
}

// subtract complex numbers
HOSTDEVICE cudacomplex operator-(const cudacomplex REF(a), const cudacomplex REF(b)) {
   cudacomplex result = {{ a.value.x - b.value.x, a.value.y  - b.value.y }};
   return result;
}

// negate a complex number
HOSTDEVICE cudacomplex operator-(const cudacomplex REF(a)) {
   cudacomplex result = {{ -a.value.x, -a.value.y }};
   return result;
}

// subtract scalar from complex
HOSTDEVICE cudacomplex operator-(const cudacomplex REF(a), const float REF(b)) {
   cudacomplex result = {{ a.value.x - b, a.value.y }};
   return result;
}

// subtract complex from scalar
HOSTDEVICE cudacomplex operator-(const float REF(a), const cudacomplex REF(b)) {
   cudacomplex result = {{ a - b.value.x, -b.value.y }};
   return result;
}

// multiply complex numbers
HOSTDEVICE cudacomplex operator*(const cudacomplex REF(a), const cudacomplex REF(b)) {
   cudacomplex result = {{ a.value.x * b.value.x - a.value.y * b.value.y,
                                                   a.value.y * b.value.x + a.value.x * b.value.y }};
   return result;
}

// multiply complex with scalar
HOSTDEVICE cudacomplex operator*(const cudacomplex REF(a), const float REF(b)) {
   cudacomplex result = {{ a.value.x * b, a.value.y * b }};
   return result;
}

// multiply scalar with complex
HOSTDEVICE cudacomplex operator*(const float REF(a), const cudacomplex REF(b)) {
   cudacomplex result = {{ a * b.value.x, a * b.value.y }};
   return result;
}

// divide complex numbers
HOSTDEVICE cudacomplex operator/(const cudacomplex REF(a), const cudacomplex REF(b)) {
   float tmp = ( b.value.x * b.value.x + b.value.y * b.value.y );
   cudacomplex result = {{ (a.value.x * b.value.x + a.value.y * b.value.y ) / tmp,
                                                   (a.value.y * b.value.x - a.value.x * b.value.y ) / tmp }};
   return result;
}

// divide complex by scalar
HOSTDEVICE cudacomplex operator/(const cudacomplex REF(a), const float REF(b)) {
   cudacomplex result = {{ a.value.x / b, a.value.y / b }};
   return result;
}

// divide scalar by complex
HOSTDEVICE cudacomplex operator/(const float REF(a), const cudacomplex REF(b)) {
   float tmp = ( b.value.x * b.value.x + b.value.y * b.value.y );
   cudacomplex result = {{ ( a * b.value.x ) / tmp, ( -a * b.value.y ) / tmp }};
   return result;
}

// complex conjugate
HOSTDEVICE cudacomplex operator~(const cudacomplex REF(a)) {
   cudacomplex result = {{ a.value.x, -a.value.y }};
   return result;
}

// complex modulus (complex absolute)
HOSTDEVICE float abs(const cudacomplex REF(a)) {
   float result = sqrt( a.value.x*a.value.x + a.value.y*a.value.y );
   return result;
}

// complex modulus (complex absolute)
HOSTDEVICE float norm(const cudacomplex REF(a)) {
   float result = a.value.x*a.value.x + a.value.y*a.value.y;
   return result;
}

// a possible alternative to a cudacomplex constructor
HOSTDEVICE cudacomplex make_cudacomplex(float a, float b)
{
        cudacomplex res;
        res.real() = a;
        res.imag() = b;
        return res;
}

namespace constants
{
        const _cudacomplex zero = make_cudacomplex(0.0f, 0.0f);
        const _cudacomplex one  = make_cudacomplex(1.0f, 0.0f);
        const _cudacomplex I    = make_cudacomplex(0.0f, 1.0f);
};

#endif // #ifndef _CUDACOMPLEX_H
