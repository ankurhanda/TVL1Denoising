/**
 * @author  Steven Lovegrove
 * Copyright (C) 2010  Steven Lovegrove
 *                     Imperial College London
 **/

#ifndef CUMATH_H
#define CUMATH_H

#include <vector_types.h>
#include <cuda_runtime_api.h>
#include <cutil_math.h>
#include <iostream>
#include <assert.h>

//namespace pangolin
//{

///////////////////////////////////////////
// Primitive indexing
///////////////////////////////////////////

inline __host__ __device__ float ix(const float4& x, unsigned i )
{
  return ((float*)&x)[i];
}

inline __host__ __device__ float ix(const float3& x, unsigned i )
{
  return ((float*)&x)[i];
}

inline __host__ __device__ unsigned char ix(const uchar4& x, unsigned i )
{
  return ((unsigned char*)&x)[i];
}

///////////////////////////////////////////
// Matrix Class
///////////////////////////////////////////

// R x C Real Matrix. Internally row-major.
template<unsigned R, unsigned C = 1, typename P = float>
struct cumat
{
  inline __device__ __host__ P operator()(int r, int c) const
  {
    return m[r*C + c];
  }

  inline __device__ __host__ P& operator()(int r, int c)
  {
    return m[r*C + c];
  }

  inline __device__ __host__ P operator()(int r) const
  {
    return m[r];
  }

  inline __device__ __host__ P& operator()(int r)
  {
    return m[r];
  }

  inline __device__ __host__ void operator+=(const cumat<R,C,P>& rhs)
  {
    #pragma unroll
    for( int i=0; i<R*C; ++i )
      m[i] += rhs.m[i];
  }

#ifdef HAVE_TOON
  inline __host__ operator TooN::Matrix<R,C,P>() const
  {
    TooN::Matrix<R,C,P> ret;
    for( int r=0; r<R; ++r )
      for( int c=0; c<C; ++c )
        ret(r,c) = m[r*C + c];
    return ret;
  }

  inline __host__ operator TooN::Vector<R*C,P>() const
  {
    TooN::Vector<R*C,P> ret;
    for( int i=0; i< R*C; ++i )
        ret[i] = m[i];
    return ret;
  }
#endif // HAVE_TOON

  P m[R*C];
};

///////////////////////////////////////////
// Aliases
///////////////////////////////////////////

typedef cumat<2> cuvec2;
typedef cumat<3> cuvec3;
typedef cumat<4> cuvec4;
typedef cumat<3,2> cumat3x2;
typedef cumat<2,3> cumat2x3;
typedef cumat<4,3> cumat4x3;
typedef cumat<3,4> cumat3x4;
typedef cumat<3,3> cumat3x3;
typedef cumat<4,4> cumat4x4;

///////////////////////////////////////////
// Construct from Zero / Identity
///////////////////////////////////////////

template<unsigned R, unsigned C>
inline __device__ __host__ cumat<R,C,float> cumat_zero()
{
  cumat<R,C,float> ret;
  #pragma unroll
  for( unsigned i=0; i<R*C; ++i )
    ret(i) = 0.0f;
  return ret;
}

template<unsigned R>
inline __device__ __host__ cumat<R,R,float> cumat_id()
{
  cumat<R,R,float> ret = cumat_zero<R,R>();
  #pragma unroll
  for( unsigned i=0; i<R; ++i )
    ret(i,i) = 1.0f;
  return ret;
}

///////////////////////////////////////////
// Construct from rows (allow size mismatch)
///////////////////////////////////////////

template<unsigned R, typename P, unsigned V>
inline __device__ __host__ cumat<R,1,P> cumat_from(const cumat<V,1,P>& r1)
{
  const unsigned RV = (R<V) ? R : V;
  cumat<R,1,P> ret;
  #pragma unroll
  for( unsigned i=0; i<RV; ++i )
    ret(i) = r1(i);
  return ret;
}

template<unsigned R, unsigned C, typename P, unsigned V>
inline __device__ __host__ cumat<R,C,P> cumat_rows(const cumat<V,1,P>& r0, const cumat<V,1,P>& r1)
{
  const unsigned CV = (C<V) ? C : V;
  cumat<R,C,P> ret;
  #pragma unroll
  for( unsigned c=0; c<CV; ++c )
  {
    ret(0,c) = r0(c);
    ret(1,c) = r1(c);
  }
  return ret;
}

template<unsigned R, unsigned C, typename P, unsigned V>
inline __device__ __host__ cumat<R,C,P> cumat_rows(const cumat<V,1,P>& r0, const cumat<V,1,P>& r1, const cumat<V,1,P>& r2)
{
  const unsigned CV = (C<V) ? C : V;
  cumat<R,C,P> ret;
  #pragma unroll
  for( unsigned c=0; c<CV; ++c )
  {
    ret(0,c) = r0(c);
    ret(1,c) = r1(c);
    ret(2,c) = r2(c);
  }
  return ret;
}

template<unsigned R, unsigned C, typename P, unsigned V>
inline __device__ __host__ cumat<R,C,P> cumat_rows(const cumat<V,1,P>& r0, const cumat<V,1,P>& r1, const cumat<V,1,P>& r2, const cumat<V,1,P>& r3)
{
  const unsigned CV = (C<V) ? C : V;
  cumat<R,C,P> ret;
  #pragma unroll
  for( unsigned c=0; c<CV; ++c )
  {
    ret(0,c) = r0(c);
    ret(1,c) = r1(c);
    ret(2,c) = r2(c);
    ret(3,c) = r3(c);
  }
  return ret;
}

///////////////////////////////////////////
// Construct from other types
///////////////////////////////////////////

#ifdef HAVE_TOON
template<unsigned R, unsigned C, typename P, typename TP>
inline __host__ cumat<R,C,P> cumat_from(const TooN::Matrix<R,C,TP>& m)
{
  cumat<R,C,P> ret;
  for( int r=0; r<R; ++r )
    for( int c=0; c<C; ++c )
      ret(r,c) = m(r,c);
  return ret;
}
#endif // HAVE_TOON

///////////////////////////////////////////
// Construct from primitives (allow size mismatch)
///////////////////////////////////////////

template<unsigned R>
inline __device__ __host__ cumat<R> cumat_from(const float2& x)
{
  cumat<R> ret;
  if(0<R) ret(0) = x.x;
  if(1<R) ret(1) = x.y;
  return ret;
}

template<unsigned R>
inline __device__ __host__ cumat<R> cumat_from(const float3& x)
{
  cumat<R> ret;
  if(0<R) ret(0) = x.x;
  if(1<R) ret(1) = x.y;
  if(2<R) ret(2) = x.z;
  return ret;
}

template<unsigned R>
inline __device__ __host__ cumat<R> cumat_from(const float4& x)
{
  cumat<R> ret;
  if(0<R) ret(0) = x.x;
  if(1<R) ret(1) = x.y;
  if(2<R) ret(2) = x.z;
  if(3<R) ret(3) = x.w;
  return ret;
}

template<unsigned R, unsigned C>
inline __device__ __host__ cumat<R,C,float> cumat_rows(const float4& r0, const float4& r1)
{
  cumat<R,C,float> ret;
  if(0<C) {
    ret(0,0) = r0.x;
    ret(1,0) = r1.x;
  }
  if(1<C) {
    ret(0,1) = r0.y;
    ret(1,1) = r1.y;
  }
  if(2<C) {
    ret(0,2) = r0.z;
    ret(1,2) = r1.z;
  }
  if(3<C) {
    ret(0,3) = r0.w;
    ret(1,3) = r1.w;
  }
  return ret;
}

template<unsigned R, unsigned C>
inline __device__ __host__ cumat<R,C,float> cumat_rows(const float4& r0, const float4& r1, const float4& r2)
{
  cumat<R,C,float> ret;
  if(0<C) {
    ret(0,0) = r0.x;
    ret(1,0) = r1.x;
    ret(2,0) = r2.x;
  }
  if(1<C) {
    ret(0,1) = r0.y;
    ret(1,1) = r1.y;
    ret(2,1) = r2.y;
  }
  if(2<C) {
    ret(0,2) = r0.z;
    ret(1,2) = r1.z;
    ret(2,2) = r2.z;
  }
  if(3<C) {
    ret(0,3) = r0.w;
    ret(1,3) = r1.w;
    ret(2,3) = r2.w;
  }
  return ret;
}

inline __device__ __host__ cumat<3,2> cumat_cols(const float4& r0, const float4& r1)
{
  return (cumat<3,2>){
    r0.x, r1.x,
    r0.y, r1.y,
    r0.z, r1.z
  };
}

inline __device__ __host__ cumat<3,2> cumat_cols(const cumat<4>& r0, const cumat<4>& r1)
{
  return (cumat<3,2>){
    r0(0), r1(0),
    r0(1), r1(1),
    r0(2), r1(2)
  };
}

///////////////////////////////////////////
// Matrix Matrix operations
///////////////////////////////////////////

template<unsigned R, unsigned CR, unsigned C, typename P>
inline __device__ __host__ cumat<R,C,P> operator*(const cumat<R,CR,P>& lhs, const cumat<CR,C,P>& rhs)
{
  cumat<R,C,P> ret;

  for( unsigned r=0; r<R; ++r) {
    for( unsigned c=0; c<C; ++c) {
      ret(r,c) = 0;
      #pragma unroll
      for( unsigned k=0; k<CR; ++k)  {
        ret(r,c) += lhs(r,k) * rhs(k,c);
      }
    }
  }
  return ret;
}

// Specialisation for scalar product
template<unsigned CR, typename P>
inline __device__ __host__ P operator*(const cumat<1,CR,P>& lhs, const cumat<CR,1,P>& rhs)
{
  P ret = 0;
  for( int i=0; i<CR; ++i)
    ret += lhs(i) * rhs(i);
  return ret;
}

// Specialisation for scalar product
template<unsigned R, typename P>
inline __device__ __host__ P operator*(const cumat<R,1,P>& lhs, const cumat<R,1,P>& rhs)
{
  P ret = 0;
  for( int i=0; i<R; ++i)
    ret += lhs(i) * rhs(i);
  return ret;
}

template<unsigned R, unsigned CR, unsigned C, typename P>
inline __device__ __host__ cumat<R,C,P> mul_aTb(const cumat<CR,R,P>& a, const cumat<CR,C>& b)
{
  cumat<R,C,P> ret;

  for( unsigned r=0; r<R; ++r) {
    for( unsigned c=0; c<C; ++c) {
      ret(r,c) = 0;
      #pragma unroll
      for( unsigned k=0; k<CR; ++k)  {
        ret(r,c) += a(k,r) * b(k,c);
      }
    }
  }
  return ret;
}

template<unsigned R, unsigned CR, unsigned C, typename P>
inline __device__ __host__ cumat<R,C,P> mul_abT(const cumat<R,CR,P>& a, const cumat<C,CR>& b)
{
  cumat<R,C,P> ret;

  for( unsigned r=0; r<R; ++r) {
    for( unsigned c=0; c<C; ++c) {
      ret(r,c) = 0;
      #pragma unroll
      for( unsigned k=0; k<CR; ++k)  {
        ret(r,c) += a(r,k) * b(c,k);
      }
    }
  }
  return ret;
}

template<unsigned R, unsigned C, typename P>
inline __device__ __host__ cumat<R,C,P> operator+(const cumat<R,C,P>& lhs, const cumat<R,C>& rhs)
{
  cumat<R,C,P> ret;
  #pragma unroll
  for( unsigned i=0; i<R*C; ++i )
    ret(i) = lhs(i) + rhs(i);
  return ret;
}

template<unsigned R, unsigned C, typename P>
inline __device__ __host__ cumat<R,C,P> operator-(const cumat<R,C,P>& lhs, const cumat<R,C>& rhs)
{
  cumat<R,C,P> ret;
  #pragma unroll
  for( unsigned i=0; i<R*C; ++i )
    ret(i) = lhs(i) - rhs(i);
  return ret;
}

///////////////////////////////////////////
// Matrix Scalar operations
///////////////////////////////////////////

template<unsigned R, unsigned C, typename P>
inline __device__ __host__ cumat<R,C,P> operator*(const cumat<R,C,P>& lhs, const P rhs)
{
  cumat<R,C,P> ret;
  #pragma unroll
  for( unsigned i=0; i<R*C; ++i )
    ret(i) = lhs(i) * rhs;
  return ret;
}

template<unsigned R, unsigned C, typename P>
inline __device__ __host__ cumat<R,C,P> operator-(const P lhs, const cumat<R,C>& rhs)
{
  cumat<R,C,P> ret;
  #pragma unroll
  for( unsigned i=0; i<R*C; ++i )
    ret(i) = lhs * rhs(i);
  return ret;
}

template<unsigned R, unsigned C, typename P>
inline __device__ __host__ cumat<R,C,P> operator/(const cumat<R,C,P>& lhs, const P rhs)
{
  cumat<R,C,P> ret;
  #pragma unroll
  for( unsigned i=0; i<R*C; ++i )
    ret(i) = lhs(i) / rhs;
  return ret;
}

///////////////////////////////////////////
// Symmetric Matrix class & utilities
///////////////////////////////////////////

// N x N Real symetric matrix. Internally row major for
// each unique element
template<unsigned N, typename P = float>
struct cusymmat
{
  static const int unique = N*(N+1)/2;

  inline __device__ __host__ operator cumat<N,N,P>()
  {
    cumat<N,N,P> ret;

    int i = 0;
    for( int r=0; r<N; ++r )
    {
      for( int c=0; c<=r; ++c )
      {
        P elem = m[i++];
        ret(r,c) = elem;
        ret(c,r) = elem;
      }
    }
    return ret;
  }

#ifdef HAVE_TOON
  inline __host__ operator TooN::Matrix<N,N,P>()
  {
    TooN::Matrix<N,N,P> ret;

    int i = 0;
    for( int r=0; r<N; ++r )
    {
      for( int c=0; c<=r; ++c )
      {
        P elem = m[i++];
        ret(r,c) = elem;
        ret(c,r) = elem;
      }
    }
    return ret;
  }
#endif // HAVE_TOON

  inline __device__ __host__ void operator+=(const cusymmat<N,P>& rhs)
  {
    #pragma unroll
    for( unsigned i=0; i<unique; ++i )
      m[i] += rhs.m[i];
  }

  P m[unique];
};

template<unsigned N, typename P>
inline __device__ __host__ cusymmat<N,P> operator+(const cusymmat<N,P>& lhs, const cusymmat<N,P>& rhs)
{
  cusymmat<N,P> ret;
  #pragma unroll
  for( unsigned i=0; i<cusymmat<N,P>::unique; ++i )
    ret.m[i] = lhs.m[i] + rhs.m[i];
  return ret;
}

template<unsigned N, typename P>
inline __device__ __host__ cusymmat<N,P> operator*(const cusymmat<N,P>& lhs, const float rhs)
{
  cusymmat<N,P> ret;
  #pragma unroll
  for( unsigned i=0; i<cusymmat<N,P>::unique; ++i )
    ret.m[i] = lhs.m[i] * rhs;
  return ret;
}

template<unsigned N, typename P>
inline __device__ __host__ cusymmat<N,P> OuterProduct(const cumat<N,1,P>& M)
{
  cusymmat<N,P> ret;
  int i=0;
  for( int r=0; r<N; ++r )
    #pragma unroll
    for( int c=0; c<=r; ++c )
      ret.m[i++] = M(r) * M(c);
  return ret;
}

template<unsigned N>
inline __device__ __host__ cusymmat<N,float> cusymmat_zero()
{
  cusymmat<N> ret;
  #pragma unroll
  for( unsigned i=0; i< cusymmat<N>::unique; ++i )
    ret.m[i] = 0.0f;
  return ret;
}

///////////////////////////////////////////
// Matrix specialisations
///////////////////////////////////////////

inline __device__ __host__ cumat<3> operator*(const cumat<3,3>& lhs, const float3& rhs )
{
  return (cumat<3>) {
      lhs(0,0)*rhs.x + lhs(0,1)*rhs.y + lhs(0,2)*rhs.z,
      lhs(1,0)*rhs.x + lhs(1,1)*rhs.y + lhs(1,2)*rhs.z,
      lhs(2,0)*rhs.x + lhs(2,1)*rhs.y + lhs(2,2)*rhs.z
  };
}

inline __device__ __host__ cumat<3> operator*(const cumat<3,4>& lhs, const float4& rhs )
{
  return (cumat<3>) {
      lhs(0,0)*rhs.x + lhs(0,1)*rhs.y + lhs(0,2)*rhs.z + lhs(0,3)*rhs.w,
      lhs(1,0)*rhs.x + lhs(1,1)*rhs.y + lhs(1,2)*rhs.z + lhs(1,3)*rhs.w,
      lhs(2,0)*rhs.x + lhs(2,1)*rhs.y + lhs(2,2)*rhs.z + lhs(2,3)*rhs.w
  };
}

inline __device__ __host__ cumat<4> operator*(const cumat<4,4>& lhs, const float4& rhs )
{
  return (cumat<4>) {
      lhs(0,0)*rhs.x + lhs(0,1)*rhs.y + lhs(0,2)*rhs.z + lhs(0,3)*rhs.w,
      lhs(1,0)*rhs.x + lhs(1,1)*rhs.y + lhs(1,2)*rhs.z + lhs(1,3)*rhs.w,
      lhs(2,0)*rhs.x + lhs(2,1)*rhs.y + lhs(2,2)*rhs.z + lhs(2,3)*rhs.w,
      lhs(3,0)*rhs.x + lhs(3,1)*rhs.y + lhs(3,2)*rhs.z + lhs(3,3)*rhs.w
  };
}

inline __device__ __host__ cumat<3> operator*(const cumat<2,3>& lhs, const float3& rhs )
{
  return (cumat<3>) {
      lhs(0,0)*rhs.x + lhs(0,1)*rhs.y + lhs(0,2)*rhs.z,
      lhs(1,0)*rhs.x + lhs(1,1)*rhs.y + lhs(1,2)*rhs.z
  };
}

///////////////////////////////////////////
// Vector project / unproject
///////////////////////////////////////////

inline __device__ __host__ cumat<3> up( const cumat<2>& x )
{
  return (cumat<3>){x(0),x(1),1};
}

inline __device__ __host__ cumat<4> up( const cumat<3>& x )
{
  return (cumat<4>){x(0),x(1),x(2),1};
}

inline __device__ __host__ cumat<2> dn( const cumat<3>& x )
{
  return (cumat<2>){x(0)/x(2), x(1)/x(2)};
}

inline __device__ __host__ cumat<3> dn( const cumat<4>& x )
{
  return (cumat<3>){x(0)/x(3), x(1)/x(3), x(2)/x(3)};
}


///////////////////////////////////////////
// Primitive project / unproject
///////////////////////////////////////////

inline __device__ __host__ float3 up( const float2& x )
{
  return (float3){x.x, x.y, 1};
}

inline __device__ __host__ float4 up( const float3& x )
{
  return (float4){x.x, x.y, x.z, 1};
}

inline __device__ __host__ float2 dn( const float3& x )
{
  return (float2){x.x/x.z, x.y/x.z};
}

inline __device__ __host__ float3 dn( const float4& x )
{
  return (float3){x.x/x.w, x.y/x.w, x.z/x.z};
}


///////////////////////////////////////////
// IO
///////////////////////////////////////////

template<unsigned R, unsigned C>
inline __host__ std::ostream& operator<<( std::ostream& os, const cumat<R,C>& m )
{
  for( unsigned r=0; r<R; ++r)
  {
    for( unsigned c=0; c<C; ++c)
      std::cout << m(r,c) << " ";
    std::cout << std::endl;
  }
  return os;
}

//template<unsigned N>
//inline __host__ std::ostream& operator<<( std::ostream& os, const cusymmat<N>& m )
//{
//  for( unsigned r=0; r<N; ++r)
//  {
//    for( unsigned c=0; c<N; ++c)
//      std::cout << m(r,c) << " ";
//    std::cout << std::endl;
//  }
//  return os;
//}


//}

#endif // CUMATH_H
