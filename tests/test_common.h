#ifndef SVE_TEST_COMMON_H
#define SVE_TEST_COMMON_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SVE_ASSERT(cond)                                                     \
  do {                                                                       \
    if (!(cond)) {                                                           \
      fprintf(stderr, "ASSERT FAILED at %s:%d: %s\n", __FILE__, __LINE__,    \
              #cond);                                                        \
      abort();                                                               \
    }                                                                        \
  } while (0)

#define SVE_ASSERT_EQ_INT(a, b)                                              \
  do {                                                                       \
    long long _va = (long long)(a);                                          \
    long long _vb = (long long)(b);                                          \
    if (_va != _vb) {                                                        \
      fprintf(stderr, "ASSERT_EQ_INT FAILED at %s:%d: %lld != %lld\n",       \
              __FILE__, __LINE__, _va, _vb);                                 \
      abort();                                                               \
    }                                                                        \
  } while (0)

#define SVE_ASSERT_NEAR(a, b, eps)                                           \
  do {                                                                       \
    double _va = (double)(a);                                                \
    double _vb = (double)(b);                                                \
    if (isnan(_va) || isnan(_vb) || fabs(_va - _vb) > (eps)) {               \
      fprintf(stderr, "ASSERT_NEAR FAILED at %s:%d: %.12g != %.12g\n",       \
              __FILE__, __LINE__, _va, _vb);                                 \
      abort();                                                               \
    }                                                                        \
  } while (0)

#endif /* SVE_TEST_COMMON_H */
