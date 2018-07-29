/*

The MIT License (MIT)

Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include "fname.h"

#ifndef ERRMEM_H
#define ERRMEM_H

/* requires:
     <stdlib.h> for malloc, calloc, realloc, free
*/

/*--------------------------------------------------------------------------
   Error Reporting
   Memory Allocation Wrappers to Catch Out-of-memory
  --------------------------------------------------------------------------*/

#ifdef __GNUC__
void fail(const char *fmt, ...) __attribute__ ((noreturn));
#else
void fail(const char *fmt, ...);
#endif

static void failwith(const char *string)
{
  fail("%s\n",string);
}

static void *smalloc(size_t size, const char *file)
{
  void *res = malloc(size);
  if(!res && size) fail("%s: allocation of %d bytes failed\n",file,(int)size);
  return res;
}

static void *scalloc(size_t nmemb, size_t size, const char *file)
{
  void *res = calloc(nmemb, size);
  if(!res && nmemb)
    fail("%s: allocation of %d bytes failed\n",file,(int)size*nmemb);
  return res;
}

static void *srealloc(void *ptr, size_t size, const char *file)
{
  void *res = realloc(ptr, size);
  if(!res && size) fail("%s: allocation of %d bytes failed\n",file,(int)size);
  return res;
}

#define tmalloc(type, count) \
  ((type*) smalloc((count)*sizeof(type),__FILE__) )
#define tcalloc(type, count) \
  ((type*) scalloc((count),sizeof(type),__FILE__) )
#define trealloc(type, ptr, count) \
  ((type*) srealloc((ptr),(count)*sizeof(type),__FILE__) )

typedef struct { size_t size; void *ptr; } buffer;
static void buffer_init_(buffer *b, size_t size, const char *file)
{
  b->size=size, b->ptr=smalloc(size,file);
}
static void buffer_reserve_(buffer *b, size_t min, const char *file)
{
  size_t size = b->size;
  if(size<min) {
    size+=size/2+1;
    if(size<min) size=min;
    b->ptr=srealloc(b->ptr,size,file);
    b->size=size;
  }
}
static void buffer_free(buffer *b) { free(b->ptr); }

#define buffer_init(b,size) buffer_init_(b,size,__FILE__)
#define buffer_reserve(b,min) buffer_reserve_(b,min,__FILE__)

#endif

#define nek_exitt FORTRAN_NAME(exitt,EXITT)
void nek_exitt(void);
void eexit(void);
