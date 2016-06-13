#ifndef VIENNACL_TOOLS_ENTRY_PROXY_HPP_
#define VIENNACL_TOOLS_ENTRY_PROXY_HPP_

/* =========================================================================
   Copyright (c) 2010-2016, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/** @file viennacl/tools/entry_proxy.hpp
    @brief A proxy class for entries in a vector
*/


#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"

namespace viennacl
{

//proxy class for single vector entries (this is a slow operation!!)
/**
  * @brief A proxy class for a single element of a vector or matrix. This proxy should not be noticed by end-users of the library.
  *
  * This proxy provides access to a single entry of a vector. If the element is assigned to a GPU object, no unnecessary transfers to the CPU and back to GPU are initiated.
  *
  * @tparam NumericT Either float or double
  */
template<typename NumericT>
class entry_proxy
{
public:
  typedef viennacl::backend::mem_handle      handle_type;

  /** @brief The constructor for the proxy class. Declared explicit to avoid any surprises created by the compiler.
      *
      * @param mem_offset The memory offset in multiples of sizeof(NumericT) relative to the memory pointed to by the handle
      * @param mem_handle A viennacl::ocl::handle for the memory buffer on the GPU.
      */
  explicit entry_proxy(vcl_size_t mem_offset,
                       handle_type & mem_handle)
    : index_(mem_offset), mem_handle_(mem_handle) {}


  //operators:
  /** @brief Inplace addition of a CPU floating point value
      */
  entry_proxy & operator+=(NumericT value)
  {
    NumericT temp = read();
    temp += value;
    write(temp);
    return *this;
  }

  /** @brief Inplace subtraction of a CPU floating point value
      */
  entry_proxy &  operator-=(NumericT value)
  {
    NumericT temp = read();
    temp -= value;
    write(temp);
    return *this;
  }

  /** @brief Inplace multiplication with a CPU floating point value
      */
  entry_proxy &  operator*=(NumericT value)
  {
    NumericT temp = read();
    temp *= value;
    write(temp);
    return *this;
  }

  /** @brief Inplace division by a CPU floating point value
      */
  entry_proxy &  operator/=(NumericT value)
  {
    NumericT temp = read();
    temp /= value;
    write(temp);
    return *this;
  }

  /** @brief Assignment of a CPU floating point value
      */
  entry_proxy &  operator=(NumericT value)
  {
    write(value);
    return *this;
  }

  /** @brief Assignment of a GPU floating point value. Avoids unnecessary GPU->CPU->GPU transfers
      */
  entry_proxy & operator=(scalar<NumericT> const & value)
  {
    viennacl::backend::memory_copy(value.handle(), mem_handle_, 0, sizeof(NumericT)*index_, sizeof(NumericT));
    return *this;
  }

  /** @brief Assignment of another GPU value.
      */
  entry_proxy &  operator=(entry_proxy const & other)
  {
    viennacl::backend::memory_copy(other.handle(), mem_handle_, sizeof(NumericT) * other.index_, sizeof(NumericT)*index_, sizeof(NumericT));
    return *this;
  }

  //type conversion:
  // allows to write something like:
  //  double test = vector(4);
  /** @brief Conversion to a CPU floating point value.
      *
      *  This conversion allows to write something like
      *    double test = vector(4);
      *  However, one has to keep in mind that CPU<->GPU transfers are very slow compared to CPU<->CPU operations.
      */
  operator NumericT () const
  {
    NumericT temp = read();
    return temp;
  }

  /** @brief Returns the index of the represented element
      */
  vcl_size_t index() const { return index_; }

  /** @brief Returns the memory viennacl::ocl::handle
      */
  handle_type const & handle() const { return mem_handle_; }

private:
  /** @brief Reads an element from the GPU to the CPU
      */
  NumericT read() const
  {
    NumericT temp;
    viennacl::backend::memory_read(mem_handle_, sizeof(NumericT)*index_, sizeof(NumericT), &temp);
    return temp;
  }

  /** @brief Writes a floating point value to the GPU
      */
  void write(NumericT value)
  {
    viennacl::backend::memory_write(mem_handle_, sizeof(NumericT)*index_, sizeof(NumericT), &value);
  }

  vcl_size_t index_;
  viennacl::backend::mem_handle & mem_handle_;
}; //entry_proxy







/**
  * @brief A proxy class for a single element of a vector or matrix. This proxy should not be noticed by end-users of the library.
  *
  * This proxy provides access to a single entry of a vector. If the element is assigned to a GPU object, no unnecessary transfers to the CPU and back to GPU are initiated.
  *
  * @tparam NumericT Either float or double
  */
template<typename NumericT>
class const_entry_proxy
{
  typedef const_entry_proxy<NumericT>      self_type;
public:
  typedef viennacl::backend::mem_handle      handle_type;

  /** @brief The constructor for the proxy class. Declared explicit to avoid any surprises created by the compiler.
      *
      * @param mem_offset The memory offset in multiples of sizeof(NumericT) relative to the memory pointed to by the handle
      * @param mem_handle A viennacl::ocl::handle for the memory buffer on the GPU.
      */
  explicit const_entry_proxy(vcl_size_t mem_offset,
                             handle_type const & mem_handle)
    : index_(mem_offset), mem_handle_(mem_handle) {}


  //type conversion:
  // allows to write something like:
  //  double test = vector(4);
  /** @brief Conversion to a CPU floating point value.
      *
      *  This conversion allows to write something like
      *    double test = vector(4);
      *  However, one has to keep in mind that CPU<->GPU transfers are very slow compared to CPU<->CPU operations.
      */
  operator NumericT () const
  {
    NumericT temp = read();
    return temp;
  }

  /** @brief Returns the index of the represented element
      */
  unsigned int index() const { return index_; }

  /** @brief Returns the memory handle
      */
  handle_type const & handle() const { return mem_handle_; }

private:
  /** @brief Reads an element from the GPU to the CPU
      */
  NumericT read() const
  {
    NumericT temp;
    viennacl::backend::memory_read(mem_handle_, sizeof(NumericT)*index_, sizeof(NumericT), &temp);
    return temp;
  }

  vcl_size_t index_;
  viennacl::backend::mem_handle const & mem_handle_;
}; //entry_proxy

}

#endif
