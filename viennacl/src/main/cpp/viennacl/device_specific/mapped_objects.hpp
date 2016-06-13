#ifndef VIENNACL_DEVICE_SPECIFIC_MAPPED_TYPE_HPP
#define VIENNACL_DEVICE_SPECIFIC_MAPPED_TYPE_HPP

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


/** @file viennacl/device_specific/mapped_objects.hpp
    @brief Map ViennaCL objects to generator wrappers
*/

#include <string>

#include "viennacl/scheduler/forwards.h"
#include "viennacl/device_specific/forwards.h"
#include "viennacl/device_specific/utils.hpp"

namespace viennacl
{

namespace device_specific
{

/** @brief Mapped Object
*
* This object populates the symbolic mapping associated with a statement. (root_id, LHS|RHS|PARENT) => mapped_object
* The tree can then be reconstructed in its symbolic form
*/
class mapped_object
{
private:
  virtual void postprocess(std::string &) const { }

protected:
  struct MorphBase { virtual ~MorphBase(){} };
  struct MorphBase1D : public MorphBase { public: virtual std::string operator()(std::string const & i) const = 0; };
  struct MorphBase2D : public MorphBase { public: virtual std::string operator()(std::string const & i, std::string const & j) const = 0; };

  static void replace_offset(std::string & str, MorphBase const & morph)
  {
    vcl_size_t pos = 0;
    while ((pos=str.find("$OFFSET", pos))!=std::string::npos)
    {
      std::string postprocessed;
      vcl_size_t pos_po = str.find('{', pos);
      vcl_size_t pos_pe = str.find('}', pos_po);

      if (MorphBase2D const * p2d = dynamic_cast<MorphBase2D const *>(&morph))
      {
        vcl_size_t pos_comma = str.find(',', pos_po);
        std::string i = str.substr(pos_po + 1, pos_comma - pos_po - 1);
        std::string j = str.substr(pos_comma + 1, pos_pe - pos_comma - 1);
        postprocessed = (*p2d)(i, j);
      }
      else if (MorphBase1D const * p1d = dynamic_cast<MorphBase1D const *>(&morph))
      {
        std::string i = str.substr(pos_po + 1, pos_pe - pos_po - 1);
        postprocessed = (*p1d)(i);
      }

      str.replace(pos, pos_pe + 1 - pos, postprocessed);
      pos = pos_pe;
    }
  }

  void register_attribute(std::string & attribute, std::string const & key, std::string const & value)
  {
    attribute = value;
    keywords_[key] = attribute;
  }

public:
  struct node_info
  {
    node_info(mapping_type const * _mapping, scheduler::statement const * _statement, vcl_size_t _root_idx) :
      mapping(_mapping), statement(_statement), root_idx(_root_idx) { }
    mapping_type const * mapping;
    scheduler::statement const * statement;
    vcl_size_t root_idx;
  };

public:
  mapped_object(std::string const & scalartype, unsigned int id, std::string const & type_key) : type_key_(type_key)
  {
    register_attribute(scalartype_, "#scalartype", scalartype);
    register_attribute(name_, "#name", "obj" + tools::to_string(id));
  }

  virtual ~mapped_object(){ }

  virtual std::string & append_kernel_arguments(std::set<std::string> &, std::string & str, unsigned int) const { return str; }

  std::string type_key() const { return type_key_; }

  std::string const & name() const { return name_; }

  std::string process(std::string const & in) const
  {
    std::string res(in);
    for (std::map<std::string,std::string>::const_iterator it = keywords_.begin(); it != keywords_.end(); ++it)
      tools::find_and_replace(res, it->first, it->second);
    postprocess(res);
    return res;
  }

  std::string evaluate(std::map<std::string, std::string> const & accessors) const
  {
    if (accessors.find(type_key_)==accessors.end())
      return name_;
    return process(at(accessors, type_key_));
  }


protected:
  std::string name_;
  std::string scalartype_;
  std::string type_key_;
  std::map<std::string, std::string> keywords_;
};


/** @brief Binary leaf interface
*
*  Some subtrees have to be interpret at leaves when reconstructing the final expression. It is the case of trans(), diag(), prod(), etc...
*  This interface stores basic infos about the sub-trees
*/
class binary_leaf
{
public:
  binary_leaf(mapped_object::node_info info) : info_(info){ }

  void process_recursive(utils::kernel_generation_stream & stream, leaf_t leaf, std::string const & key, std::string const & process_str, std::set<std::string> & already_fetched)
  {
    tree_parsing::process(stream, leaf, key, process_str, *info_.statement, info_.root_idx, *info_.mapping, already_fetched);
  }

  std::string evaluate_recursive(leaf_t leaf, std::map<std::string, std::string> const & accessors)
  {
    return tree_parsing::evaluate(leaf, accessors, *info_.statement, info_.root_idx, *info_.mapping);
  }

protected:
  mapped_object::node_info info_;
};

/** @brief Matrix product
  *
  * Maps prod(matrix_expression, matrix_expression)
  */
class mapped_matrix_product : public mapped_object, public binary_leaf
{
public:
  mapped_matrix_product(std::string const & scalartype, unsigned int id, node_info info) : mapped_object(scalartype, id, "matrix_product"), binary_leaf(info) { }
};

/** @brief Reduction
*
* Base class for mapping a reduction
*/
class mapped_reduction : public mapped_object, public binary_leaf
{
public:
  mapped_reduction(std::string const & scalartype, unsigned int id, node_info info, std::string const & type_key) : mapped_object(scalartype, id, type_key), binary_leaf(info){ }

  vcl_size_t root_idx() const { return info_.root_idx; }
  scheduler::statement const & statement() const { return *info_.statement; }
  scheduler::statement_node root_node() const { return statement().array()[root_idx()]; }
  bool is_index_reduction() const { return utils::is_index_reduction(info_.statement->array()[info_.root_idx].op); }

  scheduler::op_element root_op() const
  {
    scheduler::op_element res = info_.statement->array()[info_.root_idx].op;
    if (res.type==scheduler::OPERATION_BINARY_MAT_VEC_PROD_TYPE
        ||res.type==scheduler::OPERATION_BINARY_INNER_PROD_TYPE)
      res.type        = scheduler::OPERATION_BINARY_ADD_TYPE;
    return res;
  }
};

/** @brief Scalar reduction
*
* Maps a scalar reduction (max, min, argmax, inner_prod, etc..)
*/
class mapped_scalar_reduction : public mapped_reduction
{
public:
  mapped_scalar_reduction(std::string const & scalartype, unsigned int id, node_info info) : mapped_reduction(scalartype, id, info, "scalar_reduction"){ }
};

/** @brief Vector reduction
*
* Maps a row-wise reduction (max, min, argmax, matrix-vector product, etc..)
*/
class mapped_row_wise_reduction : public mapped_reduction
{
public:
  mapped_row_wise_reduction(std::string const & scalartype, unsigned int id, node_info info) : mapped_reduction(scalartype, id, info, "row_wise_reduction") { }
};

/** @brief Host scalar
 *
 * Maps a host scalar (passed by value)
 */
class mapped_host_scalar : public mapped_object
{
public:
  mapped_host_scalar(std::string const & scalartype, unsigned int id) : mapped_object(scalartype, id, "host_scalar"){ }

  std::string & append_kernel_arguments(std::set<std::string> & already_generated, std::string & str, unsigned int width) const
  {
    if (already_generated.insert(name_).second)
      str += generate_value_kernel_argument(utils::append_width(scalartype_, width), name_);
    return str;
  }
};

/** @brief Handle
*
* Maps an object passed by pointer
*/
class mapped_handle : public mapped_object
{
private:
  virtual void append_optional_arguments(std::string &) const = 0;

public:
  mapped_handle(std::string const & scalartype, unsigned int id, std::string const & type_key) : mapped_object(scalartype, id, type_key)
  {
    register_attribute(pointer_, "#pointer", name_ + "_pointer");
  }

  std::string & append_kernel_arguments(std::set<std::string> & already_generated, std::string & str, unsigned int width) const
  {
    if (already_generated.insert(name_).second)
    {
      str += generate_pointer_kernel_argument("__global", utils::append_width(scalartype_, width), pointer_);
      append_optional_arguments(str);
    }
    return str;
  }

private:
  std::string pointer_;
};


/** @brief Scalar
 *
 * Maps a scalar passed by pointer
 */
class mapped_scalar : public mapped_handle
{
private:
  void append_optional_arguments(std::string &) const{ }

public:
  mapped_scalar(std::string const & scalartype, unsigned int id) : mapped_handle(scalartype, id, "scalar") { }
};

/** @brief Buffered
 *
 * Maps a buffered object (vector, matrix)
 */
class mapped_buffer : public mapped_handle
{
public:
  mapped_buffer(std::string const & scalartype, unsigned int id, std::string const & type_key) : mapped_handle(scalartype, id, type_key){ }
};

/** @brief Vector
 *
 * Maps a vector
 */
class mapped_vector : public mapped_buffer
{
  void append_optional_arguments(std::string & str) const
  {
    str += generate_value_kernel_argument("unsigned int", start_);
    str += generate_value_kernel_argument("unsigned int", stride_);
  }

public:
  mapped_vector(std::string const & scalartype, unsigned int id) : mapped_buffer(scalartype, id, "vector")
  {
    register_attribute(start_, "#start", name_ + "_start");
    register_attribute(stride_, "#stride", name_ + "_stride");
  }

private:
  std::string start_;
  std::string stride_;
};

/** @brief Matrix
 *
 * Maps a matrix
 */
class mapped_matrix : public mapped_buffer
{
private:
  void append_optional_arguments(std::string & str) const
  {
    str += generate_value_kernel_argument("unsigned int", ld_);
    str += generate_value_kernel_argument("unsigned int", start1_);
    str += generate_value_kernel_argument("unsigned int", start2_);
    str += generate_value_kernel_argument("unsigned int", stride1_);
    str += generate_value_kernel_argument("unsigned int", stride2_);
  }

  void postprocess(std::string & str) const
  {
    struct Morph : public MorphBase2D
    {
      Morph(bool _is_row_major, std::string const & _ld) : is_row_major(_is_row_major), ld(_ld){ }
      std::string operator()(std::string const & i, std::string const & j) const
      {
        if (is_row_major)
          return "(" + i + ") * " + ld +  " + (" + j + ")";
        return "(" + i + ") +  (" + j + ") * " + ld;
      }
    private:
      bool is_row_major;
      std::string const & ld;
    };
    replace_offset(str, Morph(row_major_, ld_));
  }

public:
  mapped_matrix(std::string const & scalartype, unsigned int id, bool row_major) : mapped_buffer(scalartype, id, "matrix"), row_major_(row_major)
  {
    register_attribute(ld_, "#ld", name_ + "_ld");
    register_attribute(start1_, "#start1", name_ + "_start1");
    register_attribute(start2_, "#start2", name_ + "_start2");
    register_attribute(stride1_, "#stride1", name_ + "_stride1");
    register_attribute(stride2_, "#stride2", name_ + "_stride2");
    if (row_major_)
      keywords_["#nldstride"] = "#stride1";
    else
      keywords_["#nldstride"] = "#stride2";

    if (row_major_)
    {
      std::swap(start1_, start2_);
      std::swap(stride1_, stride2_);
    }
  }

  bool row_major() const
  {
    return row_major_;
  }

private:
  std::string ld_;
  std::string start1_;
  std::string start2_;
  std::string stride1_;
  std::string stride2_;
  bool row_major_;
};

/** @brief Vector diag
*
*  Maps a diag(vector_expression) node into a diagonal matrix
*/
class mapped_vector_diag : public mapped_object, public binary_leaf
{
private:
  void postprocess(std::string &res) const
  {
    std::map<std::string, std::string> accessors;
    tools::find_and_replace(res, "#diag_offset", tree_parsing::evaluate(RHS_NODE_TYPE, accessors, *info_.statement, info_.root_idx, *info_.mapping));
    accessors["vector"] = res;
    res = tree_parsing::evaluate(LHS_NODE_TYPE, accessors, *info_.statement, info_.root_idx, *info_.mapping);
  }

public:
  mapped_vector_diag(std::string const & scalartype, unsigned int id, node_info info) : mapped_object(scalartype, id, "vector_diag"), binary_leaf(info){ }
};


/** @brief Trans
*
*  Maps trans(matrix_expression) into the transposed of matrix_expression
*/
class mapped_trans: public mapped_object, public binary_leaf
{
private:
  void postprocess(std::string &res) const
  {
    std::map<std::string, std::string> accessors;
    accessors["matrix"] = res;
    res = tree_parsing::evaluate(LHS_NODE_TYPE, accessors, *info_.statement, info_.root_idx, *info_.mapping);
  }

public:
  mapped_trans(std::string const & scalartype, unsigned int id, node_info info) : mapped_object(scalartype, id, "matrix_trans"), binary_leaf(info){ }
};

/** @brief Matrix row
*
*  Maps row(matrix_expression, scalar_expression) into the scalar_expression's row of matrix_expression
*/
class mapped_matrix_row : public mapped_object, binary_leaf
{
private:
  void postprocess(std::string &res) const
  {
    std::map<std::string, std::string> accessors;
    tools::find_and_replace(res, "#row", tree_parsing::evaluate(RHS_NODE_TYPE, accessors, *info_.statement, info_.root_idx, *info_.mapping));
    accessors["matrix"] = res;
    res = tree_parsing::evaluate(LHS_NODE_TYPE, accessors, *info_.statement, info_.root_idx, *info_.mapping);
  }

public:
  mapped_matrix_row(std::string const & scalartype, unsigned int id, node_info info) : mapped_object(scalartype, id, "matrix_row"), binary_leaf(info)
  { }
};


/** @brief Matrix column
*
*  Maps column(matrix_expression, scalar_expression) into the scalar_expression's column of matrix_expression
*/
class mapped_matrix_column : public mapped_object, binary_leaf
{
private:
  void postprocess(std::string &res) const
  {
    std::map<std::string, std::string> accessors;
    tools::find_and_replace(res, "#column", tree_parsing::evaluate(RHS_NODE_TYPE, accessors, *info_.statement, info_.root_idx, *info_.mapping));
    accessors["matrix"] = res;
    res = tree_parsing::evaluate(LHS_NODE_TYPE, accessors, *info_.statement, info_.root_idx, *info_.mapping);
  }

public:
  mapped_matrix_column(std::string const & scalartype, unsigned int id, node_info info) : mapped_object(scalartype, id, "matrix_column"), binary_leaf(info)
  { }
};

/** @brief Matrix diag
*
*  Maps a diag(matrix_expression) node into the vector of its diagonal elements
*/
class mapped_matrix_diag : public mapped_object, binary_leaf
{
private:
  void postprocess(std::string &res) const
  {
    std::map<std::string, std::string> accessors;
    tools::find_and_replace(res, "#diag_offset", tree_parsing::evaluate(RHS_NODE_TYPE, accessors, *info_.statement, info_.root_idx, *info_.mapping));
    accessors["matrix"] = res;
    res = tree_parsing::evaluate(LHS_NODE_TYPE, accessors, *info_.statement, info_.root_idx, *info_.mapping);
  }

public:
  mapped_matrix_diag(std::string const & scalartype, unsigned int id, node_info info) : mapped_object(scalartype, id, "matrix_diag"), binary_leaf(info)
  { }
};

/** @brief Implicit vector
 *
 * Maps an implicit vector
 */
class mapped_implicit_vector : public mapped_object
{
public:
  mapped_implicit_vector(std::string const & scalartype, unsigned int id) : mapped_object(scalartype, id, "implicit_vector")
  { }

  std::string & append_kernel_arguments(std::set<std::string> & /*already_generated*/, std::string & str, unsigned int width) const
  {
    str += generate_value_kernel_argument(utils::append_width(scalartype_, width), name_);
    return str;
  }
};

/** @brief Implicit matrix
 *
 * Maps an implicit matrix
 */
class mapped_implicit_matrix : public mapped_object
{
public:
  mapped_implicit_matrix(std::string const & scalartype, unsigned int id) : mapped_object(scalartype, id, "implicit_matrix")
  { }

  std::string & append_kernel_arguments(std::set<std::string> & /*already_generated*/, std::string & str, unsigned int width) const
  {
    str += generate_value_kernel_argument(utils::append_width(scalartype_, width), name_);
    return str;
  }
};

}

}
#endif
