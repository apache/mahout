package org.apache.mahout.viennacl.vcl.javacpp

import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.annotation.{Name, Namespace, Platform, Properties}


@Properties(inherit = Array(classOf[Context]),
  value = Array(new Platform(
    include = Array("matrix.hpp"),
    library = "jniViennaCL")
  ))
@Namespace("viennacl")
@Name(Array("matrix_expression<const viennacl::matrix_base<double>, " +
  "const viennacl::matrix_base<double>, " +
  "viennacl::op_trans>"))
class MatrixTransExpression extends Pointer {

}
