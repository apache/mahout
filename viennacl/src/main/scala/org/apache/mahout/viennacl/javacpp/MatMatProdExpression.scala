package org.apache.mahout.viennacl.javacpp

import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.annotation.{Name, Namespace, Platform, Properties}


@Properties(inherit = Array(classOf[Context]),
  value = Array(new Platform(
    library = "jniViennaCL")
  ))
@Namespace("viennacl")
@Name(Array("matrix_expression<const viennacl::matrix_base<double>, " +
  "const viennacl::matrix_base<double>, " +
  "viennacl::op_mat_mat_prod>"))
class MatMatProdExpression extends Pointer {

}
