package org.apache.mahout.viennacl.omp.javacpp

import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.annotation.{Name, Namespace, Platform, Properties}


@Properties(inherit = Array(classOf[Context]),
  value = Array(new Platform(
    library = "jniViennaCL")
  ))
@Namespace("viennacl")
@Name(Array("matrix_expression<const viennacl::compressed_matrix<double>, " +
  "const viennacl::compressed_matrix<double>, " +
  "viennacl::op_prod>"))
class ProdExpression extends Pointer {

}
