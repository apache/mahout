package org.apache.mahout.viennacl.javacpp

import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.annotation.{Name, Namespace, Platform, Properties}


@Properties(inherit = Array(classOf[Context]),
  value = Array(new Platform(
    include = Array("forwards.h"),
    library = "jniViennaCL")
  ))
@Namespace("viennacl")
@Name(Array("vector_expression<const viennacl::vector_base<double>," +
  "const double, viennacl::op_mult >"))
class VecMultExpression extends Pointer {

}
