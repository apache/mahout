package org.apache.mahout.javacpp.linalg;

import org.apache.mahout.javacpp.viennacl.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;


@Platform(include={"viennacl/linalg/norm_2.hpp","viennacl/vector.hpp"})

public class linalg  {

//    @Name("scalar_expression < const viennacl::vector_base<double>," +
//            " const viennacl::vector_base<double>, viennacl::op_norm_2 > viennacl::linalg::norm_2<double>")
//@Name("linalg::norm_2<double>")
//    public static class norm_2 extends Pointer {
//        static {
//            Loader.load();
//        }
//
//        public norm_2() {
//            allocate();
//        }
//    }
//        @Namespace("viennacl::linalg")
//        public static native double norm_2(@Cast("viennacl::vector_base<double>") VCLVector_double_1 vec); //@Name("scalar_expression < const viennacl::vector_base<double>," +
//                " const viennacl::vector_base<double>, viennacl::op_norm_2 > viennacl::linalg::norm_2<double>")
//                     double norm_2(@ByRef @Cast("vector<double,1>") Object vec);
//    }

}
