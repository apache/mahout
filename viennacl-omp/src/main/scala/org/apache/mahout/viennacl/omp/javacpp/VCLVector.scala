package org.apache.mahout.viennacl.omp.javacpp

import org.bytedeco.javacpp._
import org.bytedeco.javacpp.annotation._


@Properties(inherit = Array(classOf[Context]),
  value = Array(new Platform(
    library="jniViennaCL"
  )))
@Name(Array("viennacl::vector<double>"))
final class VCLVector(defaultCtr: Boolean = true) extends VectorBase {

  if (defaultCtr) allocate()

  def this(){
    this(false)
    allocate()
  }

  def this(i: Int){
    this(false)
    allocate(i)
  }

  def this(size: Int, ctx: Context = new Context(Context.MAIN_MEMORY)) {
    this(false)
    allocate(size, ctx)
  }

  def this(@Const @ByRef ve: VecMultExpression) {
    this(false)
    allocate(ve)
  }

  def this(@Const @ByRef vmp: MatVecProdExpression) {
    this(false)
    allocate(vmp)
  }

//   conflicting with the next signature as MemHandle is a pointer and so is a DoublePointer..
//   leave out for now.
//
//   def this(h: MemHandle , vec_size: Int, vec_start: Int = 0, vec_stride: Int = 1) {
//      this(false)
//      allocate(h, vec_size, vec_start, vec_stride)
//    }

  def this(ptr_to_mem: DoublePointer,
           @Cast(Array("viennacl::memory_types"))mem_type : Int,
           vec_size: Int,
           start: Int = 0,
           stride: Int = 1) {

    this(false)
    allocate(ptr_to_mem, mem_type, vec_size, start, stride)
    ptrs += ptr_to_mem
  }

  def this(@Const @ByRef vc: VCLVector) {
    this(false)
    allocate(vc)
  }
  def this(@Const @ByRef vb: VectorBase) {
    this(false)
    allocate(vb)
  }

  @native protected def allocate()

  @native protected def allocate(size: Int)

  @native protected def allocate(size: Int, @ByVal ctx: Context)

  @native protected def allocate(@Const @ByRef ve: VecMultExpression)

  @native protected def allocate(@Const @ByRef ve: MatVecProdExpression)

  @native protected def allocate(@Const @ByRef vb: VCLVector)

  @native protected def allocate(@Const @ByRef vb: VectorBase)


//  @native protected def allocate(h: MemHandle , vec_size: Int,
//                                 vec_start: Int,
//                                 vec_stride: Int)

  @native protected def allocate(ptr_to_mem: DoublePointer,
                                 @Cast(Array("viennacl::memory_types"))mem_type : Int,
                                 vec_size: Int,
                                 start: Int,
                                 stride: Int)

  @Name(Array("viennacl::vector<double>::self_type"))
  def selfType:VectorBase = this.asInstanceOf[VectorBase]


  @native def switch_memory_context(@ByVal context: Context): Unit

//  Swaps the handles of two vectors by swapping the OpenCL handles only, no data copy.
//  @native def fast_swap(@ByVal other: VCLVector): VectorBase

// add this operator in for tests many more can be added
//  @Name(Array("operator*"))
//  @native @ByPtr def *(i: Int): VectorMultExpression



}

object VCLVector {
  Context.loadLib()
}


