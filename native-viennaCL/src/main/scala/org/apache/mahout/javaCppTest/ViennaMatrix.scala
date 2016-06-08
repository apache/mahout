package org.apache.mahout.javaCppTest

import org.bytedeco.javacpp.{Loader, Pointer}
import org.bytedeco.javacpp.annotation.{Namespace, Platform, StdString}

/**
  * Created by andy on 6/7/16.
  */
//@Platform(include=Array(HelloNative.PLATFORM_HEADER_FILE_ONE))
//@Namespace("viennacl")
class VieannaMatrix extends Pointer {

//  Loader.load()
//
//  allocate()
//
//  @native def allocate(): Unit


//
//  @native @StdString def get_property(): String
//  @native def set_property(property: String ): Unit
//
//  // to access the member variable directly
//  @native @StdString def property(): String
//  @native def property(property: String): Unit



}

object VieannaMatrix {
  final val PLATFORM_HEADER_FILE_ONE = "Matrix.hpp"
}
