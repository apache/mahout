package org.apache.mahout.javaCppTest

import org.bytedeco.javacpp._
import org.bytedeco.javacpp.annotation._


@Platform(include=Array(HelloNative.PLATFORM_HEADER_FILE_ONE))
@Namespace("HelloNative")
class HelloNative extends Pointer {

  Loader.load()

  allocate()

  @native def allocate(): Unit

  @native @StdString def get_property(): String
  @native def set_property(property: String ): Unit

  // to access the member variable directly
  @native @StdString def property(): String
  @native def property(property: String): Unit

}

object HelloNative {
  final val PLATFORM_HEADER_FILE_ONE = "HelloNative.h"
}