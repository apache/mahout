/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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
  final val PLATFORM_HEADER_FILE_ONE = "viennacl/HelloNative.h"
}