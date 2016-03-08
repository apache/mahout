/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.math.drm

object CacheHint extends Enumeration {

  type CacheHint = Value

  val NONE,
  DISK_ONLY,
  DISK_ONLY_2,
  MEMORY_ONLY,
  MEMORY_ONLY_2,
  MEMORY_ONLY_SER,
  MEMORY_ONLY_SER_2,
  MEMORY_AND_DISK,
  MEMORY_AND_DISK_2,
  MEMORY_AND_DISK_SER,
  MEMORY_AND_DISK_SER_2 = Value

}
