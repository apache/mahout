/**
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

package org.apache.mahout.examples;

/**
 * This class should not exist. It's here to work around some bizarre problem in Maven
 * dependency management wherein it can see methods in {@link org.apache.mahout.common.MahoutTestCase}
 * but not constants. Duplicated here to make it jive.
 */
public abstract class MahoutTestCase extends org.apache.mahout.common.MahoutTestCase {

  /** "Close enough" value for floating-point comparisons. */
  public static final double EPSILON = 0.000001;

}
