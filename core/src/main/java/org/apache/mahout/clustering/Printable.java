/* Licensed to the Apache Software Foundation (ASF) under one or more
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
package org.apache.mahout.clustering;


/**
 * Implementations of this interface have a printable representation. This representation 
 * may be enhanced by an optional Vector label bindings dictionary.
 *
 */
public interface Printable {

  /**
   * Produce a custom, printable representation of the receiver.
   * 
   * @param bindings an optional String[] containing labels used to format the primary 
   *    Vector/s of this implementation.
   * @return a String
   */
  public String asFormatString(String[] bindings);

  /**
   * Produce a printable representation of the receiver using Json. (Label bindings
   * are transient and not part of the Json representation)
   * 
   * @return a Json String
   */
  public String asJsonString();

}
