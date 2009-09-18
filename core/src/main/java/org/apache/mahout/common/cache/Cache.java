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

package org.apache.mahout.common.cache;

public interface Cache<K, V> {
  /**
   * Gets the Value from the Cache, If the object doesnt exist default behaviour
   * is to return null.
   * 
   * @param key
   * @return V
   */
  V get(K key);

  /**
   * returns true if the Cache contains the key
   * @param key
   * @return boolean
   */
  boolean contains(K key);

  /**
   * puts the key and its value into the cache
   * @param key
   * @param value
   */
  void set(K key, V value);

  /**
   * returns the current size of the cache
   * @return long
   */
  long size();

  /**
   * returns the total capacity of the cache defined at contruction time
   * @return long
   */
  long capacity();
}
