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

package org.apache.mahout.classifier.bayes;

import java.util.Collection;

/**
 * The Datastore interface for the {@link Algorithm} to use
 * 
 */
public interface Datastore {
  /**
   * Gets a double value from the Matrix pointed to by the {@code matrixName} from its cell pointed to by
   * the {@code row} and {@code column} string
   *
   * @param matrixName
   * @param row
   * @param column
   * @return double value
   * @throws InvalidDatastoreException
   */
  double getWeight(String matrixName, String row, String column) throws InvalidDatastoreException;
  
  /**
   * Gets a double value from the Vector pointed to by the {@code vectorName} from its cell pointed to by
   * the {@code index}
   *
   * @param vectorName
   * @param index
   * @return double value
   * @throws InvalidDatastoreException
   */
  double getWeight(String vectorName, String index) throws InvalidDatastoreException;
  
  /**
   * get the keySet of a given Matrix/Vector as given by {@code name}
   *
   * @param name
   * @return Collection of keys of Matrix/Vector
   * @throws InvalidDatastoreException
   */
  Collection<String> getKeys(String name) throws InvalidDatastoreException;
  
  /**
   * Initializes the  and loads the model into memory/cache if necessary
   * 
   * @throws InvalidDatastoreException
   */
  void initialize() throws InvalidDatastoreException;
}
