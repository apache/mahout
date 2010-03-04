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

package org.apache.mahout.classifier.discriminative;

/**
 * This exception is thrown in case training fails. E.g. training with an algorithm
 * that can find linear separating hyperplanes only on a training set that is not
 * linearly separable.
 * */
public class TrainingException extends Exception {
  
  /**
   * Init with message string describing the cause of the exception.
   * */
  public TrainingException(String message) {
    super(message);
  }
}
