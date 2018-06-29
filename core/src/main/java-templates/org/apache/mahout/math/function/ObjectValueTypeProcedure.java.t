/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.mahout.math.function;

/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/

/**
 * Interface that represents a procedure object: a procedure that takes two arguments and does not return a value.
 *
 */
public interface Object${valueTypeCap}Procedure<T> {

  /**
   * Applies a procedure to two arguments. Optionally can return a boolean flag to inform the object calling the
   * procedure.
   *
   * <p>Example: forEach() methods often use procedure objects. To signal to a forEach() method whether iteration should
   * continue normally or terminate (because for example a matching element has been found), a procedure can return
   * <tt>false</tt> to indicate termination and <tt>true</tt> to indicate continuation.
   *
   * @param first  first argument passed to the procedure.
   * @param second second argument passed to the procedure.
   * @return a flag  to inform the object calling the procedure.
   */
  boolean apply(T first, ${valueType} second);
}
