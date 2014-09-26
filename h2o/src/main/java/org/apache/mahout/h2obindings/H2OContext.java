/*
 *  Licensed to the Apache Software Foundation (ASF) under one or more
 *  contributor license agreements.  See the NOTICE file distributed with
 *  this work for additional information regarding copyright ownership.
 *  The ASF licenses this file to You under the Apache License, Version 2.0
 *  (the "License"); you may not use this file except in compliance with
 *  the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package org.apache.mahout.h2obindings;

import water.H2O;

/**
 * Context to an H2O Cloud.
 */
public class H2OContext {
  /**
   * Class constructor.
   *
   * @param masterURL The cloud name (name of cluster) to which all the H2O
   *                   worker nodes "join into". This is not a hostname or IP
   *                   address of a server, but a string which all cluster
   *                   members agree on.
   */
  public H2OContext(String masterURL) {
    H2O.main(new String[]{"-md5skip", "-name", masterURL});
    H2O.joinOthers();
  }
}
