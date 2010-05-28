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
package org.apache.mahout.df.node;

import org.apache.mahout.df.data.Instance;

/**
 * Custom Leaf node that returns for each instance its own label. Used mainly for testing purposes
 * 
 */
public class MockLeaf extends Leaf {
  
  @Override
  public int classify(Instance instance) {
    return instance.getLabel();
  }
  
  @Override
  protected Type getType() {
    return Type.MOCKLEAF;
  }
  
  @Override
  protected String getString() {
    return "[MockLeaf]";
  }
}
