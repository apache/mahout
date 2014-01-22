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

package org.apache.mahout.fpm.pfpgrowth.convertors;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Reducer;

/**
 * Updates the Context object of a {@link Reducer} class
 * 
 * @param <IK>
 * @param <IV>
 * @param <K>
 * @param <V>
 */
public class ContextStatusUpdater<IK extends Writable,IV extends Writable,K extends Writable,V extends Writable>
    implements StatusUpdater {
  
  private static final long PERIOD = 10000; // Update every 10 seconds
  
  private final Reducer<IK,IV,K,V>.Context context;
  
  private long time = System.currentTimeMillis();
  
  public ContextStatusUpdater(Reducer<IK,IV,K,V>.Context context) {
    this.context = context;
  }
  
  @Override
  public void update(String status) {
    long curTime = System.currentTimeMillis();
    if (curTime - time > PERIOD && context != null) {
      time = curTime;
      context.setStatus("Processing FPTree: " + status);
    }
    
  }
  
}
