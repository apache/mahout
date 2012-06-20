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

package org.apache.mahout.vectorizer.encoders;

import com.google.common.base.Charsets;
import org.apache.mahout.math.map.OpenIntIntHashMap;

import com.google.common.base.Preconditions;

public class CachingStaticWordValueEncoder extends StaticWordValueEncoder {
  private final int dataSize;
  private OpenIntIntHashMap[] caches;
//  private TIntIntHashMap[] caches;

  public CachingStaticWordValueEncoder(String name, int dataSize) {
    super(name);
    this.dataSize = dataSize;
    initCaches();
  }

  private void initCaches() {
    this.caches = new OpenIntIntHashMap[getProbes()];
    for (int ii = 0; ii < getProbes(); ii++) {
      caches[ii] = new OpenIntIntHashMap();
    }
  }

  protected OpenIntIntHashMap[] getCaches() {
    return caches;
  }

  @Override
  public void setProbes(int probes) {
    super.setProbes(probes);
    initCaches();
  }

  protected int hashForProbe(String originalForm, int dataSize, String name, int probe) {
    Preconditions.checkArgument(dataSize == this.dataSize,
        "dataSize argument [" + dataSize + "] does not match expected dataSize [" + this.dataSize + ']');
    if (caches[probe].containsKey(originalForm.hashCode())) {
      return caches[probe].get(originalForm.hashCode());
    }
    int hash = hashForProbe(originalForm.getBytes(Charsets.UTF_8), dataSize, name, probe);
    caches[probe].put(originalForm.hashCode(), hash);
    return hash;
  }
}

