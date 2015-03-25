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

import java.util.Arrays;

import com.google.common.base.Preconditions;
import org.apache.mahout.math.map.OpenIntIntHashMap;

public class CachingStaticWordValueEncoder extends StaticWordValueEncoder {

  private final int dataSize;
  private OpenIntIntHashMap[] caches;

  public CachingStaticWordValueEncoder(String name, int dataSize) {
    super(name);
    this.dataSize = dataSize;
    initCaches();
  }

  private void initCaches() {
    caches = new OpenIntIntHashMap[getProbes()];
    for (int probe = 0; probe < getProbes(); probe++) {
      caches[probe] = new OpenIntIntHashMap();
    }
  }

  OpenIntIntHashMap[] getCaches() {
    return caches;
  }

  @Override
  public void setProbes(int probes) {
    super.setProbes(probes);
    initCaches();
  }

  @Override
  protected int hashForProbe(byte[] originalForm, int dataSize, String name, int probe) {
    Preconditions.checkArgument(dataSize == this.dataSize,
        "dataSize argument [" + dataSize + "] does not match expected dataSize [" + this.dataSize + ']');
    int originalHashcode = Arrays.hashCode(originalForm);
    if (caches[probe].containsKey(originalHashcode)) {
      return caches[probe].get(originalHashcode);
    }
    int hash = super.hashForProbe(originalForm, dataSize, name, probe);
    caches[probe].put(originalHashcode, hash);
    return hash;
  }
}

