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

package org.apache.mahout.common;

import java.util.Map;

import com.google.common.collect.Maps;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.StatusReporter;

public final class DummyStatusReporter extends StatusReporter {

  private final Map<Enum<?>, Counter> counters = Maps.newHashMap();
  private final Map<String, Counter> counterGroups = Maps.newHashMap();

  @Override
  public Counter getCounter(Enum<?> name) {
    if (!counters.containsKey(name)) {
      counters.put(name, new DummyCounter());
    }
    return counters.get(name);
  }


  @Override
  public Counter getCounter(String group, String name) {
    if (!counterGroups.containsKey(group + name)) {
      counterGroups.put(group + name, new DummyCounter());
    }
    return counterGroups.get(group+name);
  }

  @Override
  public void progress() {
  }

  @Override
  public void setStatus(String status) {
  }

}
