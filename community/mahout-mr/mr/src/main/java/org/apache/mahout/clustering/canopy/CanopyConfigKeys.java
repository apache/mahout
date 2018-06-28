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

package org.apache.mahout.clustering.canopy;

import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.distance.DistanceMeasure;

@Deprecated
public final class CanopyConfigKeys {

  private CanopyConfigKeys() {}

  public static final String T1_KEY = "org.apache.mahout.clustering.canopy.t1";

  public static final String T2_KEY = "org.apache.mahout.clustering.canopy.t2";

  public static final String T3_KEY = "org.apache.mahout.clustering.canopy.t3";

  public static final String T4_KEY = "org.apache.mahout.clustering.canopy.t4";

  // keys used by Driver, Mapper, Combiner & Reducer
  public static final String DISTANCE_MEASURE_KEY = "org.apache.mahout.clustering.canopy.measure";

  public static final String CF_KEY = "org.apache.mahout.clustering.canopy.canopyFilter";

  /**
   * Create a {@link CanopyClusterer} from the Hadoop configuration.
   *
   * @param configuration Hadoop configuration
   *
   * @return CanopyClusterer
   */
  public static CanopyClusterer configureCanopyClusterer(Configuration configuration) {
    double t1 = Double.parseDouble(configuration.get(T1_KEY));
    double t2 = Double.parseDouble(configuration.get(T2_KEY));

    DistanceMeasure measure = ClassUtils.instantiateAs(configuration.get(DISTANCE_MEASURE_KEY), DistanceMeasure.class);
    measure.configure(configuration);

    CanopyClusterer canopyClusterer = new CanopyClusterer(measure, t1, t2);

    String d = configuration.get(T3_KEY);
    if (d != null) {
      canopyClusterer.setT3(Double.parseDouble(d));
    }

    d = configuration.get(T4_KEY);
    if (d != null) {
      canopyClusterer.setT4(Double.parseDouble(d));
    }
    return canopyClusterer;
  }

}
