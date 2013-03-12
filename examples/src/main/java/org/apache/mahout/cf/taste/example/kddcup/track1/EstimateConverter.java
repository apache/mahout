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

package org.apache.mahout.cf.taste.example.kddcup.track1;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class EstimateConverter {

  private static final Logger log = LoggerFactory.getLogger(EstimateConverter.class);

  private EstimateConverter() {}

  public static byte convert(double estimate, long userID, long itemID) {
    if (Double.isNaN(estimate)) {
      log.warn("Unable to compute estimate for user {}, item {}", userID, itemID);
      return 0x7F;
    } else {
      int scaledEstimate = (int) (estimate * 2.55);
      if (scaledEstimate > 255) {
        scaledEstimate = 255;
      } else if (scaledEstimate < 0) {
        scaledEstimate = 0;
      }
      return (byte) scaledEstimate;
    }
  }
}
