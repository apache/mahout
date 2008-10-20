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

package org.apache.mahout.ga.watchmaker.travellingsalesman;

import java.util.List;

/**
 * Strategy interface for providing distances between cities in the
 * Travelling Salesman problem.
 * 
 * <br>
 * The original code is from <b>the Watchmaker project</b>
 * (https://watchmaker.dev.java.net/).
 */
public interface DistanceLookup {

    /**
     * @return The list of cities that this object knows about.
     */
    List<String> getKnownCities();

    /**
     * Looks-up the distance between two cities.
     * @param startingCity The city to start from.
     * @param destinationCity The city to end in.
     * @return The distance (in kilometres) between the two cities.
     */
    int getDistance(String startingCity, String destinationCity);
}
