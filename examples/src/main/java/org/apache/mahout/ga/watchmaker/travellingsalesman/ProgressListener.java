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

/**
 * Call-back interface for keeping track of the progress of a
 * {@link TravellingSalesmanStrategy} implementation.
 * 
 * <br>
 * The original code is from <b>the Watchmaker project</b>
 * (https://watchmaker.dev.java.net/).
 */
public interface ProgressListener
{
    /**
     * Call-back method that informs the implementing object
     * of the current completion percentage.
     * @param percentComplete A percentage between 0 and 100.
     */
    void updateProgress(double percentComplete);
}
