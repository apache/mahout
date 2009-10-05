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

package org.apache.mahout.df.mapred;

import java.io.IOException;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.mahout.df.builder.TreeBuilder;
import org.apache.mahout.df.data.Dataset;

/**
 * Base class for Mapred mappers. Loads common parameters from the job
 */
public class MapredMapper extends MapReduceBase {

  private boolean noOutput;

  private boolean oobEstimate;

  private TreeBuilder treeBuilder;

  private Dataset dataset;

  /**
   * 
   * @return if false, the mapper does not output
   */
  protected boolean isOobEstimate() {
    return oobEstimate;
  }

  /**
   * 
   * @return if false, the mapper does not estimate and output predictions
   */
  protected boolean isNoOutput() {
    return noOutput;
  }

  protected TreeBuilder getTreeBuilder() {
    return treeBuilder;
  }

  protected Dataset getDataset() {
    return dataset;
  }

  @Override
  public void configure(JobConf conf) {
    super.configure(conf);

    try {
      configure(!Builder.isOutput(conf), Builder.isOobEstimate(conf), Builder
          .getTreeBuilder(conf), Builder.loadDataset(conf));
    } catch (IOException e) {
      throw new IllegalStateException("Exception caught while configuring the mapper: ", e);
    }
  }

  /**
   * Useful for testing
   * 
   * @param noOutput
   * @param oobEstimate
   * @param treeBuilder
   * @param dataset
   */
  protected void configure(boolean noOutput, boolean oobEstimate,
      TreeBuilder treeBuilder, Dataset dataset) {
    this.noOutput = noOutput;
    this.oobEstimate = oobEstimate;

    if (treeBuilder == null) {
      throw new IllegalArgumentException("TreeBuilder not found in the Job parameters");
    }
    this.treeBuilder = treeBuilder;

    this.dataset = dataset;
  }
}
