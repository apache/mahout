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

package org.apache.mahout.classifier.df.mapreduce;

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.classifier.df.builder.TreeBuilder;
import org.apache.mahout.classifier.df.data.Dataset;

import java.io.IOException;

/**
 * Base class for Mapred mappers. Loads common parameters from the job
 */
public class MapredMapper<KEYIN,VALUEIN,KEYOUT,VALUEOUT> extends Mapper<KEYIN,VALUEIN,KEYOUT,VALUEOUT> {
  
  private boolean noOutput;
  
  private TreeBuilder treeBuilder;
  
  private Dataset dataset;
  
  /**
   * 
   * @return whether the mapper does estimate and output predictions
   */
  protected boolean isOutput() {
    return !noOutput;
  }
  
  protected TreeBuilder getTreeBuilder() {
    return treeBuilder;
  }
  
  protected Dataset getDataset() {
    return dataset;
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    
    Configuration conf = context.getConfiguration();
    
    configure(!Builder.isOutput(conf), Builder.getTreeBuilder(conf), Builder
        .loadDataset(conf));
  }
  
  /**
   * Useful for testing
   */
  protected void configure(boolean noOutput, TreeBuilder treeBuilder, Dataset dataset) {
    Preconditions.checkArgument(treeBuilder != null, "TreeBuilder not found in the Job parameters");
    this.noOutput = noOutput;
    this.treeBuilder = treeBuilder;
    this.dataset = dataset;
  }
}
