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

package org.apache.mahout.ga.watchmaker;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.StringUtils;
import org.uncommons.watchmaker.framework.FitnessEvaluator;

import com.google.common.base.Preconditions;

/**
 * <p>
 * Generic Mapper class for fitness evaluation. Works with the following :
 * {@code <key, candidate, key, fitness>}
 * , where :
 * </p>
 * key: position of the current candidate in the input file. <br>
 * candidate: candidate solution to evaluate. <br>
 * fitness: evaluated fitness for the given candidate.
 */
public class EvalMapper extends Mapper<LongWritable,Text,LongWritable,DoubleWritable> {
  
  /** Parameter used to store the "stringified" evaluator */
  public static final String MAHOUT_GA_EVALUATOR = "mahout.ga.evaluator";
  
  private FitnessEvaluator<Object> evaluator;
  
  @Override
  public void setup(Context context) throws IOException, InterruptedException {
    Configuration conf = context.getConfiguration();
    String evlstr = conf.get(MAHOUT_GA_EVALUATOR);
    Preconditions.checkArgument(evlstr != null, "'MAHOUT_GA_EVALUATOR' job parameter not found");
    
    evaluator = StringUtils.fromString(evlstr);
    
    super.setup(context);
  }
  
  @Override
  public void map(LongWritable key,
                  Text value,
                  Context context) throws IOException, InterruptedException {
    Object candidate = StringUtils.fromString(value.toString());
    
    double fitness = evaluator.getFitness(candidate, null);
    
    context.write(key, new DoubleWritable(fitness));
  }
  
}
