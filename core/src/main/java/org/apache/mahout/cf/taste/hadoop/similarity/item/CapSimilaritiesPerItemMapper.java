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

package org.apache.mahout.cf.taste.hadoop.similarity.item;

import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.cf.taste.hadoop.EntityEntityWritable;

/**
 * maps out all pairs of similar items so that all similar items for an item can be collected in
 * the {@link CapSimilaritiesPerItemReducer}
 *
 */
public class CapSimilaritiesPerItemMapper
    extends Mapper<EntityEntityWritable,DoubleWritable,CapSimilaritiesPerItemKeyWritable,SimilarItemWritable> {

  @Override
  protected void map(EntityEntityWritable itemPair, DoubleWritable similarity, Context ctx)
      throws IOException, InterruptedException {

    long itemIDA = itemPair.getAID();
    long itemIDB = itemPair.getBID();
    double value = similarity.get();

    ctx.write(new CapSimilaritiesPerItemKeyWritable(itemIDA, value), new SimilarItemWritable(itemIDB, value));
    ctx.write(new CapSimilaritiesPerItemKeyWritable(itemIDB, value), new SimilarItemWritable(itemIDA, value));
  }
}
