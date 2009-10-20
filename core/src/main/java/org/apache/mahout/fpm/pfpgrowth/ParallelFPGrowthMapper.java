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

package org.apache.mahout.fpm.pfpgrowth;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.IntegerTuple;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.Parameters;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;
import java.util.regex.Pattern;

/**
 * {@link ParallelFPGrowthMapper} maps each transaction to all unique items
 * groups in the transaction. mapper outputs the group id as key and the
 * transaction as value
 * 
 */
public class ParallelFPGrowthMapper extends
    Mapper<LongWritable, Text, LongWritable, IntegerTuple> {

  private final Map<String, Integer> fMap = new HashMap<String, Integer>();

  private final Map<Integer, Long> gListInt = new HashMap<Integer, Long>();

  private Pattern splitter = null;

  @Override
  protected void map(LongWritable offset, Text input, Context context)
      throws IOException, InterruptedException {

    String[] items = splitter.split(input.toString());

    List<Integer> itemSet = new ArrayList<Integer>();
    for (String item : items) // remove items not in the fList
    {
      if (fMap.containsKey(item) && item.trim().length() != 0)
        itemSet.add(fMap.get(item));
    }

    Collections.sort(itemSet);

    Integer[] prunedItems = itemSet.toArray(new Integer[itemSet.size()]);
    
    Set<Long> groups = new HashSet<Long>();
    for (int j = prunedItems.length - 1; j >= 0; j--) { // generate group
                                                        // dependent
                                                        // shards
      Integer item = prunedItems[j];
      Long groupID = gListInt.get(item);
      if (groups.contains(groupID) == false) {
        Integer[] tempItems = new Integer[j + 1];
        System.arraycopy(prunedItems, 0, tempItems, 0, j + 1);
        context
            .setStatus("Parallel FPGrowth: Generating Group Dependent transactions for: "
                + item);
        context.write(new LongWritable(groupID), new IntegerTuple(tempItems));
      }
      groups.add(groupID);
    }

  }

  @Override
  protected void setup(Context context) throws IOException,
      InterruptedException {
    super.setup(context);
    Parameters params = Parameters.fromString(context.getConfiguration().get(
        "pfp.parameters", ""));

    int i = 0;
    for (Pair<String, Long> e : PFPGrowth.deserializeList(params, "fList",
        context.getConfiguration())) {
      fMap.put(e.getFirst(), i++);
    }
    
    for (Entry<String, Long> e : PFPGrowth.deserializeMap(params, "gList",
        context.getConfiguration()).entrySet()) {
      gListInt.put(fMap.get(e.getKey()), e.getValue());
    }
    splitter = Pattern.compile(params.get("splitPattern", PFPGrowth.SPLITTER
        .toString()));

  }
}
