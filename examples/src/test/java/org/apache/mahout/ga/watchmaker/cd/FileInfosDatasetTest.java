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

package org.apache.mahout.ga.watchmaker.cd;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.iterator.FileLineIterable;

import com.google.common.io.Resources;
import org.apache.mahout.examples.MahoutTestCase;
import org.junit.Test;

import java.io.File;

public final class FileInfosDatasetTest extends MahoutTestCase {

  @Test
  public void testRanges() throws Exception {
    FileSystem fs = FileSystem.get(new Configuration());
    Path inpath = fs.makeQualified(new Path(Resources.getResource("wdbc").toString()));
    
    DataSet dataset = FileInfoParser.parseFile(fs, inpath);
    DataSet.initialize(dataset);

    DataLine dl = new DataLine();
    for (CharSequence line : new FileLineIterable(new File(Resources.getResource("wdbc/wdbc.data").getPath()))) {
      dl.set(line);
      for (int index = 0; index < dataset.getNbAttributes(); index++) {
        if (dataset.isNumerical(index)) {
          CDMutationTest.assertInRange(dl.getAttribute(index), dataset.getMin(index), dataset
              .getMax(index));
        } else {
          CDMutationTest.assertInRange(dl.getAttribute(index), 0, dataset.getNbValues(index));
        }
      }
    }
  }


}
