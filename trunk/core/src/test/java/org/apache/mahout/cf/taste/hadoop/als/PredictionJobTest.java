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

package org.apache.mahout.cf.taste.hadoop.als;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.math.hadoop.MathHelper;
import org.junit.Test;

import java.io.File;

public class PredictionJobTest extends TasteTestCase {

  @Test
  public void smallIntegration() throws Exception {

    File pairs = getTestTempFile("pairs.txt");
    File userFeatures = getTestTempFile("userFeatures.seq");
    File itemFeatures = getTestTempFile("itemFeatures.seq");
    File tempDir = getTestTempDir("temp");
    File outputDir = getTestTempDir("out");
    outputDir.delete();

    Configuration conf = new Configuration();
    Path inputPath = new Path(pairs.getAbsolutePath());
    FileSystem fs = FileSystem.get(inputPath.toUri(), conf);

    MathHelper.writeEntries(new double[][]{
        new double[] {  1.5, -2,    0.3 },
        new double[] { -0.7,  2,    0.6 },
        new double[] { -1,    2.5,  3   } }, fs, conf, new Path(userFeatures.getAbsolutePath()));

    MathHelper.writeEntries(new double [][] {
        new double[] {  2.3,  0.5, 0   },
        new double[] {  4.7, -1,   0.2 },
        new double[] {  0.6,  2,   1.3 } }, fs, conf, new Path(itemFeatures.getAbsolutePath()));

    writeLines(pairs, "0,0", "2,1", "1,0");

    PredictionJob predictor = new PredictionJob();
    predictor.setConf(conf);
    predictor.run(new String[] { "--output", outputDir.getAbsolutePath(), "--pairs", pairs.getAbsolutePath(),
        "--userFeatures", userFeatures.getAbsolutePath(), "--itemFeatures", itemFeatures.getAbsolutePath(),
        "--tempDir", tempDir.getAbsolutePath() });

    FileDataModel dataModel = new FileDataModel(new File(outputDir, "part-r-00000"));
    assertEquals(3, dataModel.getNumUsers());
    assertEquals(2, dataModel.getNumItems());
    assertEquals(2.45f, dataModel.getPreferenceValue(0, 0), EPSILON);
    assertEquals(-6.6f, dataModel.getPreferenceValue(2, 1), EPSILON);
    assertEquals(-0.61f, dataModel.getPreferenceValue(1, 0), EPSILON);
  }

}
