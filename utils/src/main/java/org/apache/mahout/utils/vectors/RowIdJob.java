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

package org.apache.mahout.utils.vectors;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RowIdJob extends AbstractJob {
  private static final Logger log = LoggerFactory.getLogger(RowIdJob.class);

  @Override
  public int run(String[] strings) throws Exception {
    Configuration conf = getConf();
    FileSystem fs = FileSystem.get(conf);
    Path inputPath = fs.makeQualified(new Path(conf.get("mapred.input.dir")));
    Path outputPath = fs.makeQualified(new Path(conf.get("mapred.output.dir")));
    Path indexPath = new Path(outputPath, "docIndex");
    Path matrixPath = new Path(outputPath, "matrix");
    SequenceFile.Writer indexWriter = SequenceFile.createWriter(fs,
                                                                conf,
                                                                indexPath,
                                                                IntWritable.class,
                                                                Text.class);
    SequenceFile.Writer matrixWriter = SequenceFile.createWriter(fs,
                                                                 conf,
                                                                 matrixPath,
                                                                 IntWritable.class,
                                                                 VectorWritable.class);
    IntWritable docId = new IntWritable();
    Text inputKey = new Text();
    VectorWritable v = new VectorWritable();

    int i = 0;
    for (FileStatus status : fs.listStatus(inputPath)) {
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, status.getPath(), conf);
      while (reader.next(inputKey, v)) {
        docId.set(i);
        indexWriter.append(docId, inputKey);
        matrixWriter.append(docId, v);
        i++;
      }
      reader.close();
    }
    
    int numCols = v.get().size();
    matrixWriter.close();
    indexWriter.close();
    log.info("Wrote out matrix with {} rows and {} columns to " + matrixPath, i, numCols);
    return 0;
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new RowIdJob(), args);
  }

}
