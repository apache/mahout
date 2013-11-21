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
package org.apache.mahout.text;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.JobID;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.text.doc.SingleFieldDocument;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.Collections;
import java.util.List;

public class LuceneSegmentInputFormatTest extends AbstractLuceneStorageTest {

  private LuceneSegmentInputFormat inputFormat;
  private JobContext jobContext;
  private Configuration conf;

  @Before
  public void before() throws Exception {
    inputFormat = new LuceneSegmentInputFormat();
    LuceneStorageConfiguration lucene2SeqConf = new
    LuceneStorageConfiguration(getConfiguration(), Collections.singletonList(indexPath1), new Path("output"), "id", Collections.singletonList("field"));
    conf = lucene2SeqConf.serialize();

    jobContext = getJobContext(conf, new JobID());
  }

  @After
  public void after() throws IOException {
    HadoopUtil.delete(conf, indexPath1);
  }

  @Test
  public void testGetSplits() throws IOException, InterruptedException {
    SingleFieldDocument doc1 = new SingleFieldDocument("1", "This is simple document 1");
    SingleFieldDocument doc2 = new SingleFieldDocument("2", "This is simple document 2");
    SingleFieldDocument doc3 = new SingleFieldDocument("3", "This is simple document 3");

    //generate 3 segments
    commitDocuments(getDirectory(getIndexPath1AsFile()), doc1);
    commitDocuments(getDirectory(getIndexPath1AsFile()), doc2);
    commitDocuments(getDirectory(getIndexPath1AsFile()), doc3);

    List<LuceneSegmentInputSplit> splits = inputFormat.getSplits(jobContext);
    Assert.assertEquals(3, splits.size());
  }

  // Use reflection to abstract this incompatibility between Hadoop 1 & 2 APIs.
  private JobContext getJobContext(Configuration conf, JobID jobID) throws
      ClassNotFoundException, NoSuchMethodException, IllegalAccessException,
      InvocationTargetException, InstantiationException {
    Class<? extends JobContext> clazz;
    if (!JobContext.class.isInterface()) {
      clazz = JobContext.class;
    } else {
      clazz = (Class<? extends JobContext>)
          Class.forName("org.apache.hadoop.mapreduce.task.JobContextImpl");
    }
    return clazz.getConstructor(Configuration.class, JobID.class)
        .newInstance(conf, jobID);
  }
}
