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

import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.text.doc.SingleFieldDocument;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.Iterator;
import java.util.Map;

import static java.util.Arrays.asList;

public class SequenceFilesFromLuceneStorageMRJobTest extends AbstractLuceneStorageTest {

  private SequenceFilesFromLuceneStorageMRJob lucene2seq;
  private LuceneStorageConfiguration lucene2SeqConf;

  @Before
  public void before() throws IOException {
    lucene2seq = new SequenceFilesFromLuceneStorageMRJob();
    Configuration configuration = getConfiguration();
    Path seqOutputPath = new Path(getTestTempDirPath(), "seqOutputPath");//don't make the output directory
    lucene2SeqConf = new LuceneStorageConfiguration(configuration, asList(getIndexPath1(), getIndexPath2()),
            seqOutputPath, SingleFieldDocument.ID_FIELD, asList(SingleFieldDocument.FIELD));
  }

  @After
  public void after() throws IOException {
    HadoopUtil.delete(lucene2SeqConf.getConfiguration(), lucene2SeqConf.getSequenceFilesOutputPath());
    HadoopUtil.delete(lucene2SeqConf.getConfiguration(), lucene2SeqConf.getIndexPaths());
  }

  @Test
  public void testRun() throws IOException {
    //Two commit points, each in two diff. Directories
    commitDocuments(getDirectory(getIndexPath1AsFile()), docs.subList(0, 500));
    commitDocuments(getDirectory(getIndexPath1AsFile()), docs.subList(1000, 1500));

    commitDocuments(getDirectory(getIndexPath2AsFile()), docs.subList(500, 1000));
    commitDocuments(getDirectory(getIndexPath2AsFile()), docs.subList(1500, 2000));
    commitDocuments(getDirectory(getIndexPath1AsFile()), misshapenDocs);
    lucene2seq.run(lucene2SeqConf);

    Iterator<Pair<Text, Text>> iterator = lucene2SeqConf.getSequenceFileIterator();
    Map<String, Text> map = Maps.newHashMap();
    while (iterator.hasNext()) {
      Pair<Text, Text> next = iterator.next();
      map.put(next.getFirst().toString(), next.getSecond());
    }
    assertEquals(docs.size() + misshapenDocs.size(), map.size());
    for (SingleFieldDocument doc : docs) {
      Text value = map.get(doc.getId());
      assertNotNull(value);
      assertEquals(value.toString(), doc.getField());
    }
    for (SingleFieldDocument doc : misshapenDocs) {
      Text value = map.get(doc.getId());
      assertNotNull(value);
      assertEquals(value.toString(), doc.getField());
    }
  }
}
