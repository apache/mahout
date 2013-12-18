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

package org.apache.mahout.text;

import com.google.common.base.Strings;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.DocumentStoredFieldVisitor;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.store.IOContext;

import java.io.IOException;
import java.util.List;

import static org.apache.commons.lang.StringUtils.isBlank;

/**
 * Maps document IDs to key value pairs with ID field as the key and the concatenated stored field(s)
 * as value.
 */
public class SequenceFilesFromLuceneStorageMapper extends Mapper<Text, NullWritable, Text, Text> {

  public enum DataStatus { EMPTY_KEY, EMPTY_VALUE, EMPTY_BOTH }

  private LuceneStorageConfiguration l2sConf;
  private SegmentReader segmentReader;

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    Configuration configuration = context.getConfiguration();
    l2sConf = new LuceneStorageConfiguration(configuration);
    LuceneSegmentInputSplit inputSplit = (LuceneSegmentInputSplit) context.getInputSplit();
    SegmentCommitInfo segmentInfo = inputSplit.getSegment(configuration);
    segmentReader = new SegmentReader(segmentInfo, LuceneSeqFileHelper.USE_TERM_INFOS, IOContext.READ);
  }

  @Override
  protected void map(Text key, NullWritable text, Context context) throws IOException, InterruptedException {
    int docId = Integer.valueOf(key.toString());
    DocumentStoredFieldVisitor storedFieldVisitor = l2sConf.getStoredFieldVisitor();
    segmentReader.document(docId, storedFieldVisitor);
    Document document = storedFieldVisitor.getDocument();
    List<String> fields = l2sConf.getFields();
    Text theKey = new Text(Strings.nullToEmpty(document.get(l2sConf.getIdField())));
    Text theValue = new Text();
    LuceneSeqFileHelper.populateValues(document, theValue, fields);
    //if they are both empty, don't write
    if (isBlank(theKey.toString()) && isBlank(theValue.toString())) {
      context.getCounter(DataStatus.EMPTY_BOTH).increment(1);
      return;
    }
    if (isBlank(theKey.toString())) {
      context.getCounter(DataStatus.EMPTY_KEY).increment(1);
    } else if (isBlank(theValue.toString())) {
      context.getCounter(DataStatus.EMPTY_VALUE).increment(1);
    }
    context.write(theKey, theValue);
  }

  @Override
  protected void cleanup(Context context) throws IOException, InterruptedException {
    segmentReader.close();
  }
}
