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
import org.apache.hadoop.util.ToolRunner;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.util.Version;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.commandline.DefaultOptionCreator;

import java.util.ArrayList;
import java.util.List;

import static java.util.Arrays.asList;

/**
 * Driver class for the lucene2seq program. Converts text contents of stored fields of a lucene index into a Hadoop
 * SequenceFile. The key of the sequence file is the document ID and the value is the concatenated text of the specified
 * stored field(s).
 */
public class SequenceFilesFromLuceneStorageDriver extends AbstractJob {

  static final String OPTION_LUCENE_DIRECTORY = "dir";
  static final String OPTION_ID_FIELD = "idField";
  static final String OPTION_FIELD = "fields";
  static final String OPTION_QUERY = "query";
  static final String OPTION_MAX_HITS = "maxHits";

  static final Query DEFAULT_QUERY = new MatchAllDocsQuery();
  static final int DEFAULT_MAX_HITS = Integer.MAX_VALUE;

  static final String SEPARATOR_FIELDS = ",";
  static final String QUERY_DELIMITER = "'";

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new SequenceFilesFromLuceneStorageDriver(), args);
  }

  @Override
  public int run(String[] args) throws Exception {
    addOutputOption();

    addOption(OPTION_LUCENE_DIRECTORY, "d", "Lucene directory / directories. Comma separated.", true);
    addOption(OPTION_ID_FIELD, "i", "The field in the index containing the id", true);
    addOption(OPTION_FIELD, "f", "The stored field(s) in the index containing text", true);

    addOption(OPTION_QUERY, "q", "(Optional) Lucene query. Defaults to " + DEFAULT_QUERY.getClass().getSimpleName());
    addOption(OPTION_MAX_HITS, "n", "(Optional) Max hits. Defaults to " + DEFAULT_MAX_HITS);
    addOption(DefaultOptionCreator.methodOption().create());

    if (parseArguments(args) == null) {
      return -1;
    }

    Configuration configuration = getConf();
    if (configuration == null) {
      configuration = new Configuration();
    }

    String[] paths = getOption(OPTION_LUCENE_DIRECTORY).split(",");
    List<Path> indexPaths = new ArrayList<Path>();
    for (String path : paths) {
      indexPaths.add(new Path(path));
    }

    Path sequenceFilesOutputPath = new Path((getOption(DefaultOptionCreator.OUTPUT_OPTION)));

    String idField = getOption(OPTION_ID_FIELD);
    String fields = getOption(OPTION_FIELD);

    LuceneStorageConfiguration lucene2SeqConf = newLucene2SeqConfiguration(configuration,
            indexPaths,
            sequenceFilesOutputPath,
            idField,
            asList(fields.split(SEPARATOR_FIELDS)));

    Query query = DEFAULT_QUERY;
    if (hasOption(OPTION_QUERY)) {
      try {
        String queryString = getOption(OPTION_QUERY).replaceAll(QUERY_DELIMITER, "");
        QueryParser queryParser = new QueryParser(Version.LUCENE_35, queryString, new StandardAnalyzer(Version.LUCENE_35));
        query = queryParser.parse(queryString);
      } catch (ParseException e) {
        throw new IllegalArgumentException(e.getMessage(), e);
      }
    }
    lucene2SeqConf.setQuery(query);

    int maxHits = DEFAULT_MAX_HITS;
    if (hasOption(OPTION_MAX_HITS)) {
      String maxHitsString = getOption(OPTION_MAX_HITS);
      maxHits = Integer.valueOf(maxHitsString);
    }
    lucene2SeqConf.setMaxHits(maxHits);

    if (hasOption(DefaultOptionCreator.METHOD_OPTION) && getOption(DefaultOptionCreator.METHOD_OPTION).equals("sequential")) {
      new SequenceFilesFromLuceneStorage().run(lucene2SeqConf);
    } else {
      new SequenceFilesFromLuceneStorageMRJob().run(lucene2SeqConf);
    }
    return 0;
  }

  public LuceneStorageConfiguration newLucene2SeqConfiguration(Configuration configuration,
                                                               List<Path> indexPaths,
                                                               Path sequenceFilesOutputPath,
                                                               String idField,
                                                               List<String> fields) {
    return new LuceneStorageConfiguration(
            configuration,
            indexPaths,
            sequenceFilesOutputPath,
            idField,
            fields);
  }
}
