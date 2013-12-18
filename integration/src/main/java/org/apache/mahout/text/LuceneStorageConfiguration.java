package org.apache.mahout.text;
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

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DefaultStringifier;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.DocumentStoredFieldVisitor;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import static org.apache.lucene.util.Version.LUCENE_46;

/**
 * Holds all the configuration for {@link SequenceFilesFromLuceneStorage}, which generates a sequence file
 * with id as the key and a content field as value.
 */
public class LuceneStorageConfiguration implements Writable {

  private static final Query DEFAULT_QUERY = new MatchAllDocsQuery();
  private static final int DEFAULT_MAX_HITS = Integer.MAX_VALUE;

  static final String KEY = "org.apache.mahout.text.LuceneIndexToSequenceFiles";

  static final String SEPARATOR_FIELDS = ",";
  static final String SEPARATOR_PATHS = ",";

  private Configuration configuration;
  private List<Path> indexPaths;
  private Path sequenceFilesOutputPath;
  private String idField;
  private List<String> fields;
  private Query query;
  private int maxHits;

  /**
   * Create a configuration bean with all mandatory parameters.
   *
   * @param configuration           Hadoop configuration for writing sequencefiles
   * @param indexPaths              paths to the index
   * @param sequenceFilesOutputPath path to output the sequence file
   * @param idField                 field used for the key of the sequence file
   * @param fields                  field(s) used for the value of the sequence file
   */
  public LuceneStorageConfiguration(Configuration configuration, List<Path> indexPaths, Path sequenceFilesOutputPath,
                                    String idField, List<String> fields) {
    Preconditions.checkArgument(configuration != null, "Parameter 'configuration' cannot be null");
    Preconditions.checkArgument(indexPaths != null, "Parameter 'indexPaths' cannot be null");
    Preconditions.checkArgument(indexPaths != null && !indexPaths.isEmpty(), "Parameter 'indexPaths' cannot be empty");
    Preconditions.checkArgument(sequenceFilesOutputPath != null, "Parameter 'sequenceFilesOutputPath' cannot be null");
    Preconditions.checkArgument(idField != null, "Parameter 'idField' cannot be null");
    Preconditions.checkArgument(fields != null, "Parameter 'fields' cannot be null");
    Preconditions.checkArgument(fields != null && !fields.isEmpty(), "Parameter 'fields' cannot be empty");

    this.configuration = configuration;
    this.indexPaths = indexPaths;
    this.sequenceFilesOutputPath = sequenceFilesOutputPath;
    this.idField = idField;
    this.fields = fields;

    setQuery(DEFAULT_QUERY);
    setMaxHits(DEFAULT_MAX_HITS);
  }

  public LuceneStorageConfiguration() {
    // Used during serialization. Do not use.
  }

  /**
   * Deserializes a {@link LuceneStorageConfiguration} from a {@link Configuration}.
   *
   * @param conf the {@link Configuration} object with a serialized {@link LuceneStorageConfiguration}
   * @throws IOException if deserialization fails
   */
  public LuceneStorageConfiguration(Configuration conf) throws IOException {
    Preconditions.checkNotNull(conf, "Parameter 'configuration' cannot be null");

    String serializedConfigString = conf.get(KEY);

    if (serializedConfigString == null) {
      throw new IllegalArgumentException("Parameter 'configuration' does not contain a serialized " + this.getClass());
    }

    LuceneStorageConfiguration luceneStorageConf = DefaultStringifier.load(conf, KEY, LuceneStorageConfiguration.class);

    this.configuration = conf;
    this.indexPaths = luceneStorageConf.getIndexPaths();
    this.sequenceFilesOutputPath = luceneStorageConf.getSequenceFilesOutputPath();
    this.idField = luceneStorageConf.getIdField();
    this.fields = luceneStorageConf.getFields();
    this.query = luceneStorageConf.getQuery();
    this.maxHits = luceneStorageConf.getMaxHits();
  }

  /**
   * Serializes this object in a Hadoop {@link Configuration}
   *
   * @return a {@link Configuration} object with a String serialization
   * @throws IOException if serialization fails
   */
  public Configuration serialize() throws IOException {
    DefaultStringifier.store(configuration, this, KEY);

    return new Configuration(configuration);
  }

  /**
   * Returns an {@link Iterator} which returns (Text, Text) {@link Pair}s of the produced sequence files.
   *
   * @return iterator
   */
  public Iterator<Pair<Text, Text>> getSequenceFileIterator() {
    return new SequenceFileDirIterable<Text, Text>(sequenceFilesOutputPath, PathType.LIST, PathFilters.logsCRCFilter(),
                                                   configuration).iterator();
  }

  public Configuration getConfiguration() {
    return configuration;
  }

  public Path getSequenceFilesOutputPath() {
    return sequenceFilesOutputPath;
  }

  public List<Path> getIndexPaths() {
    return indexPaths;
  }

  public String getIdField() {
    return idField;
  }

  public List<String> getFields() {
    return fields;
  }

  public void setQuery(Query query) {
    this.query = query;
  }

  public Query getQuery() {
    return query;
  }

  public void setMaxHits(int maxHits) {
    this.maxHits = maxHits;
  }

  public int getMaxHits() {
    return maxHits;
  }

  public DocumentStoredFieldVisitor getStoredFieldVisitor() {
    Set<String> fieldSet = Sets.newHashSet(idField);
    fieldSet.addAll(fields);
    return new DocumentStoredFieldVisitor(fieldSet);
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeUTF(sequenceFilesOutputPath.toString());
    out.writeUTF(StringUtils.join(indexPaths, SEPARATOR_PATHS));
    out.writeUTF(idField);
    out.writeUTF(StringUtils.join(fields, SEPARATOR_FIELDS));
    out.writeUTF(query.toString());
    out.writeInt(maxHits);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    try {
      sequenceFilesOutputPath = new Path(in.readUTF());
      indexPaths = Lists.newArrayList();
      String[] indexPaths = in.readUTF().split(SEPARATOR_PATHS);
      for (String indexPath : indexPaths) {
        this.indexPaths.add(new Path(indexPath));
      }
      idField = in.readUTF();
      fields = Arrays.asList(in.readUTF().split(SEPARATOR_FIELDS));
      query = new QueryParser(LUCENE_46, "query", new StandardAnalyzer(LUCENE_46)).parse(in.readUTF());
      maxHits = in.readInt();
    } catch (ParseException e) {
      throw new RuntimeException("Could not deserialize " + this.getClass().getName(), e);
    }
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }

    LuceneStorageConfiguration that = (LuceneStorageConfiguration) o;

    if (maxHits != that.maxHits) {
      return false;
    }
    if (fields != null ? !fields.equals(that.fields) : that.fields != null) {
      return false;
    }
    if (idField != null ? !idField.equals(that.idField) : that.idField != null) {
      return false;
    }
    if (indexPaths != null ? !indexPaths.equals(that.indexPaths) : that.indexPaths != null) {
      return false;
    }
    if (query != null ? !query.equals(that.query) : that.query != null) {
      return false;
    }
    if (sequenceFilesOutputPath != null
        ? !sequenceFilesOutputPath.equals(that.sequenceFilesOutputPath)
        : that.sequenceFilesOutputPath != null) {
      return false;
    }

    return true;
  }

  @Override
  public int hashCode() {
    int result = indexPaths != null ? indexPaths.hashCode() : 0;
    result = 31 * result + (sequenceFilesOutputPath != null ? sequenceFilesOutputPath.hashCode() : 0);
    result = 31 * result + (idField != null ? idField.hashCode() : 0);
    result = 31 * result + (fields != null ? fields.hashCode() : 0);
    result = 31 * result + (query != null ? query.hashCode() : 0);
    result = 31 * result + maxHits;
    return result;
  }
}
