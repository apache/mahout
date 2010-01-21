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

package org.apache.mahout.utils.vectors.lucene;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.lucene.document.FieldSelector;
import org.apache.lucene.document.SetBasedFieldSelector;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermDocs;
import org.apache.lucene.index.TermEnum;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.OpenBitSet;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.math.stats.LogLikelihood;
import org.apache.mahout.utils.clustering.ClusterDumper;
import org.apache.mahout.utils.vectors.TermEntry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Get labels for the cluster using Log Likelihood Ratio (LLR).
 * <p/>
 * "The most useful way to think of this (LLR) is as the percentage of in-cluster
 * documents that have the feature (term) versus the percentage out, keeping in
 * mind that both percentages are uncertain since we have only a sample of all
 * possible documents." - Ted Dunning
 * <p/>
 * More about LLR can be found at :
 * http://tdunning.blogspot.com/2008/03/surprise-and-coincidence.html
 */
public class ClusterLabels {

  class TermInfoClusterInOut implements Comparable<TermInfoClusterInOut> {
    public final String term;
    public final int inClusterDF;
    public final int outClusterDF;
    public double logLikelihoodRatio;

    public TermInfoClusterInOut(String term, int inClusterDF, int outClusterDF) {
      this.term = term;
      this.inClusterDF = inClusterDF;
      this.outClusterDF = outClusterDF;
    }

    @Override
    public int compareTo(TermInfoClusterInOut that) {
      int res = -Double.compare(logLikelihoodRatio, that.logLikelihoodRatio);
      if (res == 0) {
        res = term.compareTo(that.term);
      }
      return res;
    }

    public int getInClusterDiff() {
      return this.inClusterDF - this.outClusterDF;
    }
  }

  private static final Logger log = LoggerFactory.getLogger(ClusterLabels.class);
  public static final int DEFAULT_MIN_IDS = 50;
  public static final int DEFAULT_MAX_LABELS = 25;

  private final String seqFileDir;
  private final String pointsDir;
  private final String indexDir;
  private final String contentField;
  private String idField;
  private Map<String, List<String>> clusterIdToPoints = null;
  private String output;
  private int minNumIds = DEFAULT_MIN_IDS;
  private int maxLabels = DEFAULT_MAX_LABELS;

  public ClusterLabels(String seqFileDir, String pointsDir, String indexDir, String contentField, int minNumIds, int maxLabels) throws IOException {
    this.seqFileDir = seqFileDir;
    this.pointsDir = pointsDir;
    this.indexDir = indexDir;
    this.contentField = contentField;
    this.minNumIds = minNumIds;
    this.maxLabels = maxLabels;
    init();
  }

  private void init() throws IOException {
    ClusterDumper clusterDumper = new ClusterDumper(seqFileDir, pointsDir);
    this.clusterIdToPoints = clusterDumper.getClusterIdToPoints();
  }

  public void getLabels() throws IOException {

    Writer writer;
    if (this.output != null) {
      writer = new FileWriter(this.output);
    } else {
      writer = new OutputStreamWriter(System.out);
    }

    for (String clusterID : clusterIdToPoints.keySet()) {
      List<String> ids = clusterIdToPoints.get(clusterID);
      List<TermInfoClusterInOut> termInfos = getClusterLabels(clusterID, ids);
      if (termInfos != null) {
        writer.write('\n');
        writer.write("Top labels for Cluster " + clusterID + " containing " + ids.size() + " vectors");
        writer.write('\n');
        writer.write("Term \t\t LLR \t\t In-ClusterDF \t\t Out-ClusterDF ");
        writer.write('\n');
        for (TermInfoClusterInOut termInfo : termInfos) {
          writer.write(termInfo.term + "\t\t" + termInfo.logLikelihoodRatio + "\t\t" + termInfo.inClusterDF + "\t\t" + termInfo.outClusterDF);
          writer.write('\n');
        }
      }
    }
    writer.flush();
    if (this.output != null) {
      writer.close();
    }
  }

  /**
   * Get the list of labels, sorted by best score.
   *
   * @param clusterID
   * @param ids
   * @return
   * @throws CorruptIndexException
   * @throws IOException
   */
  protected List<TermInfoClusterInOut> getClusterLabels(String clusterID, List<String> ids) throws IOException {

    if (ids.size() < minNumIds) {
      log.info("Skipping small cluster {} with size: {}", clusterID, ids.size());
      return null;
    }

    log.info("Processing Cluster {} with {} documents", clusterID, ids.size());
    Directory dir = FSDirectory.open(new File(this.indexDir));
    IndexReader reader = IndexReader.open(dir, false);

    log.info("# of documents in the index {}", reader.numDocs());

    Set<String> idSet = new HashSet<String>();
    idSet.addAll(ids);

    int numDocs = reader.numDocs();

    OpenBitSet clusterDocBitset = getClusterDocBitset(reader, idSet, this.idField);

    log.info("Populating term infos from the index");

    /**
     * This code is as that of CachedTermInfo, with one major change, which is to get the document frequency.
     *
     * Since we have deleted the documents out of the cluster, the document frequency for a term should 
     * only include the in-cluster documents. The document frequency obtained from TermEnum reflects the 
     * frequency in the entire index. To get the in-cluster frequency, we need to query the index to get
     * the term frequencies in each document. The number of results of this call will be the in-cluster 
     * document frequency.
     */

    TermEnum te = reader.terms(new Term(contentField, ""));
    int count = 0;

    Map<String, TermEntry> termEntryMap = new LinkedHashMap<String, TermEntry>();
    do {
      Term term = te.term();
      if (term == null || term.field().equals(contentField) == false) {
        break;
      }
      OpenBitSet termBitset = new OpenBitSet(reader.maxDoc());

      // Generate bitset for the term
      TermDocs termDocs = reader.termDocs(term);

      while (termDocs.next()) {
        termBitset.set(termDocs.doc());
      }

      // AND the term's bitset with cluster doc bitset to get the term's in-cluster frequency.
      // This modifies the termBitset, but that's fine as we are not using it anywhere else.
      termBitset.and(clusterDocBitset);
      int inclusterDF = (int) termBitset.cardinality();

      TermEntry entry = new TermEntry(term.text(), count++, inclusterDF);
      termEntryMap.put(entry.term, entry);
    } while (te.next());
    te.close();

    List<TermInfoClusterInOut> clusteredTermInfo = new LinkedList<TermInfoClusterInOut>();

    int clusterSize = ids.size();
    int corpusSize = numDocs;

    for (TermEntry termEntry : termEntryMap.values()) {
      int corpusDF = reader.terms(new Term(this.contentField, termEntry.term)).docFreq();
      int outDF = corpusDF - termEntry.docFreq;
      int inDF = termEntry.docFreq;
      TermInfoClusterInOut termInfoCluster = new TermInfoClusterInOut(termEntry.term, inDF, outDF);
      double llr = scoreDocumentFrequencies(inDF, outDF, clusterSize, corpusSize);
      termInfoCluster.logLikelihoodRatio = llr;
      clusteredTermInfo.add(termInfoCluster);
    }

    Collections.sort(clusteredTermInfo);
    // Cleanup
    reader.close();
    termEntryMap.clear();

    return clusteredTermInfo.subList(0, Math.min(clusteredTermInfo.size(), maxLabels));
  }


  private OpenBitSet getClusterDocBitset(IndexReader reader, Set<String> idSet, String idField) throws IOException {
    int numDocs = reader.numDocs();

    OpenBitSet bitset = new OpenBitSet(numDocs);

    FieldSelector idFieldSelector = new SetBasedFieldSelector(Collections.singleton(idField), Collections.emptySet());

    for (int i = 0; i < numDocs; i++) {
      String id = null;
      // Use Lucene's internal ID if idField is not specified. Else, get it from the document.
      if (idField == null) {
        id = Integer.toString(i);
      } else {
        id = reader.document(i, idFieldSelector).get(idField);
      }
      if (idSet.contains(id)) {
        bitset.set(i);
      }
    }
    log.info("Created bitset for in-cluster documents : {}", bitset.cardinality());
    return bitset;
  }

  private double scoreDocumentFrequencies(int inDF, int outDF, int clusterSize, int corpusSize) {
    int k12 = clusterSize - inDF;
    int k22 = corpusSize - clusterSize - outDF;

    return LogLikelihood.logLikelihoodRatio(inDF, k12, outDF, k22);
  }

  public String getIdField() {
    return idField;
  }

  public void setIdField(String idField) {
    this.idField = idField;
  }

  public String getOutput() {
    return output;
  }

  public void setOutput(String output) {
    this.output = output;
  }

  public static void main(String[] args) {

    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option indexOpt = obuilder.withLongName("dir").withRequired(true).withArgument(
            abuilder.withName("dir").withMinimum(1).withMaximum(1).create()).
            withDescription("The Lucene index directory").withShortName("d").create();

    Option outputOpt = obuilder.withLongName("output").withRequired(false).withArgument(
            abuilder.withName("output").withMinimum(1).withMaximum(1).create()).
            withDescription("The output file. If not specified, the result is printed on console.").withShortName("o").create();

    Option fieldOpt = obuilder.withLongName("field").withRequired(true).withArgument(
            abuilder.withName("field").withMinimum(1).withMaximum(1).create()).
            withDescription("The content field in the index").withShortName("f").create();

    Option idFieldOpt = obuilder.withLongName("idField").withRequired(false).withArgument(
            abuilder.withName("idField").withMinimum(1).withMaximum(1).create()).
            withDescription("The field for the document ID in the index.  If null, then the Lucene internal doc " +
                    "id is used which is prone to error if the underlying index changes").withShortName("i").create();

    Option seqOpt = obuilder.withLongName("seqFileDir").withRequired(true).withArgument(
            abuilder.withName("seqFileDir").withMinimum(1).withMaximum(1).create()).
            withDescription("The directory containing Sequence Files for the Clusters").withShortName("s").create();

    Option pointsOpt = obuilder.withLongName("pointsDir").withRequired(true).withArgument(
            abuilder.withName("pointsDir").withMinimum(1).withMaximum(1).create()).
            withDescription("The directory containing points sequence files mapping input vectors to their cluster.  ").withShortName("p").create();
    Option minClusterSizeOpt = obuilder.withLongName("minClusterSize").withRequired(false).withArgument(
            abuilder.withName("minClusterSize").withMinimum(1).withMaximum(1).create()).
            withDescription("The minimum number of points required in a cluster to print the labels for").withShortName("m").create();
    Option maxLabelsOpt = obuilder.withLongName("maxLabels").withRequired(false).withArgument(
            abuilder.withName("maxLabels").withMinimum(1).withMaximum(1).create()).
            withDescription("The maximum number of labels to print per cluster").withShortName("x").create();
    Option helpOpt = obuilder.withLongName("help").
            withDescription("Print out help").withShortName("h").create();

    Group group = gbuilder.withName("Options").withOption(indexOpt).withOption(idFieldOpt).withOption(outputOpt)
            .withOption(fieldOpt).withOption(seqOpt).withOption(pointsOpt).withOption(helpOpt).withOption(maxLabelsOpt).withOption(minClusterSizeOpt).create();

    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);

      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }

      String seqFileDir = cmdLine.getValue(seqOpt).toString();
      String pointsDir = cmdLine.getValue(pointsOpt).toString();
      String indexDir = cmdLine.getValue(indexOpt).toString();
      String contentField = cmdLine.getValue(fieldOpt).toString();


      String idField = null;
      String output = null;

      if (cmdLine.hasOption(idFieldOpt)) {
        idField = cmdLine.getValue(idFieldOpt).toString();
      }
      if (cmdLine.hasOption(outputOpt)) {
        output = cmdLine.getValue(outputOpt).toString();
      }
      int maxLabels = DEFAULT_MAX_LABELS;
      if (cmdLine.hasOption(maxLabelsOpt)) {
        maxLabels = Integer.parseInt(cmdLine.getValue(maxLabelsOpt).toString());
      }
      int minSize = DEFAULT_MIN_IDS;
      if (cmdLine.hasOption(minClusterSizeOpt)) {
        minSize = Integer.parseInt(cmdLine.getValue(minClusterSizeOpt).toString());
      }
      ClusterLabels clusterLabel = new ClusterLabels(seqFileDir, pointsDir, indexDir, contentField, minSize, maxLabels);

      if (idField != null) {
        clusterLabel.setIdField(idField);
      }
      if (output != null) {
        clusterLabel.setOutput(output);
      }

      clusterLabel.getLabels();

    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
    } catch (IOException e) {
      log.error("Exception", e);
    }
  }

}
