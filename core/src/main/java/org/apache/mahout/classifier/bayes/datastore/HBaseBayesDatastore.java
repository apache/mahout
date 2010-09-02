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

package org.apache.mahout.classifier.bayes.datastore;

import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.client.Get;
import org.apache.hadoop.hbase.client.HTable;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.mahout.classifier.bayes.exceptions.InvalidDatastoreException;
import org.apache.mahout.classifier.bayes.interfaces.Datastore;
import org.apache.mahout.classifier.bayes.mapreduce.common.BayesConstants;
import org.apache.mahout.common.Parameters;
import org.apache.mahout.common.cache.Cache;
import org.apache.mahout.common.cache.HybridCache;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Class implementing the Datastore for Algorithms to read HBase based model
 * 
 */
public class HBaseBayesDatastore implements Datastore {
  
  private static final Logger log = LoggerFactory.getLogger(HBaseBayesDatastore.class);
  
  private HBaseConfiguration config;
  
  private HTable table;
  
  private final Cache<String,Result> tableCache;
  
  private final String hbaseTable;
  
  private final Parameters parameters;
  
  private double thetaNormalizer = 1.0;
  
  private double alphaI = 1.0;
  
  private final Map<String,Set<String>> keys = new HashMap<String,Set<String>>();
  
  private double vocabCount = -1.0;
  
  private double sigmaJSigmaK = -1.0;
  
  public HBaseBayesDatastore(String hbaseTable, Parameters params) {
    this.hbaseTable = hbaseTable;
    this.parameters = params;
    this.tableCache = new HybridCache<String,Result>(50000, 100000);
    alphaI = Double.valueOf(parameters.get("alpha_i", "1.0"));
  }
  
  protected HBaseConfiguration getConfig() {
    return config;
  }
  
  protected HTable getTable() {
    return table;
  }
  
  protected Cache<String,Result> getTableCache() {
    return tableCache;
  }
  
  protected String getHbaseTable() {
    return hbaseTable;
  }
  
  protected Parameters getParameters() {
    return parameters;
  }
  
  protected double getThetaNormalizer() {
    return thetaNormalizer;
  }
  
  protected double getAlphaI() {
    return alphaI;
  }
  
  Map<String,Set<String>> getKeys() {
    return keys;
  }
  
  protected double getVocabCount() {
    return vocabCount;
  }
  
  protected double getSigmaJSigmaK() {
    return sigmaJSigmaK;
  }
  
  @Override
  public void initialize() throws InvalidDatastoreException {
    config = new HBaseConfiguration(new Configuration());
    try {
      table = new HTable(config, hbaseTable);
    } catch (IOException e) {
      throw new InvalidDatastoreException(e.getMessage());
    }
    Collection<String> labels = getKeys("thetaNormalizer");
    for (String label : labels) {
      thetaNormalizer = Math.max(thetaNormalizer, Math.abs(getWeightFromHbase(
        BayesConstants.LABEL_THETA_NORMALIZER, label)));
    }
    for (String label : labels) {
      log.info("{} {} {} {}", new Object[] {
        label,
        getWeightFromHbase(BayesConstants.LABEL_THETA_NORMALIZER, label),
        thetaNormalizer,
        getWeightFromHbase(BayesConstants.LABEL_THETA_NORMALIZER, label) / thetaNormalizer
      });
    }
  }
  
  @Override
  public Collection<String> getKeys(String name) throws InvalidDatastoreException {
    if (keys.containsKey(name)) {
      return keys.get(name);
    }
    Result r;
    if ("labelWeight".equals(name)) {
      r = getRowFromHbase(BayesConstants.LABEL_SUM);
    } else if ("thetaNormalizer".equals(name)) {
      r = getRowFromHbase(BayesConstants.LABEL_THETA_NORMALIZER);
    } else {
      r = getRowFromHbase(name);
    }
    
    if (r == null) {
      log.error("Encountered NULL");
      throw new InvalidDatastoreException("Encountered NULL");
    }
    
    Set<byte[]> labelBytes = r.getNoVersionMap().get(Bytes.toBytes(BayesConstants.HBASE_COLUMN_FAMILY))
        .keySet();
    Set<String> keySet = new HashSet<String>();
    for (byte[] key : labelBytes) {
      keySet.add(Bytes.toString(key));
    }
    keys.put(name, keySet);
    return keySet;
  }
  
  @Override
  public double getWeight(String matrixName, String row, String column) throws InvalidDatastoreException {
    if ("weight".equals(matrixName)) {
      if ("sigma_j".equals(column)) {
        return getSigmaJFromHbase(row);
      } else {
        return getWeightFromHbase(row, column);
      }
    } else {
      throw new InvalidDatastoreException();
    }
  }
  
  @Override
  public double getWeight(String vectorName, String index) throws InvalidDatastoreException {
    if ("sumWeight".equals(vectorName)) {
      if ("vocabCount".equals(index)) {
        return getVocabCountFromHbase();
      } else if ("sigma_jSigma_k".equals(index)) {
        return getSigmaJSigmaKFromHbase();
      } else {
        throw new InvalidDatastoreException();
      }
      
    } else if ("labelWeight".equals(vectorName)) {
      return getWeightFromHbase(BayesConstants.LABEL_SUM, index);
    } else if ("thetaNormalizer".equals(vectorName)) {
      return getWeightFromHbase(BayesConstants.LABEL_THETA_NORMALIZER, index) / thetaNormalizer;
    } else if ("params".equals(vectorName)) {
      if ("alpha_i".equals(index)) {
        return alphaI;
      } else {
        throw new InvalidDatastoreException();
      }
    } else {
      
      throw new InvalidDatastoreException();
    }
  }
  
  protected double getCachedCell(String row, String family, String column) {
    Result r = tableCache.get(row);
    
    if (r == null) {
      Get g = new Get(Bytes.toBytes(row));
      g.addFamily(Bytes.toBytes(family));
      try {
        r = table.get(g);
      } catch (IOException e) {
        return 0.0;
      }
      tableCache.set(row, r);
    }
    byte[] value = r.getValue(Bytes.toBytes(BayesConstants.HBASE_COLUMN_FAMILY), Bytes.toBytes(column));
    if (value == null) {
      return 0.0;
    }
    return Bytes.toDouble(value);
    
  }
  
  protected double getWeightFromHbase(String feature, String label) {
    return getCachedCell(feature, BayesConstants.HBASE_COLUMN_FAMILY, label);
  }
  
  protected Result getRowFromHbase(String feature) {
    Result r = tableCache.get(feature);
    try {
      if (r == null) {
        Get g = new Get(Bytes.toBytes(feature));
        g.addFamily(Bytes.toBytes(BayesConstants.HBASE_COLUMN_FAMILY));
        r = table.get(g);
        tableCache.set(feature, r);
        return r;
      } else {
        return r;
      }
      
    } catch (IOException e) {
      return r;
    }
  }
  
  protected double getSigmaJFromHbase(String feature) {
    return getCachedCell(feature, BayesConstants.HBASE_COLUMN_FAMILY, BayesConstants.FEATURE_SUM);
  }
  
  protected double getVocabCountFromHbase() {
    if (vocabCount == -1.0) {
      vocabCount = getCachedCell(BayesConstants.HBASE_COUNTS_ROW, BayesConstants.HBASE_COLUMN_FAMILY,
        BayesConstants.FEATURE_SET_SIZE);
      return vocabCount;
    } else {
      return vocabCount;
    }
  }
  
  protected double getSigmaJSigmaKFromHbase() {
    if (sigmaJSigmaK == -1.0) {
      sigmaJSigmaK = getCachedCell(BayesConstants.HBASE_COUNTS_ROW, BayesConstants.HBASE_COLUMN_FAMILY,
        BayesConstants.TOTAL_SUM);
      return sigmaJSigmaK;
    } else {
      return sigmaJSigmaK;
    }
  }
  
}
