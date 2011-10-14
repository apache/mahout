package org.apache.mahout.cf.taste.example.email;


import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.map.OpenObjectIntHashMap;

import java.io.IOException;
import java.net.URI;

/**
 *
 *
 **/
public final class EmailUtility {
  public static final String SEPARATOR = "separator";
  public static final String MSG_IDS_PREFIX = "msgIdsPrefix";
  public static final String FROM_PREFIX = "fromPrefix";
  public static final String MSG_ID_DIMENSION = "msgIdDim";
  public static final String FROM_INDEX = "fromIdx";
  public static final String REFS_INDEX = "refsIdx";

  private EmailUtility() {

  }

  /**
   * Strip off some spurious characters that make it harder to dedup
   *
   * @param address
   * @return
   */
  public static String cleanUpEmailAddress(String address) {
    //do some cleanup to normalize some things, like: Key: karthik ananth <karthik.jcecs@gmail.com>: Value: 178
    //Key: karthik ananth [mailto:karthik.jcecs@gmail.com]=20: Value: 179
    //TODO: is there more to clean up here?
    address = address.replaceAll("mailto:|<|>|\\[|\\]|\\=20", "");
    return address;
  }


  public static void loadDictionaries(Configuration conf, String fromPrefix,
                                      OpenObjectIntHashMap<String> fromDictionary,
                                      String msgIdPrefix,
                                      OpenObjectIntHashMap<String> msgIdDictionary) throws IOException {

    URI[] localFiles = DistributedCache.getCacheFiles(conf);
    Preconditions.checkArgument(localFiles != null,
            "missing paths from the DistributedCache");
    for (int i = 0; i < localFiles.length; i++) {
      URI localFile = localFiles[i];
      Path dictionaryFile = new Path(localFile.getPath());
      // key is word value is id

      OpenObjectIntHashMap<String> dictionary = null;
      if (dictionaryFile.getName().startsWith(fromPrefix)) {
        dictionary = fromDictionary;
      } else if (dictionaryFile.getName().startsWith(msgIdPrefix)) {
        dictionary = msgIdDictionary;
      }
      if (dictionary != null) {
        for (Pair<Writable, IntWritable> record
                : new SequenceFileIterable<Writable, IntWritable>(dictionaryFile, true, conf)) {
          dictionary.put(record.getFirst().toString(), record.getSecond().get());
        }
      }
    }

  }

  private static final String [] EMPTY = new String[0];

  public static String[] parseReferences(String rawRefs) {
    String[] splits = null;
    if (rawRefs != null && rawRefs.length() > 0) {
      splits = rawRefs.split(">|\\s+");
      for (int i = 0; i < splits.length; i++) {
        splits[i] = splits[i].replaceAll("<|>", "");
      }
    } else {
      splits = EMPTY;
    }
    return splits;
  }
}
