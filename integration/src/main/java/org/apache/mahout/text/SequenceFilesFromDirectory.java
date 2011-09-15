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

import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.Map;

import com.google.common.collect.Maps;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;

/**
 * Converts a directory of text documents into SequenceFiles of Specified chunkSize. This class takes in a
 * parent directory containing sub folders of text documents and recursively reads the files and creates the
 * {@link SequenceFile}s of docid => content. The docid is set as the relative path of the document from the
 * parent directory prepended with a specified prefix. You can also specify the input encoding of the text
 * files. The content of the output SequenceFiles are encoded as UTF-8 text.
 */
public class SequenceFilesFromDirectory extends AbstractJob {

  private static final String PREFIX_ADDITION_FILTER = PrefixAdditionFilter.class.getName();
  
  private static final String[] CHUNK_SIZE_OPTION = {"chunkSize", "chunk"};
  static final String[] FILE_FILTER_CLASS_OPTION = {"fileFilterClass","filter"};
  private static final String[] KEY_PREFIX_OPTION = {"keyPrefix", "prefix"};
  static final String[] CHARSET_OPTION = {"charset", "c"};

  public static void run(Configuration conf,
                         String keyPrefix,
                         Map<String, String> options,
                         Path input,
                         Path output)
    throws InstantiationException, IllegalAccessException, InvocationTargetException, IOException,
           NoSuchMethodException, ClassNotFoundException {
    FileSystem fs = FileSystem.get(input.toUri(), conf);
    ChunkedWriter writer = new ChunkedWriter(conf, Integer.parseInt(options.get(CHUNK_SIZE_OPTION[0])), output);

    try {
      SequenceFilesFromDirectoryFilter pathFilter;
      String fileFilterClassName = options.get(FILE_FILTER_CLASS_OPTION[0]);
      if (PrefixAdditionFilter.class.getName().equals(fileFilterClassName)) {
        pathFilter = new PrefixAdditionFilter(conf, keyPrefix, options, writer, fs);
      } else {
        Class<? extends SequenceFilesFromDirectoryFilter> pathFilterClass =
            Class.forName(fileFilterClassName).asSubclass(SequenceFilesFromDirectoryFilter.class);
        Constructor<? extends SequenceFilesFromDirectoryFilter> constructor =
            pathFilterClass.getConstructor(Configuration.class,
                                           String.class,
                                           Map.class,
                                           ChunkedWriter.class,
                                           FileSystem.class);
        pathFilter = constructor.newInstance(conf, keyPrefix, options, writer, fs);
      }
      fs.listStatus(input, pathFilter);
    } finally {
      Closeables.closeQuietly(writer);
    }
  }
  
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new SequenceFilesFromDirectory(), args);
  }
  
  /*
   * callback main after processing hadoop parameters
   */
  @Override
  public int run(String[] args)
    throws IOException, ClassNotFoundException, InstantiationException, IllegalAccessException, NoSuchMethodException,
           InvocationTargetException {
    addOptions();    
    
    if (parseArguments(args) == null) {
      return -1;
    }
   
    Map<String, String> options = parseOptions();
    Path input = getInputPath();
    Path output = getOutputPath();
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      Configuration conf = new Configuration();
      HadoopUtil.delete(conf, output);
    }
    String keyPrefix = getOption(KEY_PREFIX_OPTION[0]);

    run(getConf(), keyPrefix, options, input, output);
    return 0;
  }

  /**
   * Override this method in order to add additional options to the command line of the SequenceFileFromDirectory job.
   * Do not forget to call super() otherwise all standard options (input/output dirs etc) will not be available.
   * */
  protected void addOptions() {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption(CHUNK_SIZE_OPTION[0], CHUNK_SIZE_OPTION[1], "The chunkSize in MegaBytes. Defaults to 64", "64");
    addOption(FILE_FILTER_CLASS_OPTION[0], FILE_FILTER_CLASS_OPTION[1],
        "The name of the class to use for file parsing. Default: " + PREFIX_ADDITION_FILTER, PREFIX_ADDITION_FILTER);
    addOption(KEY_PREFIX_OPTION[0], KEY_PREFIX_OPTION[1], "The prefix to be prepended to the key", "");
    addOption(CHARSET_OPTION[0], CHARSET_OPTION[1],
        "The name of the character encoding of the input files. Default to UTF-8", "UTF-8");
  }

  /**
   * Override this method in order to parse your additional options from the command line. Do not forget to call
   * super() otherwise standard options (input/output dirs etc) will not be available.
   */
  protected Map<String, String> parseOptions() throws IOException {
    Map<String, String> options = Maps.newHashMap();
    options.put(CHUNK_SIZE_OPTION[0], getOption(CHUNK_SIZE_OPTION[0]));
    options.put(FILE_FILTER_CLASS_OPTION[0], getOption(FILE_FILTER_CLASS_OPTION[0]));
    options.put(CHARSET_OPTION[0], getOption(CHARSET_OPTION[0]));
    return options;
  }
}
