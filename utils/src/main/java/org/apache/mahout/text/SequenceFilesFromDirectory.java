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

import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.nio.charset.Charset;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.FileLineIterable;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Converts a directory of text documents into SequenceFiles of Specified chunkSize. This class takes in a
 * parent directory containing sub folders of text documents and recursively reads the files and creates the
 * {@link SequenceFile}s of docid => content. The docid is set as the relative path of the document from the
 * parent directory prepended with a specified prefix. You can also specify the input encoding of the text
 * files. The content of the output SequenceFiles are encoded as UTF-8 text.
 */
public final class SequenceFilesFromDirectory extends AbstractJob {

  private static final Logger log = LoggerFactory.getLogger(SequenceFilesFromDirectory.class);

  private static final String PREFIX_ADDITION_FILTER = PrefixAdditionFilter.class.getName();
  
  public static final String[] CHUNK_SIZE_OPTION = {"chunkSize", "chunk"};
  public static final String[] FILE_FILTER_CLASS_OPTION = {"fileFilterClass","filter"};
  public static final String[] KEY_PREFIX_OPTION = {"keyPrefix", "prefix"};
  public static final String[] CHARSET_OPTION = {"charset", "c"};

  public void run(Configuration conf,
                  Path input,
                  Path output,
                  String prefix,
                  int chunkSizeInMB,
                  Charset charset,
                  String fileFilterClassName)
    throws IllegalArgumentException, InstantiationException, IllegalAccessException, InvocationTargetException,
           IOException, SecurityException, NoSuchMethodException, ClassNotFoundException {
    FileSystem fs = FileSystem.get(conf);
    ChunkedWriter writer = new ChunkedWriter(conf, chunkSizeInMB, output);
    
    PathFilter pathFilter;
    
    if (PrefixAdditionFilter.class.getName().equals(fileFilterClassName)) {
      pathFilter = new PrefixAdditionFilter(conf, prefix, writer, charset);
    } else {
      Class<? extends PathFilter> pathFilterClass = Class.forName(fileFilterClassName).asSubclass(PathFilter.class);
      Constructor<? extends PathFilter> constructor =
          pathFilterClass.getConstructor(Configuration.class, String.class, ChunkedWriter.class, Charset.class);
      pathFilter = constructor.newInstance(conf, prefix, writer, charset);
    }
    fs.listStatus(input, pathFilter);
    writer.close();
  }
  
  private static final class ChunkedWriter implements Closeable {

    private final int maxChunkSizeInBytes;
    private final Path output;
    private SequenceFile.Writer writer;
    private int currentChunkID;
    private int currentChunkSize;
    private final FileSystem fs;
    private final Configuration conf;
    
    private ChunkedWriter(Configuration conf, int chunkSizeInMB, Path output) throws IOException {
      this.output = output;
      this.conf = conf;
      if (chunkSizeInMB > 1984) {
        chunkSizeInMB = 1984;
      }
      maxChunkSizeInBytes = chunkSizeInMB * 1024 * 1024;
      fs = FileSystem.get(conf);
      currentChunkID = 0;
      writer = new SequenceFile.Writer(fs, conf, getPath(currentChunkID), Text.class, Text.class);
    }
    
    private Path getPath(int chunkID) {
      return new Path(output, "chunk-" + chunkID);
    }
    
    public void write(String key, String value) throws IOException {
      if (currentChunkSize > maxChunkSizeInBytes) {
        writer.close();
        writer = new SequenceFile.Writer(fs, conf, getPath(currentChunkID++), Text.class, Text.class);
        currentChunkSize = 0;
      }
      
      Text keyT = new Text(key);
      Text valueT = new Text(value);
      currentChunkSize += keyT.getBytes().length + valueT.getBytes().length; // Overhead
      writer.append(keyT, valueT);
    }
    
    @Override
    public void close() throws IOException {
      writer.close();
    }
  }
  
  private final class PrefixAdditionFilter implements PathFilter {

    private final String prefix;
    private final ChunkedWriter writer;
    private final Charset charset;
    private final Configuration conf;
    private final FileSystem fs;
    
    private PrefixAdditionFilter(Configuration conf, String prefix, ChunkedWriter writer, Charset charset)
      throws IOException {
      this.conf = conf;
      this.prefix = prefix;
      this.writer = writer;
      this.charset = charset;
      this.fs = FileSystem.get(conf);
    }
    
    @Override
    public boolean accept(Path current) {
      log.debug("CURRENT: {}", current.getName());
      try {
        FileStatus[] fstatus = fs.listStatus(current);
        for (FileStatus fst : fstatus) {
          log.debug("CHILD: {}", fst.getPath().getName());
          if (fst.isDir()) {
            fs.listStatus(fst.getPath(),
                          new PrefixAdditionFilter(conf, prefix + Path.SEPARATOR + current.getName(), writer, charset));
          } else {
            StringBuilder file = new StringBuilder();
            InputStream in = fs.open(fst.getPath());
            for (String aFit : new FileLineIterable(in, charset, false)) {
              file.append(aFit).append('\n');
            }
            String name = current.getName().equals(fst.getPath().getName())
                ? current.getName()
                : current.getName() + Path.SEPARATOR + fst.getPath().getName();
            writer.write(prefix + Path.SEPARATOR + name, file.toString());
          }
        }
      } catch (Exception e) {
        throw new IllegalStateException(e);
      }
      return false;
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
    
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption(CHUNK_SIZE_OPTION[0], CHUNK_SIZE_OPTION[1], "The chunkSize in MegaBytes. Defaults to 64", "64");
    addOption(FILE_FILTER_CLASS_OPTION[0], FILE_FILTER_CLASS_OPTION[1],
        "The name of the class to use for file parsing. Default: " + PREFIX_ADDITION_FILTER, PREFIX_ADDITION_FILTER);
    addOption(KEY_PREFIX_OPTION[0], KEY_PREFIX_OPTION[1], "The prefix to be prepended to the key", "");
    addOption(CHARSET_OPTION[0], CHARSET_OPTION[1],
        "The name of the character encoding of the input files. Default to UTF-8", "UTF-8");
    
    if (parseArguments(args) == null) {
      return -1;
    }
    
    Path input = getInputPath();
    Path output = getOutputPath();
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.overwriteOutput(output);
    }
    int chunkSize = Integer.parseInt(getOption(CHUNK_SIZE_OPTION[0]));
    String fileFilterClassName = getOption(FILE_FILTER_CLASS_OPTION[0]);
    String keyPrefix = getOption(KEY_PREFIX_OPTION[0]);
    Charset charset = Charset.forName(getOption(CHARSET_OPTION[0]));
    
    run(getConf(), input, output, keyPrefix, chunkSize, charset, fileFilterClassName);
    return 0;
  }
}
