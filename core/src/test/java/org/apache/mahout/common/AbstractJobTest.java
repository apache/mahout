/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.mahout.common;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import com.google.common.collect.Maps;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.junit.Test;

public final class AbstractJobTest extends MahoutTestCase {
  
  interface AbstractJobFactory {
    AbstractJob getJob();
  }
  
  @Test
  public void testFlag() throws Exception {
    final Map<String,List<String>> testMap = Maps.newHashMap();
    
    AbstractJobFactory fact = new AbstractJobFactory() {
      @Override
      public AbstractJob getJob() {
        return new AbstractJob() {
          @Override
          public int run(String[] args) throws IOException {
            addFlag("testFlag", "t", "a simple test flag");
            
            Map<String,List<String>> argMap = parseArguments(args);
            testMap.clear();
            testMap.putAll(argMap);
            return 1;
          }
        };
      }
    };
    
    // testFlag will only be present if specified on the command-line
    
    ToolRunner.run(fact.getJob(), new String[0]);
    assertFalse("test map for absent flag", testMap.containsKey("--testFlag"));
    
    String[] withFlag = { "--testFlag" };
    ToolRunner.run(fact.getJob(), withFlag);
    assertTrue("test map for present flag", testMap.containsKey("--testFlag"));
  }
  
  @Test
  public void testOptions() throws Exception {
    final Map<String,List<String>> testMap = Maps.newHashMap();
    
    AbstractJobFactory fact = new AbstractJobFactory() {
      @Override
      public AbstractJob getJob() {
        return new AbstractJob() {
          @Override
          public int run(String[] args) throws IOException {
            this.addOption(DefaultOptionCreator.overwriteOption().create());
            this.addOption("option", "o", "option");
            this.addOption("required", "r", "required", true /* required */);
            this.addOption("notRequired", "nr", "not required", false /* not required */);
            this.addOption("hasDefault", "hd", "option w/ default", "defaultValue");


            Map<String,List<String>> argMap = parseArguments(args);
            if (argMap == null) {
              return -1;
            }
            
            testMap.clear();
            testMap.putAll(argMap);

            return 0;
          }
        };
      }
    };

    int ret = ToolRunner.run(fact.getJob(), new String[0]);
    assertEquals("-1 for missing required options", -1, ret);
    
    ret = ToolRunner.run(fact.getJob(), new String[]{
      "--required", "requiredArg"
    });
    assertEquals("0 for no missing required options", 0, ret);
    assertEquals(Collections.singletonList("requiredArg"), testMap.get("--required"));
    assertEquals(Collections.singletonList("defaultValue"), testMap.get("--hasDefault"));
    assertNull(testMap.get("--option"));
    assertNull(testMap.get("--notRequired"));
    assertFalse(testMap.containsKey("--overwrite"));
    
    ret = ToolRunner.run(fact.getJob(), new String[]{
      "--required", "requiredArg",
      "--unknownArg"
    });
    assertEquals("-1 for including unknown options", -1, ret);

    ret = ToolRunner.run(fact.getJob(), new String[]{
      "--required", "requiredArg",
      "--required", "requiredArg2",
    });
    assertEquals("-1 for including duplicate options", -1, ret);
    
    ret = ToolRunner.run(fact.getJob(), new String[]{
      "--required", "requiredArg", 
      "--overwrite",
      "--hasDefault", "nonDefault",
      "--option", "optionValue",
      "--notRequired", "notRequired"
    });
    assertEquals("0 for no missing required options", 0, ret);
    assertEquals(Collections.singletonList("requiredArg"), testMap.get("--required"));
    assertEquals(Collections.singletonList("nonDefault"), testMap.get("--hasDefault"));
    assertEquals(Collections.singletonList("optionValue"), testMap.get("--option"));
    assertEquals(Collections.singletonList("notRequired"), testMap.get("--notRequired"));
    assertTrue(testMap.containsKey("--overwrite"));
    
    ret = ToolRunner.run(fact.getJob(), new String[]{
      "-r", "requiredArg", 
      "-ow",
      "-hd", "nonDefault",
      "-o", "optionValue",
      "-nr", "notRequired"
    });
    assertEquals("0 for no missing required options", 0, ret);
    assertEquals(Collections.singletonList("requiredArg"), testMap.get("--required"));
    assertEquals(Collections.singletonList("nonDefault"), testMap.get("--hasDefault"));
    assertEquals(Collections.singletonList("optionValue"), testMap.get("--option"));
    assertEquals(Collections.singletonList("notRequired"), testMap.get("--notRequired"));
    assertTrue(testMap.containsKey("--overwrite"));
    
  }
  
  @Test
  public void testInputOutputPaths() throws Exception {
    
    AbstractJobFactory fact = new AbstractJobFactory() {
      @Override
      public AbstractJob getJob() {
        return new AbstractJob() {
          @Override
          public int run(String[] args) throws IOException {
            addInputOption();
            addOutputOption();
            
            // arg map should be null if a required option is missing.
            Map<String, List<String>> argMap = parseArguments(args);
            
            if (argMap == null) {
              return -1;
            }
            
            Path inputPath = getInputPath();
            assertNotNull("getInputPath() returns non-null", inputPath);
            
            Path outputPath = getInputPath();
            assertNotNull("getOutputPath() returns non-null", outputPath);
            return 0;
          }
        };
      }
    };

    int ret = ToolRunner.run(fact.getJob(), new String[0]);
    assertEquals("-1 for missing input option", -1, ret);
    
    String testInputPath = "testInputPath";

    AbstractJob job = fact.getJob();
    ret = ToolRunner.run(job, new String[]{
        "--input", testInputPath });
    assertEquals("-1 for missing output option", -1, ret);
    assertEquals("input path is correct", testInputPath, job.getInputPath().toString());
    
    job = fact.getJob();
    String testOutputPath = "testOutputPath";
    ret = ToolRunner.run(job, new String[]{
        "--output", testOutputPath });
    assertEquals("-1 for missing input option", -1, ret);
    assertEquals("output path is correct", testOutputPath, job.getOutputPath().toString());
    
    job = fact.getJob();
    ret = ToolRunner.run(job, new String[]{
        "--input", testInputPath, "--output", testOutputPath });
    assertEquals("0 for complete options", 0, ret);
    assertEquals("input path is correct", testInputPath, job.getInputPath().toString());
    assertEquals("output path is correct", testOutputPath, job.getOutputPath().toString());
    
    job = fact.getJob();
    ret = ToolRunner.run(job, new String[]{ 
        "--input", testInputPath, "--output", testOutputPath });
    assertEquals("0 for complete options", 0, ret);
    assertEquals("input path is correct", testInputPath, job.getInputPath().toString());
    assertEquals("output path is correct", testOutputPath, job.getOutputPath().toString());
    
    job = fact.getJob();
    String testInputPropertyPath = "testInputPropertyPath";
    String testOutputPropertyPath = "testOutputPropertyPath";
    ret = ToolRunner.run(job, new String[]{
        "-Dmapred.input.dir=" + testInputPropertyPath, 
        "-Dmapred.output.dir=" + testOutputPropertyPath });
    assertEquals("0 for complete options", 0, ret);
    assertEquals("input path from property is correct", testInputPropertyPath, job.getInputPath().toString());
    assertEquals("output path from property is correct", testOutputPropertyPath, job.getOutputPath().toString());
    
    job = fact.getJob();
    ret = ToolRunner.run(job, new String[]{ 
        "-Dmapred.input.dir=" + testInputPropertyPath,
        "-Dmapred.output.dir=" + testOutputPropertyPath,
        "--input", testInputPath,
        "--output", testOutputPath });
    assertEquals("0 for complete options", 0, ret);
    assertEquals("input command-line option precedes property",
        testInputPath, job.getInputPath().toString());
    assertEquals("output command-line option precedes property",
        testOutputPath, job.getOutputPath().toString());
  }
}
