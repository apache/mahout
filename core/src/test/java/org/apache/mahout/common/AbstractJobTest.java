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

import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.junit.Test;

import junit.framework.TestCase;

/**
 * 
 */
public class AbstractJobTest {
  
  interface AbstractJobFactory {
    public AbstractJob getJob();
  }
  
  @Test
  public void testFlag() throws Exception {
    final Map<String,String> testMap = new HashMap<String,String>();
    
    final AbstractJobFactory fact = new AbstractJobFactory() {
      public AbstractJob getJob() {
        return new AbstractJob() {
          @Override
          public int run(String[] args) throws Exception {
            addFlag("testFlag", "t", "a simple test flag");
            
            Map<String,String> argMap = parseArguments(args);
            testMap.clear();
            testMap.putAll(argMap);
            return 1;
          }
        };
      }
    };
    
    // testFlag will only be present if speciied on the command-line
    
    String[] noFlag   = new String[0];
    ToolRunner.run(fact.getJob(), noFlag);
    TestCase.assertFalse("test map for absent flag",
        testMap.containsKey("--testFlag"));
    
    String[] withFlag = { "--testFlag" };
    ToolRunner.run(fact.getJob(), withFlag);
    TestCase.assertTrue("test map for present flag",
        testMap.containsKey("--testFlag"));
  }
  
  @Test
  public void testOptions() throws Exception {
    final Map<String,String> testMap = new HashMap<String,String>();
    
    final AbstractJobFactory fact = new AbstractJobFactory() {
      public AbstractJob getJob() {
        return new AbstractJob() {
          @Override
          public int run(String[] args) throws Exception {
            this.addOption(DefaultOptionCreator.overwriteOption().create());
            this.addOption("option", "o", "option");
            this.addOption("required", "r", "required", true /* required */);
            this.addOption("notRequired", "nr", "not required", false /* not required */);
            this.addOption("hasDefault", "hd", "option w/ default", "defaultValue");
            
            
            Map<String,String> argMap = parseArguments(args);
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
    
    int ret;
    
    ret = ToolRunner.run(fact.getJob(), new String[0]);
    TestCase.assertEquals("-1 for missing required options", -1, ret);
    
    ret = ToolRunner.run(fact.getJob(), new String[]{
      "--required", "requiredArg"
    });
    TestCase.assertEquals("0 for no missing required options", 0, ret);
    TestCase.assertEquals("requiredArg", testMap.get("--required"));
    TestCase.assertEquals("defaultValue", testMap.get("--hasDefault"));
    TestCase.assertNull(testMap.get("--option"));
    TestCase.assertNull(testMap.get("--notRequired"));
    TestCase.assertFalse(testMap.containsKey("--overwrite"));
    
    ret = ToolRunner.run(fact.getJob(), new String[]{
      "--required", "requiredArg",
      "--unknownArg"
    });
    TestCase.assertEquals("-1 for including unknown options", -1, ret);

    ret = ToolRunner.run(fact.getJob(), new String[]{
      "--required", "requiredArg",
      "--required", "requiredArg2",
    });
    TestCase.assertEquals("-1 for including duplicate options", -1, ret);
    
    ret = ToolRunner.run(fact.getJob(), new String[]{
      "--required", "requiredArg", 
      "--overwrite",
      "--hasDefault", "nonDefault",
      "--option", "optionValue",
      "--notRequired", "notRequired"
    });
    TestCase.assertEquals("0 for no missing required options", 0, ret);
    TestCase.assertEquals("requiredArg", testMap.get("--required"));
    TestCase.assertEquals("nonDefault", testMap.get("--hasDefault"));
    TestCase.assertEquals("optionValue", testMap.get("--option"));
    TestCase.assertEquals("notRequired", testMap.get("--notRequired"));
    TestCase.assertTrue(testMap.containsKey("--overwrite"));
    
    ret = ToolRunner.run(fact.getJob(), new String[]{
      "-r", "requiredArg", 
      "-ow",
      "-hd", "nonDefault",
      "-o", "optionValue",
      "-nr", "notRequired"
    });
    TestCase.assertEquals("0 for no missing required options", 0, ret);
    TestCase.assertEquals("requiredArg", testMap.get("--required"));
    TestCase.assertEquals("nonDefault", testMap.get("--hasDefault"));
    TestCase.assertEquals("optionValue", testMap.get("--option"));
    TestCase.assertEquals("notRequired", testMap.get("--notRequired"));
    TestCase.assertTrue(testMap.containsKey("--overwrite"));
    
  }
  
  @Test
  public void testInputOutputPaths() throws Exception {
    
    final AbstractJobFactory fact = new AbstractJobFactory() {
      public AbstractJob getJob() {
        return new AbstractJob() {
          @Override
          public int run(String[] args) throws Exception {
            addInputOption();
            addOutputOption();
            
            // arg map should be null if a required option is missing.
            Map<String, String> argMap = parseArguments(args);
            
            if (argMap == null) {
              return -1;
            }
            
            Path inputPath = getInputPath();
            TestCase.assertNotNull("getInputPath() returns non-null", inputPath);
            
            Path outputPath = getInputPath();
            TestCase.assertNotNull("getOutputPath() returns non-null", outputPath);
            return 0;
          }
        };
      }
    };
    
    AbstractJob job;
    int ret;
    
    ret = ToolRunner.run(fact.getJob(), new String[0]);
    TestCase.assertEquals("-1 for missing input option", -1, ret);
    
    final String testInputPath = "testInputPath";
    final String testOutputPath = "testOutputPath";
    final String testInputPropertyPath = "testInputPropertyPath";
    final String testOutputPropertyPath = "testOutputPropertyPath";
    
    job = fact.getJob();
    ret = ToolRunner.run(job, new String[]{ 
        "--input", testInputPath });
    TestCase.assertEquals("-1 for missing output option", -1, ret);
    TestCase.assertEquals("input path is correct", testInputPath, 
        job.getInputPath().toString());
    
    job = fact.getJob();
    ret = ToolRunner.run(job, new String[]{ 
        "--output", testOutputPath });
    TestCase.assertEquals("-1 for missing input option", -1, ret);
    TestCase.assertEquals("output path is correct", testOutputPath, 
        job.getOutputPath().toString());
    
    job = fact.getJob();
    ret = ToolRunner.run(job, new String[]{ 
        "--input", testInputPath, "--output", testOutputPath });
    TestCase.assertEquals("0 for complete options", 0, ret);
    TestCase.assertEquals("input path is correct", testInputPath, 
        job.getInputPath().toString());
    TestCase.assertEquals("output path is correct", testOutputPath, 
        job.getOutputPath().toString());
    
    job = fact.getJob();
    ret = ToolRunner.run(job, new String[]{ 
        "--input", testInputPath, "--output", testOutputPath });
    TestCase.assertEquals("0 for complete options", 0, ret);
    TestCase.assertEquals("input path is correct", testInputPath, 
        job.getInputPath().toString());
    TestCase.assertEquals("output path is correct", testOutputPath, 
        job.getOutputPath().toString());
    
    job = fact.getJob();
    ret = ToolRunner.run(job, new String[]{ 
        "-Dmapred.input.dir=" + testInputPropertyPath, 
        "-Dmapred.output.dir=" + testOutputPropertyPath });
    TestCase.assertEquals("0 for complete options", 0, ret);
    TestCase.assertEquals("input path from property is correct", 
        testInputPropertyPath, job.getInputPath().toString());
    TestCase.assertEquals("output path from property is correct", 
        testOutputPropertyPath, job.getOutputPath().toString());
    
    job = fact.getJob();
    ret = ToolRunner.run(job, new String[]{ 
        "-Dmapred.input.dir=" + testInputPropertyPath,
        "-Dmapred.output.dir=" + testOutputPropertyPath,
        "--input", testInputPath,
        "--output", testOutputPath });
    TestCase.assertEquals("input command-line option precedes property", 
        testInputPath, job.getInputPath().toString());
    TestCase.assertEquals("output command-line option precedes property", 
        testOutputPath, job.getOutputPath().toString());
	}
}
