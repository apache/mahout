@echo off

echo "===============DEPRECATION WARNING==============="
echo "This script is no longer supported for new drivers as of Mahout 0.10.0"
echo "Mahout's bash script is supported and if someone wants to contribute a fix for this"
echo "it would be appreciated."


@rem
@rem The Mahout command script
@rem
@rem Environment Variables
@rem
@rem MAHOUT_JAVA_HOME The java implementation to use. Overrides JAVA_HOME.
@rem
@rem MAHOUT_HEAPSIZE The maximum amount of heap to use, in MB.
@rem Default is 1000.
@rem
@rem HADOOP_CONF_DIR The location of a hadoop config directory
@rem
@rem MAHOUT_OPTS Extra Java runtime options.
@rem
@rem MAHOUT_CONF_DIR The location of the program short-name to class name
@rem mappings and the default properties files
@rem defaults to "$MAHOUT_HOME/src/conf"
@rem
@rem MAHOUT_LOCAL set to anything other than an empty string to force
@rem mahout to run locally even if
@rem HADOOP_CONF_DIR and HADOOP_HOME are set
@rem
@rem MAHOUT_CORE set to anything other than an empty string to force
@rem mahout to run in developer 'core' mode, just as if the
@rem -core option was presented on the command-line
@rem Commane-line Options
@rem
@rem -core -core is used to switch into 'developer mode' when
@rem running mahout locally. If specified, the classes
@rem from the 'target/classes' directories in each project
@rem are used. Otherwise classes will be retrived from
@rem jars in the binary releas collection or *-job.jar files
@rem found in build directories. When running on hadoop
@rem the job files will always be used.

@rem
@rem /*
@rem * Licensed to the Apache Software Foundation (ASF) under one or more
@rem * contributor license agreements. See the NOTICE file distributed with
@rem * this work for additional information regarding copyright ownership.
@rem * The ASF licenses this file to You under the Apache License, Version 2.0
@rem * (the "License"); you may not use this file except in compliance with
@rem * the License. You may obtain a copy of the License at
@rem *
@rem * http://www.apache.org/licenses/LICENSE-2.0
@rem *
@rem * Unless required by applicable law or agreed to in writing, software
@rem * distributed under the License is distributed on an "AS IS" BASIS,
@rem * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@rem * See the License for the specific language governing permissions and
@rem * limitations under the License.
@rem */

setlocal enabledelayedexpansion

@rem disable "developer mode"
set IS_CORE=0
if [%1] == [-core] (
  set IS_CORE=1
  shift
)

if not [%MAHOUT_CORE%] == [] (
set IS_CORE=1
)

if [%MAHOUT_HOME%] == [] set MAHOUT_HOME=%~dp0..

echo "Mahout home set %MAHOUT_HOME%"

@rem some Java parameters
if not [%MAHOUT_JAVA_HOME%] == [] (
@rem echo run java in %MAHOUT_JAVA_HOME%
set JAVA_HOME=%MAHOUT_JAVA_HOME%
)

if [%JAVA_HOME%] == [] (
    echo Error: JAVA_HOME is not set.
    exit /B 1
)

set JAVA=%JAVA_HOME%\bin\java
set JAVA_HEAP_MAX=-Xmx3g

@rem check envvars which might override default args
if not [%MAHOUT_HEAPSIZE%] == [] (
@rem echo run with heapsize %MAHOUT_HEAPSIZE%
set JAVA_HEAP_MAX=-Xmx%MAHOUT_HEAPSIZE%m
@rem echo %JAVA_HEAP_MAX%
)

if [%MAHOUT_CONF_DIR%] == [] (
set MAHOUT_CONF_DIR=%MAHOUT_HOME%\conf
)

:main
@rem MAHOUT_CLASSPATH initially contains $MAHOUT_CONF_DIR, or defaults to $MAHOUT_HOME/src/conf
set CLASSPATH=%CLASSPATH%;%MAHOUT_CONF_DIR%

if not [%MAHOUT_LOCAL%] == [] (
echo "MAHOUT_LOCAL is set, so we do not add HADOOP_CONF_DIR to classpath."
) else (
if not [%HADOOP_CONF_DIR%] == [] (
echo "MAHOUT_LOCAL is not set; adding HADOOP_CONF_DIR to classpath."
set CLASSPATH=%CLASSPATH%;%HADOOP_CONF_DIR%
)
)

set CLASSPATH=%CLASSPATH%;%JAVA_HOME%\lib\tools.jar

if %IS_CORE% == 0 (
@rem add release dependencies to CLASSPATH
for %%f in (%MAHOUT_HOME%\mahout-*.jar) do (
set CLASSPATH=!CLASSPATH!;%%f
)
@rem add dev targets if they exist
for %%f in (%MAHOUT_HOME%\examples\target\mahout-examples-*-job.jar) do (
set CLASSPATH=!CLASSPATH!;%%f
)
for %%f in (%MAHOUT_HOME%\mahout-examples-*-job.jar) do (
set CLASSPATH=!CLASSPATH!;%%f
)
@rem add release dependencies to CLASSPATH
for %%f in (%MAHOUT_HOME%\lib\*.jar) do (
set CLASSPATH=!CLASSPATH!;%%f
)
) else (
set CLASSPATH=!CLASSPATH!;%MAHOUT_HOME%\math\target\classes
set CLASSPATH=!CLASSPATH!;%MAHOUT_HOME%\core\target\classes
set CLASSPATH=!CLASSPATH!;%MAHOUT_HOME%\integration\target\classes
set CLASSPATH=!CLASSPATH!;%MAHOUT_HOME%\examples\target\classes
@rem set CLASSPATH=%CLASSPATH%;%MAHOUT_HOME%\core\src\main\resources
)

@rem add development dependencies to CLASSPATH
for %%f in (%MAHOUT_HOME%\examples\target\dependency\*.jar) do (
set CLASSPATH=!CLASSPATH!;%%f
)

@rem default log directory & file
if [%MAHOUT_LOG_DIR%] == [] (
set MAHOUT_LOG_DIR=%MAHOUT_HOME%\logs
)
if [%MAHOUT_LOGFILE%] == [] (
set MAHOUT_LOGFILE=mahout.log
)

set MAHOUT_OPTS=%MAHOUT_OPTS% -Dhadoop.log.dir=%MAHOUT_LOG_DIR%
set MAHOUT_OPTS=%MAHOUT_OPTS% -Dhadoop.log.file=%MAHOUT_LOGFILE%

if not [%JAVA_LIBRARY_PATH%] == [] (
set MAHOUT_OPTS=%MAHOUT_OPTS% -Djava.library.path=%JAVA_LIBRARY_PATH%
)

set CLASS=org.apache.mahout.driver.MahoutDriver

for %%f in (%MAHOUT_HOME%\examples\target\mahout-examples-*-job.jar) do (
set MAHOUT_JOB=%%f
)

@rem run it

if not [%MAHOUT_LOCAL%] == [] (
    echo "MAHOUT_LOCAL is set, running locally"
    %JAVA% %JAVA_HEAP_MAX% %MAHOUT_OPTS% -classpath %MAHOUT_CLASSPATH% %CLASS% %*
) else (
    if [%MAHOUT_JOB%] == [] (
        echo "ERROR: Could not find mahout-examples-*.job in %MAHOUT_HOME% or %MAHOUT_HOME%/examples/target, please run 'mvn install' to create the .job file"
        exit /B 1
    ) else (
        set HADOOP_CLASSPATH=%MAHOUT_CLASSPATH%
        if /i [%1] == [hadoop] (
shift
set HADOOP_CLASSPATH=%MAHOUT_CONF_DIR%;%HADOOP_CLASSPATH%
            call %HADOOP_HOME%\bin\%*
        ) else (
if /i [%1] == [classpath] (
echo %CLASSPATH%
) else (
echo MAHOUT_JOB: %MAHOUT_JOB%
set HADOOP_CLASSPATH=%MAHOUT_CONF_DIR%;%HADOOP_CLASSPATH%
set HADOOP_CLIENT_OPTS=%JAVA_HEAP_MAX%
call %HADOOP_HOME%\bin\hadoop jar %MAHOUT_JOB% %CLASS% %*
)
            
        )
    )
)
@echo off

@rem
@rem The Mahout command script
@rem
@rem Environment Variables
@rem
@rem MAHOUT_JAVA_HOME The java implementation to use. Overrides JAVA_HOME.
@rem
@rem MAHOUT_HEAPSIZE The maximum amount of heap to use, in MB.
@rem Default is 1000.
@rem
@rem HADOOP_CONF_DIR The location of a hadoop config directory
@rem
@rem MAHOUT_OPTS Extra Java runtime options.
@rem
@rem MAHOUT_CONF_DIR The location of the program short-name to class name
@rem mappings and the default properties files
@rem defaults to "$MAHOUT_HOME/src/conf"
@rem
@rem MAHOUT_LOCAL set to anything other than an empty string to force
@rem mahout to run locally even if
@rem HADOOP_CONF_DIR and HADOOP_HOME are set
@rem
@rem MAHOUT_CORE set to anything other than an empty string to force
@rem mahout to run in developer 'core' mode, just as if the
@rem -core option was presented on the command-line
@rem Commane-line Options
@rem
@rem -core -core is used to switch into 'developer mode' when
@rem running mahout locally. If specified, the classes
@rem from the 'target/classes' directories in each project
@rem are used. Otherwise classes will be retrived from
@rem jars in the binary releas collection or *-job.jar files
@rem found in build directories. When running on hadoop
@rem the job files will always be used.

@rem
@rem /*
@rem * Licensed to the Apache Software Foundation (ASF) under one or more
@rem * contributor license agreements. See the NOTICE file distributed with
@rem * this work for additional information regarding copyright ownership.
@rem * The ASF licenses this file to You under the Apache License, Version 2.0
@rem * (the "License"); you may not use this file except in compliance with
@rem * the License. You may obtain a copy of the License at
@rem *
@rem * http://www.apache.org/licenses/LICENSE-2.0
@rem *
@rem * Unless required by applicable law or agreed to in writing, software
@rem * distributed under the License is distributed on an "AS IS" BASIS,
@rem * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@rem * See the License for the specific language governing permissions and
@rem * limitations under the License.
@rem */

setlocal enabledelayedexpansion

@rem disable "developer mode"
set IS_CORE=0
if [%1] == [-core] (
  set IS_CORE=1
  shift
)

if not [%MAHOUT_CORE%] == [] (
set IS_CORE=1
)

if [%MAHOUT_HOME%] == [] set MAHOUT_HOME=%~dp0..

echo "Mahout home set %MAHOUT_HOME%"

@rem some Java parameters
if not [%MAHOUT_JAVA_HOME%] == [] (
@rem echo run java in %MAHOUT_JAVA_HOME%
set JAVA_HOME=%MAHOUT_JAVA_HOME%
)

if [%JAVA_HOME%] == [] (
    echo Error: JAVA_HOME is not set.
    exit /B 1
)

set JAVA=%JAVA_HOME%\bin\java
set JAVA_HEAP_MAX=-Xmx3g

@rem check envvars which might override default args
if not [%MAHOUT_HEAPSIZE%] == [] (
@rem echo run with heapsize %MAHOUT_HEAPSIZE%
set JAVA_HEAP_MAX=-Xmx%MAHOUT_HEAPSIZE%m
@rem echo %JAVA_HEAP_MAX%
)

if [%MAHOUT_CONF_DIR%] == [] (
set MAHOUT_CONF_DIR=%MAHOUT_HOME%\conf
)

:main
@rem MAHOUT_CLASSPATH initially contains $MAHOUT_CONF_DIR, or defaults to $MAHOUT_HOME/src/conf
set CLASSPATH=%CLASSPATH%;%MAHOUT_CONF_DIR%

if not [%MAHOUT_LOCAL%] == [] (
echo "MAHOUT_LOCAL is set, so we do not add HADOOP_CONF_DIR to classpath."
) else (
if not [%HADOOP_CONF_DIR%] == [] (
echo "MAHOUT_LOCAL is not set; adding HADOOP_CONF_DIR to classpath."
set CLASSPATH=%CLASSPATH%;%HADOOP_CONF_DIR%
)
)

set CLASSPATH=%CLASSPATH%;%JAVA_HOME%\lib\tools.jar

if %IS_CORE% == 0 (
@rem add release dependencies to CLASSPATH
for %%f in (%MAHOUT_HOME%\mahout-*.jar) do (
set CLASSPATH=!CLASSPATH!;%%f
)
@rem add dev targets if they exist
for %%f in (%MAHOUT_HOME%\examples\target\mahout-examples-*-job.jar) do (
set CLASSPATH=!CLASSPATH!;%%f
)
for %%f in (%MAHOUT_HOME%\mahout-examples-*-job.jar) do (
set CLASSPATH=!CLASSPATH!;%%f
)
@rem add release dependencies to CLASSPATH
for %%f in (%MAHOUT_HOME%\lib\*.jar) do (
set CLASSPATH=!CLASSPATH!;%%f
)
) else (
set CLASSPATH=!CLASSPATH!;%MAHOUT_HOME%\math\target\classes
set CLASSPATH=!CLASSPATH!;%MAHOUT_HOME%\core\target\classes
set CLASSPATH=!CLASSPATH!;%MAHOUT_HOME%\integration\target\classes
set CLASSPATH=!CLASSPATH!;%MAHOUT_HOME%\examples\target\classes
@rem set CLASSPATH=%CLASSPATH%;%MAHOUT_HOME%\core\src\main\resources
)

@rem add development dependencies to CLASSPATH
for %%f in (%MAHOUT_HOME%\examples\target\dependency\*.jar) do (
set CLASSPATH=!CLASSPATH!;%%f
)

@rem default log directory & file
if [%MAHOUT_LOG_DIR%] == [] (
set MAHOUT_LOG_DIR=%MAHOUT_HOME%\logs
)
if [%MAHOUT_LOGFILE%] == [] (
set MAHOUT_LOGFILE=mahout.log
)

set MAHOUT_OPTS=%MAHOUT_OPTS% -Dhadoop.log.dir=%MAHOUT_LOG_DIR%
set MAHOUT_OPTS=%MAHOUT_OPTS% -Dhadoop.log.file=%MAHOUT_LOGFILE%
set MAHOUT_OPTS=%MAHOUT_OPTS% -Dmapred.min.split.size=512MB
set MAHOUT_OPTS=%MAHOUT_OPTS% -Dmapred.map.child.java.opts=-Xmx4096m
set MAHOUT_OPTS=%MAHOUT_OPTS% -Dmapred.reduce.child.java.opts=-Xmx4096m
set MAHOUT_OPTS=%MAHOUT_OPTS% -Dmapred.output.compress=true
set MAHOUT_OPTS=%MAHOUT_OPTS% -Dmapred.compress.map.output=true
set MAHOUT_OPTS=%MAHOUT_OPTS% -Dmapred.map.tasks=1
set MAHOUT_OPTS=%MAHOUT_OPTS% -Dmapred.reduce.tasks=1
set MAHOUT_OPTS=%MAHOUT_OPTS% -Dio.sort.factor=30
set MAHOUT_OPTS=%MAHOUT_OPTS% -Dio.sort.mb=1024
set MAHOUT_OPTS=%MAHOUT_OPTS% -Dio.file.buffer.size=32786
set HADOOP_OPTS=%HADOOP_OPTS% -Djava.library.path=%HADOOP_HOME%\bin

if not [%JAVA_LIBRARY_PATH%] == [] (
set MAHOUT_OPTS=%MAHOUT_OPTS% -Djava.library.path=%JAVA_LIBRARY_PATH%
)

set CLASS=org.apache.mahout.driver.MahoutDriver

for %%f in (%MAHOUT_HOME%\examples\target\mahout-examples-*-job.jar) do (
set MAHOUT_JOB=%%f
)

@rem run it

if not [%MAHOUT_LOCAL%] == [] (
    echo "MAHOUT_LOCAL is set, running locally"
    %JAVA% %JAVA_HEAP_MAX% %MAHOUT_OPTS% -classpath %MAHOUT_CLASSPATH% %CLASS% %*
) else (
    if [%MAHOUT_JOB%] == [] (
        echo "ERROR: Could not find mahout-examples-*.job in %MAHOUT_HOME% or %MAHOUT_HOME%/examples/target, please run 'mvn install' to create the .job file"
        exit /B 1
    ) else (
        set HADOOP_CLASSPATH=%MAHOUT_CLASSPATH%
        if /i [%1] == [hadoop] (
shift
set HADOOP_CLASSPATH=%MAHOUT_CONF_DIR%;%HADOOP_CLASSPATH%
            call %HADOOP_HOME%\bin\%*
        ) else (
if /i [%1] == [classpath] (
echo %CLASSPATH%
) else (
echo MAHOUT_JOB: %MAHOUT_JOB%
set HADOOP_CLASSPATH=%MAHOUT_CONF_DIR%;%HADOOP_CLASSPATH%
set HADOOP_CLIENT_OPTS=%JAVA_HEAP_MAX%
call %HADOOP_HOME%\bin\hadoop jar %MAHOUT_JOB% %CLASS% %*
)
            
        )
    )
)
