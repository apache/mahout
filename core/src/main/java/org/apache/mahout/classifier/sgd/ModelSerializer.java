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

package org.apache.mahout.classifier.sgd;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.InstanceCreator;
import com.google.gson.JsonArray;
import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;
import com.google.gson.reflect.TypeToken;
import org.apache.mahout.ep.EvolutionaryProcess;
import org.apache.mahout.ep.Mapping;
import org.apache.mahout.ep.State;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.stats.OnlineAuc;

import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Reader;
import java.lang.reflect.Type;
import java.util.List;

/**
 * Provides the ability to store SGD model-related objects as JSON.
 */
public final class ModelSerializer {

  // thread-local singleton json (de)serializer
  private static final ThreadLocal<Gson> GSON;
  static {
    final GsonBuilder gb = new GsonBuilder();
    gb.registerTypeAdapter(AdaptiveLogisticRegression.class, new AdaptiveLogisticRegressionTypeAdapter());
    gb.registerTypeAdapter(Mapping.class, new MappingTypeAdapter());
    gb.registerTypeAdapter(PriorFunction.class, new PriorTypeAdapter());
    gb.registerTypeAdapter(CrossFoldLearner.class, new CrossFoldLearnerTypeAdapter());
    gb.registerTypeAdapter(Vector.class, new VectorTypeAdapter());
    gb.registerTypeAdapter(Matrix.class, new MatrixTypeAdapter());
    gb.registerTypeAdapter(EvolutionaryProcess.class, new EvolutionaryProcessTypeAdapter());
    gb.registerTypeAdapter(State.class, new StateTypeAdapter());
    GSON = new ThreadLocal<Gson>() {
      @Override
      protected Gson initialValue() {
        return gb.create();
      }
    };
  }

  // static class ... don't instantiate
  private ModelSerializer() {
  }

  public static Gson gson() {
    return GSON.get();
  }

  public static void writeJson(String path, AdaptiveLogisticRegression model) throws IOException {
    OutputStreamWriter out = new FileWriter(path);
    try {
      out.write(gson().toJson(model));
    } finally {
      out.close();
    }
  }

  /**
   * Reads a model in JSON format.
   *
   * @param in Where to read the model from.
   * @param clazz
   * @return The LogisticModelParameters object that we read.
   */
  public static AdaptiveLogisticRegression loadJsonFrom(Reader in, Class<AdaptiveLogisticRegression> clazz) {
    return gson().fromJson(in, clazz);
  }

  private static class MappingTypeAdapter implements JsonDeserializer<Mapping>, JsonSerializer<Mapping> {
    @Override
    public Mapping deserialize(JsonElement jsonElement,
                               Type type,
                               JsonDeserializationContext jsonDeserializationContext) {
      JsonObject x = jsonElement.getAsJsonObject();
      try {
        return jsonDeserializationContext.deserialize(x.get("value"), Class.forName(x.get("class").getAsString()));
      } catch (ClassNotFoundException e) {
        throw new IllegalStateException("Can't understand serialized data, found bad type: "
            + x.get("class").getAsString());
      }
    }

    @Override
    public JsonElement serialize(Mapping mapping, Type type, JsonSerializationContext jsonSerializationContext) {
      JsonObject r = new JsonObject();
      r.add("class", new JsonPrimitive(mapping.getClass().getName()));
      r.add("value", jsonSerializationContext.serialize(mapping));
      return r;
    }
  }

  private static class PriorTypeAdapter implements JsonDeserializer<PriorFunction>, JsonSerializer<PriorFunction> {
    @Override
    public PriorFunction deserialize(JsonElement jsonElement,
                                     Type type,
                                     JsonDeserializationContext jsonDeserializationContext) {
      JsonObject x = jsonElement.getAsJsonObject();
      try {
        return jsonDeserializationContext.deserialize(x.get("value"), Class.forName(x.get("class").getAsString()));
      } catch (ClassNotFoundException e) {
        throw new IllegalStateException("Can't understand serialized data, found bad type: "
            + x.get("class").getAsString());
      }
    }

    @Override
    public JsonElement serialize(PriorFunction priorFunction,
                                 Type type,
                                 JsonSerializationContext jsonSerializationContext) {
      JsonObject r = new JsonObject();
      r.add("class", new JsonPrimitive(priorFunction.getClass().getName()));
      r.add("value", jsonSerializationContext.serialize(priorFunction));
      return r;
    }
  }

  private static class CrossFoldLearnerTypeAdapter implements JsonDeserializer<CrossFoldLearner> {
    @Override
    public CrossFoldLearner deserialize(JsonElement jsonElement,
                                        Type type,
                                        JsonDeserializationContext jsonDeserializationContext) {
      CrossFoldLearner r = new CrossFoldLearner();
      JsonObject x = jsonElement.getAsJsonObject();
      r.setRecord(x.get("record").getAsInt());
      r.setAuc(jsonDeserializationContext.<OnlineAuc>deserialize(x.get("auc"), OnlineAuc.class));
      r.setLogLikelihood(x.get("logLikelihood").getAsDouble());

      JsonArray models = x.get("models").getAsJsonArray();
      for (JsonElement model : models) {
        r.addModel(
            jsonDeserializationContext.<OnlineLogisticRegression>deserialize(model, OnlineLogisticRegression.class));
      }

      r.setParameters(asArray(x, "parameters"));
      r.setNumFeatures(x.get("numFeatures").getAsInt());
      r.setPrior(jsonDeserializationContext.<PriorFunction>deserialize(x.get("prior"), PriorFunction.class));
      return r;
    }
  }

  /**
   * Tells GSON how to (de)serialize a Mahout matrix.  We assume on deserialization that the matrix
   * is dense.
   */
  private static class MatrixTypeAdapter
    implements JsonDeserializer<Matrix>, JsonSerializer<Matrix>, InstanceCreator<Matrix> {
    @Override
    public JsonElement serialize(Matrix m, Type type, JsonSerializationContext jsonSerializationContext) {
      JsonObject r = new JsonObject();
      r.add("rows", new JsonPrimitive(m.numRows()));
      r.add("cols", new JsonPrimitive(m.numCols()));
      JsonArray v = new JsonArray();
      for (int row = 0; row < m.numRows(); row++) {
        JsonArray rowData = new JsonArray();
        for (int col = 0; col < m.numCols(); col++) {
          rowData.add(new JsonPrimitive(m.get(row, col)));
        }
        v.add(rowData);
      }
      r.add("data", v);
      return r;
    }

    @Override
    public Matrix deserialize(JsonElement x, Type type, JsonDeserializationContext jsonDeserializationContext) {
      JsonObject data = x.getAsJsonObject();
      Matrix r = new DenseMatrix(data.get("rows").getAsInt(), data.get("cols").getAsInt());
      int i = 0;
      for (JsonElement row : data.get("data").getAsJsonArray()) {
        int j = 0;
        for (JsonElement element : row.getAsJsonArray()) {
          r.set(i, j, element.getAsDouble());
          j++;
        }
        i++;
      }
      return r;
    }

    @Override
    public Matrix createInstance(Type type) {
      return new DenseMatrix();
    }
  }


  /**
   * Tells GSON how to (de)serialize a Mahout matrix.  We assume on deserialization that the
   * matrix is dense.
   */
  private static class VectorTypeAdapter
    implements JsonDeserializer<Vector>, JsonSerializer<Vector>, InstanceCreator<Vector> {
    @Override
    public JsonElement serialize(Vector m, Type type, JsonSerializationContext jsonSerializationContext) {
      JsonObject r = new JsonObject();
      JsonArray v = new JsonArray();
      for (int i = 0; i < m.size(); i++) {
        v.add(new JsonPrimitive(m.get(i)));
      }
      r.add("data", v);
      return r;
    }

    @Override
    public Vector deserialize(JsonElement x, Type type, JsonDeserializationContext jsonDeserializationContext) {
      JsonArray data = x.getAsJsonObject().get("data").getAsJsonArray();
      Vector r = new DenseVector(data.size());
      int i = 0;
      for (JsonElement v : data) {
        r.set(i, v.getAsDouble());
        i++;
      }
      return r;
    }

    @Override
    public Vector createInstance(Type type) {
      return new DenseVector();
    }
  }

  private static class StateTypeAdapter implements JsonSerializer<State<AdaptiveLogisticRegression.Wrapper>>,
    JsonDeserializer<State<AdaptiveLogisticRegression.Wrapper>> {
    @Override
    public State<AdaptiveLogisticRegression.Wrapper> deserialize(
      JsonElement jsonElement, Type type, JsonDeserializationContext jsonDeserializationContext) {

      JsonObject v = (JsonObject) jsonElement;
      double[] params = asArray(v, "params");
      double omni = v.get("omni").getAsDouble();
      State<AdaptiveLogisticRegression.Wrapper> r = new State<AdaptiveLogisticRegression.Wrapper>(params, omni);

      double[] step = asArray(v, "step");
      r.setId(v.get("id").getAsInt());
      r.setStep(step);
      r.setValue(v.get("value").getAsDouble());

      Type mapListType = new TypeToken<List<Mapping>>() {}.getType();
      r.setMaps(jsonDeserializationContext.<List<Mapping>>deserialize(v.get("maps"), mapListType));

      r.setPayload(
          jsonDeserializationContext.<AdaptiveLogisticRegression.Wrapper>deserialize(
              v.get("payload"),
              AdaptiveLogisticRegression.Wrapper.class));
      return r;
    }

    @Override
    public JsonElement serialize(State<AdaptiveLogisticRegression.Wrapper> state,
                                 Type type,
                                 JsonSerializationContext jsonSerializationContext) {
      JsonObject r = new JsonObject();
      r.add("id", new JsonPrimitive(state.getId()));
      JsonArray v = new JsonArray();
      for (double x : state.getParams()) {
        v.add(new JsonPrimitive(x));
      }
      r.add("params", v);

      v = new JsonArray();
      for (Mapping mapping : state.getMaps()) {
        v.add(jsonSerializationContext.serialize(mapping, Mapping.class));
      }
      r.add("maps", v);
      r.add("omni", new JsonPrimitive(state.getOmni()));
      r.add("step", jsonSerializationContext.serialize(state.getStep()));
      r.add("value", new JsonPrimitive(state.getValue()));
      r.add("payload", jsonSerializationContext.serialize(state.getPayload()));

      return r;
    }
  }

  private static class AdaptiveLogisticRegressionTypeAdapter implements JsonSerializer<AdaptiveLogisticRegression>,
    JsonDeserializer<AdaptiveLogisticRegression> {

    @Override
    public AdaptiveLogisticRegression deserialize(JsonElement element, Type type, JsonDeserializationContext jdc) {
      JsonObject x = element.getAsJsonObject();
      AdaptiveLogisticRegression r =
          new AdaptiveLogisticRegression(x.get("numCategories").getAsInt(),
                                         x.get("numFeatures").getAsInt(),
                                         jdc.<PriorFunction>deserialize(x.get("prior"), PriorFunction.class));
      Type stateType = new TypeToken<State<AdaptiveLogisticRegression.Wrapper>>() {}.getType();
      r.setEvaluationInterval(x.get("evaluationInterval").getAsInt());
      r.setRecord(x.get("record").getAsInt());

      Type epType = new TypeToken<EvolutionaryProcess<AdaptiveLogisticRegression.Wrapper>>() {}.getType();
      r.setEp(jdc.<EvolutionaryProcess<AdaptiveLogisticRegression.Wrapper>>deserialize(x.get("ep"), epType));
      r.setSeed(jdc.<State<AdaptiveLogisticRegression.Wrapper>>deserialize(x.get("seed"), stateType));
      r.setBest(jdc.<State<AdaptiveLogisticRegression.Wrapper>>deserialize(x.get("best"), stateType));

      r.setBuffer(jdc.<List<AdaptiveLogisticRegression.TrainingExample>>deserialize(x.get("buffer"),
                  new TypeToken<List<AdaptiveLogisticRegression.TrainingExample>>() {}.getType()));
      return r;
    }

    @Override
    public JsonElement serialize(AdaptiveLogisticRegression x, Type type, JsonSerializationContext jsc) {
      JsonObject r = new JsonObject();
      r.add("ep", jsc.serialize(x.getEp(),
          new TypeToken<EvolutionaryProcess<AdaptiveLogisticRegression.Wrapper>>() {}.getType()));
      r.add("buffer", jsc.serialize(x.getBuffer(),
          new TypeToken<List<AdaptiveLogisticRegression.TrainingExample>>() {}.getType()));
      r.add("evaluationInterval", jsc.serialize(x.getEvaluationInterval()));
      Type stateType = new TypeToken<State<AdaptiveLogisticRegression.Wrapper>>() {}.getType();
      r.add("best", jsc.serialize(x.getBest(), stateType));
      r.add("numFeatures", jsc.serialize(x.getNumFeatures()));
      r.add("numCategories", jsc.serialize(x.getNumCategories()));
      PriorFunction prior = x.getPrior();
      JsonElement pf = jsc.serialize(prior, PriorFunction.class);
      r.add("prior", pf);
      r.add("record", jsc.serialize(x.getRecord()));
      r.add("seed", jsc.serialize(x.getSeed(), stateType));
      return r;
    }
  }

  private static class EvolutionaryProcessTypeAdapter implements
    InstanceCreator<EvolutionaryProcess<AdaptiveLogisticRegression.Wrapper>>,
    JsonDeserializer<EvolutionaryProcess<AdaptiveLogisticRegression.Wrapper>>,
    JsonSerializer<EvolutionaryProcess<AdaptiveLogisticRegression.Wrapper>> {
    private static final Type STATE_TYPE = new TypeToken<State<AdaptiveLogisticRegression.Wrapper>>() {}.getType();

    @Override
    public EvolutionaryProcess<AdaptiveLogisticRegression.Wrapper> createInstance(Type type) {
      return new EvolutionaryProcess<AdaptiveLogisticRegression.Wrapper>();
    }

    @Override
    public EvolutionaryProcess<AdaptiveLogisticRegression.Wrapper> deserialize(
        JsonElement jsonElement, Type type, JsonDeserializationContext jsonDeserializationContext) {
      JsonObject x = (JsonObject) jsonElement;
      int threadCount = x.get("threadCount").getAsInt();

      EvolutionaryProcess<AdaptiveLogisticRegression.Wrapper> r =
          new EvolutionaryProcess<AdaptiveLogisticRegression.Wrapper>();
      r.setThreadCount(threadCount);

      for (JsonElement element : x.get("population").getAsJsonArray()) {
        State<AdaptiveLogisticRegression.Wrapper> state = jsonDeserializationContext.deserialize(element, STATE_TYPE);
        r.add(state);
      }
      return r;
    }

    @Override
    public JsonElement serialize(EvolutionaryProcess<AdaptiveLogisticRegression.Wrapper> x,
                                 Type type,
                                 JsonSerializationContext jsc) {
      JsonObject r = new JsonObject();
      r.add("threadCount", new JsonPrimitive(x.getThreadCount()));
      JsonArray v = new JsonArray();
      for (State<AdaptiveLogisticRegression.Wrapper> state : x.getPopulation()) {
        v.add(jsc.serialize(state, STATE_TYPE));
      }
      r.add("population", v);
      return r;
    }
  }

  public static double[] asArray(JsonObject v, String name) {
    JsonArray x = v.get(name).getAsJsonArray();
    double[] params = new double[x.size()];
    int i = 0;
    for (JsonElement element : x) {
      params[i++] = element.getAsDouble();
    }
    return params;
  }

}
