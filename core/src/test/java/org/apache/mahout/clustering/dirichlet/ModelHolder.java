package org.apache.mahout.clustering.dirichlet;

import org.apache.mahout.clustering.dirichlet.models.Model;

class ModelHolder<Observation> {
  public Model<Observation> model;

  public ModelHolder() {
  }

  public ModelHolder(Model<Observation> model) {
    this.model = model;
  }
}
