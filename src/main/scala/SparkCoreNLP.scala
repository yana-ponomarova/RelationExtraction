import java.util.Properties

import edu.stanford.nlp.pipeline.StanfordCoreNLP

/**
  * Created by nico on 04/01/2016.
  */
class SparkCoreNLP (properties : Properties) extends Serializable {
  @transient private var pipeline: StanfordCoreNLP = _

  def get: StanfordCoreNLP  = {
    if (pipeline  == null) {
      pipeline = new StanfordCoreNLP(properties)
    }
    pipeline
  }
}
