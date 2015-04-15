package org.elte.hadoop.loris.spark;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.DistributedLDAModel;
import org.apache.spark.mllib.clustering.LDA;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import scala.Tuple2;

/**
 * Hello world!
 *
 */
public class App {
	public static void main(String[] args) {
		try {
			String ip = new String("172.31.14.253");
			String port = new String("7077");
			SparkConf conf = new SparkConf().setAppName("LDA example")
					.setMaster("spark://" + ip + ":" + port + "");

			JavaSparkContext sc = new JavaSparkContext(conf);

			String path = "resources/lda_data.txt";

			JavaRDD<String> data = sc.textFile(path);
			JavaRDD<Vector> parsedData = data
					.map(new Function<String, Vector>() {
						// call?
						public Vector call(String s) {
							// space mentén végni a szavakat
							String[] sarray = s.trim().split(" ");
							// szóhossz méretű vector
							double[] values = new double[sarray.length];

							for (int i = 0; i < sarray.length; i++) {
								// double értékké alakítja a szavakat
								values[i] = Double.parseDouble(sarray[i]);
							}

							// sűrűség?
							return Vectors.dense(values);
						}
					});

			JavaPairRDD<Long, Vector> corpus = JavaPairRDD
					.fromJavaRDD(parsedData
							.zipWithIndex()
							.map(new Function<Tuple2<Vector, Long>, Tuple2<Long, Vector>>() {
								// függvény:
								public Tuple2<Long, Vector> call(
										Tuple2<Vector, Long> doc_id) {
									return doc_id.swap();
								}
							}));

			DistributedLDAModel ldaModel = new LDA().setK(3).run(corpus);

			System.out
					.println("Learned topics (as distributions over vocab of "
							+ ldaModel.vocabSize() + " words):");

			Matrix topics = ldaModel.topicsMatrix();
			for (int topic = 0; topic < 3; topic++) {
				System.out.print("Topic " + topic + ":");
				for (int word = 0; word < ldaModel.vocabSize(); word++) {
					System.out.print(" " + topics.apply(word, topic));
				}
				System.out.println();
			}

			sc.stop();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
