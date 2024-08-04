package com.example;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.PipelineStage;

public class WineQualityPrediction {
    public static void main(String[] args) {
        try {
            // Configure Spark
            SparkConf conf = new SparkConf().setAppName("WineQualityPrediction").setMaster("local[*]");
            JavaSparkContext sc = new JavaSparkContext(conf);
            SparkSession spark = SparkSession.builder().config(conf).getOrCreate();

            // Load training data
            Dataset<Row> trainingData = spark.read().option("header", "true").option("inferSchema", "true").csv("s3a://wine-quality-data-bucket/TrainingDataset.csv");

            // Feature engineering
            VectorAssembler assembler = new VectorAssembler().setInputCols(new String[]{
                    "fixed_acidity", "volatile_acidity", "citric_acid", 
                    "residual_sugar", "chlorides", "free_sulfur_dioxide",
                    "total_sulfur_dioxide", "density", "pH", 
                    "sulphates", "alcohol"}) // Adjust these features
                                                              .setOutputCol("features");

            // Train a logistic regression model
            LogisticRegression lr = new LogisticRegression().setLabelCol("quality").setFeaturesCol("features");

            // Create a pipeline
            Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{assembler, lr});

            // Train model
            PipelineModel model = pipeline.fit(trainingData);

            // Save the model
            model.write().overwrite().save("s3a://wine-quality-data-bucket/wine-quality-model");

            // Stop Spark
            sc.stop();
            spark.stop();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
