##############
'''
https://spark.apache.org/docs/latest/ml-features.html#onehotencoder
'''

import os
import pyspark
from pyspark import *
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import *
from pyspark.ml.tuning import *
from pyspark.ml.feature import *
from pyspark.ml.classification import *
from pyspark.ml.evaluation import *
from pyspark.ml.classification import *
from pyspark.mllib.evaluation import *

sc = SparkContext("local")
sqlContext = SparkSession.builder.getOrCreate()

df = sqlContext.createDataFrame(
    [
    ("positive", "a", 1.0,170, "m"), 
    ("negatvie", "b", 0.3,159, "f"), 
    ("positive", "c", 0.4,180, "m"), 
    ("negatvie", "d", 0.0,194, "f"), 
    ("positive", "a", 1.0,170, "m"), 
    ("negatvie", "b", 0.3,159, "f"), 
    ("positive", "c", 0.4,180, "m"), 
    ("negatvie", "d", 0.0,194, "f"), 
    ("positive", "a", 1.0,170, "m"), 
    ("negatvie", "b", 0.3,159, "f"), 
    ("positive", "c", 0.4,180, "m"), 
    ("negatvie", "d", 0.0,194, "f"), 
    ("positive", "a", 1.0,170, "m"), 
    ("negatvie", "b", 0.3,159, "f"), 
    ("positive", "c", 0.4,180, "m"), 
    ("negatvie", "d", 0.0,194, "f"), 
    ("positive", "a", 1.0,170, "m"), 
    ("negatvie", "b", 0.3,159, "f"), 
    ("positive", "c", 0.4,180, "m"), 
    ("negatvie", "d", 0.0,194, "f"), 
    ("positive", "a", 1.0,170, "m"), 
    ("negatvie", "b", 0.3,159, "f"), 
    ("positive", "c", 0.4,180, "m"), 
    ("negatvie", "d", 0.0,194, "f"), 
    ],
    ["label_str", "country", "income", "hight", "gender"])


############

label_index = StringIndexer(
	inputCol="label_str", 
	outputCol="label")

country_index = StringIndexer(
	inputCol="country", 
	outputCol="country_index")

gender_index = StringIndexer(
	inputCol="gender", 
	outputCol="gender_index")

country_vector = OneHotEncoder(
	inputCols=["country_index", "gender_index"],
	outputCols=["country_vector", "gender_vector"])

assembler = VectorAssembler(
    inputCols=[
    "income",
    "hight",
    "country_vector", 
    "gender_vector"],
    outputCol="features_original")

scaler = StandardScaler(
	inputCol="features_original", 
	outputCol="features",
	withStd = True, 
	withMean = False)

lr = LogisticRegression(
	maxIter = 10,
	regParam = 0.001,
	labelCol="label",
	featuresCol="features")


pipeline = Pipeline(
	stages=[
		label_index,
		country_index, 
		gender_index, 
		country_vector, 
		assembler,
		scaler,
		lr
	])


'''
model = pipeline.fit(df)
df1 = model.transform(df)
df1.show()

df1.select(
	df1["label"],
	df1["features_original"],
	df1["features"],
	).write.mode('Overwrite').json('df1')

os.system(u"""
	cat df1/*
	""")
'''

##########

paramGrid = ParamGridBuilder()\
    .addGrid(lr.regParam, [1, 0.1, 0.01]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
    .build()

crossval = CrossValidator(
	estimator = pipeline,
	estimatorParamMaps = paramGrid,
	evaluator = BinaryClassificationEvaluator(),
	numFolds = 3)

train, test = df.randomSplit([0.5, 0.5], seed=12345)

cvModel = crossval.fit(train)

cvModel.transform(train)\
    .select("features", "label", "prediction")\
    .show()

cvModel.write().overwrite().save("model")

########

model_best = CrossValidatorModel.load("model").bestModel

test_prediction = model_best.transform(test)\
    .select("features", "label", "prediction")

test_prediction.show()
