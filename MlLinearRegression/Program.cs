using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;
using System;
using System.Linq;

namespace MlLinearRegression
{
    class Program
    {
        static readonly string path = "..\\..\\..\\poverty.csv";
        static void Main(string[] args)
        {
            var context = new MLContext(seed: 0);

            //Train the model
            var data = context.Data.LoadFromTextFile<MyInput>(path, hasHeader: true, separatorChar: ',');
            var traintestdata = context.Data.TrainTestSplit(data:data,testFraction:0.1, seed:0 );

            var traindata = traintestdata.TrainSet;
            var testdata = traintestdata.TestSet;


            var options = new FastForestRegressionTrainer.Options
            {
                //Only use 80% of features to reduce overfitting
                FeatureFraction = 0.9,

                //Simplify the model by penalizing the usage of new features
                FeatureFirstUsePenalty = 0.1,

                //Limit the number of trees to 60
                NumberOfTrees = 60
            };

            var pipeline = context.Transforms.NormalizeMinMax("PovertyRate")
                .Append(context.Transforms.Concatenate("Features", "PovertyRate"))
                .Append(context.Regression.Trainers.Ols());

            var model = pipeline.Fit(traindata);

            var random = new Random();
            for (int i = 1; i <= 3; i++)
            {
                //use the model to make the prediction

                /*var predictor = context.Model.CreatePredictionEngine<MyInput, MyOutput>(model);
                var pr = float.Parse(Math.Round(MathF.E * random.Next(6, 10), 2).ToString());

                Console.WriteLine($"Predicted poverty rate: {pr}");

                var input = new MyInput { PovertyRate = pr} ;
                var prediction = predictor.Predict(input);

                Console.WriteLine($"Predicted birth rate {prediction.BirthRate:0.##}");
                Console.WriteLine($"Actual birth rate 58.10 This is hard coded. \n");*/



                //evaluate the model
                var predictions = model.Transform(testdata);
                var metrics = context.Regression.Evaluate(predictions);
               // Console.WriteLine($"Accuracy R2 {metrics.RSquared:0.##} ");

                //evaluation the model by cross validiating
                var scores = context.Regression.CrossValidate(data, pipeline, numberOfFolds:5);
                var mean = scores.Average(x => x.Metrics.RSquared);
                Console.WriteLine($"Mean cross validated R2 {mean:0.##} ");
            }
        }
    }
}
