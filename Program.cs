using System;
using Microsoft.ML;

namespace MatrixFactorisation
{
    class Program
    {
        static void Main(string[] args)
        {
            //STEP 1: Create MLContext to be shared across the model creation workflow objects 
            MLContext mlContext = new MLContext();

            ITransformer model = ModelHelper.Load(mlContext);

            var predictionengine = mlContext.Model.CreatePredictionEngine<ProductEntry, Copurchase_prediction>(model);
            var prediction = predictionengine.Predict(
                new ProductEntry()
                {
                    ProductID = 3,
                    CoPurchaseProductID = 23
                }
            );

            Console.WriteLine(prediction.Score.ToString());
            Console.ReadLine();
        }
    }
}
