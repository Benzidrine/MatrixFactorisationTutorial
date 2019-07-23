using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace MatrixFactorisation
{
    public class ModelHelper
    {        
        
        private static string BaseDataSetRelativePath = @"/Users/benzidrine/Repos/MLTireweb/MatrixFactorisation/Data";
        private static string TrainingDataRelativePath = $"{BaseDataSetRelativePath}/Amazon0302.txt";
        private static string TrainingDataLocation = Helper.GetAbsolutePath(TrainingDataRelativePath);

        private static string BaseModelRelativePath = @"/Users/benzidrine/Repos/MLTireweb/MatrixFactorisation/Model";
        private static string ModelRelativePath = $"{BaseModelRelativePath}/model.zip";
        private static string ModelPath = Helper.GetAbsolutePath(ModelRelativePath);
        
        public static ITransformer Build(MLContext mlContext )
        {
            //Read the trained data using TextLoader by defining the schema for reading the product co-purchase dataset
            var traindata = mlContext.Data.LoadFromTextFile(path:TrainingDataLocation,
                columns: new[]
                            {
                                new TextLoader.Column("Label", DataKind.Single, 0),
                                new TextLoader.Column(name:nameof(ProductEntry.ProductID), dataKind:DataKind.UInt32, source: new [] { new TextLoader.Range(0) }, keyCount: new KeyCount(262111)), 
                                new TextLoader.Column(name:nameof(ProductEntry.CoPurchaseProductID), dataKind:DataKind.UInt32, source: new [] { new TextLoader.Range(1) }, keyCount: new KeyCount(262111))
                            },
                hasHeader: true,
                separatorChar: '\t');

            //Your data is already encoded so all you need to do is specify options for MatrxiFactorizationTrainer with a few extra hyperparameters
            MatrixFactorizationTrainer.Options options = new MatrixFactorizationTrainer.Options();
            options.MatrixColumnIndexColumnName = nameof(ProductEntry.ProductID);
            options.MatrixRowIndexColumnName = nameof(ProductEntry.CoPurchaseProductID);
            options.LabelColumnName= "Label";
            options.LossFunction = MatrixFactorizationTrainer.LossFunctionType.SquareLossOneClass;
            options.Alpha = 0.01;
            options.Lambda = 0.025;
            // For better results use the following parameters
            //options.K = 100;
            //options.C = 0.00001;

            //Call the MatrixFactorization trainer by passing options.
            var est = mlContext.Recommendation().Trainers.MatrixFactorization(options);
                    
            //Train the model fitting to the DataSet
            ITransformer model = est.Fit(traindata);
            
            mlContext.Model.Save(model,traindata.Schema,ModelPath);

            return model;
        }

        public static ITransformer Load(MLContext mlContext )
        {
            return mlContext.Model.Load(ModelPath, out DataViewSchema schema);
        }
    }
}