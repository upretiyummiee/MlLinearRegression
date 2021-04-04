using Microsoft.ML.Data;

namespace MlLinearRegression
{
    class MyOutput
    {
        [ColumnName("Score")]
        public float BirthRate { get; set; }
    }
}
