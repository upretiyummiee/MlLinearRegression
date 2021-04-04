using Microsoft.ML.Data;

namespace MlLinearRegression
{
    class MyInput
    {
        [LoadColumn(1)]
        public float PovertyRate { get; set; }

        [LoadColumn(5), ColumnName("Label")]
        public float BirthRate { get; set; }
    }
}
