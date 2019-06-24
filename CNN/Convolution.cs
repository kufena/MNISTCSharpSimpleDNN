using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace CNN
{
    public class Convolution
    {

        Matrix<double> convolution;

        public Convolution(int rows, int cols)
        {
            convolution = Matrix<double>.Build.Dense();
        }
    }
}
