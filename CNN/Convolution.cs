/**

The MIT License (MIT)

Copyright (c) 2019 Andrew Douglas. 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

The Software is provided “as is”, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, 
fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other 
liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings 
in the Software.

Line by line description: https://writing.kemitchell.com/2016/09/21/MIT-License-Line-by-Line.html

**/
using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace CNN
{
    public class Convolution
    {

        Matrix<double> convolution;
        int rows;
        int cols;

        public Convolution(int rows, int cols)
        {
            this.rows = rows;
            this.cols = cols;
            convolution = Matrix<double>.Build.Dense(rows, cols, (x, y) => 0.5);
        }

        public Matrix<double> apply(Matrix<double> m)
        {
            if (m.RowCount < rows || m.ColumnCount < cols)
                throw new ArgumentException("applying convolution to a matrix which is too small.");

            var result = Matrix<double>.Build.Dense(m.RowCount - rows + 1, m.ColumnCount - cols + 1);
            int i = 0;
            int j = 0;
            Matrix<double> sub;
            Matrix<double> res = Matrix<double>.Build.Dense(rows, cols);

            while (i < m.ColumnCount - cols + 1)
            {
                while (j < m.RowCount - rows + 1)
                {
                    sub = m.SubMatrix(j, rows, i, cols);
                    sub.PointwiseMultiply(convolution,res);
                    double point = res.ColumnSums().Sum();
                    result[j, i] = point;
                    j++;
                }
                i++;
                j = 0;
            }
            return result;
        }
    }
}
