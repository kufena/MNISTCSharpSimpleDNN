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

namespace DNN
{
    public class Utils
    {
        public static Vector<double> haddamardProduct(Vector<double> x, Vector<double> y)
        {
            if (x.Count != y.Count)
                throw new ArgumentException("unequal length vectors passed");

            Vector<double> result = Vector<double>.Build.Dense(x.Count);
            for (int i = 0; i < x.Count; i++)
                result[i] = x[i] * y[i];
            return result;
        }

        public static Vector<double> collapseMatrixSumRows(Matrix<double> m)
        {
            Vector<double> res = Vector<double>.Build.Dense(m.ColumnCount);
            for(int i = 0; i < m.ColumnCount; i++)
                for(int j = 0; j < m.RowCount; j++)
                {
                    res[i] += m[j, i];
                }
            return res;
        }

        public static Vector<double> collapseMatrixSumCols(Matrix<double> m)
        {
            Vector<double> res = Vector<double>.Build.Dense(m.RowCount);
            for (int i = 0; i < m.RowCount; i++)
                for (int j = 0; j < m.ColumnCount; j++)
                {
                    res[i] += m[i,j];
                }
            return res;
        }

    }
}
