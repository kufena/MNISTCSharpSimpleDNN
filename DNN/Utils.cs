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
