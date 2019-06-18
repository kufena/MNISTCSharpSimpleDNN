using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using MathNet.Numerics.LinearAlgebra;

namespace MNISTCSharpSimpleDNN
{
    public class SimpleData
    {
        StreamReader sr;

        public SimpleData(string filename)
        {
            FileStream f = File.OpenRead(filename);
            sr = new System.IO.StreamReader(f);
        }

        public (int, Vector<double>) getTrainingImage()
        {
            if (sr.EndOfStream)
                return (-1, Vector<double>.Build.Dense(new double[] { -1, -1, -1 }));

            string s = sr.ReadLine();
            var splits = s.Split(new char[] { '{', '}' }, StringSplitOptions.RemoveEmptyEntries);
            int label;
            if (!Int32.TryParse(splits[1], out label))
                throw new Exception("odd data:: int expected found " + splits[1]);
            var vals = splits[0].Split(new char[] { ',' }, StringSplitOptions.None);
            double[] dbals = new double[vals.Length];
            for(int i = 0; i < vals.Length; i++)
            {
                dbals[i] = double.Parse(vals[i]);
            }
            return (label, Vector<double>.Build.Dense(dbals));
        }
    }
}
