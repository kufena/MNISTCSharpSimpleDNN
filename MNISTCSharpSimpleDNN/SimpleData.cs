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
