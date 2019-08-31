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
    class MNISTData
    {

        BinaryReader trainLabels;
        BinaryReader trainImages;
        public MNISTData(string dir) : this(dir + @"/train-images.idx3-ubyte", dir + @"/train-labels.idx1-ubyte")
        { }

        public MNISTData(string imgfile, string labelfile)
        {

            
            trainImages = new BinaryReader(File.OpenRead(imgfile));
            trainLabels = new BinaryReader(File.OpenRead(labelfile));

            trainImages.ReadBytes(4); // magic
            trainImages.ReadBytes(4); // num of images
            trainImages.ReadBytes(4); // rows
            trainImages.ReadBytes(4); // columns

            trainLabels.ReadBytes(4); // magic
            trainLabels.ReadBytes(4); // num of labels

        }

        public (int, Vector<double>) getTrainingImage()
        {
            int label = (int)trainLabels.ReadByte();
            byte[] imageB = trainImages.ReadBytes(28 * 28);
            double[] imageD = new double[imageB.Length];
            for (int i = 0; i < imageB.Length; i++)
                imageD[i] = ((double)imageB[i]) / 255.0;
            Vector<double> image = Vector<double>.Build.Dense(imageD);
            return (label, image);
        }
    }
}
