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
        public MNISTData(string dir)
        {
            trainImages = new BinaryReader(File.OpenRead(dir + @"\train-images.idx3-ubyte"));
            trainLabels = new BinaryReader(File.OpenRead(dir + @"\train-labels.idx1-ubyte"));

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
                imageD[i] = ((double)imageB[i])/255.0;
            Vector<double> image = Vector<double>.Build.Dense(imageD);
            return (label, image);
        }
    }
}
