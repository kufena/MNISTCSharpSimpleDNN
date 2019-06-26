using NUnit.Framework;
using CNN;
using MathNet.Numerics.LinearAlgebra;

namespace Tests
{
    public class Tests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void Test1()
        {
            Convolution cv = new Convolution(2, 2);
            var m = Matrix<double>.Build.Dense(5,5,0.5);
            var r = cv.apply(m);
            Assert.Pass();
        }
    }
}