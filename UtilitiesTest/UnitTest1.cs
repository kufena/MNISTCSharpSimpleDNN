using NUnit.Framework;
using Utilities.RandomVariables;

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
            GaussianRV rv = new GaussianRV(17, 5);
            double var = 0.0;
            double mean = 0.0;
            double[] scores = new double[10000];
            for (int i = 0; i < 10000; i++)
                scores[i] = rv.next();

            for (int i = 0; i < 10000; i++)
                mean += scores[i];

            mean = mean / 10000.0;
            for(int i = 0; i < 10000; i++)
            {
                double x = scores[i] - mean;
                var += (x * x);
            }

            var = var / 10000.0;
            Assert.AreEqual(17, mean, 0.2);
            Assert.AreEqual(5, var, 0.2);
        }
    }
}