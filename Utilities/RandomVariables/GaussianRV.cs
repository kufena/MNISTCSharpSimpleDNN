using System;
using System.Collections.Generic;
using System.Text;

namespace Utilities.RandomVariables
{
    public class GaussianRV : IRandomVariable
    {
        double mean { get; set; }
        double var { get; set; }

        private Random rand;
        private double sd;
        private double sqrt2 = Math.Sqrt(2.0);
        private const double a = 0.140012;

        public GaussianRV(double mean, double variance)
        {
            this.mean = mean;
            this.var = variance;
            this.sd = Math.Sqrt(variance);
            this.rand = new Random();
        }

        public GaussianRV(double mean, double variance, int seed) : this(mean, variance)
        {
            this.rand = new Random(seed);
        }

        public double next()
        {
            double u = rand.NextDouble();
            double x = sqrt2 * inv_erf((2 * u) - 1);
            return mean + (sd * x);
        }

        private double inv_erf(double x)
        {
            double minusxsquared = 1 - (x * x);
            double twoover = 2.0 / (Math.PI * a);
            double logged = Math.Log(minusxsquared);
            double mid1 = twoover + (logged / 2);
            double mid2 = (mid1 * mid1) - (logged / a);
            double mid3 = Math.Sqrt(mid2) - (twoover + (logged / 2));
            double mid4 = Math.Sqrt(mid3);
            return Math.Sign(x) * mid4;
        }
    }
}
